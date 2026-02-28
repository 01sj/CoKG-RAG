#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
面向中文法律知识图谱的检索脚本

默认从 /newdataf/SJ/LeanRAG/law_kg_output_processed/ 读取：
  - entity.jsonl / relation.jsonl
  - 依赖已完成的层级聚类、关系与向量索引（由 build_law_graph.py / build_graph.py 生成）

功能：
  1) 向量检索 Top-K 实体/节点
  2) 基于候选实体构造推理路径并聚合社区信息
  3) 结合 chunks 提取文本单元
  4) 组织上下文交给 LLM 生成最终回答

注意：本脚本默认在 CPU 上串行做 embedding（稳定优先）。如需 GPU/并发，可通过环境变量覆盖。
"""

import argparse
import json
import os
from itertools import combinations
from collections import defaultdict

import numpy as np
import tiktoken
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from vllm import LLM, SamplingParams
from database_utils import (
    search_vector_search,
    find_tree_root,
    search_nodes_link,
    search_community,
    get_text_units,
)
from prompt import PROMPTS


# ---------- 环境/设备与 embedding ----------
_force_cpu = os.environ.get("FORCE_CPU", "1") == "1"  # 缺省 CPU，更稳
_device = "cpu" if _force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
_st_model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

try:
    _ST_EMB = SentenceTransformer(_st_model_name, device=_device)
except Exception as e:
    print(f"Failed to load {_st_model_name} on {_device}: {e}")
    print("Falling back to CPU + BAAI/bge-m3")
    _device = "cpu"
    _ST_EMB = SentenceTransformer("BAAI/bge-m3", device=_device)

_ST_EMB.max_seq_length = 4096

_emb_batch = max(1, int(os.environ.get("EMB_BATCH", "8")))

tokenizer = tiktoken.get_encoding("cl100k_base")


def truncate_text(text: str, max_tokens: int = 4096) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)


def embedding(texts) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    batch_size = max(1, min(_emb_batch, len(texts)))
    vectors = _ST_EMB.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    result = np.array(vectors)
    if len(texts) == 1:
        return [result[0].tolist()]
    return result


# ---------- 推理链与社区聚合 ----------
def get_reasoning_chain(working_dir: str, entities_set: list[str]):
    maybe_edges = list(combinations(entities_set, 2))
    reasoning_path = []
    reasoning_path_information = []
    db_name = os.path.basename(working_dir.rstrip("/"))

    for edge in maybe_edges:
        a_path = []
        b_path = []
        node1 = edge[0]
        node2 = edge[1]
        node1_tree = find_tree_root(db_name, node1)
        node2_tree = find_tree_root(db_name, node2)

        for i, j in zip(node1_tree, node2_tree):
            if i == j:
                a_path.append(i)
                break
            if i in b_path or j in a_path:
                break
            if i != j:
                a_path.append(i)
                b_path.append(j)

        reasoning_path.append(a_path + [b_path[len(b_path) - 1 - i] for i in range(len(b_path))])
        a_path = list(set(a_path))
        b_path = list(set(b_path))
        for maybe_edge in list(combinations(a_path + b_path, 2)):
            if maybe_edge[0] == maybe_edge[1]:
                continue
            information = search_nodes_link(maybe_edge[0], maybe_edge[1], working_dir)
            if information is None:
                continue
            reasoning_path_information.append([maybe_edge[0], maybe_edge[1], information[2]])

    temp_relations_information = list(set([info[2] for info in reasoning_path_information]))
    reasoning_path_information_description = "\n".join(temp_relations_information)
    return reasoning_path, reasoning_path_information_description


def get_aggregation_description(working_dir: str, reasoning_path):
    aggregation_results = []
    communities = set([community for each_path in reasoning_path for community in each_path])
    for community in communities:
        temp = search_community(community, working_dir)
        if temp == "":
            continue
        aggregation_results.append(temp)

    columns = ["entity_name", "entity_description"]
    aggregation_descriptions = "\t\t".join(columns) + "\n"
    aggregation_descriptions += "\n".join([info[0] + "\t\t" + str(info[1]) for info in aggregation_results])
    return aggregation_descriptions, communities


def search_article_in_chunks(chunks_file: str, article_pattern: str, max_results: int = 5) -> list[dict]:
    """
    在文本块中直接搜索包含指定法条的内容
    
    参数:
        chunks_file: 文本块文件路径
        article_pattern: 法条模式（如"第一条"、"第二十三条"）
        max_results: 最大返回结果数
    
    返回:
        包含法条的文本块列表，每个元素包含 hash_code 和 text
    """
    import json
    import re
    
    results = []
    
    if not os.path.exists(chunks_file):
        return results
    
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # 处理不同的数据格式
        if isinstance(chunks_data, list):
            chunks_list = chunks_data
        elif isinstance(chunks_data, dict):
            chunks_list = [{"hash_code": k, "text": v} for k, v in chunks_data.items()]
        else:
            return results
        
        # 在文本块中搜索包含法条的内容
        for chunk in chunks_list:
            if not isinstance(chunk, dict):
                continue
            
            text = chunk.get("text", "")
            hash_code = chunk.get("hash_code", "")
            
            # 检查文本中是否包含法条模式
            # 支持多种格式：第一条、第1条、第一条（标题）等
            if article_pattern in text:
                # 提取包含法条的完整段落
                # 尝试提取法条及其后续内容（直到下一个法条或段落结束）
                # 使用传入的 article_pattern 作为起点，匹配到下一个法条或文本结束
                pattern = re.escape(article_pattern) + r'[^第]*?(?=第[一二三四五六七八九十百千万零\d]+条|$)'
                match = re.search(pattern, text, re.DOTALL)
                
                if match:
                    article_text = match.group(0).strip()
                    results.append({
                        "hash_code": hash_code,
                        "text": article_text,
                        "match_type": "direct_search"
                    })
                else:
                    # 如果正则匹配失败，至少包含法条模式的文本块
                    results.append({
                        "hash_code": hash_code,
                        "text": text,
                        "match_type": "contains_pattern"
                    })
                
                if len(results) >= max_results:
                    break
        
        # 按相关性排序（包含完整法条内容的优先）
        results.sort(key=lambda x: 1 if x["match_type"] == "direct_search" else 2)
        
    except Exception as e:
        print(f"Warning: 搜索法条时出错: {e}")
    
    return results


def get_entity_description(entity_results: list[tuple]):
    columns = ["entity_name", "parent", "description"]
    entity_descriptions = "\t\t".join(columns) + "\n"
    entity_descriptions += "\n".join([info[0] + "\t\t" + info[1] + "\t\t" + info[2] for info in entity_results])
    return entity_descriptions


def query_law_graph(global_config: dict, query: str, return_structured: bool = False):
    """
    查询法律知识图谱并生成答案
    
    参数:
        global_config: 全局配置字典
        query: 查询问题
        return_structured: 是否返回结构化结果（包含文本块等详细信息）
    
    返回:
        如果 return_structured=False: (describe, response) - 描述文本和LLM回答
        如果 return_structured=True: dict - 包含所有检索信息和文本块的字典
    """
    use_llm_func = global_config["use_llm_func"]
    working_dir = global_config["working_dir"]
    level_mode = global_config.get("level_mode", 1)
    topk = global_config.get("topk", 10)
    chunks_file = global_config.get("chunks_file")

    # 标准化路径并打印调试信息
    working_dir = os.path.normpath(working_dir.rstrip("/"))
    print(f"[DEBUG] Query working directory: {working_dir}")

    # 1. 向量检索相关实体
    print(f"[步骤 1/9] 生成查询向量...")
    query_embedding = embedding(query)
    print(f"[步骤 2/9] 向量检索 Top-{topk} 实体...")
    entity_results = search_vector_search(working_dir, query_embedding, topk=topk, level_mode=level_mode)
    print(f"✅ 检索到 {len(entity_results)} 个相关实体")
    res_entity = [i[0] for i in entity_results]
    chunks = [i[-1] for i in entity_results]  # 获取实体的 source_id（对应文本块的 hash_code）

    # 2. 获取实体描述
    print(f"[步骤 3/9] 获取实体描述...")
    entity_descriptions = get_entity_description(entity_results)
    
    # 3. 构建推理路径
    print(f"[步骤 4/9] 构建推理路径...")
    reasoning_path, reasoning_path_information_description = get_reasoning_chain(working_dir, res_entity)
    print(f"✅ 找到 {len(reasoning_path)} 条推理路径")
    
    # 4. 获取聚合实体描述
    print(f"[步骤 5/9] 获取聚合实体描述...")
    aggregation_descriptions, aggregation = get_aggregation_description(working_dir, reasoning_path)
    
    # 5. 检测是否为法条查询
    is_article_query = any(keyword in query for keyword in ["法条", "第", "条", "条款", "条文", "规定", "内容"])
    
    # 5.1 如果是法条查询，尝试从文本块中直接搜索法条
    direct_article_chunks = []
    article_pattern = None
    if is_article_query and chunks_file and os.path.exists(chunks_file):
        # 提取法条编号（如"第一条"、"第二十三条"、"第1条"、"第23条"等）
        import re
        # 匹配中文数字+条，如：第一条、第二十三条、第一百条等
        # 也匹配阿拉伯数字+条，如：第1条、第23条等
        article_match = re.search(r'第[一二三四五六七八九十百千万零\d]+条', query)
        if article_match:
            article_pattern = article_match.group(0)
            print(f"[DEBUG] 检测到法条查询: {article_pattern}")
            # 在文本块中直接搜索包含该法条的内容
            direct_article_chunks = search_article_in_chunks(chunks_file, article_pattern)
            if direct_article_chunks:
                print(f"[DEBUG] 在文本块中找到 {len(direct_article_chunks)} 个包含该法条的文本块")
    
    # 6. 获取文本块（实体原本的文本块）
    print(f"[步骤 6/9] 获取文本块...")
    # 如果是法条查询，增加文本块数量以确保包含完整法条内容
    text_units_k = global_config.get("text_units_k")
    if text_units_k is None:
        text_units_k = 10 if is_article_query else 5
    text_units_from_entities = get_text_units(working_dir, chunks, chunks_file, k=text_units_k)
    print(f"✅ 获取到文本块 (长度: {len(text_units_from_entities)} 字符)")
    
    # 6.1 如果通过直接搜索找到了法条文本块，优先使用这些文本块
    if direct_article_chunks and article_pattern:
        # 只使用直接搜索到的法条文本块（这些已经过滤过，只包含查询的法条）
        direct_text_units = "\n".join([chunk["text"] for chunk in direct_article_chunks])
        print(f"[DEBUG] 使用直接搜索到的法条文本块（只包含查询的法条）")
        # 对于法条查询，优先使用直接搜索的结果，避免包含其他法条
        text_units = direct_text_units
        # 如果直接搜索的结果中没有找到完整的法条内容，再尝试从实体检索的结果中查找
        if article_pattern not in direct_text_units:
            print(f"[DEBUG] 直接搜索未找到完整法条，尝试从实体检索结果中查找")
            # 从实体检索的 text_units 中提取包含查询法条的部分
            import re
            pattern = re.escape(article_pattern) + r'[^第]*?(?=第[一二三四五六七八九十百千万零\d]+条|$)'
            match = re.search(pattern, text_units_from_entities, re.DOTALL)
            if match:
                extracted_article = match.group(0).strip()
                text_units = extracted_article
                print(f"[DEBUG] 从实体检索结果中提取到法条内容")
            else:
                # 如果还是找不到，使用直接搜索的结果（至少包含法条模式）
                text_units = direct_text_units
    else:
        # 如果不是法条查询或没有找到直接搜索的结果，使用实体检索的结果
        text_units = text_units_from_entities
    
    # 7. 获取详细的文本块信息（用于结构化返回）
    text_chunks_detail = []
    if chunks_file and os.path.exists(chunks_file):
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            chunks_dict = {item["hash_code"]: item["text"] for item in chunks_data}
            
            # 统计每个 chunk 的出现次数
            from collections import Counter
            chunks_list = []
            for chunk_id in chunks:
                if "|" in chunk_id:
                    chunks_list.extend(chunk_id.split("|"))
                else:
                    chunks_list.append(chunk_id)
            counter = Counter(chunks_list)
            
            # 获取最相关的文本块（按出现次数排序）
            sorted_chunks = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            for chunk_id, count in sorted_chunks[:topk]:
                if chunk_id in chunks_dict:
                    text_chunks_detail.append({
                        "hash_code": chunk_id,
                        "text": chunks_dict[chunk_id],
                        "relevance_count": count  # 该文本块被多少个相关实体引用
                    })
        except Exception as e:
            print(f"Warning: Failed to load chunks file for detailed info: {e}")

    # 8. 组织上下文描述（如果是法条查询，添加特殊标记）
    print(f"[步骤 7/9] 组织上下文...")
    # 如果是法条查询，在 describe 中明确标注用户查询的法条
    if is_article_query and article_pattern:
        describe = f"""
    用户查询的法条: {article_pattern}
    
    entity_information:
    {entity_descriptions}
    aggregation_entity_information:
    {aggregation_descriptions}
    reasoning_path_information:
    {reasoning_path_information_description}
    text_units:
    {text_units}
    
    重要提示: 用户询问的是"{article_pattern}"，请只返回该法条的内容，不要返回 text_units 中的其他法条。
    """
    else:
        describe = f"""
    entity_information:
    {entity_descriptions}
    aggregation_entity_information:
    {aggregation_descriptions}
    reasoning_path_information:
    {reasoning_path_information_description}
    text_units:
    {text_units}
    """

    # 8.1 检查并截断过长的上下文，避免超过 max_model_len
    # 预留空间给 prompt 模板和生成（约 2000 tokens）
    no_truncate = global_config.get("no_truncate", False)
    max_context_tokens = global_config.get("max_model_len", 32768) - 2000
    describe_tokens = len(tokenizer.encode(describe))
    
    if no_truncate:
        print(f"ℹ️  禁用截断模式：上下文长度 {describe_tokens} tokens，最大支持 {max_context_tokens + 2000} tokens")
        if describe_tokens > max_context_tokens + 2000:
            print(f"⚠️  警告: 上下文长度 ({describe_tokens} tokens) 超过模型最大长度 ({max_context_tokens + 2000} tokens)，可能导致错误")
    elif describe_tokens > max_context_tokens:
        print(f"⚠️  警告: 上下文长度 ({describe_tokens} tokens) 超过限制 ({max_context_tokens} tokens)，进行截断")
        # 按优先级截断：优先保留 text_units 和 entity_information
        # 1. 先截断 reasoning_path_information（通常较长且重要性较低）
        if len(reasoning_path_information_description) > 0:
            reasoning_tokens = len(tokenizer.encode(reasoning_path_information_description))
            if reasoning_tokens > max_context_tokens * 0.3:  # 如果推理路径信息超过30%，进行截断
                # 只保留前一部分
                reasoning_lines = reasoning_path_information_description.split('\n')
                truncated_reasoning = []
                current_tokens = 0
                max_reasoning_tokens = int(max_context_tokens * 0.2)  # 最多保留20%的tokens给推理路径
                for line in reasoning_lines:
                    line_tokens = len(tokenizer.encode(line))
                    if current_tokens + line_tokens <= max_reasoning_tokens:
                        truncated_reasoning.append(line)
                        current_tokens += line_tokens
                    else:
                        break
                reasoning_path_information_description = '\n'.join(truncated_reasoning)
                print(f"   截断推理路径信息，保留 {len(truncated_reasoning)}/{len(reasoning_lines)} 行")
        
        # 2. 重新组装 describe 并检查是否还需要进一步截断
        if is_article_query and article_pattern:
            describe_temp = f"""
    用户查询的法条: {article_pattern}
    
    entity_information:
    {entity_descriptions}
    aggregation_entity_information:
    {aggregation_descriptions}
    reasoning_path_information:
    {reasoning_path_information_description}
    text_units:
    {text_units}
    
    重要提示: 用户询问的是"{article_pattern}"，请只返回该法条的内容，不要返回 text_units 中的其他法条。
    """
        else:
            describe_temp = f"""
    entity_information:
    {entity_descriptions}
    aggregation_entity_information:
    {aggregation_descriptions}
    reasoning_path_information:
    {reasoning_path_information_description}
    text_units:
    {text_units}
    """
        
        describe_tokens = len(tokenizer.encode(describe_temp))
        
        # 3. 如果还是太长，截断 aggregation_entity_information
        if describe_tokens > max_context_tokens:
            if len(aggregation_descriptions) > 0:
                # 只保留前几个聚合实体
                agg_lines = aggregation_descriptions.split('\n')
                truncated_agg = agg_lines[:min(10, len(agg_lines))]  # 最多保留10行
                aggregation_descriptions = '\n'.join(truncated_agg)
                print(f"   截断聚合实体信息，保留 {len(truncated_agg)}/{len(agg_lines)} 行")
                # 重新组装
                if is_article_query and article_pattern:
                    describe_temp = f"""
    用户查询的法条: {article_pattern}
    
    entity_information:
    {entity_descriptions}
    aggregation_entity_information:
    {aggregation_descriptions}
    reasoning_path_information:
    {reasoning_path_information_description}
    text_units:
    {text_units}
    
    重要提示: 用户询问的是"{article_pattern}"，请只返回该法条的内容，不要返回 text_units 中的其他法条。
    """
                else:
                    describe_temp = f"""
    entity_information:
    {entity_descriptions}
    aggregation_entity_information:
    {aggregation_descriptions}
    reasoning_path_information:
    {reasoning_path_information_description}
    text_units:
    {text_units}
    """
                describe_tokens = len(tokenizer.encode(describe_temp))
        
        # 4. 如果还是太长，截断 text_units（保留最重要的部分）
        if describe_tokens > max_context_tokens:
            if len(text_units) > 0:
                # 截断 text_units，保留前面的部分
                text_units_tokens = len(tokenizer.encode(text_units))
                max_text_units_tokens = int(max_context_tokens * 0.4)  # 最多保留40%的tokens给文本块
                if text_units_tokens > max_text_units_tokens:
                    # 保留前一部分的文本块
                    text_units_lines = text_units.split('\n')
                    truncated_text_units = []
                    current_tokens = 0
                    for line in text_units_lines:
                        line_tokens = len(tokenizer.encode(line))
                        if current_tokens + line_tokens <= max_text_units_tokens:
                            truncated_text_units.append(line)
                            current_tokens += line_tokens
                        else:
                            break
                    text_units = '\n'.join(truncated_text_units)
                    print(f"   截断文本块，保留 {len(truncated_text_units)}/{len(text_units_lines)} 行")
                    # 重新组装
                    if is_article_query and article_pattern:
                        describe_temp = f"""
    用户查询的法条: {article_pattern}
    
    entity_information:
    {entity_descriptions}
    aggregation_entity_information:
    {aggregation_descriptions}
    reasoning_path_information:
    {reasoning_path_information_description}
    text_units:
    {text_units}
    
    重要提示: 用户询问的是"{article_pattern}"，请只返回该法条的内容，不要返回 text_units 中的其他法条。
    """
                    else:
                        describe_temp = f"""
    entity_information:
    {entity_descriptions}
    aggregation_entity_information:
    {aggregation_descriptions}
    reasoning_path_information:
    {reasoning_path_information_description}
    text_units:
    {text_units}
    """
                    describe_tokens = len(tokenizer.encode(describe_temp))
        
        # 使用最终的 describe
        describe = describe_temp
        final_tokens = len(tokenizer.encode(describe))
        print(f"   截断后上下文长度: {final_tokens} tokens")
    else:
        print(f"[步骤 8/9] 上下文长度检查通过 ({describe_tokens} tokens)")

    # 9. 生成LLM回答
    print(f"[步骤 9/9] 调用 LLM 生成回答...")
    # 优先使用中文版本的 prompt，如果没有则使用英文版本
    # 如果是法条查询，使用专门的法条查询 prompt
    if is_article_query:
        rag_prompt = PROMPTS.get("rag_response_article_zh", PROMPTS.get("rag_response_zh", PROMPTS.get("rag_response", "")))
    else:
        rag_prompt = PROMPTS.get("rag_response_zh", PROMPTS.get("rag_response", ""))
    sys_prompt = rag_prompt.format(context_data=describe)
    
    # 调试：打印文本块信息（可选）
    if global_config.get("debug", False):
        print(f"\n[DEBUG] 文本块数量: {len(text_chunks_detail)}")
        print(f"[DEBUG] 文本块总长度: {len(text_units)} 字符")
        print(f"[DEBUG] 上下文总长度: {len(describe)} 字符")
        if text_units:
            print(f"[DEBUG] 文本块预览:\n{text_units[:500]}...")
    
    response = use_llm_func(query, system_prompt=sys_prompt)
    print(f"✅ LLM 回答生成完成 (长度: {len(response)} 字符)")
    
    # 10. 根据参数决定返回格式
    if return_structured:
        return {
            "query": query,
            "answer": response,
            "retrieved_entities": [
                {
                    "entity_name": i[0],
                    "parent": i[1],
                    "description": i[2],
                    "source_ids": i[-1] if len(i) > 3 else ""  # source_id 可能包含多个 chunk hash
                }
                for i in entity_results
            ],
            "text_chunks": text_chunks_detail,  # 实体原本的文本块
            "reasoning_path": reasoning_path,
            "reasoning_path_information": reasoning_path_information_description,
            "aggregation_entities": list(aggregation) if aggregation else [],
            "context_summary": describe  # 完整的上下文摘要
        }
    else:
        return describe, response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="/newdataf/SJ/LeanRAG/basicLaw_doc_output_7B_processed/",
        help="法律知识图谱工作目录（包含生成的向量/社区/关系等）",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default=None,
        help="chunks.json 路径（可选，不提供会影响文本证据提取）",
    )
    parser.add_argument("-q", "--query", type=str, required=False, help="查询问题（中文/英文均可）。若提供 --input-json 可不填")
    parser.add_argument("--input-json", type=str, default=None, help="批量查询的输入 JSON 文件路径（数组，每条包含 question 字段）")
    parser.add_argument("--output-json", type=str, default=None, help="批量查询的输出 JSON 文件路径（写入 prediction 字段）")
    parser.add_argument("-k", "--topk", type=int, default=10, help="检索 Top-K 实体/节点")
    parser.add_argument("-l", "--level", type=int, default=2, help="检索层级：0原始节点/1聚合节点/2全部")
    parser.add_argument("-n", "--num", type=int, default=1, help="LLM 并发实例数量（用于生成回答）")
    parser.add_argument("--model", type=str, default=os.environ.get("VLLM_MODEL", "/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct"), help="vLLM 本地模型名称或权重路径（默认：/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct）")
    parser.add_argument("--tp", type=int, default=int(os.environ.get("VLLM_TP", "2")), help="vLLM 张量并行度 tensor_parallel_size（默认2，使用两张GPU卡；可设置为4使用四张卡以支持更长序列）")
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("VLLM_MAX_NEW_TOKENS", "512")), help="生成的最大新token数")
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("VLLM_TEMPERATURE", "0.3")), help="采样温度（DeepSeek-R1推荐0.3，Qwen推荐0.2）")
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("VLLM_TOP_P", "0.9")), help="top_p 采样阈值")
    parser.add_argument("--gpu-mem-util", type=float, default=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.8")), help="每张 GPU 目标显存占用比例，降低可缓解 OOM")
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384")), help="最大模型序列长度（默认16384，如需更长上下文可调高，显存不足时可降到8192）")
    parser.add_argument("--no-truncate", action="store_true", help="禁用上下文截断（保留完整上下文，需要足够的显存和 max_model_len）")
    parser.add_argument("--quantization", type=str, default=os.environ.get("VLLM_QUANT", None), help="量化方式，如 awq/gptq/bitsandbytes（根据模型提供情况）")
    parser.add_argument("--dtype", type=str, default=os.environ.get("VLLM_DTYPE", None), help="精度：bfloat16/float16/float32（不指定则由 vLLM 自动选择）")
    parser.add_argument("--structured", action="store_true", help="返回结构化结果（包含文本块等详细信息）")
    parser.add_argument("--text-units-k", type=int, default=None, help="文本块数量（默认：法条查询10个，其他5个）")
    args = parser.parse_args()

    working_dir = os.path.normpath(args.path.rstrip("/"))
    chunks_file = args.chunks

    # 打印路径信息用于调试
    print(f"[DEBUG] Command line path argument: {args.path}")
    print(f"[DEBUG] Normalized working directory: {working_dir}")

    # 模型路径检测和配置
    # 优先使用环境变量指定的模型路径
    if os.environ.get('VLLM_MODEL_PATH'):
        model_path = os.environ.get('VLLM_MODEL_PATH')
        print(f"📁 使用环境变量指定的模型路径: {model_path}")
    elif os.path.exists(args.model):
        # 如果参数是存在的路径，直接使用
        model_path = args.model
        print(f"📁 使用指定的本地模型路径: {model_path}")
    elif args.model.startswith('/'):
        # 如果是绝对路径但不存在，仍然尝试使用（可能路径配置错误）
        model_path = args.model
        print(f"⚠️  警告: 指定的模型路径不存在: {model_path}")
        print(f"   将尝试使用该路径（如果失败，vLLM 可能会报错）")
    else:
        # 尝试从本地缓存中查找模型
        # 模型缓存目录列表（按优先级排序）
        model_cache_dirs = [
            "/root/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct",   # 用户指定的模型
            "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct",  # 备选模型
            "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat",  # 备选模型
            "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B",  # 备选模型
        ]
        
        model_path = None
        found_model = None
        
        # 首先检查用户指定的模型（Qwen2-7B-Instruct）
        target_cache_dir = "/root/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct"
        if os.path.exists(target_cache_dir):
            snapshots_dir = os.path.join(target_cache_dir, "snapshots")
            if os.path.exists(snapshots_dir):
                try:
                    snapshots = [d for d in os.listdir(snapshots_dir) 
                                if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        # 按修改时间排序，取最新的
                        snapshots.sort(key=lambda x: os.path.getmtime(os.path.join(snapshots_dir, x)), reverse=True)
                        model_path = os.path.join(snapshots_dir, snapshots[0])
                        found_model = "Qwen2-7B-Instruct"
                        print(f"✅ 找到本地缓存模型: {found_model}")
                        print(f"📁 模型路径: {model_path}")
                except Exception as e:
                    print(f"⚠️  检测模型路径时出错 ({target_cache_dir}): {e}")
        
        # 如果没找到，尝试其他模型
        if model_path is None:
            for cache_dir in model_cache_dirs:
                if os.path.exists(cache_dir):
                    snapshots_dir = os.path.join(cache_dir, "snapshots")
                    if os.path.exists(snapshots_dir):
                        try:
                            snapshots = [d for d in os.listdir(snapshots_dir) 
                                        if os.path.isdir(os.path.join(snapshots_dir, d))]
                            if snapshots:
                                # 按修改时间排序，取最新的
                                snapshots.sort(key=lambda x: os.path.getmtime(os.path.join(snapshots_dir, x)), reverse=True)
                                model_path = os.path.join(snapshots_dir, snapshots[0])
                                found_model = os.path.basename(cache_dir)
                                print(f"✅ 找到可用模型: {found_model}")
                                print(f"📁 模型路径: {model_path}")
                                break
                        except Exception as e:
                            print(f"⚠️  检测模型路径时出错 ({cache_dir}): {e}")
                            continue
        
        # 如果所有路径都不存在，使用命令行参数指定的路径或模型名称
        if model_path is None:
            model_path = args.model  # 使用命令行参数指定的模型路径或名称
            if os.path.exists(model_path):
                print(f"✅ 使用指定的本地模型路径: {model_path}")
            elif model_path.startswith('/'):
                print(f"⚠️  警告: 指定的模型路径不存在: {model_path}")
                print(f"   将尝试使用该路径（如果失败，vLLM 可能会报错）")
            else:
                print(f"⚠️  本地路径不存在，将使用 HuggingFace 模型名称: {model_path}")
                print(f"💡 提示: 如果模型未下载，vLLM 会自动从 HuggingFace 下载")

    # vLLM 本地引擎
    # 默认使用两张GPU卡（tensor_parallel_size=2），可以支持更长的序列长度
    max_model_len = args.max_model_len
    tensor_parallel_size = max(1, args.tp)
    
    # 检查 max_model_len 是否超过模型限制
    # Qwen2-7B-Instruct 的最大长度是 32768
    # 如果用户设置的值超过 32768，需要设置环境变量允许覆盖
    if max_model_len > 32768:
        # 设置环境变量允许超过模型限制（需要谨慎使用）
        if not os.environ.get('VLLM_ALLOW_LONG_MAX_MODEL_LEN'):
            os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
            print(f"⚠️  警告: max_model_len ({max_model_len}) 超过模型默认最大长度 (32768)")
            print(f"   已设置 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 以允许覆盖")
            print(f"   注意: 如果模型使用 RoPE 位置编码，超过限制可能导致 NaN")
            print(f"   如果模型使用绝对位置编码，超过限制可能导致 CUDA 错误")
    
    print(f"📏 最大序列长度: {max_model_len}")
    
    # 设置使用第1张GPU卡（索引为0）
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        print(f"✅ 设置 CUDA_VISIBLE_DEVICES=1，使用第2张GPU卡")
        # 如果只使用1张卡，自动调整 tensor_parallel_size 为1
        if tensor_parallel_size > 1:
            print(f"⚠️  检测到只使用1张GPU卡，自动将 tensor_parallel_size 从 {tensor_parallel_size} 调整为 1")
            tensor_parallel_size = 1
    else:
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        print(f"ℹ️  使用环境变量指定的 CUDA_VISIBLE_DEVICES={cuda_devices}")
        # 检查可见的GPU数量，如果少于 tensor_parallel_size，则调整
        visible_gpu_count = len([x for x in cuda_devices.split(',') if x.strip()])
        if visible_gpu_count < tensor_parallel_size:
            print(f"⚠️  可见GPU数量 ({visible_gpu_count}) 少于 tensor_parallel_size ({tensor_parallel_size})，自动调整为 {visible_gpu_count}")
            tensor_parallel_size = visible_gpu_count
    
    # 在确定 tensor_parallel_size 之后，根据实际使用的卡数设置显存利用率
    # 单卡时使用 0.8（默认），双卡 0.75，4 卡 0.85；若命令行更低/更高会取 min/max 范围
    if tensor_parallel_size >= 4:
        gpu_mem_util = min(0.85, max(0.1, args.gpu_mem_util))  # 4卡时可以使用更高的显存利用率
        print(f"✅ 使用 {tensor_parallel_size} 张 GPU 卡，显存利用率: {gpu_mem_util}")
    elif tensor_parallel_size >= 2:
        gpu_mem_util = min(0.75, max(0.1, args.gpu_mem_util))  # 双卡时使用中等显存利用率
        print(f"✅ 使用 {tensor_parallel_size} 张 GPU 卡，显存利用率: {gpu_mem_util}")
    else:
        gpu_mem_util = min(0.8, max(0.1, args.gpu_mem_util))  # 单卡时默认提高到 0.8，避免 KV cache 为负
        print(f"✅ 使用 {tensor_parallel_size} 张 GPU 卡，显存利用率: {gpu_mem_util}")
    
    llm = LLM(
        model=model_path,  # 使用检测到的模型路径
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,  # 限制最大序列长度，避免显存不足
        quantization=(args.quantization or None),
        dtype=(args.dtype or "auto"),
    )

    sampling_params = SamplingParams(
        temperature=max(0.0, args.temperature),
        top_p=min(1.0, max(0.0, args.top_p)),
        max_tokens=max(1, args.max_new_tokens),
        repetition_penalty=1.1,  # 添加重复惩罚，防止重复生成
        stop=["---", "## 用户:", "## 法条查询"],  # 添加停止序列，防止重复格式
    )

    def vllm_generate_text(user_prompt: str, system_prompt: str = ""):
        # 简单指令拼接，适配大多数 Instruct 模型
        composed = (system_prompt.strip() + "\n\n" + user_prompt.strip()).strip()
        outputs = llm.generate([composed], sampling_params=sampling_params)
        if not outputs:
            return ""
        response_text = outputs[0].outputs[0].text
        
        # 后处理：去除重复内容
        import re
        
        # 如果回答中包含重复的"## 法条查询"标记，只保留第一次出现的内容
        if "## 法条查询" in response_text:
            # 找到第一个"## 法条查询"之前的所有内容（如果没有"## 法条查询"标记，保留全部）
            parts = response_text.split("## 法条查询")
            if len(parts) > 1:
                # 保留第一个部分（第一个"## 法条查询"之前的内容）
                response_text = parts[0].strip()
        
        # 如果回答中包含重复的"### 用户询问"标记，只保留第一次出现的内容
        if "### 用户询问" in response_text:
            # 找到第一个"### 用户询问"之前的所有内容
            parts = response_text.split("### 用户询问")
            if len(parts) > 1:
                # 保留第一个部分
                response_text = parts[0].strip()
        
        # 如果回答中包含重复的"---"分隔符，只保留第一个完整回答
        if response_text.count("---") > 2:
            # 找到第一个"---"到第二个"---"之间的内容
            parts = response_text.split("---")
            if len(parts) >= 3:
                # 保留第一个完整回答（第一部分 + 第二部分）
                response_text = (parts[0] + "---" + parts[1]).strip()
        
        # 检测并去除完全重复的段落（如果整个回答重复了多次）
        # 通过检测"## 法条内容"出现的次数来判断
        if response_text.count("## 法条内容") > 1:
            # 找到第一个"## 法条内容"到第二个"## 法条内容"之间的内容
            parts = response_text.split("## 法条内容")
            if len(parts) >= 3:
                # 保留第一个完整的法条内容块
                response_text = ("## 法条内容" + parts[1]).strip()
        
        return response_text

    global_config = {
        "working_dir": working_dir,
        "chunks_file": chunks_file,
        "embeddings_func": embedding,
        "use_llm_func": vllm_generate_text,
        "topk": max(1, args.topk),
        "level_mode": max(0, min(2, args.level)),
        "text_units_k": args.text_units_k,  # 文本块数量
        "max_model_len": max_model_len,  # 传递给 query_law_graph，用于上下文截断
        "no_truncate": args.no_truncate,  # 是否禁用截断
    }

    # 批处理模式（优先）
    if args.input_json:
        input_path = args.input_json
        output_path = args.output_json or (os.path.splitext(input_path)[0] + "_pred.json")
        print(f"\n{'='*70}")
        print(f"📂 读取输入文件: {input_path}")
        print(f"{'='*70}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("输入 JSON 必须是数组，每个元素为包含 question 的对象")
        print(f"✅ 成功读取 {len(data)} 个问题")
        print(f"📝 输出文件: {output_path}")
        print(f"{'='*70}\n")
        results = []
        for idx, item in enumerate(tqdm(data, desc="处理问题", unit="个"), 1):
            question = item.get("question", "").strip()
            if not question:
                # 空问题直接透传
                new_item = dict(item)
                new_item["prediction"] = ""
                results.append(new_item)
                continue
            print(f"\n[{idx}/{len(data)}] 处理问题: {question[:50]}{'...' if len(question) > 50 else ''}")
            try:
                result = query_law_graph(global_config, question, return_structured=args.structured)
                new_item = dict(item)
                if args.structured:
                    # 结构化返回：保存完整结果
                    new_item["prediction"] = result["answer"]
                    new_item["retrieved_entities"] = result["retrieved_entities"]
                    new_item["text_chunks"] = result["text_chunks"]
                    new_item["reasoning_path"] = result["reasoning_path"]
                else:
                    # 简单返回：只保存答案
                    _, resp = result
                    new_item["prediction"] = resp
                results.append(new_item)
                print(f"✅ 问题 {idx} 处理完成")
            except Exception as e:
                print(f"❌ 处理问题 {idx} 时出错: {e}")
                import traceback
                traceback.print_exc()
                # 即使出错也保存空结果，继续处理下一个
                new_item = dict(item)
                new_item["prediction"] = f"[错误: {str(e)}]"
                results.append(new_item)
        print(f"\n{'='*70}")
        print(f"💾 保存结果到: {output_path}")
        print(f"{'='*70}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ 成功保存 {len(results)} 条结果到 {output_path}")
        print(f"{'='*70}\n")
        return

    # 单问模式
    if not args.query:
        raise SystemExit("必须提供 -q/--query，或提供 --input-json 进行批处理")
    
    result = query_law_graph(global_config, args.query, return_structured=args.structured)
    
    if args.structured:
        # 结构化输出
        print("\n" + "=" * 70)
        print("[查询问题]")
        print("=" * 70)
        print(result["query"])
        
        print("\n" + "=" * 70)
        print("[检索到的实体]")
        print("=" * 70)
        for i, entity in enumerate(result["retrieved_entities"], 1):
            print(f"\n{i}. {entity['entity_name']}")
            print(f"   父节点: {entity['parent']}")
            print(f"   描述: {entity['description'][:100]}..." if len(entity['description']) > 100 else f"   描述: {entity['description']}")
            print(f"   来源ID: {entity['source_ids']}")
        
        print("\n" + "=" * 70)
        print("[实体原本的文本块]")
        print("=" * 70)
        for i, chunk in enumerate(result["text_chunks"], 1):
            print(f"\n[文本块 {i}] (hash: {chunk['hash_code']}, 被 {chunk['relevance_count']} 个实体引用)")
            print("-" * 70)
            print(chunk['text'])
            print("-" * 70)
        
        print("\n" + "=" * 70)
        print("[推理路径]")
        print("=" * 70)
        for i, path in enumerate(result["reasoning_path"], 1):
            print(f"路径 {i}: {' -> '.join(path)}")
        
        print("\n" + "=" * 70)
        print("[LLM 生成的回答]")
        print("=" * 70)
        print(result["answer"])
        
        # 可选：保存结构化结果到文件
        output_file = f"query_result_{args.query[:20].replace(' ', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[结构化结果已保存到: {output_file}]")
    else:
        # 简单输出（保持向后兼容）
        ref, resp = result
    print("\n[Retrieved Context]\n" + ref)
    print("\n" + "#" * 50)
    print("\n[LLM Response]\n" + str(resp))


if __name__ == "__main__":
    main()


