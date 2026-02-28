#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
面向中文法律知识图谱的检索脚本

默认从 /newdataf/SJ/LeanRAG/KG_output/social_law_7B_processed/ 读取：
  - entity.jsonl / relation.jsonl
  - 依赖已完成的层级聚类、关系与向量索引（由 build_law_graph.py / build_graph.py 生成）

功能：
  1) 向量检索 Top-K 实体/节点
  2) 基于候选实体构造推理路径并聚合社区信息
  3) 结合 chunks 提取文本单元
  4) 组织上下文交给 LLM 生成最终回答

注意：本脚本默认在 CPU 上串行做 embedding（稳定优先）。如需 GPU/并发，可通过环境变量覆盖。


仅致知识图谱回答，生成文件是social_QWen2_7B_chunks.json
"""

import argparse
import json
import os
import logging
import sys
from datetime import datetime
from itertools import combinations
from collections import defaultdict

# ⚠️ 重要：必须在导入 vLLM 之前设置 CUDA_VISIBLE_DEVICES
# 默认使用第1张GPU卡（索引0）
default_gpu_id = os.environ.get('VLLM_GPU_IDS', '2')
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = default_gpu_id
    print(f"🔧 在导入 vLLM 之前设置 CUDA_VISIBLE_DEVICES={default_gpu_id}")
else:
    print(f"✅ 使用已设置的 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

import numpy as np
import tiktoken
import torch
from sentence_transformers import SentenceTransformer
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
    """构建推理路径（并行优化版：使用线程池加速数据库查询）"""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    
    start_time = time.time()
    
    maybe_edges = list(combinations(entities_set, 2))
    reasoning_path = []
    reasoning_path_information = []
    db_name = os.path.basename(working_dir.rstrip("/"))
    
    # 线程安全的缓存
    tree_cache = {}
    link_cache = {}
    tree_lock = Lock()
    link_lock = Lock()
    
    def get_tree_root_cached(entity):
        with tree_lock:
            if entity not in tree_cache:
                tree_cache[entity] = find_tree_root(db_name, entity)
            return tree_cache[entity]
    
    def get_link_cached(e1, e2):
        key = tuple(sorted([e1, e2]))
        with link_lock:
            if key not in link_cache:
                link_cache[key] = search_nodes_link(e1, e2, working_dir)
            return link_cache[key]
    
    print(f"   需要处理 {len(maybe_edges)} 个实体对...")
    print(f"   🚀 使用并行处理加速...")
    
    # 步骤1：并行预加载所有实体的 tree_root（大幅减少等待时间）
    print(f"   📥 预加载实体树结构...")
    unique_entities = list(set(entities_set))
    max_workers = 16  # 增加到16个线程，充分利用CPU
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_tree_root_cached, entity): entity for entity in unique_entities}
        for future in as_completed(futures):
            pass  # 只是为了触发缓存
    print(f"   ✅ 预加载完成，已缓存 {len(tree_cache)} 个实体")
    
    # 步骤2：处理每个实体对
    def process_edge(edge_idx_tuple):
        idx, edge = edge_idx_tuple
        a_path = []
        b_path = []
        node1, node2 = edge
        
        # 从缓存获取（已预加载）
        node1_tree = get_tree_root_cached(node1)
        node2_tree = get_tree_root_cached(node2)

        for i, j in zip(node1_tree, node2_tree):
            if i == j:
                a_path.append(i)
                break
            if i in b_path or j in a_path:
                break
            if i != j:
                a_path.append(i)
                b_path.append(j)

        path = a_path + [b_path[len(b_path) - 1 - i] for i in range(len(b_path))]
        a_path = list(set(a_path))
        b_path = list(set(b_path))
        
        # 限制组合数量（激进优化：降低到5个节点）
        all_nodes = a_path + b_path
        if len(all_nodes) > 5:  # 降低到5，大幅加快速度
            all_nodes = all_nodes[:5]
        
        # 收集需要查询的边
        edges_to_query = []
        for maybe_edge in combinations(all_nodes, 2):
            if maybe_edge[0] != maybe_edge[1]:
                edges_to_query.append(maybe_edge)
        
        return idx, path, edges_to_query
    
    # 并行处理所有实体对
    print(f"   🔗 构建推理路径...")
    edge_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_edge, (idx, edge)): idx for idx, edge in enumerate(maybe_edges)}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0:
                elapsed = time.time() - start_time
                print(f"   进度: {completed}/{len(maybe_edges)} ({completed/len(maybe_edges)*100:.1f}%), 耗时: {elapsed:.1f}s")
            edge_results.append(future.result())
    
    # 步骤3：并行查询所有关系
    print(f"   🔍 查询实体关系...")
    all_edges_to_query = []
    for idx, path, edges in edge_results:
        reasoning_path.append(path)
        all_edges_to_query.extend(edges)
    
    # 去重
    unique_edges = list(set(all_edges_to_query))
    print(f"   需要查询 {len(unique_edges)} 个关系...")
    
    # 并行查询关系
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_link_cached, e1, e2): (e1, e2) for e1, e2 in unique_edges}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0:
                print(f"   关系查询进度: {completed}/{len(unique_edges)} ({completed/len(unique_edges)*100:.1f}%)")
            e1, e2 = futures[future]
            info = future.result()
            if info is not None:
                reasoning_path_information.append([e1, e2, info[2]])
    
    temp_relations_information = list(set([info[2] for info in reasoning_path_information]))
    reasoning_path_information_description = "\n".join(temp_relations_information)
    
    elapsed = time.time() - start_time
    print(f"   ✅ 推理路径构建完成，耗时: {elapsed:.1f}s")
    print(f"   📊 统计: tree_root={len(tree_cache)} 个实体, links={len(link_cache)} 个关系")
    
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


def get_entity_description(entity_results: list[tuple]):
    columns = ["entity_name", "parent", "description"]
    entity_descriptions = "\t\t".join(columns) + "\n"
    entity_descriptions += "\n".join([info[0] + "\t\t" + info[1] + "\t\t" + info[2] for info in entity_results])
    return entity_descriptions


def query_law_graph(global_config: dict, query: str):
    use_llm_func = global_config["use_llm_func"]
    working_dir = global_config["working_dir"]
    level_mode = global_config.get("level_mode", 1)
    topk = global_config.get("topk", 10)
    chunks_file = global_config.get("chunks_file")
    fast_mode = global_config.get("fast_mode", False)  # 快速模式

    print(f"\n{'='*60}")
    print(f"🔍 开始处理查询: {query[:50]}...")
    if fast_mode:
        print(f"⚡ 快速模式已启用（减少推理路径复杂度）")
    print(f"{'='*60}")
    
    print("📊 步骤 1/6: 向量检索实体...")
    entity_results = search_vector_search(working_dir, embedding(query), topk=topk, level_mode=level_mode)
    res_entity = [i[0] for i in entity_results]
    chunks = [i[-1] for i in entity_results]
    print(f"   ✅ 检索到 {len(res_entity)} 个相关实体")

    print("📝 步骤 2/6: 生成实体描述...")
    entity_descriptions = get_entity_description(entity_results)
    print(f"   ✅ 完成")
    
    print("🔗 步骤 3/6: 构建推理路径...")
    # 快速模式：只使用前5个最相关的实体
    if fast_mode and len(res_entity) > 5:
        print(f"   ⚡ 快速模式：使用前5个最相关实体（原{len(res_entity)}个）")
        res_entity = res_entity[:5]
    reasoning_path, reasoning_path_information_description = get_reasoning_chain(working_dir, res_entity)
    print(f"   ✅ 构建了 {len(reasoning_path)} 条推理路径")
    
    print("🏘️  步骤 4/6: 聚合社区信息...")
    aggregation_descriptions, aggregation = get_aggregation_description(working_dir, reasoning_path)
    print(f"   ✅ 聚合了 {len(aggregation)} 个社区")
    
    print("📄 步骤 5/6: 提取文本单元...")
    text_units = get_text_units(working_dir, chunks, chunks_file, k=5)
    print(f"   ✅ 完成")

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

    # 使用中文提示词（针对中文法律问答优化）
    sys_prompt = PROMPTS["rag_response_zh"].format(context_data=describe)
    
    print("🤖 步骤 6/6: LLM 生成答案...")
    response = use_llm_func(query, system_prompt=sys_prompt)
    print(f"   ✅ 生成完成")
    return describe, response


def setup_logging(log_dir="/newdataf/SJ/LeanRAG/logs"):
    """
    配置日志系统
    
    参数:
        log_dir: 日志目录
    
    返回:
        logger: 日志记录器
        log_file: 日志文件路径
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"query_{timestamp}.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 清除现有的处理器（避免重复）
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # 文件处理器
            logging.FileHandler(log_file, encoding='utf-8'),
            # 控制台处理器
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 日志文件: {log_file}")
    logger.info("="*60)
    
    return logger, log_file


def main():
    # 设置日志
    logger, log_file = setup_logging()
    logger.info("社会法知识图谱查询系统启动")
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="/newdataf/SJ/LeanRAG/KG_output/social_law_7B_processed/",
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
    parser.add_argument("-l", "--level", type=int, default=0, help="检索层级：0原始节点/1聚合节点/2全部")
    parser.add_argument("--fast", action="store_true", help="快速模式：减少推理路径复杂度，大幅提升速度（推荐）")
    parser.add_argument("--model", type=str, default=os.environ.get("VLLM_MODEL", "/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct"), help="本地模型路径")
    parser.add_argument("--tp", type=int, default=int(os.environ.get("VLLM_TP", "1")), help="tensor_parallel_size（默认1单卡）")
    parser.add_argument("--gpu-mem-util", type=float, default=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.75")), help="每张 GPU 目标显存占用比例")
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192")), help="最大模型序列长度")
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("VLLM_MAX_NEW_TOKENS", "1024")), help="生成的最大新token数")
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("VLLM_TEMPERATURE", "0.3")), help="采样温度")
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("VLLM_TOP_P", "0.9")), help="top_p 采样阈值")
    args = parser.parse_args()

    working_dir = args.path.rstrip("/")
    chunks_file = args.chunks

    # 检查 GPU 配置（CUDA_VISIBLE_DEVICES 已在文件开头设置）
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    visible_gpu_count = len([x for x in cuda_devices.split(",") if x.strip()])
    print(f"✅ 当前使用 GPU: {cuda_devices}（共 {visible_gpu_count} 张可见GPU）")
    
    # 自动调整 tensor_parallel_size
    if visible_gpu_count < args.tp:
        print(f"⚠️  可见GPU数量 ({visible_gpu_count}) 少于 tp ({args.tp})，自动调整为 {visible_gpu_count}")
        args.tp = max(1, visible_gpu_count)

    # vLLM 本地模型实例
    llm = LLM(
        model=args.model,
        tensor_parallel_size=max(1, args.tp),
        gpu_memory_utilization=max(0.1, min(0.95, args.gpu_mem_util)),
        max_model_len=args.max_model_len,
        dtype="auto",
    )
    sampling_params = SamplingParams(
        temperature=max(0.0, args.temperature),
        top_p=min(1.0, max(0.0, args.top_p)),
        max_tokens=max(1, args.max_new_tokens),
        repetition_penalty=1.1,
        stop=None,  # 不设置停止词，让模型自然结束
    )
    
    print(f"✅ 模型加载完成")
    print(f"📊 采样参数: temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_new_tokens}")

    def vllm_generate_text(user_prompt: str, system_prompt: str = ""):
        try:
            # 构建提示词
            if system_prompt:
                composed = f"{system_prompt}\n\n{user_prompt}"
            else:
                composed = user_prompt
            
            # 打印调试信息
            print(f"\n[DEBUG] 提示词长度: {len(composed)} 字符")
            print(f"[DEBUG] 前100字符: {composed[:100]}...")
            
            # 生成
            outputs = llm.generate([composed], sampling_params=sampling_params)
            
            if not outputs:
                print("[ERROR] vLLM 返回空输出")
                return ""
            
            if not outputs[0].outputs:
                print("[ERROR] vLLM 输出列表为空")
                return ""
            
            result = outputs[0].outputs[0].text.strip()
            print(f"[DEBUG] LLM 生成长度: {len(result)} 字符")
            
            if not result:
                print("[WARNING] LLM 生成了空字符串")
                return "抱歉，无法根据提供的信息生成回答。"
            
            return result
            
        except Exception as e:
            print(f"[ERROR] LLM 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return f"生成失败: {str(e)}"

    global_config = {
        "working_dir": working_dir,
        "chunks_file": chunks_file,
        "embeddings_func": embedding,
        "use_llm_func": vllm_generate_text,
        "topk": max(1, args.topk),
        "level_mode": max(0, min(2, args.level)),
        "fast_mode": args.fast,  # 快速模式
    }

    # 批处理模式（优先）
    if args.input_json:
        import time
        
        input_path = args.input_json
        output_path = args.output_json or (os.path.splitext(input_path)[0] + "_pred.json")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("输入 JSON 必须是数组，每个元素为包含 question 的对象")
        
        results = []
        total_questions = len(data)
        processed_questions = 0
        
        logger.info("="*60)
        logger.info(f"📊 开始批量处理: 共 {total_questions} 个问题")
        logger.info("="*60)
        
        # 记录总开始时间
        batch_start_time = time.time()
        
        for idx, item in enumerate(data, 1):
            question = item.get("question", "").strip()
            if not question:
                # 空问题直接透传
                new_item = dict(item)
                new_item["prediction"] = ""
                results.append(new_item)
                logger.info(f"[{idx}/{total_questions}] 跳过空问题")
                continue
            
            # 记录单个问题开始时间
            question_start_time = time.time()
            logger.info(f"\n[{idx}/{total_questions}] 处理问题: {question[:50]}...")
            
            _, resp = query_law_graph(global_config, question)
            new_item = dict(item)
            new_item["prediction"] = resp
            results.append(new_item)
            
            # 计算单个问题耗时
            question_elapsed = time.time() - question_start_time
            processed_questions += 1
            logger.info(f"[{idx}/{total_questions}] ✅ 完成，耗时: {question_elapsed:.2f}秒")
        
        # 计算总耗时
        batch_end_time = time.time()
        total_elapsed = batch_end_time - batch_start_time
        avg_time_per_question = total_elapsed / processed_questions if processed_questions > 0 else 0
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info("\n" + "="*60)
        logger.info("📊 批量处理完成统计")
        logger.info("="*60)
        logger.info(f"总问题数: {total_questions}")
        logger.info(f"处理问题数: {processed_questions}")
        logger.info(f"跳过问题数: {total_questions - processed_questions}")
        logger.info(f"总耗时: {total_elapsed:.2f} 秒 ({total_elapsed/60:.2f} 分钟)")
        logger.info(f"平均每个问题耗时: {avg_time_per_question:.2f} 秒")
        logger.info(f"输出文件: {output_path}")
        logger.info("="*60)
        
        print(f"\n已写入结果: {output_path}")
        print(f"总耗时: {total_elapsed:.2f}秒, 平均每题: {avg_time_per_question:.2f}秒")
        return

    # 单问模式
    if not args.query:
        raise SystemExit("必须提供 -q/--query，或提供 --input-json 进行批处理")
    
    import time
    
    logger.info("="*60)
    logger.info("📊 单问题查询模式")
    logger.info("="*60)
    logger.info(f"查询问题: {args.query}")
    
    # 记录开始时间
    query_start_time = time.time()
    
    ref, resp = query_law_graph(global_config, args.query)
    
    # 计算耗时
    query_elapsed = time.time() - query_start_time
    
    logger.info("\n[Retrieved Context]\n" + ref)
    logger.info("\n" + "#" * 50)
    logger.info("\n[LLM Response]\n" + str(resp))
    
    print("\n[Retrieved Context]\n" + ref)
    print("\n" + "#" * 50)
    print("\n[LLM Response]\n" + str(resp))
    
    logger.info("\n" + "="*60)
    logger.info("📊 查询完成统计")
    logger.info("="*60)
    logger.info(f"总耗时: {query_elapsed:.2f} 秒 ({query_elapsed/60:.2f} 分钟)")
    logger.info(f"日志文件: {log_file}")
    logger.info("="*60)


if __name__ == "__main__":
    main()


