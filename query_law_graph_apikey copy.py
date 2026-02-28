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

from tools.utils import InstanceManager
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

    entity_results = search_vector_search(working_dir, embedding(query), topk=topk, level_mode=level_mode)
    res_entity = [i[0] for i in entity_results]
    chunks = [i[-1] for i in entity_results]

    entity_descriptions = get_entity_description(entity_results)
    reasoning_path, reasoning_path_information_description = get_reasoning_chain(working_dir, res_entity)
    aggregation_descriptions, aggregation = get_aggregation_description(working_dir, reasoning_path)
    text_units = get_text_units(working_dir, chunks, chunks_file, k=5)

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

    sys_prompt = PROMPTS["rag_response"].format(context_data=describe)
    response = use_llm_func(query, system_prompt=sys_prompt)
    return describe, response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="/newdataf/SJ/LeanRAG/law_kg_output_processed/",
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
    parser.add_argument("-l", "--level", type=int, default=1, help="检索层级：0原始节点/1聚合节点/2全部")
    parser.add_argument("-n", "--num", type=int, default=1, help="LLM 并发实例数量（用于生成回答）")
    parser.add_argument("--base-url", type=str, default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), help="大模型服务 Base URL（默认 DeepSeek 在线）")
    parser.add_argument("--model", type=str, default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"), help="模型名称（如 deepseek-chat 或 deepseek-reasoner）")
    parser.add_argument("--api-key", type=str, default="sk-6ba1dbd9a186476ca86e3400c0dc0427", help="用于在线服务认证的 API Key（默认已写死）")
    args = parser.parse_args()

    working_dir = args.path.rstrip("/")
    chunks_file = args.chunks

    # LLM 实例（在线 DeepSeek 默认配置）
    instance_manager = InstanceManager(
        url=args.base_url,
        ports=[443 for _ in range(max(1, args.num))],
        gpus=[0 for _ in range(max(1, args.num))],
        generate_model=args.model,
        startup_delay=0,
        api_key=args.api_key,
    )

    global_config = {
        "working_dir": working_dir,
        "chunks_file": chunks_file,
        "embeddings_func": embedding,
        "use_llm_func": instance_manager.generate_text,
        "topk": max(1, args.topk),
        "level_mode": max(0, min(2, args.level)),
    }

    # 批处理模式（优先）
    if args.input_json:
        input_path = args.input_json
        output_path = args.output_json or (os.path.splitext(input_path)[0] + "_pred.json")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("输入 JSON 必须是数组，每个元素为包含 question 的对象")
        results = []
        for item in data:
            question = item.get("question", "").strip()
            if not question:
                # 空问题直接透传
                new_item = dict(item)
                new_item["prediction"] = ""
                results.append(new_item)
                continue
            _, resp = query_law_graph(global_config, question)
            new_item = dict(item)
            new_item["prediction"] = resp
            results.append(new_item)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"已写入结果: {output_path}")
        return

    # 单问模式
    if not args.query:
        raise SystemExit("必须提供 -q/--query，或提供 --input-json 进行批处理")
    ref, resp = query_law_graph(global_config, args.query)
    print("\n[Retrieved Context]\n" + ref)
    print("\n" + "#" * 50)
    print("\n[LLM Response]\n" + str(resp))


if __name__ == "__main__":
    main()


