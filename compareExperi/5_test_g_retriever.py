#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基线方法：G-Retriever (He et al., 2024)

G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering
论文: arXiv:2402.07630

核心思想：
1. 基于k-hop子图检索（固定深度，无层次感知）
2. 扁平图谱结构（实体-关系-实体）
3. 无复杂度感知，所有问题使用相同策略

与CoKG-RAG的对比：
- G-Retriever: 扁平图谱 + 固定k-hop + 无自适应
- CoKG-RAG: 层次图谱 + 双维度感知 + 自适应分流
"""

import json
import os
import sys
import logging
import time
from typing import List, Dict, Set, Tuple
import argparse
from datetime import datetime
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ⚠️ 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import torch
from vllm import LLM, SamplingParams

# 配置
VECTOR_DB_PATH = "/newdataf/SJ/LeanRAG/KG_output/social_law_7B_processed/milvus_demo.db"
COLLECTION_NAME = "entity_collection"
EMBEDDING_MODEL = "BAAI/bge-m3"

# 知识图谱数据路径
KG_DATA_DIR = "/newdataf/SJ/LeanRAG/KG_output/social_law_7B_processed/"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GRetriever:
    """
    G-Retriever 实现
    
    基于固定k-hop子图检索的扁平图谱方法
    """
    
    def __init__(self, kg_data_dir, vector_db_path, collection_name, 
                 embedding_model_name, llm, sampling_params, k_hop=2):
        """
        初始化
        
        Args:
            kg_data_dir: 知识图谱数据目录
            vector_db_path: 向量数据库路径
            collection_name: 集合名称
            embedding_model_name: Embedding模型名称
            llm: LLM模型
            sampling_params: 采样参数
            k_hop: 子图检索深度（默认2-hop）
        """
        logger.info("="*60)
        logger.info("初始化 G-Retriever 系统")
        logger.info("="*60)
        logger.info(f"方法: G-Retriever (He et al., 2024)")
        logger.info(f"特点: 固定{k_hop}-hop子图检索，扁平图谱结构")
        logger.info("="*60)
        
        self.kg_data_dir = kg_data_dir
        self.k_hop = k_hop
        
        # 连接向量数据库
        logger.info(f"连接向量数据库: {vector_db_path}")
        self.milvus_client = MilvusClient(uri=vector_db_path)
        self.collection_name = collection_name
        
        # 加载Embedding模型
        logger.info(f"加载Embedding模型: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device="cuda")
        self.embedding_model.max_seq_length = 4096
        
        # LLM配置
        self.llm = llm
        self.sampling_params = sampling_params
        
        # 加载知识图谱数据
        self._load_kg_data()
        
        logger.info("✅ 初始化完成")
    
    def _load_kg_data(self):
        """加载知识图谱数据（实体和关系）"""
        logger.info("加载知识图谱数据...")
        
        # 加载实体
        entity_file = os.path.join(self.kg_data_dir, "entity.jsonl")
        self.entities = {}
        with open(entity_file, 'r', encoding='utf-8') as f:
            for line in f:
                entity = json.loads(line)
                entity_name = entity.get('entity_name', '')
                if entity_name:
                    self.entities[entity_name] = entity
        
        logger.info(f"   ✅ 加载 {len(self.entities)} 个实体")
        
        # 加载关系（构建邻接表）
        relation_file = os.path.join(self.kg_data_dir, "relation.jsonl")
        self.relations = []
        self.adjacency = defaultdict(list)  # entity -> [(relation, target_entity)]
        
        with open(relation_file, 'r', encoding='utf-8') as f:
            for line in f:
                relation = json.loads(line)
                src = relation.get('src_id', '')
                tgt = relation.get('tgt_id', '')
                rel_type = relation.get('description', '')
                
                if src and tgt:
                    self.relations.append(relation)
                    # 双向边（无向图）
                    self.adjacency[src].append((rel_type, tgt))
                    self.adjacency[tgt].append((rel_type, src))
        
        logger.info(f"   ✅ 加载 {len(self.relations)} 个关系")
        logger.info(f"   ✅ 构建邻接表，覆盖 {len(self.adjacency)} 个节点")
    
    def _retrieve_entities(self, question: str, top_k: int = 5) -> List[str]:
        """
        步骤1：实体检索
        
        使用向量检索找到与问题最相关的实体
        
        Args:
            question: 查询问题
            top_k: 返回Top-K实体
            
        Returns:
            实体名称列表
        """
        # 向量检索（与现有系统保持一致）
        query_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # 使用与 database_utils.search_vector_search 相同的搜索方式
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            limit=top_k,
            params={"metric_type": "IP", "params": {}},
            output_fields=["entity_name", "description"],
        )
        
        # 提取实体名称
        entities = []
        for hit in results[0]:
            entity_name = hit['entity'].get('entity_name', '')
            if entity_name and entity_name in self.entities:
                entities.append(entity_name)
        
        return entities
    
    def _k_hop_subgraph(self, seed_entities: List[str]) -> Tuple[Set[str], List[Dict]]:
        """
        步骤2：k-hop子图检索
        
        从种子实体出发，检索k-hop邻居，构建子图
        
        Args:
            seed_entities: 种子实体列表
            
        Returns:
            (子图实体集合, 子图三元组列表)
        """
        visited_entities = set(seed_entities)
        subgraph_triples = []
        
        # BFS遍历k-hop
        current_level = set(seed_entities)
        
        for hop in range(self.k_hop):
            next_level = set()
            
            for entity in current_level:
                if entity not in self.adjacency:
                    continue
                
                # 遍历邻居
                for rel_type, neighbor in self.adjacency[entity]:
                    if neighbor not in visited_entities:
                        next_level.add(neighbor)
                        visited_entities.add(neighbor)
                    
                    # 记录三元组
                    triple = {
                        'source': entity,
                        'relation': rel_type,
                        'target': neighbor
                    }
                    subgraph_triples.append(triple)
            
            current_level = next_level
            
            if not current_level:
                break
        
        return visited_entities, subgraph_triples
    
    def _format_subgraph(self, entities: Set[str], triples: List[Dict]) -> str:
        """
        步骤3：格式化子图为文本
        
        Args:
            entities: 实体集合
            triples: 三元组列表
            
        Returns:
            格式化的文本
        """
        context_parts = []
        
        # 1. 实体信息
        context_parts.append("【相关实体】")
        for i, entity_name in enumerate(sorted(entities), 1):
            if entity_name in self.entities:
                entity_info = self.entities[entity_name]
                desc = entity_info.get('description', '')
                context_parts.append(f"{i}. {entity_name}: {desc}")
        
        # 2. 关系信息
        context_parts.append("\n【实体关系】")
        for i, triple in enumerate(triples, 1):
            src = triple['source']
            rel = triple['relation']
            tgt = triple['target']
            context_parts.append(f"{i}. {src} --[{rel}]--> {tgt}")
        
        return "\n".join(context_parts)
    
    def query(self, question: str, top_k_entities: int = 5) -> str:
        """
        执行查询（G-Retriever完整流程）
        
        流程：
        1. 实体检索：向量检索Top-K相关实体
        2. 子图检索：固定k-hop邻居扩展
        3. 上下文构建：格式化子图信息
        4. LLM生成：基于子图生成答案
        
        Args:
            question: 查询问题
            top_k_entities: 检索Top-K实体
            
        Returns:
            生成的答案
        """
        # 1. 实体检索
        seed_entities = self._retrieve_entities(question, top_k=top_k_entities)
        
        if not seed_entities:
            # 如果没有找到实体，返回默认回答
            return "抱歉，未能在知识图谱中找到相关信息。"
        
        # 2. k-hop子图检索
        subgraph_entities, subgraph_triples = self._k_hop_subgraph(seed_entities)
        
        # 3. 格式化上下文
        context = self._format_subgraph(subgraph_entities, subgraph_triples)
        
        # 4. LLM生成
        prompt = f"""请根据以下知识图谱信息回答问题。

知识图谱信息：
{context}

问题：{question}

请基于上述知识图谱信息，给出准确、专业的回答："""
        
        outputs = self.llm.generate([prompt], self.sampling_params)
        answer = outputs[0].outputs[0].text.strip()
        
        return answer


def run_experiment(
    input_file: str,
    output_file: str,
    llm_model_path: str,
    k_hop: int = 2,
    top_k_entities: int = 5,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 1024
):
    """运行G-Retriever实验"""
    logger.info("\n" + "="*60)
    logger.info("G-Retriever 基线实验")
    logger.info("="*60)
    logger.info(f"论文: He et al., 2024 (arXiv:2402.07630)")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"k-hop深度: {k_hop}")
    logger.info(f"Top-K实体: {top_k_entities}")
    logger.info("="*60 + "\n")
    
    # 初始化LLM
    logger.info("正在加载LLM模型...")
    llm = LLM(
        model=llm_model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        max_model_len=8192,
        dtype="auto",
    )
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=1.1,
    )
    logger.info("✅ LLM模型加载完成\n")
    
    # 初始化G-Retriever系统
    g_retriever = GRetriever(
        KG_DATA_DIR,
        VECTOR_DB_PATH,
        COLLECTION_NAME,
        EMBEDDING_MODEL,
        llm,
        sampling_params,
        k_hop=k_hop
    )
    
    # 加载问题
    logger.info(f"\n正在加载问题: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_questions = len(data)
    logger.info(f"✅ 加载了 {total_questions} 个问题\n")
    
    # 处理每个问题
    results = []
    start_time = time.time()
    
    logger.info("="*60)
    logger.info("开始处理问题")
    logger.info("="*60 + "\n")
    
    for i, item in enumerate(data, 1):
        question = item.get('question', '')
        if not question:
            logger.warning(f"[{i}/{total_questions}] 跳过空问题")
            continue
        
        logger.info(f"[{i}/{total_questions}] 处理: {question[:50]}...")
        
        question_start = time.time()
        try:
            answer = g_retriever.query(question, top_k_entities=top_k_entities)
            question_time = time.time() - question_start
            
            result_item = item.copy()
            result_item['prediction'] = answer
            result_item['method'] = 'g_retriever'
            result_item['k_hop'] = k_hop
            result_item['processing_time'] = question_time
            results.append(result_item)
            
            logger.info(f"   ✅ 完成 (耗时: {question_time:.2f}秒)")
            
        except Exception as e:
            question_time = time.time() - question_start
            logger.error(f"   ❌ 处理失败: {e}")
            
            result_item = item.copy()
            result_item['prediction'] = f"[错误: {str(e)}]"
            result_item['method'] = 'g_retriever'
            result_item['k_hop'] = k_hop
            result_item['processing_time'] = question_time
            results.append(result_item)
    
    # 计算统计信息
    total_time = time.time() - start_time
    avg_time = total_time / len(results) if results else 0
    
    # 保存结果
    logger.info(f"\n正在保存结果: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    logger.info("\n" + "="*60)
    logger.info("实验完成统计")
    logger.info("="*60)
    logger.info(f"方法: G-Retriever ({k_hop}-hop)")
    logger.info(f"总问题数: {total_questions}")
    logger.info(f"成功处理: {len(results)}")
    logger.info(f"输出文件: {output_file}")
    logger.info("="*60)
    
    # 打印时间统计
    logger.info("\n" + "="*60)
    logger.info("⏱️  时间统计")
    logger.info("="*60)
    logger.info(f"总运行时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    logger.info(f"问题总数: {len(results)}")
    logger.info(f"平均每题耗时: {avg_time:.2f}秒")
    logger.info(f"吞吐量: {len(results)/total_time:.2f} 问题/秒")
    logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="G-Retriever基线方法 (He et al., 2024)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python compareExperi/5_test_g_retriever.py \\
      --input datasets/query_social.json \\
      --output compareExperi/results/g_retriever_results.json \\
      --k-hop 2

论文引用:
  He, X., et al. (2024). G-Retriever: Retrieval-Augmented Generation 
  for Textual Graph Understanding and Question Answering. 
  arXiv preprint arXiv:2402.07630.
        """
    )
    
    parser.add_argument("--input", type=str, required=True, help="输入JSON文件")
    parser.add_argument("--output", type=str, required=True, help="输出JSON文件")
    parser.add_argument(
        "--model",
        type=str,
        default="/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct",
        help="LLM模型路径"
    )
    parser.add_argument(
        "--k-hop",
        type=int,
        default=2,
        help="子图检索深度（默认2-hop，G-Retriever标准配置）"
    )
    parser.add_argument(
        "--top-k-entities",
        type=int,
        default=5,
        help="检索Top-K实体（默认5）"
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="top_p采样")
    parser.add_argument("--max-tokens", type=int, default=1024, help="最大生成token数")
    
    args = parser.parse_args()
    
    run_experiment(
        input_file=args.input,
        output_file=args.output,
        llm_model_path=args.model,
        k_hop=args.k_hop,
        top_k_entities=args.top_k_entities,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()
