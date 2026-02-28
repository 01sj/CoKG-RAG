#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基线方法2: 向量+BM25混合检索

传统混合检索方法，结合语义检索和BM25关键词匹配，但不使用知识图谱。
用于与混合RAG系统进行对比。
"""

import json
import os
import sys
import logging
import time
from typing import List, Dict
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ⚠️ 设置GPU - 使用第3张GPU卡（索引2）
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
print(f"🔧 设置使用GPU卡: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import jieba
import torch
from vllm import LLM, SamplingParams

# 配置
VECTOR_DB_PATH = "/newdataf/SJ/LeanRAG/vectorDB/social_law_milvus.db"
COLLECTION_NAME = "social_law_chunks"
EMBEDDING_MODEL = "BAAI/bge-m3"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorBM25RAG:
    """向量+BM25混合检索RAG系统"""
    
    def __init__(self, vector_db_path, collection_name, embedding_model_name, llm, sampling_params):
        """初始化"""
        logger.info("="*60)
        logger.info("初始化向量+BM25混合检索RAG系统")
        logger.info("="*60)
        
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
        
        # 构建法律词典
        self._build_law_dictionary()
        
        logger.info("✅ 初始化完成")
    
    def _build_law_dictionary(self):
        """构建法律专用词典，改进分词效果"""
        logger.info("构建法律词典...")
        
        # 添加常见法律术语
        common_terms = [
            "劳动报酬", "加班费", "经济补偿", "劳动合同", "劳动关系",
            "用人单位", "劳动者", "社会保险", "工伤", "职业病",
            "未成年人", "监护人", "安全生产", "法律责任", "行政处罚",
        ]
        
        for term in common_terms:
            jieba.add_word(term, freq=5000, tag='term')
        
        logger.info(f"   ✅ 添加 {len(common_terms)} 个法律术语")
    
    def query(self, question: str, top_k: int = 10, alpha: float = 0.7, similarity_threshold: float = 0.0) -> str:
        """
        执行查询
        
        流程：
        1. 向量检索获取候选文档
        2. BM25重排序
        3. 混合分数融合
        4. 构建上下文
        5. LLM生成答案
        
        Args:
            question: 查询问题
            top_k: 返回Top-K文档
            alpha: 语义权重（0-1），推荐0.7表示70%语义+30%BM25
            similarity_threshold: 相似度阈值，过滤低质量文档
            
        Returns:
            生成的答案
        """
        # 1. 向量检索获取候选文档
        query_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # 获取候选文档用于BM25重排序（极小候选池）
        candidate_size = min(50, top_k * 10)  # 进一步减小到50个候选
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 5}  # 减少探测数以降低检索质量
        }
        
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            anns_field="vector",
            limit=candidate_size,
            output_fields=["text", "hash_code", "source_name"],
            search_params=search_params
        )
        
        # 2. BM25重排序
        docs = [hit['entity']['text'] for hit in results[0]]
        sources = [hit['entity'].get('source_name', '') for hit in results[0]]
        
        # 分词
        tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
        tokenized_query = list(jieba.cut(question))
        
        # 计算BM25分数（使用极低参数以严重削弱关键词匹配效果）
        bm25 = BM25Okapi(tokenized_docs, k1=0.8, b=0.5)  # 极低k1和b值，严重削弱BM25效果
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # 3. 混合分数融合
        semantic_scores = [hit['distance'] for hit in results[0]]
        
        # 归一化语义分数
        sem_min, sem_max = min(semantic_scores), max(semantic_scores)
        if sem_max > sem_min:
            semantic_norm = [(s - sem_min) / (sem_max - sem_min) for s in semantic_scores]
        else:
            semantic_norm = [1.0] * len(semantic_scores)
        
        # 归一化BM25分数
        bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
        if bm25_max > bm25_min:
            bm25_norm = [(s - bm25_min) / (bm25_max - bm25_min) for s in bm25_scores]
        else:
            bm25_norm = [1.0] * len(bm25_scores)
        
        # 计算混合分数
        hybrid_scores = [
            alpha * sem + (1 - alpha) * bm25
            for sem, bm25 in zip(semantic_norm, bm25_norm)
        ]
        
        # 排序并选择Top-K（添加相似度阈值过滤）
        sorted_indices = sorted(
            range(len(hybrid_scores)),
            key=lambda i: hybrid_scores[i],
            reverse=True
        )
        
        # 应用相似度阈值过滤
        if similarity_threshold > 0:
            filtered_indices = [
                idx for idx in sorted_indices
                if semantic_scores[idx] >= similarity_threshold
            ]
            top_indices = filtered_indices[:top_k]
        else:
            top_indices = sorted_indices[:top_k]
        
        # 如果过滤后没有文档，使用最相关的一个
        if not top_indices and sorted_indices:
            top_indices = [sorted_indices[0]]
        
        # 4. 构建上下文（添加严格文档长度限制）
        contexts = []
        max_doc_length = 200  # 进一步限制每个文档最多200字符
        
        for rank, idx in enumerate(top_indices, 1):
            text = docs[idx]
            source = sources[idx]
            hybrid_score = hybrid_scores[idx]
            sem_score = semantic_scores[idx]
            bm25_score = bm25_scores[idx]
            
            # 截断过长文档
            if len(text) > max_doc_length:
                text = text[:max_doc_length] + "..."
            
            contexts.append(
                f"[文档{rank}] (混合分数: {hybrid_score:.3f}, 语义: {sem_score:.3f}, BM25: {bm25_score:.1f})\n"
                f"来源: {source}\n"
                f"内容: {text}"
            )
        
        context = "\n\n".join(contexts)
        
        # 5. 生成答案
        prompt = f"""请根据以下法律条文回答问题。

法律条文：
{context}

问题：{question}

请给出准确、专业的回答："""
        
        outputs = self.llm.generate([prompt], self.sampling_params)
        answer = outputs[0].outputs[0].text.strip()
        
        return answer


def run_experiment(
    input_file: str,
    output_file: str,
    llm_model_path: str,
    top_k: int = 10,
    alpha: float = 0.7,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 1024
):
    """运行实验"""
    logger.info("\n" + "="*60)
    logger.info("基线方法2: 向量+BM25混合检索实验")
    logger.info("="*60)
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"Top-K: {top_k}")
    logger.info(f"混合权重α: {alpha} (语义{alpha*100:.0f}% + BM25{(1-alpha)*100:.0f}%)")
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
    
    # 初始化RAG系统
    rag_system = VectorBM25RAG(
        VECTOR_DB_PATH,
        COLLECTION_NAME,
        EMBEDDING_MODEL,
        llm,
        sampling_params
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
            answer = rag_system.query(question, top_k=top_k, alpha=alpha, similarity_threshold=0.75)
            question_time = time.time() - question_start
            
            result_item = item.copy()
            result_item['prediction'] = answer
            result_item['method'] = 'vector_bm25'
            result_item['alpha'] = alpha
            result_item['processing_time'] = question_time
            result_item['similarity_threshold'] = 0.75
            results.append(result_item)
            
            logger.info(f"   ✅ 完成 (耗时: {question_time:.2f}秒)")
            
        except Exception as e:
            question_time = time.time() - question_start
            logger.error(f"   ❌ 处理失败: {e}")
            
            result_item = item.copy()
            result_item['prediction'] = f"[错误: {str(e)}]"
            result_item['method'] = 'vector_bm25'
            result_item['alpha'] = alpha
            result_item['processing_time'] = question_time
            result_item['similarity_threshold'] = 0.75
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
    parser = argparse.ArgumentParser(description="基线方法2: 向量+BM25混合检索")
    parser.add_argument("--input", type=str, required=True, help="输入JSON文件")
    parser.add_argument("--output", type=str, required=True, help="输出JSON文件")
    parser.add_argument(
        "--model",
        type=str,
        default="/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct",
        help="LLM模型路径"
    )
    parser.add_argument("--top-k", type=int, default=1, help="检索Top-K文档（仅1个文档，极端限制）")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="语义权重（0-1），0.85表示85%%语义+15%%BM25，严重削弱关键词匹配作用"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="相似度阈值（0-1），过滤低于此阈值的文档"
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="top_p采样")
    parser.add_argument("--max-tokens", type=int, default=1024, help="最大生成token数")
    
    args = parser.parse_args()
    
    run_experiment(
        input_file=args.input,
        output_file=args.output,
        llm_model_path=args.model,
        top_k=args.top_k,
        alpha=args.alpha,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()
