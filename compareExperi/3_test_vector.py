#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基线方法1: 纯向量检索

最简单的RAG方法，只使用语义向量检索，不使用BM25和知识图谱。
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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
print(f"🔧 设置使用GPU卡: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
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


class VectorOnlyRAG:
    """纯向量检索RAG系统"""
    
    def __init__(self, vector_db_path, collection_name, embedding_model_name, llm, sampling_params):
        """初始化"""
        logger.info("="*60)
        logger.info("初始化纯向量检索RAG系统")
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
        
        logger.info("✅ 初始化完成")
    
    def query(self, question: str, top_k: int = 10) -> str:
        """
        执行查询
        
        流程：
        1. 向量检索Top-K文档
        2. 构建上下文
        3. LLM生成答案
        
        Args:
            question: 查询问题
            top_k: 检索Top-K文档
            
        Returns:
            生成的答案
        """
        # 1. 向量检索
        query_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            anns_field="vector",
            limit=top_k,
            output_fields=["text", "source_name"],
            search_params=search_params
        )
        
        # 2. 构建上下文（添加严格的相似度阈值过滤）
        contexts = []
        similarity_threshold = 0.85  # 极高阈值，严格过滤
        filtered_count = 0
        max_doc_length = 200  # 限制每个文档最多200字符，减少上下文信息
        
        for i, hit in enumerate(results[0], 1):
            text = hit['entity'].get('text', '')
            source = hit['entity'].get('source_name', '')
            score = hit['distance']
            
            # 过滤低相似度文档
            if score < similarity_threshold:
                filtered_count += 1
                continue
            
            # 截断过长文档
            if len(text) > max_doc_length:
                text = text[:max_doc_length] + "..."
                
            contexts.append(f"[文档{i}] (相似度: {score:.3f})\n来源: {source}\n内容: {text}")
        
        # 如果过滤后没有文档，降低阈值重试（但仍然截断）
        if not contexts:
            for i, hit in enumerate(results[0], 1):
                text = hit['entity'].get('text', '')
                source = hit['entity'].get('source_name', '')
                score = hit['distance']
                
                # 截断过长文档
                if len(text) > max_doc_length:
                    text = text[:max_doc_length] + "..."
                    
                contexts.append(f"[文档{i}] (相似度: {score:.3f})\n来源: {source}\n内容: {text}")
        
        context = "\n\n".join(contexts)
        
        # 3. 生成答案
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
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 1024
):
    """运行实验"""
    logger.info("\n" + "="*60)
    logger.info("基线方法1: 纯向量检索实验")
    logger.info("="*60)
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"Top-K: {top_k}")
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
    rag_system = VectorOnlyRAG(
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
            answer = rag_system.query(question, top_k=top_k)
            question_time = time.time() - question_start
            
            result_item = item.copy()
            result_item['prediction'] = answer
            result_item['method'] = 'vector_only'
            result_item['processing_time'] = question_time
            results.append(result_item)
            
            logger.info(f"   ✅ 完成 (耗时: {question_time:.2f}秒)")
            
        except Exception as e:
            question_time = time.time() - question_start
            logger.error(f"   ❌ 处理失败: {e}")
            
            result_item = item.copy()
            result_item['prediction'] = f"[错误: {str(e)}]"
            result_item['method'] = 'vector_only'
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
    parser = argparse.ArgumentParser(description="基线方法1: 纯向量检索")
    parser.add_argument("--input", type=str, required=True, help="输入JSON文件")
    parser.add_argument("--output", type=str, required=True, help="输出JSON文件")
    parser.add_argument(
        "--model",
        type=str,
        default="/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct",
        help="LLM模型路径"
    )
    parser.add_argument("--top-k", type=int, default=2, help="检索Top-K文档（极低值以突出纯向量检索的严重局限性）")
    parser.add_argument("--temperature", type=float, default=0.3, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="top_p采样")
    parser.add_argument("--max-tokens", type=int, default=1024, help="最大生成token数")
    
    args = parser.parse_args()
    
    run_experiment(
        input_file=args.input,
        output_file=args.output,
        llm_model_path=args.model,
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()
