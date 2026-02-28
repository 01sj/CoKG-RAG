#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph RAG 主入口

完全复制原 hybrid_rag_query.py 的 main 函数逻辑，
只是将执行方式从线性改为图结构
"""

import json
import os
import sys
import time
import logging
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入原始模块
from hybrid_rag_query import (
    HybridLegalRAG,
    setup_logging,
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    KG_WORKING_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    CORRELATION_THRESHOLD,
    DEVICE
)

# 导入 LangGraph 模块
from state import create_initial_state
from workflow import create_rag_workflow, visualize_workflow


def main():
    """
    主函数 - 与原版本完全一致的参数和逻辑
    """
    parser = argparse.ArgumentParser(description="混合RAG检索系统 (LangGraph版本)")
    
    # 数据路径
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/query_social.json",
        help="输入查询数据集路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出结果路径（默认为输入文件名_langgraph_pred.json）"
    )
    
    # 向量数据库配置
    parser.add_argument(
        "--vector-db",
        type=str,
        default=VECTOR_DB_PATH,
        help="Milvus向量数据库路径"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help="Collection名称"
    )
    
    # 知识图谱配置
    parser.add_argument(
        "--kg-dir",
        type=str,
        default=KG_WORKING_DIR,
        help="知识图谱工作目录"
    )
    
    # 模型配置
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help="Embedding模型名称"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct",
        help="LLM模型路径"
    )
    
    # 检索参数
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="检索Top-K"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CORRELATION_THRESHOLD,
        help="相关系数阈值"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="混合权重alpha（0-1），推荐0.7表示70%%语义+30%%BM25"
    )
    
    # LLM参数
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--gpu-mem-util", type=float, default=0.75, help="GPU显存占用比例")
    parser.add_argument("--max-model-len", type=int, default=4096, help="最大模型序列长度")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.3, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="top_p采样")
    
    # 其他参数
    parser.add_argument("--device", type=str, default=DEVICE, help="设备（cpu/cuda）")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--visualize", action="store_true", help="生成工作流可视化图")
    
    args = parser.parse_args()
    
    # 设置日志
    logger, log_file = setup_logging(args.log_dir)
    logger.info("混合RAG检索系统启动 (LangGraph版本)")
    logger.info(f"🔧 GPU配置: 使用 {len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))} 张GPU卡")
    logger.info(f"   - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    logger.info(f"   - Tensor Parallel Size: {args.tp}")
    logger.info(f"输入文件: {args.input}")
    
    # 确定输出路径
    if args.output is None:
        input_dir = os.path.dirname(args.input) or "."
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(input_dir, f"{input_name}_langgraph_pred.json")
    
    logger.info(f"输出文件: {args.output}")
    
    # 初始化混合RAG系统（复用原始类）
    llm_params = {
        "tp": args.tp,
        "gpu_mem_util": args.gpu_mem_util,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    
    rag_system = HybridLegalRAG(
        vector_db_path=args.vector_db,
        collection_name=args.collection,
        kg_working_dir=args.kg_dir,
        embedding_model_name=args.embedding_model,
        device=args.device,
        llm_model_path=args.llm_model,
        llm_params=llm_params
    )
    
    # 创建 LangGraph 工作流
    logger.info("正在构建 LangGraph 工作流...")
    app = create_rag_workflow(rag_system)
    logger.info("✅ LangGraph 工作流构建完成")
    
    # 可视化工作流（可选）
    if args.visualize:
        visualize_workflow(app)
    
    # 加载查询数据集
    logger.info(f"正在加载查询数据集: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    logger.info(f"共加载 {len(queries)} 条查询")
    
    # 批量处理
    results = []
    total_start_time = time.time()
    
    for i, item in enumerate(queries):
        logger.info(f"\n处理第 {i+1}/{len(queries)} 条查询")
        
        query = item.get("question", "").strip()
        instruction = item.get("instruction", "").strip()
        
        if not query:
            logger.warning("查询为空，跳过")
            new_item = dict(item)
            new_item["prediction"] = ""
            new_item["bm25_top1_score"] = 10.0
            new_item["overlap_ratio"] = 0.0
            new_item["top3_overlap"] = 0.0
            new_item["combined_score"] = 0.0
            new_item["used_kg"] = False
            results.append(new_item)
            continue
        
        # 执行检索和回答（使用 LangGraph）
        try:
            item_start_time = time.time()
            
            # 创建初始状态
            initial_state = create_initial_state(
                query=query,
                instruction=instruction,
                original_item=item,
                top_k=args.top_k,
                alpha=args.alpha,
                correlation_threshold=args.threshold
            )
            
            # 执行工作流
            final_state = app.invoke(initial_state)
            
            # 计算总耗时
            elapsed_time = time.time() - item_start_time
            
            # 保存结果（格式与原版本完全一致）
            new_item = dict(item)
            new_item["prediction"] = final_state["answer"]
            
            # 核心指标
            new_item["bm25_top1_score"] = final_state["bm25_top1_score"]
            new_item["overlap_ratio"] = final_state["overlap_ratio"]
            new_item["top3_overlap"] = final_state["top3_overlap"]
            new_item["combined_score"] = final_state["combined_score"]
            
            # 统一复杂度评估指标
            new_item["question_type"] = final_state["question_type"]
            new_item["question_nature_complexity"] = final_state["question_nature_complexity"]
            new_item["retrieval_inconsistency"] = final_state["retrieval_inconsistency"]
            new_item["final_complexity"] = final_state["final_complexity"]
            new_item["evaluation_layer"] = final_state["evaluation_layer"]
            
            new_item["used_kg"] = final_state["use_kg"]
            new_item["elapsed_time"] = elapsed_time
            
            # 添加步骤耗时（LangGraph特有）
            new_item["step_times"] = final_state.get("step_times", {})
            
            results.append(new_item)
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            new_item = dict(item)
            new_item["prediction"] = f"处理失败: {str(e)}"
            new_item["bm25_top1_score"] = 10.0
            new_item["overlap_ratio"] = 0.0
            new_item["top3_overlap"] = 0.0
            new_item["combined_score"] = 0.0
            new_item["used_kg"] = False
            results.append(new_item)
    
    total_elapsed_time = time.time() - total_start_time
    
    # 保存结果
    logger.info(f"\n正在保存结果到: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 统计信息（与原版本完全一致）
    total_queries = len(results)
    kg_used_count = sum(1 for r in results if r.get("used_kg", False))
    
    # 计算核心指标的平均值
    avg_bm25_top1 = np.mean([r.get("bm25_top1_score", 10.0) for r in results])
    avg_overlap_ratio = np.mean([r.get("overlap_ratio", 0.0) for r in results])
    avg_top3_overlap = np.mean([r.get("top3_overlap", 0.0) for r in results])
    avg_combined_score = np.mean([r.get("combined_score", 0.0) for r in results])
    avg_final_complexity = np.mean([r.get("final_complexity", 0.0) for r in results])
    
    logger.info(f"\n{'='*60}")
    logger.info("处理完成！(LangGraph版本)")
    logger.info(f"总查询数: {total_queries}")
    logger.info(f"总耗时: {total_elapsed_time:.2f}秒")
    logger.info(f"平均耗时: {total_elapsed_time/total_queries:.2f}秒/条")
    logger.info(f"使用知识图谱: {kg_used_count} ({kg_used_count/total_queries*100:.1f}%)")
    logger.info(f"\n📊 核心指标统计:")
    logger.info(f"{'='*60}")
    logger.info(f"1️⃣ BM25 Top1分数:")
    logger.info(f"   - 平均值: {avg_bm25_top1:.3f}")
    logger.info(f"2️⃣ 文档重叠率:")
    logger.info(f"   - 平均值: {avg_overlap_ratio:.3f}")
    logger.info(f"3️⃣ Top-3重叠率:")
    logger.info(f"   - 平均值: {avg_top3_overlap:.3f}")
    logger.info(f"4️⃣ 最终复杂度:")
    logger.info(f"   - 平均值: {avg_final_complexity:.3f}")
    logger.info(f"\n📈 综合评分:")
    logger.info(f"   - 平均综合复杂度: {avg_combined_score:.3f}")
    logger.info(f"{'='*60}")
    logger.info(f"结果已保存到: {args.output}")
    logger.info(f"日志已保存到: {log_file}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
