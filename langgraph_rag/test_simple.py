#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试脚本 - 测试单个查询

用于快速验证 LangGraph 版本是否正常工作
"""

import sys
import os
import logging

# 设置使用第一张显卡（GPU 0）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_rag_query import HybridLegalRAG, setup_logging
from state import create_initial_state
from workflow import create_rag_workflow

# 配置
VECTOR_DB_PATH = "/newdataf/SJ/LeanRAG/vectorDB/social_law_milvus.db"
COLLECTION_NAME = "social_law_chunks"
KG_WORKING_DIR = "/newdataf/SJ/LeanRAG/output/social_law_7B_processed/"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL_PATH = "/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct"


def test_single_query():
    """测试单个查询"""
    
    # 设置日志
    logger, log_file = setup_logging("logs")
    logger.info("="*60)
    logger.info("LangGraph RAG 简单测试")
    logger.info("="*60)
    
    # 初始化 RAG 系统
    logger.info("\n1. 初始化 RAG 系统...")
    llm_params = {
        "tp": 1,
        "gpu_mem_util": 0.75,
        "max_model_len": 4096,
        "max_new_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.9,
    }
    
    rag_system = HybridLegalRAG(
        vector_db_path=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME,
        kg_working_dir=KG_WORKING_DIR,
        embedding_model_name=EMBEDDING_MODEL,
        device="cuda",
        llm_model_path=LLM_MODEL_PATH,
        llm_params=llm_params
    )
    
    # 创建工作流
    logger.info("\n2. 创建 LangGraph 工作流...")
    app = create_rag_workflow(rag_system)
    logger.info("   ✅ 工作流创建完成")
    
    # 测试查询
    test_cases = [
        {
            "query": "什么是劳动合同？",
            "instruction": "",
            "expected": "简单问题，不使用KG"
        },
        {
            "query": "我在工地摔伤，老板不给赔偿，怎么办？",
            "instruction": "",
            "expected": "复杂问题，可能使用KG"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"测试案例 {i}/{len(test_cases)}")
        logger.info(f"{'='*60}")
        logger.info(f"查询: {test_case['query']}")
        logger.info(f"预期: {test_case['expected']}")
        
        # 创建初始状态
        initial_state = create_initial_state(
            query=test_case['query'],
            instruction=test_case['instruction'],
            top_k=10,
            alpha=0.7,
            correlation_threshold=0.6
        )
        
        # 执行工作流
        try:
            final_state = app.invoke(initial_state)
            
            # 输出结果
            logger.info(f"\n结果:")
            logger.info(f"  - 问题类型: {final_state['question_type']}")
            logger.info(f"  - 最终简单度: {final_state['final_simplicity']:.3f}")
            logger.info(f"  - 使用KG: {final_state['use_kg']}")
            logger.info(f"  - 答案: {final_state['answer'][:100]}...")
            
            # 步骤耗时
            if final_state.get('step_times'):
                logger.info(f"\n  步骤耗时:")
                for step, elapsed in final_state['step_times'].items():
                    logger.info(f"    - {step}: {elapsed:.3f}秒")
            
            logger.info(f"\n✅ 测试案例 {i} 通过")
            
        except Exception as e:
            logger.error(f"\n❌ 测试案例 {i} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.info("测试完成")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    test_single_query()
