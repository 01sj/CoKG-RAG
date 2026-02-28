#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph 节点函数

每个节点对应原流程中的一个步骤，复用原 HybridLegalRAG 类的方法
"""

import time
import logging
from typing import Dict
from state import RAGState

logger = logging.getLogger(__name__)


def query_rewrite_node(state: RAGState, rag_system) -> Dict:
    """
    节点0: 查询重写
    
    对应原流程的 rewrite_query_for_consistency 方法
    """
    start_time = time.time()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🔍 查询: {state['query'][:50]}...")
    if state['instruction']:
        logger.info(f"📋 指令: {state['instruction'][:50]}...")
    logger.info(f"{'='*60}")
    
    # 检测是否为分类任务
    is_classification = rag_system._is_classification_task(state['instruction'])
    
    # 分类任务调整参数
    if is_classification:
        logger.info(f"   🏷️ 检测到分类任务，使用优化参数")
        top_k = 15
        alpha = 0.75
        threshold = 0.55
        
        logger.info(f"   - Top-K: {state['top_k']} → {top_k}")
        logger.info(f"   - Alpha: {state['alpha']} → {alpha}")
        logger.info(f"   - 阈值: {state['threshold']} → {threshold}")
        
        # 添加分类示例
        instruction = rag_system._add_classification_examples(
            state['instruction'], 
            state['query']
        )
        logger.info(f"   ✅ 已添加分类示例到指令")
        
        # 检测医疗纠纷
        is_medical, medical_confidence = rag_system._detect_medical_dispute(state['query'])
        if is_medical and medical_confidence > 0.6:
            logger.info(f"   ⚕️ 检测到可能的医疗纠纷（置信度: {medical_confidence:.2f}）")
            instruction += "\n\n【特别提示】该问题可能涉及医疗纠纷，请仔细判断是否为医疗机构的医疗行为导致的损害。"
    else:
        top_k = state['top_k']
        alpha = state['alpha']
        threshold = state['threshold']  # 使用配置的阈值（默认0.35）
        instruction = state['instruction']
    
    # 查询重写
    rewritten_query = rag_system.rewrite_query_for_consistency(
        state['query'], 
        instruction
    )
    
    elapsed = time.time() - start_time
    
    return {
        "rewritten_query": rewritten_query,
        "instruction": instruction,
        "is_classification": is_classification,
        "top_k": top_k,
        "alpha": alpha,
        "threshold": threshold,
        "step_times": {"query_rewrite": elapsed}
    }


def semantic_search_node(state: RAGState, rag_system) -> Dict:
    """
    节点1: 语义向量检索
    
    对应原流程的 semantic_search 方法
    """
    start_time = time.time()
    
    semantic_results = rag_system.semantic_search(
        state['query'],
        top_k=state['top_k'],
        rewritten_query=state['rewritten_query']
    )
    
    elapsed = time.time() - start_time
    
    return {
        "semantic_results": semantic_results,
        "step_times": {"semantic_search": elapsed}
    }


def bm25_search_node(state: RAGState, rag_system) -> Dict:
    """
    节点2: BM25检索
    
    对应原流程的 bm25_search 方法
    """
    start_time = time.time()
    
    bm25_results = rag_system.bm25_search(
        state['query'],
        top_k=state['top_k'],
        rewritten_query=state['rewritten_query']
    )
    
    elapsed = time.time() - start_time
    
    return {
        "bm25_results": bm25_results,
        "step_times": {"bm25_search": elapsed}
    }


def evaluation_node(state: RAGState, rag_system) -> Dict:
    """
    节点3: 智能评估与决策
    
    对应原流程的：
    - compare_independent_rankings
    - create_hybrid_results
    - _rerank_and_select
    """
    start_time = time.time()
    
    # 步骤3: 比较两种检索结果
    metrics = rag_system.compare_independent_rankings(
        state['semantic_results'],
        state['bm25_results'],
        state['query']
    )
    
    # 步骤4: 创建混合检索结果
    logger.info(f"步骤4: 创建混合检索结果 (alpha={state['alpha']})...")
    hybrid_results = rag_system.create_hybrid_results(
        state['semantic_results'],
        state['bm25_results'],
        alpha=state['alpha']
    )
    logger.info(f"   ✅ 混合检索完成，共 {len(hybrid_results)} 个文档")
    top3_scores = [f"{r['hybrid_score']:.3f}" for r in hybrid_results[:3]]
    logger.info(f"   Top3 混合分数: {top3_scores}")
    
    # 步骤4.5: 重排序并动态选择文档
    logger.info(f"步骤4.5: 重排序并选择最相关文档...")
    max_context_docs = 12 if state['is_classification'] else 10
    
    reranked_results, selected_count = rag_system._rerank_and_select(
        hybrid_results,
        state['query'],
        metrics['combined_score'],
        is_simple=(metrics['combined_score'] < state['threshold']),  # 反转：复杂度低=简单
        max_docs=max_context_docs
    )
    logger.info(f"   ✅ 重排序完成，选择 {selected_count} 个最相关文档")
    
    # 判断是否使用KG（反转逻辑：复杂度评估）
    combined_score = metrics['combined_score']
    threshold = state['threshold']
    
    if combined_score >= threshold:
        logger.info(f"✓ 最终复杂度 {combined_score:.3f} >= {threshold}")
        logger.info(f"   → 问题复杂，需要KG辅助")
        use_kg = True
    else:
        logger.info(f"✗ 最终复杂度 {combined_score:.3f} < {threshold}")
        logger.info(f"   → 问题简单，使用传统RAG")
        use_kg = False
    
    elapsed = time.time() - start_time
    
    return {
        "hybrid_results": hybrid_results,
        "selected_docs": reranked_results,
        "bm25_top1_score": metrics['bm25_top1_score'],
        "overlap_ratio": metrics['overlap_ratio'],
        "top3_overlap": metrics['top3_overlap'],
        "combined_score": metrics['combined_score'],
        "question_type": metrics.get('question_type', 'unknown'),
        "question_nature_complexity": metrics.get('question_nature_complexity', 0.0),
        "retrieval_inconsistency": metrics.get('retrieval_inconsistency', 0.0),
        "final_complexity": metrics.get('final_complexity', 0.0),
        "evaluation_layer": metrics.get('evaluation_layer', 0),
        "metrics": metrics,
        "use_kg": use_kg,
        "step_times": {"evaluation": elapsed}
    }


def kg_search_node(state: RAGState, rag_system) -> Dict:
    """
    节点4: 知识图谱检索
    
    对应原流程的 kg_search 方法
    仅在 use_kg=True 时执行
    """
    start_time = time.time()
    
    kg_context = rag_system.kg_search(state['query'], top_k=state['top_k'])
    
    elapsed = time.time() - start_time
    
    return {
        "kg_context": kg_context,
        "step_times": {"kg_search": elapsed}
    }


def answer_generation_node(state: RAGState, rag_system) -> Dict:
    """
    节点5: 答案生成
    
    对应原流程的 generate_answer 方法
    """
    start_time = time.time()
    
    # 构建上下文
    if state['use_kg']:
        # 融合向量检索和知识图谱结果
        vector_context = "\n\n".join([
            f"【文档{i+1}】{r['source_name']}\n{r['text']}"
            for i, r in enumerate(state['selected_docs'])
        ])
        
        final_context = f"""
## 向量检索结果

{vector_context}

## 知识图谱检索结果

{state['kg_context']}
"""
    else:
        # 只使用向量检索结果
        final_context = "\n\n".join([
            f"【文档{i+1}】{r['source_name']}\n{r['text']}"
            for i, r in enumerate(state['selected_docs'])
        ])
    
    # 生成答案
    answer = rag_system.generate_answer(
        state['query'],
        final_context,
        state['instruction'],
        state['use_kg'],
        semantic_results=state['semantic_results'],
        bm25_results=state['bm25_results']
    )
    
    elapsed = time.time() - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ 完成！")
    logger.info(f"{'='*60}\n")
    
    return {
        "answer": answer,
        "step_times": {"answer_generation": elapsed}
    }
