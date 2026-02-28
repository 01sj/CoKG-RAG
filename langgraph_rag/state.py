#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph 状态定义

定义整个 RAG 流程的状态结构
"""

from typing import TypedDict, List, Dict, Optional, Any
from typing_extensions import Annotated
from operator import add


def merge_dicts(left: Dict, right: Dict) -> Dict:
    """合并两个字典"""
    result = left.copy() if left else {}
    if right:
        result.update(right)
    return result


class RAGState(TypedDict):
    """
    RAG 流程的状态定义
    
    这个状态会在所有节点之间传递，每个节点可以读取和更新状态
    """
    
    # ==================== 输入 ====================
    query: str                          # 原始查询
    instruction: str                    # 指令（来自数据集）
    original_item: Dict                 # 原始数据项（用于最后保存结果）
    
    # ==================== 查询重写 ====================
    rewritten_query: Optional[str]      # 重写后的查询
    
    # ==================== 检索结果 ====================
    semantic_results: Optional[List[Dict]]   # 语义检索结果
    bm25_results: Optional[List[Dict]]       # BM25检索结果
    hybrid_results: Optional[List[Dict]]     # 混合检索结果
    selected_docs: Optional[List[Dict]]      # 重排序后选择的文档
    
    # ==================== 评估指标 ====================
    # 核心指标
    bm25_top1_score: float              # BM25 Top1分数
    overlap_ratio: float                # 文档重叠率
    top3_overlap: float                 # Top-3重叠率
    combined_score: float               # 综合复杂度分数
    
    # 统一复杂度评估指标
    question_type: str                  # 问题类型
    question_nature_complexity: float   # 问题本质复杂度
    retrieval_inconsistency: float      # 检索不一致性
    final_complexity: float             # 最终复杂度
    evaluation_layer: int               # 评估层级
    
    # 其他评估信息
    metrics: Dict[str, Any]             # 完整的评估指标字典
    
    # ==================== 决策 ====================
    use_kg: bool                        # 是否使用知识图谱
    is_classification: bool             # 是否为分类任务
    threshold: float                    # 使用的阈值
    
    # ==================== 知识图谱检索 ====================
    kg_context: Optional[str]           # 知识图谱检索的上下文
    
    # ==================== 答案生成 ====================
    answer: str                         # 生成的答案
    
    # ==================== 元数据 ====================
    elapsed_time: float                 # 总耗时
    step_times: Annotated[Dict[str, float], merge_dicts]  # 各步骤耗时（支持并发更新）
    
    # ==================== 参数配置 ====================
    top_k: int                          # Top-K参数
    alpha: float                        # 混合权重
    correlation_threshold: float        # 相关系数阈值


def create_initial_state(
    query: str,
    instruction: str = "",
    original_item: Dict = None,
    top_k: int = 10,
    alpha: float = 0.7,
    correlation_threshold: float = 0.35
) -> RAGState:
    """
    创建初始状态
    
    Args:
        query: 用户查询
        instruction: 指令
        original_item: 原始数据项
        top_k: Top-K参数
        alpha: 混合权重
        correlation_threshold: 复杂度阈值（默认0.35，优化后）
        
    Returns:
        初始化的状态字典
    """
    return {
        # 输入
        "query": query,
        "instruction": instruction,
        "original_item": original_item or {},
        
        # 查询重写
        "rewritten_query": None,
        
        # 检索结果
        "semantic_results": None,
        "bm25_results": None,
        "hybrid_results": None,
        "selected_docs": None,
        
        # 评估指标
        "bm25_top1_score": 10.0,
        "overlap_ratio": 0.0,
        "top3_overlap": 0.0,
        "combined_score": 0.0,
        "question_type": "unknown",
        "question_nature_complexity": 0.0,
        "retrieval_inconsistency": 0.0,
        "final_complexity": 0.0,
        "evaluation_layer": 0,
        "metrics": {},
        
        # 决策
        "use_kg": False,
        "is_classification": False,
        "threshold": correlation_threshold,
        
        # 知识图谱
        "kg_context": None,
        
        # 答案
        "answer": "",
        
        # 元数据
        "elapsed_time": 0.0,
        "step_times": {},
        
        # 参数
        "top_k": top_k,
        "alpha": alpha,
        "correlation_threshold": correlation_threshold,
    }
