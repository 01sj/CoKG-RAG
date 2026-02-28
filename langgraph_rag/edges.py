#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph 条件边函数

定义图中的决策逻辑
"""

from state import RAGState


def should_use_kg(state: RAGState) -> str:
    """
    决策函数：是否使用知识图谱
    
    这是整个流程的核心决策点：
    - 如果 use_kg=True：跳转到 kg_search 节点
    - 如果 use_kg=False：直接跳转到 answer_generation 节点
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点的名称
    """
    if state['use_kg']:
        return "kg_search"
    else:
        return "answer_generation"
