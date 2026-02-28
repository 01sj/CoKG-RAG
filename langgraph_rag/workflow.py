#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph 工作流构建

将所有节点和边组装成完整的图
"""

import sys
import os

# 添加父目录到路径，以便导入原始模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from state import RAGState
from nodes import (
    query_rewrite_node,
    semantic_search_node,
    bm25_search_node,
    evaluation_node,
    kg_search_node,
    answer_generation_node
)
from edges import should_use_kg


def create_rag_workflow(rag_system):
    """
    创建 RAG 工作流图
    
    流程图：
    
    START
      ↓
    query_rewrite (查询重写)
      ↓
    ┌─────────────────┐
    │                 │
    semantic_search   bm25_search (并行执行)
    │                 │
    └─────────────────┘
      ↓
    evaluation (智能评估与决策)
      ↓
    [决策点: use_kg?]
      ↓           ↓
     YES         NO
      ↓           ↓
    kg_search    answer_generation
      ↓           ↓
    answer_generation
      ↓
    END
    
    Args:
        rag_system: HybridLegalRAG 实例
        
    Returns:
        编译后的工作流应用
    """
    
    # 创建状态图
    workflow = StateGraph(RAGState)
    
    # ==================== 添加节点 ====================
    # 每个节点都是一个函数，接收 state 并返回更新后的 state
    
    workflow.add_node(
        "query_rewrite",
        lambda state: query_rewrite_node(state, rag_system)
    )
    
    workflow.add_node(
        "semantic_search",
        lambda state: semantic_search_node(state, rag_system)
    )
    
    workflow.add_node(
        "bm25_search",
        lambda state: bm25_search_node(state, rag_system)
    )
    
    workflow.add_node(
        "evaluation",
        lambda state: evaluation_node(state, rag_system)
    )
    
    workflow.add_node(
        "kg_search",
        lambda state: kg_search_node(state, rag_system)
    )
    
    workflow.add_node(
        "answer_generation",
        lambda state: answer_generation_node(state, rag_system)
    )
    
    # ==================== 设置入口点 ====================
    workflow.set_entry_point("query_rewrite")
    
    # ==================== 添加边 ====================
    
    # 查询重写后，同时启动语义检索和BM25检索（并行）
    workflow.add_edge("query_rewrite", "semantic_search")
    workflow.add_edge("query_rewrite", "bm25_search")
    
    # 两个检索完成后，都进入评估节点
    # 注意：LangGraph 会自动等待所有前置节点完成
    workflow.add_edge("semantic_search", "evaluation")
    workflow.add_edge("bm25_search", "evaluation")
    
    # ==================== 添加条件边（决策点）====================
    # 评估完成后，根据 use_kg 决定下一步
    workflow.add_conditional_edges(
        "evaluation",
        should_use_kg,
        {
            "kg_search": "kg_search",
            "answer_generation": "answer_generation"
        }
    )
    
    # KG检索完成后，进入答案生成
    workflow.add_edge("kg_search", "answer_generation")
    
    # 答案生成完成后，流程结束
    workflow.add_edge("answer_generation", END)
    
    # ==================== 编译图 ====================
    app = workflow.compile()
    
    return app


def visualize_workflow(app, output_path: str = "langgraph_rag/workflow_graph.png"):
    """
    可视化工作流图（可选）
    
    需要安装: pip install pygraphviz
    
    Args:
        app: 编译后的工作流
        output_path: 输出图片路径
    """
    try:
        from IPython.display import Image, display
        
        # 生成图片
        graph_image = app.get_graph().draw_mermaid_png()
        
        # 保存图片
        with open(output_path, 'wb') as f:
            f.write(graph_image)
        
        print(f"✅ 工作流图已保存到: {output_path}")
        
        # 如果在 Jupyter 中，直接显示
        try:
            display(Image(graph_image))
        except:
            pass
            
    except ImportError:
        print("⚠️ 无法生成可视化图，请安装: pip install pygraphviz")
    except Exception as e:
        print(f"⚠️ 生成可视化图失败: {e}")
