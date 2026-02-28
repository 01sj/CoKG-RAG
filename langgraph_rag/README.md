# LangGraph RAG 实验流程

这是原 `hybrid_rag_query.py` 的 LangGraph 版本实现。

## 🎯 设计目标

**完全保持原流程逻辑不变**，只是将线性执行改为图结构：

1. ✅ 所有评估指标计算逻辑完全一致
2. ✅ 决策阈值和权重完全一致
3. ✅ 检索、重排序、生成逻辑完全一致
4. ✅ 输出结果格式完全一致

## 📁 文件结构

```
langgraph_rag/
├── README.md                      # 本文件
├── state.py                       # 状态定义
├── nodes.py                       # 节点函数（复用原代码）
├── edges.py                       # 条件边函数
├── workflow.py                    # 图构建
├── main.py                        # 主入口（替代原 main 函数）
├── test_simple.py                 # 简单测试
├── compare_results.py             # 结果对比
├── requirements.txt               # 依赖列表
├── USAGE.md                       # 使用指南
├── FAQ.md                         # 常见问题
├── IMPLEMENTATION_NOTES.md        # 实现说明
├── EVALUATION_LOGIC.md            # 评估逻辑详解 ⭐
├── EVALUATION_FLOWCHART.md        # 评估流程图 ⭐
└── WORKFLOW_COMPARISON.md         # 流程对比
```

## 🚀 使用方法

```bash
# 运行 LangGraph 版本
python langgraph_rag/main.py --input datasets/query_social.json

# 参数与原版本完全一致
python langgraph_rag/main.py \
    --input datasets/query_social.json \
    --output datasets/query_social_langgraph_pred.json \
    --top-k 10 \
    --threshold 0.6 \
    --alpha 0.7
```

## 🔄 与原版本的对比

| 特性 | 原版本 (hybrid_rag_query.py) | LangGraph 版本 |
|------|------------------------------|----------------|
| 执行逻辑 | ✅ 线性顺序执行 | ✅ 图结构执行（逻辑相同） |
| 评估指标 | ✅ 完全一致 | ✅ 完全一致 |
| 决策规则 | ✅ 完全一致 | ✅ 完全一致 |
| 输出格式 | ✅ 完全一致 | ✅ 完全一致 |
| 可视化 | ❌ 无 | ✅ 自动生成流程图 |
| 状态追踪 | ❌ 无 | ✅ 每步状态可查 |
| 并行执行 | ❌ 顺序执行 | ✅ 语义+BM25并行 |
| 人工干预 | ❌ 不支持 | ✅ 可在决策点暂停 |

## ⚠️ 注意事项

1. **依赖安装**：需要安装 `langgraph`
   ```bash
   pip install langgraph
   ```

2. **完全兼容**：可以直接替换原版本使用，输出结果应该完全一致

3. **性能**：LangGraph 会有轻微的额外开销（状态管理），但可以通过并行执行弥补
