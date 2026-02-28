# LangGraph RAG 使用指南

## 📦 安装依赖

```bash
# 安装 LangGraph
pip install langgraph langchain langchain-core

# 或者使用 requirements.txt
pip install -r langgraph_rag/requirements.txt
```

## 🚀 快速开始

### 1. 简单测试（单个查询）

```bash
# 测试 LangGraph 版本是否正常工作
python langgraph_rag/test_simple.py
```

这会测试两个查询：
- 简单问题："什么是劳动合同？"（预期不使用KG）
- 复杂问题："我在工地摔伤，老板不给赔偿，怎么办？"（预期可能使用KG）

### 2. 批量处理（完整数据集）

```bash
# 使用默认参数
python langgraph_rag/main.py --input datasets/query_social.json

# 自定义参数
python langgraph_rag/main.py \
    --input datasets/query_social.json \
    --output datasets/query_social_langgraph_pred.json \
    --top-k 10 \
    --threshold 0.6 \
    --alpha 0.7
```

### 3. 对比原版本和 LangGraph 版本

```bash
# 先运行原版本
python hybrid_rag_query.py --input datasets/query_social.json

# 再运行 LangGraph 版本
python langgraph_rag/main.py --input datasets/query_social.json

# 对比结果
python langgraph_rag/compare_results.py \
    datasets/query_social_hybrid_pred.json \
    datasets/query_social_langgraph_pred.json
```

## 📊 参数说明

所有参数与原版本 `hybrid_rag_query.py` 完全一致：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `datasets/query_social.json` | 输入数据集路径 |
| `--output` | 自动生成 | 输出结果路径 |
| `--top-k` | `10` | 检索Top-K |
| `--threshold` | `0.6` | 相关系数阈值 |
| `--alpha` | `0.7` | 混合权重（70%语义+30%BM25） |
| `--llm-model` | `/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct` | LLM模型路径 |
| `--temperature` | `0.3` | 采样温度 |
| `--visualize` | `False` | 是否生成工作流可视化图 |

## 🔍 工作流可视化

```bash
# 生成工作流图（需要安装 pygraphviz）
python langgraph_rag/main.py --input datasets/query_social.json --visualize
```

这会生成 `langgraph_rag/workflow_graph.png`，展示完整的执行流程。

## 📈 输出格式

输出的 JSON 文件格式与原版本完全一致：

```json
[
  {
    "question": "什么是劳动合同？",
    "instruction": "",
    "answer": "...",
    "prediction": "...",
    "bm25_top1_score": 15.234,
    "overlap_ratio": 0.8,
    "top3_overlap": 1.0,
    "combined_score": 0.75,
    "final_simplicity": 0.82,
    "question_type": "概念定义",
    "used_kg": false,
    "elapsed_time": 2.34,
    "step_times": {
      "querite": 0.05,
      "semantic_sry_rewearch": 0.8,
      "bm25_search": 0.6,
      "evaluation": 0.3,
      "answer_generation": 0.59
    }
  }
]
```

**新增字段**：
- `step_times`: 各步骤的耗时（LangGraph 特有）

## 🔄 与原版本的差异

### 相同点 ✅
1. **所有评估指标计算逻辑完全一致**
2. **决策阈值和权重完全一致**
3. **检索、重排序、生成逻辑完全一致**
4. **输出结果格式完全一致**（除了新增 `step_times`）

### 不同点 🆕
1. **执行方式**：从线性顺序执行改为图结构执行
2. **并行执行**：语义检索和BM25检索可以并行
3. **状态追踪**：每个步骤的状态都被记录
4. **可视化**：可以生成工作流图
5. **可扩展性**：更容易添加新节点或修改流程


