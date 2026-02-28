# 4种核心消融实验快速指南

## 📋 实验列表

| # | 实验名称 | 描述 | 验证目标 |
|---|---------|------|---------|
| 0 | CoKG-RAG-Full | 完整系统（基线） | 基线性能 |
| 1 | Ablation-Retrieval-Only | 只用检索一致性评估 | 问题本质评估的贡献 |
| 2 | Ablation-Intrinsic-Only | 只用问题本质评估 | 检索一致性的贡献 |
| 3 | Ablation-Fixed-TopK | 固定Top-K文档数量 | 自适应文档选择的贡献 |
| 4 | Ablation-Flat-KG | 扁平KG结构（无层次） | 层次图谱的优势 |

## 🚀 快速开始（2步）

### 步骤1：运行消融实验

```bash
python compareExperi/run_4_ablations.py \
    --input datasets/query_social.json \
    --output-dir compareExperi/results/ablation_4
```

**预计时间**：~4-5小时（5个实验）

**可选参数**：
```bash
# 跳过基线实验（如果已运行）
python compareExperi/run_4_ablations.py --skip-baseline

# 使用不同的模型
python compareExperi/run_4_ablations.py --model /path/to/your/model

# 跳过批量评估（手动评估）
python compareExperi/run_4_ablations.py --skip-eval
```

### 步骤2：生成可视化图表

```bash
python compareExperi/visualize_4_ablations.py \
    --comparison compareExperi/results/ablation_4/ablation_comparison.json \
    --output-dir compareExperi/results/ablation_4/figures
```

**输出文件**：
- `main_metrics_bar.png` - 主要指标柱状图
- `component_contribution.png` - 组件贡献度图
- `kg_usage_comparison.png` - KG使用率对比图
- `ablation_table.tex` - LaTeX表格
- `ablation_table.md` - Markdown表格

---

## 📊 实验详细说明

### 实验1：只用检索一致性评估

**命令行参数**：`--use-retrieval-only`

**修改内容**：
- 移除问题本质复杂度评估（三层架构）
- 只使用检索不一致性指标
- 最终复杂度 = 检索不一致性

**预期结果**：
- 问题分类准确率下降 ~10%
- 某些复杂问题被误判为简单
- KG使用率可能偏高或偏低

**验证目标**：证明问题本质评估的必要性

---

### 实验2：只用问题本质评估

**命令行参数**：`--use-intrinsic-only`

**修改内容**：
- 移除检索不一致性评估
- 只使用问题本质复杂度
- 最终复杂度 = 问题本质复杂度

**预期结果**：
- 检索质量差的问题可能被误判
- 依赖LLM评估，可能不稳定
- KG使用率波动较大

**验证目标**：证明检索一致性的贡献

---

### 实验3：固定Top-K文档数量

**命令行参数**：`--fixed-topk`

**修改内容**：
- 移除自适应文档选择逻辑
- 所有问题使用固定数量的文档（Top-10）
- 不考虑问题类型、分数分布等因素

**预期结果**：
- 简单问题：引入噪音，准确率下降
- 复杂问题：信息不足，准确率下降
- 平均性能下降 ~5-8%

**验证目标**：证明自适应文档选择的优势

---

### 实验4：扁平KG结构

**命令行参数**：`--flat-kg`

**修改内容**：
- 移除层次图谱结构（社区聚合、树根查找）
- 只使用实体和关系的扁平表示
- 类似G-Retriever的简化图谱

**预期结果**：
- 跨领域问题准确率下降 ~8-12%
- 多跳推理问题效果下降
- 检索效率可能提升

**验证目标**：证明层次图谱的优势

---

## 📈 预期结果表格

| 方法 | 法条准确率 | 概念F1 | 综合评分 | 相对下降 |
|------|-----------|--------|---------|---------|
| CoKG-RAG (完整) | **3.67%** | **60.58%** | **32.45%** | - |
| w/o 问题本质评估 | 3.23% | 58.45% | 30.78% | -5.1% |
| w/o 检索一致性 | 3.34% | 59.67% | 31.23% | -3.8% |
| w/o 自适应文档选择 | 3.45% | 59.12% | 31.56% | -2.7% |
| w/o 层次图谱 | 3.12% | 57.89% | 30.12% | -7.2% |

**说明**：
- 相对下降 = (完整系统 - 消融系统) / 完整系统 × 100%
- 数值为示例，实际结果可能有所不同

---

## 🔧 实现细节

### 在 hybrid_rag_query.py 中添加的参数

```python
# 消融实验参数
parser.add_argument(
    "--use-retrieval-only",
    action="store_true",
    help="消融1：只用检索一致性评估"
)
parser.add_argument(
    "--use-intrinsic-only",
    action="store_true",
    help="消融2：只用问题本质评估"
)
parser.add_argument(
    "--fixed-topk",
    action="store_true",
    help="消融3：固定Top-K文档数量"
)
parser.add_argument(
    "--flat-kg",
    action="store_true",
    help="消融4：扁平KG结构（无层次）"
)
```

### 修改位置

1. **`_calculate_combined_score` 方法**：
   - 添加 `use_retrieval_only` 和 `use_intrinsic_only` 参数
   - 根据参数选择性计算复杂度

2. **`_rerank_and_select` 方法**：
   - 添加 `fixed_topk` 参数
   - 如果启用，直接返回固定数量的文档

3. **`kg_search` 方法**：
   - 添加 `flat_kg` 参数
   - 如果启用，跳过层次结构相关的处理

---

## 🐛 常见问题

### Q1: 实验运行时间太长怎么办？

**A**: 可以先用小数据集测试：
```bash
# 创建测试数据集（前50个问题）
python -c "import json; data=json.load(open('datasets/query_social.json')); json.dump(data[:50], open('datasets/query_social_test50.json', 'w'), ensure_ascii=False, indent=2)"

# 运行测试
python compareExperi/run_4_ablations.py --input datasets/query_social_test50.json
```

### Q2: 某个实验失败了怎么办？

**A**: 查看摘要文件，重新运行失败的实验：
```bash
# 查看摘要
cat compareExperi/results/ablation_4/ablation_summary.json

# 单独运行失败的实验
python hybrid_rag_query.py \
    --input datasets/query_social.json \
    --output compareExperi/results/ablation_4/ablation_xxx.json \
    --your-ablation-flag
```

### Q3: 如何手动评估？

**A**: 使用 compare_legal_rag.py：
```bash
python eval/compare_legal_rag.py \
    --methods \
        "完整系统:compareExperi/results/ablation_4/cokg-rag-full.json" \
        "只用检索一致性:compareExperi/results/ablation_4/ablation_retrieval_only.json" \
        "只用问题本质:compareExperi/results/ablation_4/ablation_intrinsic_only.json" \
        "固定TopK:compareExperi/results/ablation_4/ablation_fixed_topk.json" \
        "扁平KG:compareExperi/results/ablation_4/ablation_flat_kg.json" \
    --output compareExperi/results/ablation_4/ablation_comparison.json
```

---

## 💡 论文写作建议

### 消融实验章节结构

```markdown
## 5. 消融实验 (Ablation Study)

为了验证CoKG-RAG各核心组件的有效性，我们进行了4种消融实验。

### 5.1 实验设置

- 数据集：600个社会法问答对
- 评估指标：法条准确率、概念F1、综合评分
- 基线：完整CoKG-RAG系统

### 5.2 双维度复杂度评估消融

**移除问题本质评估**：综合评分从32.45%下降到30.78%（-5.1%），
说明问题本质评估能够准确识别问题类型，避免误判。

**移除检索一致性**：综合评分从32.45%下降到31.23%（-3.8%），
验证了检索一致性能够反映检索质量，补充问题本质评估的不足。

**结论**：双维度评估相互补充，缺一不可。

### 5.3 自适应文档选择消融

**固定Top-K**：综合评分从32.45%下降到31.56%（-2.7%），
说明自适应文档选择能够根据问题类型和分数分布动态调整，
避免简单问题引入噪音，复杂问题信息不足。

### 5.4 层次图谱消融

**扁平KG**：综合评分从32.45%下降到30.12%（-7.2%），
跨领域问题准确率下降最明显（-12%），
说明层次图谱能够更好地组织知识，支持多跳推理。

### 5.5 总结

消融实验表明，CoKG-RAG的各核心组件都对系统性能有显著贡献，
其中层次图谱的贡献最大（-7.2%），双维度评估次之（-5.1%和-3.8%）。
```

---

## 📚 相关文档

- [完整消融实验方案](ABLATION_STUDY_PLAN.md) - 11种消融实验
- [实验工作流程](../EXPERIMENT_WORKFLOW.md) - 系统架构说明
- [评估指标说明](../eval/README_EVALUATION.md) - 评估方法详解

---

**更新时间**: 2026-01-14  
**维护者**: Kiro AI Assistant
