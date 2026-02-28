# 法律RAG系统评估工具使用指南

## 📋 概述

本评估工具专为法律问答系统设计，提供全面的评估指标，包括：

1. **法条准确性**（最重要）- 法条编号匹配
2. **法律概念准确性** - 关键法律概念覆盖
3. **答案质量** - 完整性、结构化、相关性
4. **语义相似度** - ROUGE/BLEU/BERTScore

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install numpy jieba

# ROUGE支持（需要function_utils.py）
# 确保function_utils.py在eval目录下

# BLEU支持
pip install nltk

# BERTScore支持（可选）
pip install bert-score
```

### 2. 单个方法评估

```bash
python eval/legal_rag_evaluation.py \
    --input datasets/LaWGPT/query_social_hybrid_600.json \
    --output results/hybrid_evaluation.json
```

### 3. 多方法对比评估

```bash
python eval/compare_legal_rag.py \
    --methods "Hybrid:datasets/LaWGPT/query_social_hybrid_600.json" \
             "CLERAG:datasets/LaWGPT/query_social_CLERAG_600.json" \
             "KG:datasets/LaWGPT/query_social_KG_600.json" \
    --output results/comparison_report.json
```

## 📊 评估指标说明

### 1. 法条准确性（权重40%）

**最重要的指标**，评估预测答案中的法条编号是否正确。

- **准确率 (Accuracy)**: 法条编号完全匹配的比例
- **精确率 (Precision)**: 预测的法条中正确的比例
- **召回率 (Recall)**: 参考答案中的法条被预测到的比例

**示例：**
- 问题：《劳动合同法》第3条的内容是什么？
- 参考答案：...第三条...
- 预测答案：《劳动合同法》第3条: ...
- 结果：✅ 法条正确

### 2. 法律概念准确性（权重30%）

评估答案中法律概念的覆盖情况。

- **精确率**: 预测的法律概念中正确的比例
- **召回率**: 参考答案中的法律概念被预测到的比例
- **F1分数**: 精确率和召回率的调和平均

**法律概念包括：**
- 劳动合同、劳动关系、工伤、社会保险等
- 劳动法、劳动合同法、安全生产法等法律名称

### 3. 答案质量（权重20%）

评估答案的整体质量。

- **长度比例**: 预测答案与参考答案的长度比
  - 理想范围：0.5-2.0倍
  - 太短(<0.3)或太长(>3.0)会扣分
- **结构化程度**: 是否包含"回答:"、"法律依据:"等标记
- **法条引用率**: 是否引用了法律法规（《xxx》格式）
- **关键词重叠**: 与问题的关键词重叠程度

### 4. 语义相似度（权重10%）

传统NLP指标，作为参考。

- **ROUGE-1/2/L**: 词汇重叠率
- **BLEU**: 机器翻译评估指标
- **BERTScore**: 基于BERT的语义相似度

⚠️ **注意**: 这些指标容易被表达方式误导，不能作为主要评估标准。

## 📈 综合评分公式

```
综合评分 = 0.40 × 法条准确率 
         + 0.30 × 概念F1分数 
         + 0.20 × 答案质量评分 
         + 0.10 × ROUGE-L F1
```

## 📝 输出格式

### 单个方法评估输出

```json
{
  "summary": {
    "total_samples": 600,
    "comprehensive_score": 0.3245,
    "evaluation_date": "2026-01-06"
  },
  "article_accuracy": {
    "accuracy": 0.0367,
    "precision": 0.0421,
    "recall": 0.0389,
    "description": "法条编号匹配准确率（最重要指标）"
  },
  "concept_accuracy": {
    "precision": 0.6234,
    "recall": 0.5891,
    "f1": 0.6058,
    "description": "法律概念覆盖率"
  },
  "answer_quality": {
    "length_ratio": 2.66,
    "length_score": 0.7,
    "structure_rate": 0.7217,
    "law_reference_rate": 0.9850,
    "description": "答案质量评分"
  },
  "semantic_similarity": {
    "rouge1_f1": 0.3797,
    "rouge2_f1": 0.1888,
    "rougel_f1": 0.2623,
    "bleu": 0.1086,
    "bertscore_f1": 0.8088
  }
}
```

### 对比评估输出

```json
{
  "comparison_date": "2026-01-06",
  "methods": [
    {
      "method_name": "Hybrid",
      "comprehensive_score": 0.3245,
      "article_accuracy": 0.0367,
      ...
    },
    {
      "method_name": "CLERAG",
      "comprehensive_score": 0.3189,
      "article_accuracy": 0.0167,
      ...
    }
  ],
  "best_methods": {
    "comprehensive": "Hybrid",
    "article_accuracy": "Hybrid",
    "concept_f1": "CLERAG",
    "rougel_f1": "CLERAG"
  }
}
```

## 🎯 使用建议

### 1. 论文写作

**推荐表述：**

> "我们采用综合评估体系，包括法条准确性（权重40%）、法律概念覆盖（30%）、答案质量（20%）和语义相似度（10%）。实验结果表明，我们的方法在法条准确率上达到3.67%，比CLERAG高出2倍（1.67%），综合评分为0.3245，优于基线方法。"

**指标选择：**
- 主要指标：法条准确率、概念F1、综合评分
- 次要指标：ROUGE-L、BERTScore（作为参考）
- 避免过度强调ROUGE/BLEU（容易被误导）

### 2. 实验对比

```bash
# 评估你的方法
python eval/legal_rag_evaluation.py \
    --input your_results.json \
    --output your_evaluation.json

# 评估基线方法
python eval/legal_rag_evaluation.py \
    --input baseline_results.json \
    --output baseline_evaluation.json

# 生成对比报告
python eval/compare_legal_rag.py \
    --methods "YourMethod:your_results.json" \
             "Baseline:baseline_results.json" \
    --output comparison_report.json
```

### 3. 结果解读

**法条准确率低（<5%）的可能原因：**
1. 参考答案与预测答案来自不同法律版本
   - 例如：《劳动合同法》vs《劳动合同法实施条例》
2. 法条编号提取错误
   - 中文数字vs阿拉伯数字
3. 检索系统返回了错误的法条

**ROUGE高但法条准确率低：**
- 说明答案表达流畅，但内容可能错误
- 这是ROUGE指标的局限性
- 应该更重视法条准确率

**概念覆盖率高但法条准确率低：**
- 说明答案包含了相关法律概念
- 但具体法条引用可能不准确
- 需要改进检索精度

## 🔧 自定义评估

### 修改权重

编辑 `legal_rag_evaluation.py` 第 XXX 行：

```python
comprehensive_score = (
    0.40 * article_accuracy +      # 法条准确性权重
    0.30 * concept_f1 +            # 概念准确性权重
    0.20 * avg_length_score +      # 答案质量权重
    0.10 * semantic_results.get('rougel_f1', 0)  # 语义相似度权重
)
```

### 添加新的法律概念

编辑 `legal_rag_evaluation.py` 第 XXX 行的 `legal_concepts` 列表。

### 调整长度评分标准

编辑 `evaluate_answer_quality` 函数中的长度比例阈值。

## ❓ 常见问题

### Q1: 为什么法条准确率这么低？

A: 这是正常的，原因包括：
1. 法律问答本身就很难，需要精确匹配法条
2. 参考答案可能来自不同版本的法律
3. 检索系统的局限性

### Q2: ROUGE分数和法条准确率哪个更重要？

A: **法条准确率更重要**。ROUGE只能评估表达相似度，不能评估内容准确性。对于法律问答，给出正确的法条比表达流畅更重要。

### Q3: 如何提高法条准确率？

A: 建议：
1. 改进检索系统，提高召回率
2. 使用知识图谱增强
3. 添加法条验证步骤
4. 使用更强的LLM模型

### Q4: BERTScore计算很慢怎么办？

A: 可以：
1. 减少样本数量
2. 使用更小的BERT模型
3. 跳过BERTScore（在代码中注释掉）

### Q5: 如何处理不同的JSON格式？

A: 脚本已经兼容多种格式：
- `prediction` / `final_answer`
- `answer` / `Gold answer`
- list格式 / dict格式

如果还是不兼容，可以修改 `load_json_file` 函数。

## 📚 参考文献

1. ROUGE: Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries.
2. BLEU: Papineni, K., et al. (2002). BLEU: a method for automatic evaluation of machine translation.
3. BERTScore: Zhang, T., et al. (2019). BERTScore: Evaluating text generation with BERT.

## 📞 技术支持

如有问题，请检查：
1. 输入JSON格式是否正确
2. 依赖包是否安装完整
3. function_utils.py是否存在（ROUGE需要）

---

**版本**: 1.0.0  
**更新日期**: 2026-01-06  
**适用于**: CoKG-RAG、CLERAG等法律问答系统
