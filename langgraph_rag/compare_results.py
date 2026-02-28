#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结果对比脚本

对比原版本和 LangGraph 版本的输出结果，验证逻辑是否一致
"""

import json
import sys
from typing import Dict, List


def load_results(file_path: str) -> List[Dict]:
    """加载结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_metrics(original: Dict, langgraph: Dict) -> Dict:
    """
    对比两个结果的核心指标
    
    Returns:
        差异字典
    """
    differences = {}
    
    # 核心指标列表
    metrics = [
        'bm25_top1_score',
        'overlap_ratio',
        'top3_overlap',
        'combined_score',
        'final_simplicity',
        'used_kg',
        'question_type'
    ]
    
    for metric in metrics:
        orig_val = original.get(metric)
        lang_val = langgraph.get(metric)
        
        if orig_val != lang_val:
            # 对于浮点数，允许小的误差
            if isinstance(orig_val, float) and isinstance(lang_val, float):
                diff = abs(orig_val - lang_val)
                if diff > 1e-6:  # 误差阈值
                    differences[metric] = {
                        'original': orig_val,
                        'langgraph': lang_val,
                        'diff': diff
                    }
            else:
                differences[metric] = {
                    'original': orig_val,
                    'langgraph': lang_val
                }
    
    return differences


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python compare_results.py <原版本结果.json> <LangGraph版本结果.json>")
        print("\n示例:")
        print("  python langgraph_rag/compare_results.py \\")
        print("    datasets/query_social_hybrid_pred.json \\")
        print("    datasets/query_social_langgraph_pred.json")
        sys.exit(1)
    
    original_file = sys.argv[1]
    langgraph_file = sys.argv[2]
    
    print("="*70)
    print("结果对比分析")
    print("="*70)
    print(f"原版本文件: {original_file}")
    print(f"LangGraph版本文件: {langgraph_file}")
    print()
    
    # 加载结果
    try:
        original_results = load_results(original_file)
        langgraph_results = load_results(langgraph_file)
    except FileNotFoundError as e:
        print(f"❌ 文件不存在: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        sys.exit(1)
    
    # 检查数量
    if len(original_results) != len(langgraph_results):
        print(f"⚠️ 警告: 结果数量不一致")
        print(f"  原版本: {len(original_results)} 条")
        print(f"  LangGraph版本: {len(langgraph_results)} 条")
        print()
    
    # 逐条对比
    total_items = min(len(original_results), len(langgraph_results))
    items_with_diff = 0
    all_differences = []
    
    for i in range(total_items):
        orig = original_results[i]
        lang = langgraph_results[i]
        
        # 检查查询是否一致
        if orig.get('question') != lang.get('question'):
            print(f"⚠️ 第 {i+1} 条: 查询不一致，跳过对比")
            continue
        
        # 对比指标
        diffs = compare_metrics(orig, lang)
        
        if diffs:
            items_with_diff += 1
            all_differences.append({
                'index': i + 1,
                'query': orig.get('question', '')[:50],
                'differences': diffs
            })
    
    # 输出统计
    print(f"对比完成:")
    print(f"  - 总条目数: {total_items}")
    print(f"  - 完全一致: {total_items - items_with_diff} 条 ({(total_items - items_with_diff)/total_items*100:.1f}%)")
    print(f"  - 存在差异: {items_with_diff} 条 ({items_with_diff/total_items*100:.1f}%)")
    print()
    
    # 详细差异
    if items_with_diff > 0:
        print("="*70)
        print("差异详情 (仅显示前10条)")
        print("="*70)
        
        for item in all_differences[:10]:
            print(f"\n第 {item['index']} 条: {item['query']}...")
            for metric, diff_info in item['differences'].items():
                print(f"  {metric}:")
                print(f"    原版本: {diff_info['original']}")
                print(f"    LangGraph: {diff_info['langgraph']}")
                if 'diff' in diff_info:
                    print(f"    差值: {diff_info['diff']:.6f}")
        
        if len(all_differences) > 10:
            print(f"\n... 还有 {len(all_differences) - 10} 条差异未显示")
    
    # 总结
    print("\n" + "="*70)
    if items_with_diff == 0:
        print("✅ 结论: 两个版本的结果完全一致！")
    elif items_with_diff < total_items * 0.05:  # 差异小于5%
        print("✅ 结论: 两个版本的结果基本一致（差异<5%）")
    else:
        print("⚠️ 结论: 两个版本存在较多差异，需要进一步检查")
    print("="*70)


if __name__ == "__main__":
    main()
