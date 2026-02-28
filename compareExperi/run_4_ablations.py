#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4种核心消融实验批量运行脚本

消融实验：
1. 简化复杂度评估（只用检索一致性）
2. 简化复杂度评估（只用问题本质）
3. 不自适应选择文档（固定Top-K）
4. 扁平KG（无层次结构）
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from typing import List, Dict

# 实验配置
EXPERIMENTS = [
    {
        "name": "CoKG-RAG-Full",
        "description": "完整系统（基线）",
        "args": [],
        "priority": "baseline"
    },
    {
        "name": "Ablation-Retrieval-Only",
        "description": "消融1：只用检索一致性评估",
        "args": ["--use-retrieval-only"],
        "priority": "high"
    },
    {
        "name": "Ablation-Intrinsic-Only",
        "description": "消融2：只用问题本质评估",
        "args": ["--use-intrinsic-only"],
        "priority": "high"
    },
    {
        "name": "Ablation-Fixed-TopK",
        "description": "消融3：固定Top-K文档数量",
        "args": ["--fixed-topk"],
        "priority": "high"
    },
    {
        "name": "Ablation-Flat-KG",
        "description": "消融4：扁平KG结构（无层次）",
        "args": ["--flat-kg"],
        "priority": "high"
    },
]

# 默认参数
DEFAULT_INPUT = "datasets/训练数据_基础社会法_600条.json"
DEFAULT_OUTPUT_DIR = "/newdataf/SJ/LeanRAG/datasets/ablation_4_SocialLawQA /"
DEFAULT_MODEL = "/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct"


def run_experiment(exp: Dict, input_file: str, output_dir: str, model_path: str) -> Dict:
    """运行单个消融实验"""
    print("\n" + "="*80)
    print(f"🔬 实验: {exp['name']}")
    print(f"📝 描述: {exp['description']}")
    print("="*80)
    
    # 构建输出文件路径
    output_file = os.path.join(output_dir, f"{exp['name'].lower()}.json")
    
    # 检查是否已存在结果
    if os.path.exists(output_file):
        print(f"⚠️  结果文件已存在: {output_file}")
        response = input("是否重新运行？(y/n): ")
        if response.lower() != 'y':
            print("⏭️  跳过此实验")
            return {
                "name": exp['name'],
                "status": "skipped",
                "output_file": output_file
            }
    
    # 构建命令
    cmd = [
        "python", "hybrid_rag_query.py",
        "--input", input_file,
        "--output", output_file,
        "--llm-model", model_path,
    ] + exp['args']
    
    print(f"\n📌 命令: {' '.join(cmd)}\n")
    
    # 运行实验
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # 显示实时输出
            text=True
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ 实验完成！")
        print(f"⏱️  耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
        print(f"📄 输出: {output_file}")
        
        return {
            "name": exp['name'],
            "description": exp['description'],
            "status": "success",
            "elapsed_time": elapsed_time,
            "output_file": output_file,
            "command": ' '.join(cmd)
        }
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        
        print(f"\n❌ 实验失败！")
        print(f"错误码: {e.returncode}")
        
        return {
            "name": exp['name'],
            "description": exp['description'],
            "status": "failed",
            "elapsed_time": elapsed_time,
            "error": str(e),
            "returncode": e.returncode
        }


def run_evaluation(results: List[Dict], output_dir: str):
    """运行批量评估"""
    print("\n" + "="*80)
    print("📊 开始批量评估")
    print("="*80)
    
    # 构建评估命令
    methods = []
    for result in results:
        if result['status'] == 'success':
            name = result['name']
            output_file = result['output_file']
            methods.append(f"{name}:{output_file}")
    
    if not methods:
        print("❌ 没有成功的实验结果，跳过评估")
        return
    
    eval_output = os.path.join(output_dir, "ablation_comparison.json")
    
    cmd = [
        "python", "eval/compare_legal_rag.py",
        "--methods"
    ] + methods + [
        "--output", eval_output
    ]
    
    print(f"\n📌 评估命令: python eval/compare_legal_rag.py --methods ... (共{len(methods)}个方法)\n")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 评估完成！")
        print(f"📄 对比报告: {eval_output}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 评估失败: {e}")


def generate_summary(results: List[Dict], output_dir: str):
    """生成实验摘要"""
    summary_file = os.path.join(output_dir, "ablation_summary.json")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r['status'] == 'success'),
        "failed": sum(1 for r in results if r['status'] == 'failed'),
        "skipped": sum(1 for r in results if r['status'] == 'skipped'),
        "total_time": sum(r.get('elapsed_time', 0) for r in results),
        "experiments": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print("📋 实验摘要")
    print("="*80)
    print(f"总实验数: {summary['total_experiments']}")
    print(f"✅ 成功: {summary['successful']}")
    print(f"❌ 失败: {summary['failed']}")
    print(f"⏭️  跳过: {summary['skipped']}")
    print(f"⏱️  总耗时: {summary['total_time']:.2f}秒 ({summary['total_time']/60:.2f}分钟)")
    print(f"📄 摘要文件: {summary_file}")
    print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="4种核心消融实验批量运行")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="输入数据文件"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="LLM模型路径"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="跳过批量评估"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="跳过基线实验（如果已运行）"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 过滤实验
    experiments = EXPERIMENTS
    if args.skip_baseline:
        experiments = [exp for exp in EXPERIMENTS if exp['priority'] != 'baseline']
    
    print("\n" + "="*80)
    print("🚀 4种核心消融实验批量运行")
    print("="*80)
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output_dir}")
    print(f"LLM模型: {args.model}")
    print(f"实验数量: {len(experiments)}")
    print("\n实验列表:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}: {exp['description']}")
    print("="*80)
    
    # 确认运行
    response = input("\n是否开始运行？(y/n): ")
    if response.lower() != 'y':
        print("❌ 取消运行")
        return
    
    # 运行所有实验
    results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n\n{'='*80}")
        print(f"进度: [{i}/{len(experiments)}]")
        print(f"{'='*80}")
        
        result = run_experiment(exp, args.input, args.output_dir, args.model)
        results.append(result)
        
        # 保存中间结果
        generate_summary(results, args.output_dir)
    
    # 批量评估
    if not args.skip_eval:
        run_evaluation(results, args.output_dir)
    
    # 生成最终摘要
    generate_summary(results, args.output_dir)
    
    print("\n" + "="*80)
    print("🎉 所有实验完成！")
    print("="*80)
    print(f"\n下一步：运行可视化脚本")
    print(f"python compareExperi/visualize_4_ablations.py")


if __name__ == "__main__":
    main()
