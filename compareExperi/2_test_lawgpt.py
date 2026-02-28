#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LawGPT_zh 批量测试 - 基于 ChatGLM-6B 的法律模型
"""

import sys
import os
import json
import argparse
from typing import List, Dict
from tqdm import tqdm
import time

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import torch
from transformers import AutoModel, AutoTokenizer

print("="*70)
print("LawGPT_zh 批量测试 (ChatGLM-6B 法律模型)")
print("="*70)
print(f"使用GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"可用GPU数量: {torch.cuda.device_count()}")
print("="*70)


def load_model(model_path: str):
    """加载 LawGPT 模型"""
    print(f"\n🔄 正在加载模型: {model_path}")
    
    # 先加载模型
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision='main'  # 指定版本以消除警告
    ).half().cuda()
    
    # 再加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision='main'  # 指定版本以消除警告
    )
    
    model.eval()
    
    print("✅ 模型加载完成！")
    print(f"   模型类型: ChatGLM-6B (法律微调版)")
    print(f"   参数量: 6B")
    
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    question: str,
    max_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """生成答案 - 使用 ChatGLM 的 chat 方法"""
    
    try:
        response, history = model.chat(
            tokenizer,
            question,
            history=[],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        return response.strip()
    except Exception as e:
        print(f"⚠️  生成失败: {e}")
        return f"[错误: {str(e)}]"


def main():
    parser = argparse.ArgumentParser(description="LawGPT_zh 批量测试")
    parser.add_argument("--input", type=str, required=True, help="输入JSON文件")
    parser.add_argument("--output", type=str, required=True, help="输出JSON文件")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/newdatae/model/LawGPT_zh",
        help="LawGPT模型路径"
    )
    parser.add_argument("--max-length", type=int, default=2048, help="最大长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="top_p采样")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model(args.model_path)
    
    # 加载问题
    print(f"\n🔄 正在加载问题: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_questions = len(data)
    print(f"✅ 加载了 {total_questions} 个问题\n")
    
    # 处理每个问题
    results = []
    start_time = time.time()
    question_times = []
    
    print("="*70)
    print("开始处理问题")
    print("="*70)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print("="*70 + "\n")
    
    for i, item in enumerate(tqdm(data, desc="处理问题"), 1):
        question = item.get('question', '')
        if not question:
            continue
        
        question_start = time.time()
        try:
            answer = generate_answer(
                model,
                tokenizer,
                question,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            question_time = time.time() - question_start
            question_times.append(question_time)
            
            result_item = item.copy()
            result_item['prediction'] = answer
            result_item['method'] = 'lawgpt_zh'
            result_item['processing_time'] = question_time
            results.append(result_item)
            
        except Exception as e:
            question_time = time.time() - question_start
            question_times.append(question_time)
            print(f"\n❌ [{i}/{total_questions}] 处理失败: {e}")
            
            result_item = item.copy()
            result_item['prediction'] = f"[错误: {str(e)}]"
            result_item['method'] = 'lawgpt_zh'
            result_item['processing_time'] = question_time
            results.append(result_item)
    
    # 计算统计信息
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = sum(question_times) / len(question_times) if question_times else 0
    min_time = min(question_times) if question_times else 0
    max_time = max(question_times) if question_times else 0
    
    # 保存结果
    print(f"\n💾 正在保存结果: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print("\n" + "="*70)
    print("✅ 实验完成统计")
    print("="*70)
    print(f"总问题数: {total_questions}")
    print(f"成功处理: {len(results)}")
    print(f"失败数量: {total_questions - len(results)}")
    print(f"输出文件: {args.output}")
    print("="*70)
    
    # 打印详细时间统计
    print("\n" + "="*70)
    print("⏱️  详细时间统计")
    print("="*70)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"-" * 70)
    print(f"总运行时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟 / {total_time/3600:.2f} 小时)")
    print(f"问题总数: {len(results)}")
    print(f"-" * 70)
    print(f"平均每题耗时: {avg_time:.2f} 秒")
    print(f"最快问题耗时: {min_time:.2f} 秒")
    print(f"最慢问题耗时: {max_time:.2f} 秒")
    print(f"-" * 70)
    print(f"吞吐量: {len(results)/total_time:.2f} 问题/秒")
    print(f"预计1000题耗时: {(avg_time * 1000)/60:.2f} 分钟")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
