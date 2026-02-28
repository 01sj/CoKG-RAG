#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比实验1：测试纯LLM（无RAG）

直接使用大模型回答问题，不使用任何检索增强。
这是最基础的基线方法。

使用方法：
python compareExperi/1_test_pure_llm.py \
    --input datasets/query_social.json \
    --output compareExperi/results/pure_llm_pred.json
"""

import sys
import os

# ⚠️ 重要：必须在导入任何CUDA相关库之前设置GPU
# 使用第3张GPU卡（索引2）
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
print(f"🔧 设置使用GPU卡: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

import json
import argparse
from typing import List, Dict
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 默认配置
DEFAULT_MODEL_PATH = "/newdatae/model/Qwen-7B-Chat"

# 系统提示词（用于纯LLM基线测试）
# 注意：这里不要求引用具体法条，因为模型没有检索能力，避免产生幻觉
DEFAULT_SYSTEM_PROMPT = """你是一个专业的法律助手，擅长回答社会法相关的问题。
请根据你已有的法律知识回答用户的问题。

回答要求：
1. 如果你知道相关法律规定，请说明
2. 如果不确定，请明确表示不确定，不要编造
3. 逻辑清晰，条理分明
4. 语言专业但易懂"""


def load_questions(input_file: str) -> List[Dict]:
    """加载问题数据"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def test_pure_llm(
    input_file: str,
    output_file: str,
    model_path: str,
    system_prompt: str,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 1024,
    gpu_mem_util: float = 0.75,
    max_model_len: int = 8192,
):
    """使用vLLM测试纯LLM"""
    import time
    from vllm import LLM, SamplingParams
    
    # 记录总开始时间
    total_start_time = time.time()
    
    print("\n🔄 正在加载模型...")
    print(f"   模型路径: {model_path}")
    print(f"   GPU内存使用率: {gpu_mem_util}")
    print(f"   最大模型长度: {max_model_len}")
    
    # 加载vLLM模型（Qwen模型需要trust_remote_code=True）
    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
        )
        print("✅ 模型加载完成！")
    except ValueError as e:
        if "KV cache" in str(e):
            print(f"\n❌ GPU内存不足！")
            print(f"错误信息: {e}")
            print(f"\n💡 建议解决方案：")
            print(f"1. 降低 max_model_len: --max-model-len 3600")
            print(f"2. 提高 gpu_mem_util: --gpu-mem-util 0.9")
            print(f"3. 两者结合使用")
            raise
        else:
            raise
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    # 加载问题
    print(f"\n🔄 正在加载问题: {input_file}")
    questions = load_questions(input_file)
    print(f"✅ 加载了 {len(questions)} 个问题")
    
    # 准备prompts
    prompts = []
    for item in questions:
        question = item.get('question', '')
        # 构建完整的prompt（Qwen ChatML格式）
        # Qwen-7B-Chat 使用 ChatML 格式：<|im_start|>role\ncontent<|im_end|>
        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
        prompts.append(prompt)
    
    # 批量生成
    print("\n🔄 正在生成回答...")
    generation_start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    generation_end_time = time.time()
    
    # 整理结果
    results = []
    for i, output in enumerate(tqdm(outputs, desc="处理结果")):
        item = questions[i].copy()
        generated_text = output.outputs[0].text.strip()
        item['prediction'] = generated_text  # 保存在 prediction 字段
        results.append(item)
    
    # 保存结果
    print(f"\n💾 正在保存结果: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 计算时间统计
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    generation_time = generation_end_time - generation_start_time
    avg_time_per_question = generation_time / len(results) if results else 0
    
    # 打印时间统计
    print("\n" + "="*60)
    print("⏱️  时间统计")
    print("="*60)
    print(f"总运行时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"生成回答时间: {generation_time:.2f}秒 ({generation_time/60:.2f}分钟)")
    print(f"问题总数: {len(results)}")
    print(f"平均每题耗时: {avg_time_per_question:.2f}秒")
    print(f"吞吐量: {len(results)/generation_time:.2f} 问题/秒")
    print("="*60)
    
    print(f"\n✅ 成功处理 {len(results)} 个问题")


def main():
    parser = argparse.ArgumentParser(description="测试纯LLM（无RAG）")
    parser.add_argument("--input", type=str, required=True, help="输入JSON文件")
    parser.add_argument("--output", type=str, required=True, help="输出JSON文件")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                       help="模型路径")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                       help="系统提示词")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="top_p采样")
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="最大生成token数")
    parser.add_argument("--gpu-mem-util", type=float, default=0.9,
                       help="GPU内存使用率（默认0.9，如果OOM可降低）")
    parser.add_argument("--max-model-len", type=int, default=3600,
                       help="模型最大长度（默认3600，根据GPU内存调整）")
    
    args = parser.parse_args()
    
    print("="*60)
    print("对比实验1：测试纯LLM（无RAG）")
    print("="*60)
    print(f"模型: {args.model_path}")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print("="*60)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:  # 如果有目录路径
        os.makedirs(output_dir, exist_ok=True)
    
    # 调用测试函数
    test_pure_llm(
        input_file=args.input,
        output_file=args.output,
        model_path=args.model_path,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    
    print("\n✅ 纯LLM测试完成！")
    print(f"结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
