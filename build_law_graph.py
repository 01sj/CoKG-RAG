#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建中文法律知识图谱脚本（使用本地 vLLM 模型）

来源参考：build_graph.py
输入：/newdataf/SJ/LeanRAG/output/social_law_7B_processed/ 下的 entity.jsonl 与 relation.jsonl
输出：
  - 生成的 community.json 与 generate_relations.json 写回到 working_dir
  - 构建向量检索（如可用）
  - 写入 MySQL（如可用且配置正确）

特点：
  - 使用本地 vLLM 模型，无需在线 API
  - 针对中文法律知识图谱优化
  - 支持层次聚类构建
"""

import argparse
import multiprocessing
import os
import threading
from dataclasses import field

# ⚠️ 重要：必须在导入任何 CUDA 相关库（如 vLLM、Ray、torch）之前设置 CUDA_VISIBLE_DEVICES
# 否则环境变量设置无效，会使用默认 GPU（通常是 GPU 0）
# 默认改为单卡 GPU 3（第四张卡），并行数为 1；如需改回多卡，可再调整环境变量
# 注意：服务器上建议通过 nvidia-smi/nvitop 确认 GPU 3 空闲
default_gpu_ids = os.environ.get('VLLM_GPU_IDS', '3')

# 强制设置 CUDA_VISIBLE_DEVICES（覆盖任何已有设置）
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    old_value = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['CUDA_VISIBLE_DEVICES'] = default_gpu_ids
    print(f"🔧 覆盖 CUDA_VISIBLE_DEVICES: {old_value} -> {default_gpu_ids}")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = default_gpu_ids
    print(f"🔧 在导入 vLLM 之前设置 CUDA_VISIBLE_DEVICES={default_gpu_ids}")

print(f"✅ 当前 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}，将使用 GPU 3（第四张卡，单卡）")

# 强制单卡张量并行配置，确保 vLLM/Ray 不会尝试多卡
os.environ.setdefault("VLLM_TENSOR_PARALLEL_SIZE", "1")

# 导入vLLM（直接使用，不需要API服务）
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  警告: vLLM 未安装，无法使用本地模型")

import build_graph as bg


def setup_vllm_sync(config):
    """
    直接加载vLLM模型（同步版本，用于层次图谱构建）
    
    参数:
        config: LLM配置字典
            - model: 模型路径或名称
            - tensor_parallel_size: 张量并行GPU数量（默认2）
            - gpu_memory_utilization: 每张GPU的内存利用率（默认0.80）
            - max_model_len: 最大序列长度（默认8192，层次图谱需要较长上下文）
            - temperature: 采样温度（默认0.2）
            - top_p: top_p采样阈值（默认0.9）
            - max_tokens: 最大生成token数（默认2048，社区报告可能较长）
    
    返回:
        同步LLM生成函数
    """
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM 未安装，请先安装: pip install vllm")
    
    print(f"\n{'='*60}")
    print("加载 vLLM 模型（同步模式，用于层次图谱构建）")
    print(f"{'='*60}")
    print(f"模型路径: {config['model']}")
    print(f"张量并行GPU数量: {config.get('tensor_parallel_size', 2)}")
    print(f"每张GPU内存利用率: {config.get('gpu_memory_utilization', 0.80)}")
    print(f"最大序列长度: {config.get('max_model_len', 8192)}")
    
    print("=" * 60)
    print("\n正在加载模型...（这可能需要一些时间）")
    
    # 设置PyTorch内存分配配置（避免内存碎片和OOM）
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    
    # 获取 tensor_parallel_size
    tensor_parallel_size = config.get('tensor_parallel_size', 2)
    
    # 在使用多 GPU 时，vLLM 会使用 Ray 来协调，需要正确初始化 Ray
    if tensor_parallel_size > 1:
        try:
            import ray
            # 先尝试停止可能存在的旧 Ray 实例
            try:
                if ray.is_initialized():
                    print("🔧 检测到已存在的 Ray 实例，正在关闭...")
                    ray.shutdown()
                    import time
                    time.sleep(2)  # 等待清理完成
            except:
                pass
            
            # 检查 Ray 是否已经初始化
            if not ray.is_initialized():
                print("🔧 初始化 Ray 集群（用于多 GPU 张量并行）...")
                # 显式初始化 Ray，指定 GPU 数量
                # 注意：Ray 会使用 CUDA_VISIBLE_DEVICES 中指定的 GPU
                # 减少 object_store_memory 以避免内存问题
                ray.init(
                    ignore_reinit_error=True,
                    num_gpus=tensor_parallel_size,
                    num_cpus=tensor_parallel_size,  # 每个 GPU 分配 1 个 CPU
                    object_store_memory=1 * 10**9,  # 降低到1GB对象存储，减少内存压力
                    _temp_dir="/tmp/ray",  # 指定临时目录
                    _system_config={
                        "object_timeout_milliseconds": 30000,  # 30秒超时
                    }
                )
                print("✅ Ray 集群初始化完成")
            else:
                print("✅ Ray 集群已初始化")
        except Exception as e:
            print(f"⚠️  Ray 初始化警告: {e}")
            print("   将尝试继续运行，vLLM 可能会自动初始化 Ray")
    
    # 加载vLLM模型（添加错误处理和重试）
    llm_kwargs = {
        'model': config['model'],
        'tensor_parallel_size': tensor_parallel_size,
        'gpu_memory_utilization': config.get('gpu_memory_utilization', 0.80),
        'max_model_len': config.get('max_model_len', 8192),
        'trust_remote_code': True,
        'dtype': config.get('dtype', 'auto'),
        'enforce_eager': True,  # 强制使用 eager 模式，避免 CUDA graph 导致的显存峰值
        'disable_log_stats': True,  # 禁用统计日志，减少开销
        'max_num_seqs': 16,  # 限制并发序列数，降低显存占用
    }
    
    print("正在加载 vLLM 模型...")
    print(f"配置: gpu_memory_utilization={llm_kwargs['gpu_memory_utilization']}, max_model_len={llm_kwargs['max_model_len']}")
    
    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        error_msg = str(e)
        print(f"❌ vLLM 模型加载失败: {error_msg}")
        
        # 如果是 OOM 错误，提供建议
        if "out of memory" in error_msg.lower() or "OOM" in error_msg:
            print("\n⚠️  显存不足！建议:")
            print("  1. 降低 gpu_memory_utilization (当前: {})".format(llm_kwargs['gpu_memory_utilization']))
            print("  2. 降低 max_model_len (当前: {})".format(llm_kwargs['max_model_len']))
            print("  3. 使用单 GPU 模式: --tensor-parallel-size 1")
            print("  4. 设置环境变量: export VLLM_GPU_MEM_UTIL=0.50")
            print("     设置环境变量: export VLLM_MAX_MODEL_LEN=4096")
        
        # 如果是 Ray 相关错误
        if "ray" in error_msg.lower() or "EngineCore" in error_msg:
            print("\n⚠️  Ray/EngineCore 错误！建议:")
            print("  1. 先运行: ray stop")
            print("  2. 使用单 GPU 模式: --tensor-parallel-size 1")
        
        raise
    
    # 设置采样参数（层次图谱构建需要更稳定的输出）
    sampling_params = SamplingParams(
        temperature=config.get('temperature', 0.2),
        top_p=config.get('top_p', 0.9),
        max_tokens=config.get('max_tokens', 2048),  # 社区报告可能较长
    )
    
    print("✅ 模型加载完成！")
    print("=" * 60)
    print("")
    
    # 创建线程锁，确保 vLLM 调用的线程安全
    vllm_lock = threading.Lock()
    
    # 创建同步生成函数（带错误处理和重试机制）
    def generate_text_sync(prompt, system_prompt="", **kwargs):
        """
        同步生成文本函数（包装vLLM的同步调用，带错误处理和重试）
        
        参数:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            **kwargs: 其他参数（如 response_format 等）
        
        返回:
            生成的文本字符串
        """
        import time
        
        # 构建完整的提示
        full_prompt = ""
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"
        full_prompt += f"User: {prompt}\n\nAssistant: "
        
        # 处理 JSON 格式要求（如果指定）
        current_sampling_params = sampling_params
        if kwargs.get('response_format', {}).get('type') == 'json_object':
            # 对于需要 JSON 输出的情况，在提示中添加要求
            full_prompt = full_prompt.rstrip("Assistant: ")
            full_prompt += "请以 JSON 格式输出。\n\nAssistant: "
        
        # 重试配置
        max_retries = 3
        retry_delay = 2.0  # 初始延迟2秒
        
        # 重试机制
        for attempt in range(max_retries):
            try:
                # 使用线程锁保护 vLLM 调用
                with vllm_lock:
                    outputs = llm.generate([full_prompt], current_sampling_params)
                
                if not outputs or not outputs[0].outputs:
                    if attempt < max_retries - 1:
                        print(f"⚠️  vLLM 返回空结果 (尝试 {attempt + 1}/{max_retries})，等待后重试...")
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    return ""
                
                return outputs[0].outputs[0].text.strip()
                
            except Exception as e:
                error_msg = str(e)
                
                # 检查是否是引擎错误或超时错误
                is_engine_error = "EngineCore" in error_msg or "EngineDeadError" in str(type(e))
                is_timeout_error = "timeout" in error_msg.lower() or "TimeoutError" in str(type(e))
                
                if (is_engine_error or is_timeout_error) and attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)  # 指数退避
                    print(f"⚠️  vLLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {error_msg[:150]}")
                    print(f"   等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 最后一次尝试或非可重试错误，打印错误并返回空字符串
                    print(f"❌ vLLM 调用最终失败: {error_msg[:200]}")
                    if attempt == max_retries - 1:
                        print(f"   已重试 {max_retries} 次，放弃本次请求")
                    return ""
        
        return ""
    
    return generate_text_sync


def main():
    # 先清理可能存在的 Ray 进程（避免 EngineCore 错误）
    print("🔧 清理可能存在的 Ray 进程...")
    try:
        import ray
        if ray.is_initialized():
            print("   检测到已存在的 Ray 实例，正在关闭...")
            ray.shutdown()
            import time
            time.sleep(3)
            print("   ✅ Ray 实例已关闭")
    except Exception as e:
        print(f"   Ray 清理跳过: {e}")
    
    # 尝试通过命令行清理 Ray
    try:
        import subprocess
        result = subprocess.run(["ray", "stop"], capture_output=True, timeout=10, text=True)
        if result.returncode == 0:
            print("   ✅ Ray 进程已通过命令行清理")
    except Exception as e:
        print(f"   命令行清理 Ray 跳过: {e}")
    
    # 确保在使用 CUDA 与多进程前设置为 spawn 模式
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 已设置过，忽略

    # 强制 embedding 使用 CPU，避免与 vLLM 抢占显存
    print("🔧 配置 Embedding 使用 CPU（避免显存冲突）")
    os.environ["FORCE_CPU"] = "1"
    os.environ["EMB_MAX_WORKERS"] = "0"  # 0=串行，无多进程
    os.environ["EMB_BATCH"] = "4"  # 更小的batch，降低内存占用
    parser = argparse.ArgumentParser(description="构建中文法律知识图谱（使用本地 vLLM 模型）")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="/newdataf/SJ/LeanRAG/output/social_law_7B_processed/",
        help="去重后的实体与关系输出目录，需包含 entity.jsonl 与 relation.jsonl",
    )
    # 这里参数仍保留，但默认和实际运行都强制为单卡，方便你在服务器上用第 2 张卡稳定运行
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")),
        help="张量并行GPU数量（当前脚本强制为 1，单卡运行）",
    )
    args = parser.parse_args()

    working_dir = args.path.rstrip("/")
    # 强制单卡运行，忽略传入的大于 1 的值，避免 Ray 多卡导致 EngineCore 错误
    tensor_parallel_size = 1

    # 设置 build_graph 模块的全局 WORKING_DIR（其层级聚类函数内部依赖）
    bg.WORKING_DIR = working_dir

    # ============================================
    # 配置本地 vLLM 模型
    # ============================================
    print("="*70)
    print(" 构建中文法律知识图谱 - 使用本地 vLLM 模型")
    print("="*70)
    
    # 使用本地 Qwen2-7B-Instruct 模型（与提取脚本 law_extract_graphrag_parllar2_QWen7B.py 一致）
    # 优先使用环境变量指定的模型路径
    if os.environ.get('VLLM_MODEL_PATH'):
        model_path = os.environ.get('VLLM_MODEL_PATH')
        print(f"📁 使用环境变量指定的模型路径: {model_path}")
    elif os.environ.get('VLLM_MODEL_NAME'):
        model_path = os.environ.get('VLLM_MODEL_NAME')
        print(f"📁 使用环境变量指定的模型名称: {model_path}")
    else:
        # 默认使用本地 Qwen2-7B-Instruct 模型（与提取脚本一致）
        # 本地模型路径：/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct
        model_path = '/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct'
        print(f"📁 使用本地模型: {model_path}")
        print(f"💡 提示: 使用与提取脚本相同的模型，保证一致性")
        print(f"   配置: 当前脚本固定使用 tensor_parallel_size=1，在第 2 张卡单卡运行")
    
    # LLM 配置（针对 24GB 4090 优化，避免 EngineCore 崩溃）
    llm_config = {
        'model': model_path,
        'tensor_parallel_size': tensor_parallel_size,
        'gpu_memory_utilization': float(os.environ.get('VLLM_GPU_MEM_UTIL', '0.75')),  # 设置为0.75（约18GB），为系统和缓存预留空间
        'max_model_len': int(os.environ.get('VLLM_MAX_MODEL_LEN', '4096')),  # 降低到4096，减少KV cache显存占用
        'temperature': float(os.environ.get('VLLM_TEMPERATURE', '0.2')),  # 较低温度，更稳定的输出
        'top_p': float(os.environ.get('VLLM_TOP_P', '0.9')),
        'max_tokens': int(os.environ.get('VLLM_MAX_TOKENS', '1024')),  # 降低到1024，减少生成时显存占用
        'dtype': os.environ.get('VLLM_DTYPE', 'auto'),
    }
    
    print(f"\n💡 显存配置:")
    print(f"   GPU 内存利用率: {llm_config['gpu_memory_utilization']} (约 {24 * llm_config['gpu_memory_utilization']:.1f}GB)")
    print(f"   最大序列长度: {llm_config['max_model_len']}")
    print(f"   最大生成长度: {llm_config['max_tokens']}")
    print(f"   Embedding: 强制使用 CPU（避免显存冲突）")
    
    # 设置 LLM
    print(f"\n{'='*60}")
    print("配置 LLM 用于层次图谱构建")
    print(f"{'='*60}")
    
    try:
        use_llm_func = setup_vllm_sync(llm_config)
        print("✅ LLM 配置完成")
    except Exception as e:
        print(f"❌ LLM 配置失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 组装全局配置
    global_config = {}
    global_config["max_workers"] = min(2, tensor_parallel_size * 2)  # 根据GPU数量调整
    global_config["working_dir"] = working_dir
    global_config["use_llm_func"] = use_llm_func
    global_config["embeddings_func"] = bg.embedding
    global_config["special_community_report_llm_kwargs"] = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # 基础存在性检查
    entity_path = os.path.join(working_dir, "entity.jsonl")
    relation_path = os.path.join(working_dir, "relation.jsonl")
    if not os.path.exists(entity_path) or not os.path.exists(relation_path):
        raise FileNotFoundError(
            f"未找到输入文件，请确认存在: {entity_path} 和 {relation_path}"
        )

    # 调用原有的层级聚类构图流程
    print(f"\n{'='*70}")
    print("🚀 开始构建层级知识图谱...")
    print(f"{'='*70}")
    print(f"📂 工作目录: {working_dir}")
    print(f"📊 输入文件:")
    print(f"   - {entity_path}")
    print(f"   - {relation_path}")
    print(f"{'='*70}\n")
    
    try:
        bg.hierarchical_clustering(global_config)
        
        # 检查输出文件是否生成
        community_file = os.path.join(working_dir, "community.json")
        relations_file = os.path.join(working_dir, "generate_relations.json")
        
        print(f"\n{'='*70}")
        print("🎉 知识图谱构建成功！")
        print(f"{'='*70}")
        
        if os.path.exists(community_file):
            file_size = os.path.getsize(community_file) / (1024 * 1024)  # MB
            print(f"✅ 社区文件已生成: {community_file}")
            print(f"   文件大小: {file_size:.2f} MB")
        
        if os.path.exists(relations_file):
            file_size = os.path.getsize(relations_file) / (1024 * 1024)  # MB
            print(f"✅ 关系文件已生成: {relations_file}")
            print(f"   文件大小: {file_size:.2f} MB")
        
        print(f"\n📊 输出目录: {working_dir}")
        print(f"{'='*70}")
        print("✨ 现在可以使用 query_law_graph_apikey.py 进行查询了！")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("❌ 知识图谱构建失败！")
        print(f"{'='*70}")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        raise


if __name__ == "__main__":
    main()


