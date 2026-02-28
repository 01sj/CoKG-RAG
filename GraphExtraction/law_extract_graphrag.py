#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理json，使用 vLLM 直接加载本地模型进行实体和关系提取

修改历史：
- 2024-12-12: 使用本地模型路径 /newdatad/WHH/MyEmoHH/models/Qwen2-1.5B-Instruct/
- 配置：GPU 3（单卡），Qwen2-1.5B-Instruct（约 3-4GB 显存）
- 备注：
  - DeepSeek-V2-Lite-Chat: MoE 模型，需要 23GB+ 显存，单卡无法加载
  - Qwen2-7B-Instruct: 需要 14-16GB 显存，单卡会出现 KV cache 内存不足
  - Qwen2-1.5B-Instruct: 需要 3-4GB 显存，单卡完全够用
"""

import os
import sys
import json
import asyncio
import threading
from pathlib import Path

# ⚠️ 重要：必须在导入任何 CUDA 相关库（如 vLLM、Ray、torch）之前设置 CUDA_VISIBLE_DEVICES
# 否则环境变量设置无效，会使用默认 GPU（通常是 GPU 0）
# 默认配置：使用 GPU 3（第四张卡），并行数为 1（单卡模式）
# 注意：根据实际 GPU 使用情况选择空闲的 GPU（通过 nvidia-smi 或 nvitop 查看）
# 如果环境变量已设置，会强制覆盖为 GPU 3
default_gpu_ids = os.environ.get('VLLM_GPU_IDS', '3')

# 强制设置 CUDA_VISIBLE_DEVICES（覆盖任何已有设置）
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    old_value = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['CUDA_VISIBLE_DEVICES'] = default_gpu_ids
    print(f"🔧 覆盖 CUDA_VISIBLE_DEVICES: {old_value} -> {default_gpu_ids}")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = default_gpu_ids
    print(f"🔧 在导入 vLLM 之前设置 CUDA_VISIBLE_DEVICES={default_gpu_ids}")

print(f"✅ 当前 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}，将使用 GPU 3")

# 添加项目路径
# 自动检测脚本位置并调整导入
current_file = Path(__file__).resolve()
current_dir = current_file.parent

# 如果在 GraphExtraction 目录内，项目根目录是上级目录
if current_dir.name == "GraphExtraction":
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))
    from chunk import get_chunk, triple_extraction
else:
    # 在项目根目录
    project_root = current_dir
    sys.path.insert(0, str(project_root))
    from GraphExtraction.chunk import get_chunk, triple_extraction

# 导入vLLM（直接使用，不需要API服务）
from vllm import LLM, SamplingParams


def setup_vllm_direct(config):
    """
    直接加载vLLM模型（不使用API服务）

    参数:
        config: LLM配置字典
            - model: 模型名称或路径
            - tensor_parallel_size: 张量并行GPU数量（默认2）
            - gpu_memory_utilization: 每张GPU的内存利用率（默认0.80）
            - max_model_len: 最大序列长度（默认8192）
            - temperature: 采样温度（默认0.2）
            - top_p: top_p采样阈值（默认0.9）
            - max_tokens: 最大生成token数（默认1024）
            - cache_dir: 模型缓存目录（可选，默认使用HuggingFace默认位置）

    返回:
        异步LLM生成函数
    """
    print(f"\n{'='*60}")
    print("加载 vLLM 模型（直接模式，不使用API服务）")
    print(f"{'='*60}")
    print(f"模型: {config['model']}")
    print(f"张量并行GPU数量: {config.get('tensor_parallel_size', 1)}")
    print(f"使用的GPU: {config.get('gpu_ids', '3')} (物理GPU编号)")
    # 显示实际配置的内存利用率（从 config 中获取，而不是默认值）
    actual_mem_util = config.get('gpu_memory_utilization')
    print(f"每张GPU内存利用率: {actual_mem_util}")
    actual_max_len = config.get('max_model_len')
    print(f"最大序列长度: {actual_max_len}")
    
    # 显示缓存目录信息
    cache_dir = config.get('cache_dir')
    if cache_dir:
        print(f"模型缓存目录: {cache_dir}")
        # 设置HuggingFace缓存目录
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = cache_dir
    else:
        # 使用默认缓存位置
        default_cache = os.path.expanduser("~/.cache/huggingface")
        print(f"模型缓存目录: {default_cache} (默认)")
    
    print("=" * 60)
    print("\n正在加载模型...（这可能需要一些时间）")
    
    # 设置PyTorch内存分配配置（避免内存碎片）
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 设置使用的GPU（使用单卡：GPU 3，第四张卡）
    # 注意：CUDA_VISIBLE_DEVICES 已经在文件开头设置（在导入 vLLM 之前）
    # 这里只是显示当前使用的 GPU 配置
    # 重要：确保选择的 GPU 是空闲的（通过 nvidia-smi 或 nvitop 查看）
    gpu_ids = config.get('gpu_ids', '3')  # 默认使用 GPU 3（第四张卡，单卡模式）
    tensor_parallel_size = config.get('tensor_parallel_size', 1)  # 单卡模式，并行数为 1
    
    # 检查 CUDA_VISIBLE_DEVICES 是否与配置匹配
    current_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if current_cuda_devices:
        # 解析当前设置的 GPU 数量
        current_gpu_count = len(current_cuda_devices.split(','))
        if current_gpu_count != tensor_parallel_size:
            print(f"错误: CUDA_VISIBLE_DEVICES={current_cuda_devices} 只提供 {current_gpu_count} 个 GPU")
            print(f"   但配置要求 tensor_parallel_size={tensor_parallel_size} 个 GPU")
            print(f"   请取消设置环境变量或修改配置以匹配")
            print(f"   取消方法: unset CUDA_VISIBLE_DEVICES")
            raise ValueError(
                f"GPU 数量不匹配: CUDA_VISIBLE_DEVICES 提供 {current_gpu_count} 个 GPU，"
                f"但 tensor_parallel_size 需要 {tensor_parallel_size} 个 GPU"
            )
        print(f"当前使用的 GPU: {current_cuda_devices} (物理 GPU 编号，vLLM 会将其视为 GPU 0-{current_gpu_count-1})")
    else:
        print(f"使用配置的 GPU: {gpu_ids} (物理 GPU 编号)")
    
    # 在使用多 GPU 时，vLLM 会使用 Ray 来协调，需要正确初始化 Ray
    if tensor_parallel_size > 1:
        try:
            import ray
            # 检查 Ray 是否已经初始化
            if not ray.is_initialized():
                print("🔧 初始化 Ray 集群（用于多 GPU 张量并行）...")
                # 显式初始化 Ray，指定 GPU 数量
                # 注意：Ray 会使用 CUDA_VISIBLE_DEVICES 中指定的 GPU
                ray.init(
                    ignore_reinit_error=True,
                    num_gpus=tensor_parallel_size,
                    num_cpus=tensor_parallel_size,  # 每个 GPU 分配 1 个 CPU
                    object_store_memory=2 * 10**9,  # 2GB 对象存储
                    _temp_dir="/tmp/ray",  # 指定临时目录
                )
                print("✅ Ray 集群初始化完成")
            else:
                print("✅ Ray 集群已初始化")
        except Exception as e:
            print(f"⚠️  Ray 初始化警告: {e}")
            print("   将尝试继续运行，vLLM 可能会自动初始化 Ray")
    
    # 加载vLLM模型（带自动重试机制，如果 KV cache 内存不足会自动降低 max_model_len）
    max_retries = 2
    retry_count = 0
    llm = None
    original_max_len = config.get('max_model_len', 8192)
    
    while retry_count <= max_retries and llm is None:
        try:
            current_max_len = original_max_len
            if retry_count > 0:
                # 每次重试时降低 max_model_len（每次降低到原来的 70%）
                current_max_len = int(original_max_len * (0.7 ** retry_count))
                print(f"🔄 重试 {retry_count}/{max_retries}: 降低 max_model_len 到 {current_max_len} (原始: {original_max_len})")
            
            llm_kwargs = {
                'model': config['model'],
                'tensor_parallel_size': config.get('tensor_parallel_size', 2),
                'gpu_memory_utilization': config.get('gpu_memory_utilization', 0.80),
                'max_model_len': current_max_len,
                'trust_remote_code': True,
                'dtype': config.get('dtype', 'auto'),
            }
            
            # 如果指定了缓存目录，添加到参数中
            if config.get('cache_dir'):
                llm_kwargs['download_dir'] = config['cache_dir']
            
            llm = LLM(**llm_kwargs)
            
            if retry_count > 0:
                print(f"✅ 使用降低后的 max_model_len={current_max_len} 成功加载模型")
        except (ValueError, RuntimeError) as e:
            error_msg = str(e)
            if "No available memory for the cache blocks" in error_msg or "cache blocks" in error_msg.lower():
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"⚠️  KV cache 内存不足，将尝试降低 max_model_len 后重试...")
                    continue
                else:
                    print(f"❌ 经过 {max_retries} 次重试后仍然失败")
                    print(f"   当前配置: gpu_memory_utilization={config.get('gpu_memory_utilization', 0.80)}, max_model_len={current_max_len}")
                    print(f"   建议：")
                    print(f"   1. 提高 gpu_memory_utilization（如 0.85-0.90）")
                    print(f"   2. 手动降低 max_model_len（如 1536 或 1024）")
                    print(f"   3. 检查 GPU 是否有其他进程占用（使用 nvidia-smi）")
                    raise
            else:
                # 其他错误直接抛出
                raise
    
    if llm is None:
        raise RuntimeError("无法加载 vLLM 模型")
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=config.get('temperature', 0.2),
        top_p=config.get('top_p', 0.9),
        max_tokens=config.get('max_tokens', 1024),
    )
    
    print("✅ 模型加载完成！")
    print("=" * 60)
    print("")
    
    # 创建线程锁和信号量，确保 vLLM 调用的线程安全
    # vLLM 的 generate 方法不是线程安全的，需要加锁保护
    # 同时使用信号量限制并发数量，避免过多并发请求导致内部线程通信问题
    vllm_lock = threading.Lock()
    # 限制并发数量，默认4（可根据GPU数量和内存调整）
    # 如果遇到OOM错误，可以降低到2；如果GPU利用率不高，可以提高到6-8
    # 建议值：tensor_parallel_size=2 时，并发数设为 2-4 较合适
    max_concurrent = int(os.environ.get('VLLM_MAX_CONCURRENT', '2'))
    vllm_semaphore = asyncio.Semaphore(max_concurrent)
    print(f"✅ 设置 vLLM 并发数: {max_concurrent} (可通过环境变量 VLLM_MAX_CONCURRENT 修改)")
    
    # 创建异步包装函数
    async def async_generate_text(prompt, system_prompt="", history_messages=None, **kwargs):
        """
        异步生成文本函数（包装vLLM的同步调用）
        
        参数:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            history_messages: 历史消息列表（可选），格式: [{"role": "user", "content": "..."}, ...]
            **kwargs: 其他参数（暂时忽略，使用默认sampling_params）
        """
        # 使用信号量限制并发
        async with vllm_semaphore:
            # 构建完整的消息列表
            messages = []
            
            # 添加系统提示（如果有）
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # 添加历史消息（如果有）
            if history_messages:
                # history_messages 已经是列表格式: [{"role": "user", "content": "..."}, ...]
                messages.extend(history_messages)
            
            # 添加当前用户提示
            messages.append({"role": "user", "content": prompt})
            
            # 将消息列表转换为字符串格式（适配vLLM的generate方法）
            # 对于对话模型，需要将消息格式化为模型能理解的格式
            full_prompt = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    full_prompt += f"System: {content}\n\n"
                elif role == 'user':
                    full_prompt += f"User: {content}\n\n"
                elif role == 'assistant':
                    full_prompt += f"Assistant: {content}\n\n"
            
            # 添加提示让模型继续
            full_prompt += "Assistant: "
            
            # 使用线程锁保护 vLLM 调用，避免与内部线程冲突
            # 在线程池中运行同步的vLLM调用（避免阻塞事件循环）
            # 添加重试机制处理超时和其他临时错误
            max_retries = 3
            retry_delay = 2.0  # 重试延迟（秒）
            
            def _generate_with_lock():
                with vllm_lock:
                    return llm.generate([full_prompt], sampling_params)
            
            # 在异步函数中，使用 get_running_loop() 获取当前运行的事件循环
            loop = asyncio.get_running_loop()
            
            # 重试机制
            last_exception = None
            for attempt in range(max_retries):
                try:
                    outputs = await loop.run_in_executor(None, _generate_with_lock)
                    
                    if not outputs or not outputs[0].outputs:
                        return ""
                    
                    return outputs[0].outputs[0].text
                    
                except Exception as e:
                    last_exception = e
                    error_msg = str(e)
                    
                    # 检查是否是超时错误或引擎错误
                    is_timeout_error = "timeout" in error_msg.lower() or "TimeoutError" in str(type(e))
                    is_engine_error = "EngineCore" in error_msg or "EngineDeadError" in str(type(e))
                    
                    if (is_timeout_error or is_engine_error) and attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)  # 指数退避
                        print(f"⚠️  vLLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {error_msg[:100]}")
                        print(f"   等待 {wait_time:.1f} 秒后重试...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # 最后一次尝试或非超时错误，直接抛出
                        raise
            
            # 如果所有重试都失败
            if last_exception:
                raise last_exception
    
    # 获取 tokenizer（用于文本截断）
    tokenizer = llm.get_tokenizer()
    
    return async_generate_text, tokenizer


def get_chunk_files(input_path):
    """
    获取要处理的文件列表
    
    参数:
        input_path: 文件路径或文件夹路径
    
    返回:
        文件路径列表
    """
    input_path = Path(input_path).resolve()
    
    if input_path.is_file():
        # 如果是单个文件，直接返回
        return [str(input_path)]
    elif input_path.is_dir():
        # 如果是文件夹，查找所有 JSON 文件
        json_files = list(input_path.glob("*.json"))
        if not json_files:
            raise ValueError(f"文件夹 {input_path} 中没有找到 JSON 文件")
        return sorted([str(f) for f in json_files])
    else:
        raise FileNotFoundError(f"路径不存在: {input_path}")


def main():
    """主函数 - 使用 GraphRAG 方法提取实体和关系"""
    
    print("="*70)
    print(" 法律知识图谱实体提取 - GraphRAG 方法")
    print("="*70)
    
    # ============================================
    # 配置参数 - 根据你的环境修改这里
    # ============================================
    
    # 输入配置 - 支持单个文件或文件夹路径（支持相对路径和绝对路径）
    # 方式1: 指定单个文件
    #   chunk_input = "datasets/law_test_chunk_v2.json"
    #   或
    #   chunk_input = "/newdataf/SJ/LeanRAG/datasets/law_test_chunk_v2.json"
    # 
    # 方式2: 指定文件夹（会处理文件夹中所有 JSON 文件，推荐）
    #   chunk_input = "datasets/chunks"
    #   或
    #   chunk_input = "/newdataf/SJ/LeanRAG/datasets/chunks"
    
    if current_dir.name == "GraphExtraction":
        # 在 GraphExtraction 目录内，路径需要回到上级
        chunk_input = "../datasets/chunks"  # 修改为你的 chunks 文件夹路径
        output_dir = "../law_kg_output_v2"
    else:
        # 在项目根目录
        chunk_input = "datasets/chunks"  # 修改为你的 chunks 文件夹路径
        output_dir = "law_kg_output_v2"
    
    # 也可以通过环境变量指定路径（优先级更高，推荐用于绝对路径）
    # 例如: export CHUNK_INPUT_PATH="/newdataf/SJ/LeanRAG/datasets/chunks"
    chunk_input = os.environ.get('CHUNK_INPUT_PATH', chunk_input)
    output_dir = os.environ.get('OUTPUT_DIR', output_dir)
    
    # 如果使用绝对路径，可以直接在这里设置（取消下面的注释并修改路径）
    # 注意：这会覆盖环境变量的设置
    # chunk_input = "/newdataf/SJ/LeanRAG/datasets/chunks"
    output_dir = "/newdataf/SJ/LeanRAG/basicLaw_doc_output"
    
    # 检查输出目录权限，如果无写权限则使用备用目录
    def check_and_get_output_dir(desired_dir):
        """检查目录权限，如果无写权限则使用备用目录，并确保目录属于当前用户"""
        import tempfile
        from pathlib import Path
        import getpass
        import stat
        
        current_user = getpass.getuser()
        desired_path = Path(desired_dir)
        
        try:
            # 如果目录已存在，检查所有者和权限
            if desired_path.exists():
                try:
                    # 获取目录的 stat 信息
                    dir_stat = desired_path.stat()
                    # 检查目录所有者（在 Unix 系统上）
                    if hasattr(dir_stat, 'st_uid'):
                        import pwd
                        try:
                            dir_owner = pwd.getpwuid(dir_stat.st_uid).pw_name
                            if dir_owner != current_user:
                                print(f"⚠️  目录 {desired_dir} 的所有者是 {dir_owner}，当前用户是 {current_user}")
                                print(f"   这通常是因为之前使用 sudo 运行过脚本")
                                print(f"   建议修复权限：sudo chown -R {current_user}:{current_user} {desired_dir}")
                        except (KeyError, AttributeError):
                            # 在某些系统上可能无法获取用户名，跳过
                            pass
                except Exception as e:
                    # 权限检查失败，继续尝试
                    pass
            
            # 尝试创建目录（如果不存在）或检查权限
            desired_path.mkdir(parents=True, exist_ok=True)
            
            # 尝试创建一个测试文件
            test_file = desired_path / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
                print(f"✅ 输出目录权限检查通过: {desired_dir}")
                return str(desired_path)
            except PermissionError:
                print(f"⚠️  目录 {desired_dir} 没有写权限")
                # 使用备用目录（当前用户主目录，保持与原目录相同的名称）
                fallback_dir = Path.home() / "basicLaw_doc_output"
                fallback_dir.mkdir(parents=True, exist_ok=True)
                print(f"   将使用备用目录: {fallback_dir}")
                print(f"   提示：如需使用原目录，请执行：")
                print(f"   sudo chown -R {current_user}:{current_user} {desired_dir}  # 修复所有者")
                print(f"   sudo chmod 755 {desired_dir}  # 修复权限")
                return str(fallback_dir)
        except PermissionError as e:
            print(f"⚠️  无法创建或访问目录 {desired_dir}: {e}")
            # 使用备用目录（当前用户主目录，保持与原目录相同的名称）
            fallback_dir = Path.home() / "basicLaw_doc_output"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            print(f"   将使用备用目录: {fallback_dir}")
            print(f"   提示：如需使用原目录，请执行：")
            print(f"   sudo chown -R {current_user}:{current_user} {desired_dir}  # 修复所有者")
            return str(fallback_dir)
        except Exception as e:
            print(f"⚠️  检查目录时出错: {e}")
            # 使用备用目录（当前用户主目录，保持与原目录相同的名称）
            fallback_dir = Path.home() / "basicLaw_doc_output"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            print(f"   将使用备用目录: {fallback_dir}")
            return str(fallback_dir)
    
    output_dir = check_and_get_output_dir(output_dir)
    
    # LLM 配置（直接使用vLLM，不需要启动API服务）
    # 优势：
    # 1. 无需启动API服务，直接在代码中使用
    # 2. 更高效，减少HTTP开销
    # 3. 更简单，适合脚本和批处理任务
    # 4. vLLM会自动缓存已加载的模型
    # 
    # 配置说明：
    # - tensor_parallel_size: 张量并行GPU数量（1, 2, 4, 8等）
    #   使用多GPU可以显著降低每张卡的内存压力，解决OOM问题
    # - gpu_ids: 指定使用的GPU编号（默认"2,3"使用后两张卡）
    #   可通过环境变量 VLLM_GPU_IDS 修改，例如: export VLLM_GPU_IDS="2,3"
    #   如果已设置 CUDA_VISIBLE_DEVICES，将优先使用环境变量的设置
    # - gpu_memory_utilization: 每张GPU的内存利用率（0.0-1.0）
    #   默认0.70（如果GPU有其他进程占用，需要提高以给KV cache足够空间）
    #   如果遇到"No available memory for cache blocks"错误，需要提高到0.70-0.75
    #   如果遇到OOM错误，可以降低到0.65
    # - max_model_len: 最大序列长度
    #   默认3072（需要至少容纳输入文本+提示词，如果遇到"decoder prompt is longer"错误，需要提高）
    #   如果内存充足且需要处理更长文本，可以提高到4096或6144
    #   如果遇到OOM错误，可以降低到2048，但可能无法处理较长的文本块
    #
    # 自定义模型缓存位置：
    # 方法1: 通过环境变量设置（推荐）
    #   export HF_HOME=/path/to/your/cache
    #   或
    #   export TRANSFORMERS_CACHE=/path/to/your/cache
    # 
    # 方法2: 在配置中直接指定
    #   在 llm_config 中设置 'cache_dir': '/path/to/your/cache'
    
    # 使用本地 Qwen2-7B-Instruct 模型
    # 优先使用环境变量指定的模型（如果设置了的话）
    # 原配置记录：
    # - DeepSeek-V2-Lite-Chat (MoE 模型，需要 23GB+ 显存，单卡无法加载)
    # - Qwen/Qwen2.5-7B-Instruct (从 HuggingFace 下载)
    if os.environ.get('VLLM_MODEL_PATH'):
        model_path = os.environ.get('VLLM_MODEL_PATH')
        print(f"📁 使用环境变量指定的模型路径: {model_path}")
    elif os.environ.get('VLLM_MODEL_NAME'):
        # 如果指定了模型名称，使用模型名称（会自动从缓存或下载）
        model_path = os.environ.get('VLLM_MODEL_NAME')
        print(f"📁 使用环境变量指定的模型名称: {model_path}")
    else:
        # 默认使用本地 Qwen2-1.5B-Instruct 模型（约 3-4GB 显存，单张 24GB 显卡完全够用）
        # 本地模型路径：/newdatad/WHH/MyEmoHH/models/Qwen2-1.5B-Instruct/
        model_path = '/newdatad/WHH/MyEmoHH/models/Qwen2-1.5B-Instruct/'
        print(f"📁 使用本地模型: {model_path}")
        print(f"💡 提示: 使用服务器本地模型，无需从 HuggingFace 下载")
        print(f"   注意: 改用 Qwen2-1.5B-Instruct (3-4GB) 替代 Qwen2-7B (14-16GB) 以避免内存不足")
    
    # Qwen2-1.5B-Instruct 模型配置
    # 该模型约需 3-4GB 显存，单张 24GB 显卡有充足空间用于模型和 KV cache
    # 相比 Qwen2-7B (14-16GB) 显著降低显存需求
    default_mem_util = '0.85'  # 1.5B 模型可以使用较高的内存利用率
    default_max_len = '4096'  # 1.5B 模型可以使用较长的序列长度
    print(f"📌 使用 Qwen2-1.5B-Instruct 配置（mem_util=0.85, max_len=4096）")
    print(f"   模型大小: 约 3-4GB，单张 24GB 显卡有充足空间")
    
    llm_config = {
        'model': model_path,
        'tensor_parallel_size': int(os.environ.get('VLLM_TENSOR_PARALLEL_SIZE', '1')),  # 默认使用1张GPU（单卡模式，GPU 3）
        'gpu_ids': os.environ.get('VLLM_GPU_IDS', '3'),  # 默认使用 GPU 3（第四张卡），可通过环境变量 VLLM_GPU_IDS 修改
        'gpu_memory_utilization': float(os.environ.get('VLLM_GPU_MEM_UTIL', default_mem_util)),  # 根据模型类型自动调整
        'max_model_len': int(os.environ.get('VLLM_MAX_MODEL_LEN', default_max_len)),  # 根据模型类型自动调整
        'temperature': float(os.environ.get('VLLM_TEMPERATURE', '0.2')),  # 默认0.2
        'top_p': float(os.environ.get('VLLM_TOP_P', '0.9')),  # 默认0.9
        'max_tokens': int(os.environ.get('VLLM_MAX_TOKENS', '1024')),  # 默认1024
        'dtype': os.environ.get('VLLM_DTYPE', 'auto'),  # 默认auto
        'cache_dir': os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or os.environ.get('VLLM_CACHE_DIR'),  # 自定义缓存目录
    }
    
    # ============================================
    # 步骤1: 检查输入文件/文件夹
    # ============================================
    print(f"\n{'='*60}")
    print("步骤1: 检查输入文件/文件夹")
    print(f"{'='*60}")
    
    try:
        chunk_files = get_chunk_files(chunk_input)
        print(f"✅ 找到 {len(chunk_files)} 个文件需要处理:")
        total_size = 0
        for i, file_path in enumerate(chunk_files, 1):
            file_size = os.path.getsize(file_path) / 1024  # KB
            total_size += file_size
            file_name = os.path.basename(file_path)
            print(f"   [{i}] {file_name} ({file_size:.2f} KB)")
        print(f"\n   总大小: {total_size:.2f} KB ({total_size/1024:.2f} MB)")
    except Exception as e:
        print(f"❌ 错误: {e}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"输入路径: {chunk_input}")
        return
    
    # ============================================
    # 步骤2: 加载所有分块数据
    # ============================================
    print(f"\n{'='*60}")
    print("步骤2: 加载分块数据")
    print(f"{'='*60}")
    
    try:
        all_chunks = {}
        total_chunks = 0
        
        for i, chunk_file in enumerate(chunk_files, 1):
            file_name = os.path.basename(chunk_file)
            print(f"\n正在加载文件 [{i}/{len(chunk_files)}]: {file_name}")
            
            file_chunks = get_chunk(chunk_file)
            print(f"  ✅ 加载了 {len(file_chunks)} 个文本块")
            
            # 合并到总字典中（如果 hash_code 重复，后面的会覆盖前面的）
            # 由于 hash_code 是基于内容生成的，重复的可能性很小
            before_count = len(all_chunks)
            all_chunks.update(file_chunks)
            after_count = len(all_chunks)
            
            if after_count - before_count < len(file_chunks):
                duplicate_count = len(file_chunks) - (after_count - before_count)
                print(f"  ⚠️  发现 {duplicate_count} 个重复的 hash_code（已去重）")
            
            total_chunks += len(file_chunks)
        
        print(f"\n✅ 总共加载 {len(all_chunks)} 个唯一文本块（原始总数: {total_chunks}）")
        
        # 显示第一个块的信息
        if all_chunks:
            first_key = list(all_chunks.keys())[0]
            first_text = all_chunks[first_key]
            print(f"\n第一个块信息:")
            print(f"  Hash ID: {first_key[:32]}...")
            print(f"  文本长度: {len(first_text)} 字符")
            print(f"  文本预览: {first_text[:100]}...")
        
        chunks = all_chunks
        
    except Exception as e:
        print(f"❌ 加载分块文件失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================
    # 步骤3: 加载 vLLM 模型（直接模式）
    # ============================================
    print(f"\n{'='*60}")
    print("步骤3: 加载 vLLM 模型")
    print(f"{'='*60}")
    
    try:
        use_llm_func, tokenizer = setup_vllm_direct(llm_config)
        print("✅ vLLM 模型加载完成")
    except Exception as e:
        print(f"❌ vLLM 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================
    # 步骤4: 提取实体和关系
    # ============================================
    print(f"\n{'='*60}")
    print("步骤4: 提取实体和关系 (使用 GraphRAG)")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    print("\n开始提取... (这可能需要一些时间)")
    print("-" * 60)
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 运行异步提取
        # main() 是同步函数，应该使用 asyncio.run() 来运行异步函数
        # asyncio.run() 会自动创建新的事件循环并运行，完成后清理
        
        # 追加模式配置（默认True，支持追加新数据并去重）
        # 设置为False将覆盖现有文件
        append_mode = os.environ.get('APPEND_MODE', 'true').lower() == 'true'
        if append_mode:
            print(f"📝 模式: 追加模式（将去重并追加新数据）")
        else:
            print(f"📝 模式: 覆盖模式（将覆盖现有文件）")
        
        # 获取 max_model_len 配置，传递给 triple_extraction 用于文本截断
        max_model_len = llm_config.get('max_model_len', 3072)
        
        asyncio.run(
            triple_extraction(chunks, use_llm_func, output_dir, append_mode=append_mode, max_model_len=max_model_len, tokenizer=tokenizer)
        )
        
        print("-" * 60)
        print("✅ 实体和关系提取完成!")
        
    except Exception as e:
        print(f"\n❌ 提取过程出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================
    # 步骤5: 检查输出结果
    # ============================================
    print(f"\n{'='*60}")
    print("步骤5: 输出结果统计")
    print(f"{'='*60}")
    
    entity_file = f"{output_dir}/entity.jsonl"
    relation_file = f"{output_dir}/relation.jsonl"
    
    # 统计实体
    if os.path.exists(entity_file):
        with open(entity_file, 'r', encoding='utf-8') as f:
            entities = [json.loads(line) for line in f if line.strip()]
        print(f"✅ 实体文件: {entity_file}")
        print(f"   实体数量: {len(entities)}")
        
        if entities:
            print(f"\n   实体示例:")
            for i, entity in enumerate(entities[:3], 1):
                print(f"     [{i}] {entity.get('entity_name', 'N/A')}")
                print(f"         类型: {entity.get('entity_type', 'N/A')}")
                print(f"         描述: {entity.get('description', 'N/A')[:60]}...")
    else:
        print(f"⚠️  实体文件不存在: {entity_file}")
    
    # 统计关系
    if os.path.exists(relation_file):
        with open(relation_file, 'r', encoding='utf-8') as f:
            raw_relations = [json.loads(line) for line in f if line.strip()]
        # 处理可能存在的列表格式（向后兼容）
        relations = []
        for rel in raw_relations:
            if isinstance(rel, list):
                # 如果是列表，展平它
                relations.extend(rel)
            else:
                # 如果是字典，直接添加
                relations.append(rel)
        print(f"\n✅ 关系文件: {relation_file}")
        print(f"   关系数量: {len(relations)}")
        
        if relations:
            print(f"\n   关系示例:")
            for i, rel in enumerate(relations[:3], 1):
                src = rel.get('src_id', 'N/A')
                tgt = rel.get('tgt_id', 'N/A')
                desc = rel.get('description', 'N/A')
                print(f"     [{i}] {src} -> {tgt}")
                print(f"         描述: {desc[:60]}...")
    else:
        print(f"⚠️  关系文件不存在: {relation_file}")
    
    # ============================================
    # 步骤6: 去重处理
    # ============================================
    print(f"\n{'='*60}")
    print("步骤6: 去重和后处理")
    print(f"{'='*60}")
    print("接下来需要运行去重脚本:")
    print(f"  python GraphExtraction/deal_triple.py")
    print("\n或者修改 deal_triple.py 中的路径后运行:")
    print(f"  working_dir='{output_dir}'")
    print(f"  output_path='{output_dir}_processed'")
    
    # ============================================
    # 完成
    # ============================================
    print(f"\n{'='*70}")
    print(" 实体提取完成!")
    print(f"{'='*70}")
    print(f"\n输出文件位置:")
    print(f"  - 实体: {entity_file}")
    print(f"  - 关系: {relation_file}")
    print(f"\n后续步骤:")
    print(f"  1. 运行去重脚本: python law_deal_triple.py")
    print(f"  2. 构建知识图谱: python build_graph.py")
    print(f"  3. 查询测试: python query_graph.py")


if __name__ == "__main__":
    main()

