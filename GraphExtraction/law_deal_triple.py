#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
法律实体去重和后处理脚本
基于 GraphExtraction/deal_triple.py 修改
使用本地Qwen2-7B-Instruct模型，双卡并行
"""

import json
import os
import sys
import threading
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken

# ⚠️ 重要：必须在导入任何 CUDA 相关库（如 vLLM、Ray、torch）之前设置 CUDA_VISIBLE_DEVICES
# 否则环境变量设置无效，会使用默认 GPU（通常是 GPU 0）
# 默认配置：使用 GPU 3（第四张卡），单卡运行
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

print(f"✅ 当前 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}，将使用 GPU 3（第四张卡）")

# 添加项目根路径（上一级目录）到 sys.path，确保可导入根目录下模块
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tools.utils import read_jsonl, write_jsonl, create_if_not_exist, InstanceManager
from prompt import PROMPTS

# 导入vLLM（直接使用，不需要API服务）
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  警告: vLLM 未安装，无法使用本地模型")


# 描述摘要的token阈值
THRESHOLD = 50


def setup_vllm_sync(config):
    """
    直接加载vLLM模型（同步版本，用于摘要任务）
    
    参数:
        config: LLM配置字典
            - model: 模型路径或名称
            - tensor_parallel_size: 张量并行GPU数量（默认1，单卡模式）
            - gpu_memory_utilization: 每张GPU的内存利用率（默认0.80）
            - max_model_len: 最大序列长度（默认4096）
            - temperature: 采样温度（默认0.2）
            - top_p: top_p采样阈值（默认0.9）
            - max_tokens: 最大生成token数（默认1024）
    
    返回:
        同步LLM生成函数
    """
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM 未安装，请先安装: pip install vllm")
    
    print(f"\n{'='*60}")
    print("加载 vLLM 模型（同步模式，用于摘要）")
    print(f"{'='*60}")
    print(f"模型路径: {config['model']}")
    print(f"张量并行GPU数量: {config.get('tensor_parallel_size', 1)}")
    print(f"每张GPU内存利用率: {config.get('gpu_memory_utilization', 0.80)}")
    print(f"最大序列长度: {config.get('max_model_len', 4096)}")
    
    print("=" * 60)
    print("\n正在加载模型...（这可能需要一些时间）")
    
    # 设置PyTorch内存分配配置（避免内存碎片）
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 获取 tensor_parallel_size
    tensor_parallel_size = config.get('tensor_parallel_size', 1)
    
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
    
    # 加载vLLM模型
    llm_kwargs = {
        'model': config['model'],
        'tensor_parallel_size': tensor_parallel_size,
        'gpu_memory_utilization': config.get('gpu_memory_utilization', 0.80),
        'max_model_len': config.get('max_model_len', 4096),
        'trust_remote_code': True,
        'dtype': config.get('dtype', 'auto'),
    }
    
    llm = LLM(**llm_kwargs)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=config.get('temperature', 0.2),
        top_p=config.get('top_p', 0.9),
        max_tokens=config.get('max_tokens', 1024),  # 与提取脚本一致
    )
    
    print("✅ 模型加载完成！")
    print("=" * 60)
    print("")
    
    # 创建线程锁，确保 vLLM 调用的线程安全
    vllm_lock = threading.Lock()
    
    # 创建同步生成函数
    def generate_text_sync(prompt, system_prompt="", **kwargs):
        """
        同步生成文本函数（包装vLLM的同步调用）
        
        参数:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            **kwargs: 其他参数（暂时忽略，使用默认sampling_params）
        
        返回:
            生成的文本字符串
        """
        # 构建完整的提示
        full_prompt = ""
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"
        full_prompt += f"User: {prompt}\n\nAssistant: "
        
        # 使用线程锁保护 vLLM 调用
        with vllm_lock:
            outputs = llm.generate([full_prompt], sampling_params)
        
        if not outputs or not outputs[0].outputs:
            return ""
        
        return outputs[0].outputs[0].text.strip()
    
    return generate_text_sync


def summarize_entity(entity_name, description, summary_prompt, threshold, tokenizer, use_llm):
    """
    如果实体描述超过阈值，使用LLM进行摘要
    
    参数:
        entity_name: 实体名称
        description: 实体描述
        summary_prompt: 摘要提示词模板
        threshold: token阈值
        tokenizer: tiktoken tokenizer
        use_llm: LLM函数
    """
    tokens = len(tokenizer.encode(description))
    if tokens > threshold:
        exact_prompt = summary_prompt.format(entity_name=entity_name, description=description)
        response = use_llm(exact_prompt)
        return entity_name, response
    return entity_name, description  # 不需要摘要则返回原始描述


def deal_duplicate_entity(working_dir, output_path, use_llm=None):
    """
    处理重复实体，合并描述，并进行摘要
    
    参数:
        working_dir: 输入目录（包含 entity.jsonl 和 relation.jsonl）
        output_path: 输出目录
        use_llm: LLM函数（可选，用于摘要）
    """
    print(f"\n{'='*60}")
    print("去重和后处理")
    print(f"{'='*60}")
    
    relation_path = f"{working_dir}/relation.jsonl"
    relation_output_path = f"{output_path}/relation.jsonl"
    entity_path = f"{working_dir}/entity.jsonl"
    entity_output_path = f"{output_path}/entity.jsonl"
    
    # 检查文件是否存在
    if not os.path.exists(entity_path):
        print(f"❌ 错误: 实体文件不存在: {entity_path}")
        return False
    
    if not os.path.exists(relation_path):
        print(f"❌ 错误: 关系文件不存在: {relation_path}")
        return False
    
    # 创建输出目录
    create_if_not_exist(output_path)
    
    # ============================================
    # 处理实体
    # ============================================
    print(f"\n处理实体文件: {entity_path}")
    
    all_entities = []
    e_dic = {}
    # 优先使用中文摘要提示词，如果没有则使用英文版本
    summary_prompt = PROMPTS.get('summary_entities_zh', PROMPTS.get('summary_entities', ''))
    
    # 读取并合并重复实体
    with open(entity_path, "r", encoding='utf-8') as f:
        for line_num, xline in enumerate(f, 1):
            try:
                line = json.loads(xline)
                entity_name = str(line['entity_name']).replace('"', '')
                entity_type = line.get('entity_type', '').replace('"', '')
                description = line['description'].replace('"', '')
                source_id = line['source_id']
                
                if entity_name not in e_dic:
                    e_dic[entity_name] = dict(
                        entity_name=str(entity_name),
                        entity_type=entity_type,
                        description=description,
                        source_id=source_id,
                        degree=0,
                    )
                else:
                    # 合并描述
                    e_dic[entity_name]['description'] += " | " + description
                    # 合并来源ID
                    if e_dic[entity_name]['source_id'] != source_id:
                        e_dic[entity_name]['source_id'] += "|" + source_id
            
            except Exception as e:
                print(f"⚠️  警告: 第 {line_num} 行解析失败: {e}")
                continue
    
    print(f"   原始实体数: {line_num}")
    print(f"   去重后实体数: {len(e_dic)}")
    
    # 去重来源ID
    for k, v in e_dic.items():
        v['source_id'] = "|".join(set(v['source_id'].split("|")))
    
    # ============================================
    # 摘要长描述
    # ============================================
    tokenizer = tiktoken.get_encoding("cl100k_base")
    to_summarize = []
    
    for k, v in e_dic.items():
        description = v['description']
        tokens = len(tokenizer.encode(description))
        if tokens > THRESHOLD:
            to_summarize.append((k, description))
        else:
            all_entities.append(v)
    
    print(f"   需要摘要的实体数: {len(to_summarize)}")
    
    if to_summarize and use_llm:
        print("   开始摘要处理...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(summarize_entity, k, desc, summary_prompt, THRESHOLD, tokenizer, use_llm): k
                for k, desc in to_summarize
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="   摘要进度"):
                k, summarized_desc = future.result()
                e_dic[k]['description'] = summarized_desc
                all_entities.append(e_dic[k])
    elif to_summarize:
        print("   ⚠️  未提供LLM函数，跳过摘要，直接使用原始描述")
        for k, _ in to_summarize:
            all_entities.append(e_dic[k])
    
    # 保存实体
    write_jsonl(all_entities, entity_output_path)
    print(f"✅ 实体已保存: {entity_output_path}")
    print(f"   最终实体数: {len(all_entities)}")
    
    # ============================================
    # 处理关系
    # ============================================
    print(f"\n处理关系文件: {relation_path}")
    
    all_relations = []
    r_dic = {}  # 用于关系去重: key = (src_tgt, tgt_src)
    raw_relation_count = 0
    
    with open(relation_path, "r", encoding='utf-8') as f:
        for line_num, xline in enumerate(f, 1):
            try:
                data = json.loads(xline)
                raw_relation_count += 1
                
                # 处理两种数据格式：列表或字典
                if isinstance(data, list):
                    line = data[0]
                else:
                    line = data
                
                # 根据数据格式选择正确的字段名
                if 'src_id' in line:
                    src_tgt = str(line['src_id']).replace('"', '')
                    tgt_src = str(line['tgt_id']).replace('"', '')
                else:
                    src_tgt = str(line.get('src_tgt', '')).replace('"', '')
                    tgt_src = str(line.get('tgt_src', '')).replace('"', '')
                
                description = line['description'].replace('"', '')
                # 如果原数据有weight字段，使用它；否则默认为1
                weight = line.get('weight', 1)
                if isinstance(weight, (int, float)):
                    weight = float(weight)
                else:
                    weight = 1.0
                source_id = line['source_id']
                
                # 关系去重：使用 (src_tgt, tgt_src) 作为唯一键
                relation_key = (src_tgt, tgt_src)
                
                if relation_key not in r_dic:
                    # 新关系
                    r_dic[relation_key] = dict(
                        src_tgt=src_tgt,
                        tgt_src=tgt_src,
                        description=description,
                        weight=weight,
                        source_id=source_id
                    )
                else:
                    # 重复关系：合并描述、累加权重、合并来源ID
                    r_dic[relation_key]['description'] += " | " + description
                    r_dic[relation_key]['weight'] += weight  # 累加权重
                    if r_dic[relation_key]['source_id'] != source_id:
                        r_dic[relation_key]['source_id'] += "|" + source_id
            
            except Exception as e:
                print(f"⚠️  警告: 第 {line_num} 行解析失败: {e}")
                continue
    
    # 去重来源ID
    for k, v in r_dic.items():
        v['source_id'] = "|".join(set(v['source_id'].split("|")))
        all_relations.append(v)
    
    print(f"   原始关系数: {raw_relation_count}")
    print(f"   去重后关系数: {len(all_relations)}")
    
    # 保存关系
    write_jsonl(all_relations, relation_output_path)
    print(f"✅ 关系已保存: {relation_output_path}")
    print(f"   最终关系数: {len(all_relations)}")
    
    return True


def main():
    """主函数"""
    
    print("="*70)
    print(" 法律实体去重和后处理")
    print("="*70)
    
    # ============================================
    # 配置参数 - 根据你的环境修改这里
    # ============================================
    
    # 输入输出目录
    # 注意：这里应该指向 law_extract_graphrag_parllar2.py 的输出目录
    working_dir = "/newdataf/SJ/LeanRAG/output/social_law_7B"            # GraphRAG 输出的原始目录
    output_path = "/newdataf/SJ/LeanRAG/output/social_law_7B_processed"  # 处理后的输出目录
    
    # 是否使用LLM进行摘要（可选）
    use_llm_for_summary = True  # 改为 True，启用摘要功能并使用本地 vLLM 模型
    
    # LLM 配置（本地 vLLM，与提取脚本 law_extract_graphrag_parllar2_QWen7B.py 使用相同的模型）
    # 使用本地 Qwen2-7B-Instruct 模型（与提取脚本一致）
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
        print(f"   配置: 使用 tensor_parallel_size=1 单卡模式，需要约 14-16GB 显存")
    
    # Qwen2-7B-Instruct 模型配置（单卡模式）
    # 该模型约需 14-16GB 显存，使用单卡时需要较高的内存利用率
    default_mem_util = '0.85'  # 单卡模式使用较高的内存利用率
    default_max_len = '3072'  # 单卡模式使用适中的序列长度
    print(f"📌 使用 Qwen2-7B-Instruct 配置（单卡模式：mem_util=0.85, max_len=3072）")
    print(f"   模型大小: 约 14-16GB，单卡运行需要24GB显存")
    
    llm_config = {
        'model': model_path,
        'tensor_parallel_size': int(os.environ.get('VLLM_TENSOR_PARALLEL_SIZE', '1')),  # 默认使用1张GPU（单卡模式，GPU 3）
        'gpu_ids': os.environ.get('VLLM_GPU_IDS', '3'),  # 默认使用 GPU 3（第四张卡），可通过环境变量 VLLM_GPU_IDS 修改
        'gpu_memory_utilization': float(os.environ.get('VLLM_GPU_MEM_UTIL', default_mem_util)),  # 与提取脚本一致
        'max_model_len': int(os.environ.get('VLLM_MAX_MODEL_LEN', default_max_len)),  # 与提取脚本一致
        'temperature': float(os.environ.get('VLLM_TEMPERATURE', '0.2')),  # 默认0.2
        'top_p': float(os.environ.get('VLLM_TOP_P', '0.9')),  # 默认0.9
        'max_tokens': int(os.environ.get('VLLM_MAX_TOKENS', '1024')),  # 默认512（摘要任务使用较短输出）
        'dtype': os.environ.get('VLLM_DTYPE', 'auto'),  # 默认auto
        'cache_dir': os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or os.environ.get('VLLM_CACHE_DIR'),  # 自定义缓存目录
    }
    
    # ============================================
    # 设置 LLM（如果需要）
    # ============================================
    use_llm = None
    
    if use_llm_for_summary:
        print(f"\n{'='*60}")
        print("配置 LLM 用于描述摘要（使用本地 vLLM）")
        print(f"{'='*60}")
        
        try:
            use_llm = setup_vllm_sync(llm_config)
            print("✅ LLM 配置完成")
        except Exception as e:
            print(f"❌ LLM 配置失败: {e}")
            print("⚠️  将跳过摘要功能，直接使用原始描述")
            use_llm = None
    else:
        print(f"\n⚠️  未启用LLM摘要功能（use_llm_for_summary=False）")
    
    # ============================================
    # 执行去重处理
    # ============================================
    success = deal_duplicate_entity(working_dir, output_path, use_llm)
    
    if success:
        print(f"\n{'='*70}")
        print(" 处理完成!")
        print(f"{'='*70}")
        print(f"\n输出文件:")
        print(f"  - 实体: {output_path}/entity.jsonl")
        print(f"  - 关系: {output_path}/relation.jsonl")
        print(f"\n后续步骤:")
        print(f"  1. 将处理后的文件用于构建知识图谱")
        print(f"  2. 运行: python build_graph.py")
    else:
        print(f"\n❌ 处理失败")


if __name__ == "__main__":
    main()


