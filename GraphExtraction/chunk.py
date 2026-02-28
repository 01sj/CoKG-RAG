import os
import json
from _utils import split_string_by_multi_markers,_handle_single_entity_extraction,\
    _handle_single_relationship_extraction,clean_str,pack_user_ass_to_openai_messages
import sys
from pathlib import Path
# 动态加入项目根目录，避免硬编码路径导致导入失败
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from tools.utils import InstanceManager,write_jsonl
from collections import Counter, defaultdict
from prompt import PROMPTS
import asyncio
import re
import copy
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 用于文本长度检查和截断
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️  tiktoken 未安装，将使用字符数估算（可能不够准确）")


class LocalGenerator:
    def __init__(self, model_id: str, max_new_tokens: int = 1024, temperature: float = 0.2, top_p: float = 0.95):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.generation_kwargs = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

    def __call__(self, prompt: str, system_prompt: str = None, history_messages = [], **kwargs):
        args = dict(self.generation_kwargs)
        if "max_new_tokens" in kwargs:
            args["max_new_tokens"] = kwargs["max_new_tokens"]
        if "temperature" in kwargs:
            args["temperature"] = kwargs["temperature"]
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=args.get("temperature", 0.0) > 0,
                temperature=args.get("temperature", 0.2),
                top_p=args.get("top_p", 0.95),
                max_new_tokens=args.get("max_new_tokens", 1024),
                repetition_penalty=args.get("repetition_penalty", 1.1),
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)



def get_chunk(chunk_file):
    doc_name=os.path.basename(chunk_file).rsplit(".",1)[0]
    
    # 首先检查文件是否存在
    if not os.path.exists(chunk_file):
        raise FileNotFoundError(f"文件不存在: {chunk_file}")
    
    print(f"正在读取文件: {chunk_file}")
    
    # 使用二进制模式读取，然后尝试不同的解码方式
    try:
        with open(chunk_file, "rb") as f:
            raw_data = f.read()
        
        print(f"文件大小: {len(raw_data)} 字节")
        
        # 尝试不同的编码，使用errors='ignore'来跳过无法解码的字符
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252', 'utf-16', 'utf-32']
        
        for encoding in encodings:
            try:
                print(f"尝试编码: {encoding}")
                
                # 使用errors='ignore'来忽略无法解码的字符
                content = raw_data.decode(encoding, errors='ignore')
                
                # 清理控制字符，但保留必要的JSON字符
                cleaned_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
                
                # 尝试解析JSON
                corpus = json.loads(cleaned_content)
                print(f"成功使用编码 {encoding} 读取文件 (忽略错误字符)")
                chunks = {item["hash_code"]: item["text"] for item in corpus}
                return chunks
                
            except json.JSONDecodeError as e:
                print(f"编码 {encoding} JSON解析失败: {e}")
                continue
            except Exception as e:
                print(f"编码 {encoding} 其他错误: {e}")
                continue
        
        # 如果所有编码都失败，尝试更激进的清理方法
        print("尝试激进的字符清理方法...")
        try:
            # 使用latin-1读取（它可以读取任何字节），然后清理
            content = raw_data.decode('latin-1', errors='ignore')
            
            # 只保留ASCII可打印字符和基本的JSON字符
            import string
            allowed_chars = string.printable + '""''—–'  # 包含一些常见的Unicode引号和破折号
            cleaned_content = ''.join(char for char in content if char in allowed_chars)
            
            # 修复可能的JSON格式问题
            cleaned_content = re.sub(r'[\x00-\x1F\x7F]', '', cleaned_content)  # 移除控制字符
            cleaned_content = re.sub(r'([^\\])"([^"]*?)"', r'\1"\2"', cleaned_content)  # 修复引号
            
            corpus = json.loads(cleaned_content)
            print("通过激进清理方法成功解析JSON")
            chunks = {item["hash_code"]: item["text"] for item in corpus}
            return chunks
            
        except Exception as e:
            print(f"激进清理方法也失败: {e}")
    
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    
    raise ValueError(f"无法读取文件 {chunk_file}，尝试了所有可能的方法都失败")

def truncate_text_by_tokens(text, max_tokens, tokenizer=None, reserve_ratio=0.8):
    """
    根据 token 数量截断文本，避免超出 max_model_len
    
    参数:
        text: 要截断的文本
        max_tokens: 最大 token 数量
        tokenizer: tiktoken tokenizer（如果可用）
        reserve_ratio: 保留比例（0.8 表示保留 80% 的 token 空间给提示词和输出）
    
    返回:
        (truncated_text, was_truncated, original_tokens, truncated_tokens)
    """
    if not text:
        return text, False, 0, 0
    
    # 计算实际可用的 token 数量（保留一部分给提示词和输出）
    available_tokens = int(max_tokens * reserve_ratio)
    
    if TIKTOKEN_AVAILABLE and tokenizer is None:
        try:
            # 使用 cl100k_base（GPT-4/DeepSeek 等模型常用）
            tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            tokenizer = None
    
    if tokenizer:
        # 使用 tiktoken 精确计算
        tokens = tokenizer.encode(text)
        original_token_count = len(tokens)
        
        if original_token_count <= available_tokens:
            return text, False, original_token_count, original_token_count
        
        # 截断到可用 token 数量
        truncated_tokens = tokens[:available_tokens]
        truncated_text = tokenizer.decode(truncated_tokens)
        return truncated_text, True, original_token_count, available_tokens
    else:
        # 回退方案：使用字符数估算（中文约 1.5 字符/token，英文约 4 字符/token）
        # 保守估计：按 2 字符/token 计算
        estimated_tokens = len(text) // 2
        if estimated_tokens <= available_tokens:
            return text, False, estimated_tokens, estimated_tokens
        
        # 截断文本
        max_chars = available_tokens * 2
        truncated_text = text[:max_chars]
        return truncated_text, True, estimated_tokens, available_tokens


async def triple_extraction(chunks,use_llm_func,output_dir,append_mode=True,max_model_len=3072,tokenizer=None):
    
    # extract entities
    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    
    # 性能监控：记录开始时间
    total_start_time = time.time()
    
    already_processed = 0
    already_entities = 0
    already_relations = 0
    ordered_chunks = list(chunks.items())
    print(f"\n开始处理 {len(ordered_chunks)} 个文本块...")
    async def _process_single_content_entity(chunk_key_dp,use_llm_func):           # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        content = chunk_key_dp[1]
        entity_extract_prompt = PROMPTS["entity_extraction"]        # give 3 examples in the prompt context
        relation_extract_prompt = PROMPTS["relation_extraction"]
        continue_prompt = PROMPTS["entiti_continue_extraction"]     # means low quality in the last extraction
        if_loop_prompt = PROMPTS["entiti_if_loop_extraction"] 
        context_base_entity = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["META_ENTITY_TYPES"])
    )
        entity_extract_max_gleaning=1
        hint_prompt = entity_extract_prompt.format(**context_base_entity, input_text=content)      # fill in the parameter
        final_result = await use_llm_func(hint_prompt)                                      # feed into LLM with the prompt

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)               # set as history
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)      # add to history
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(                                            # split entities from result --> list of entities
            final_result,
            [context_base_entity["record_delimiter"], context_base_entity["completion_delimiter"]],
        )
        # resolve the entities
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(          # split entity
                record, [context_base_entity["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(       # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1                                      # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][                     # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)
    
    # 实体提取阶段
    entity_start_time = time.time()
    entity_results = await asyncio.gather(
        *[_process_single_content_entity(c,use_llm_func) for c in ordered_chunks]
    )
    entity_end_time = time.time()
    entity_duration = entity_end_time - entity_start_time
    print()  # clear the progress bar
    print(f"✅ 实体提取完成，耗时: {entity_duration:.2f} 秒 ({entity_duration/60:.2f} 分钟)")
    print(f"   平均每个块: {entity_duration/len(ordered_chunks):.2f} 秒")

    # fetch all entities from results
    all_entities = {}
    for item in entity_results:
        for k, v in item[0].items():
            value = v[0]
            all_entities[k] = v[0]
    context_entities = {key[0]: list(x[0].keys()) for key, x in zip(ordered_chunks, entity_results)}
    already_processed = 0
    async def _process_single_content_relation(chunk_key_dp,use_llm_func):           # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        content = chunk_key_dp[1]
        
        # 检查并截断输入文本，避免超出 max_model_len
        truncated_content, was_truncated, orig_tokens, trunc_tokens = truncate_text_by_tokens(
            content, max_model_len, tokenizer
        )
        if was_truncated:
            print(f"⚠️  文本块 {chunk_key[:16]}... 超出长度限制，已截断: {orig_tokens} -> {trunc_tokens} tokens")
            content = truncated_content
        entity_extract_prompt = PROMPTS["entity_extraction"]        # give 3 examples in the prompt context
        relation_extract_prompt = PROMPTS["relation_extraction"]
        continue_prompt = PROMPTS["entiti_continue_extraction"]     # means low quality in the last extraction
        if_loop_prompt = PROMPTS["entiti_if_loop_extraction"] 
        entities = context_entities[chunk_key]
        context_base_relation = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entities=",".join(entities)
            )
        entity_extract_max_gleaning=1
        hint_prompt = relation_extract_prompt.format(**context_base_relation, input_text=content)      # fill in the parameter
        final_result = await use_llm_func(hint_prompt)                                      # feed into LLM with the prompt

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)               # set as history
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)      # add to history
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(                                            # split entities from result --> list of entities
            final_result,
            [context_base_relation["record_delimiter"], context_base_relation["completion_delimiter"]],
        )
        # resolve the entities
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(          # split entity
                record, [context_base_relation["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(       # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1                                      # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][                     # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)
    # 关系提取阶段
    relation_start_time = time.time()
    relation_results = await asyncio.gather(
        *[_process_single_content_relation(c,use_llm_func) for c in ordered_chunks]
    )
    relation_end_time = time.time()
    relation_duration = relation_end_time - relation_start_time
    print()
    print(f"✅ 关系提取完成，耗时: {relation_duration:.2f} 秒 ({relation_duration/60:.2f} 分钟)")
    print(f"   平均每个块: {relation_duration/len(ordered_chunks):.2f} 秒")
    
    all_relations = {}
    for item in relation_results:
        for k, v in item[1].items():
            all_relations[k] = v
    save_entity=[]
    save_relation=[]
    for k,v in copy.deepcopy(all_entities).items():
    #     del v['embedding']
        save_entity.append(v)
    for k,v in copy.deepcopy(all_relations).items():
        # v 是一个列表，需要展平
        if isinstance(v, list):
            save_relation.extend(v)
        else:
            save_relation.append(v)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    save_start_time = time.time()
    
    # 文件路径
    entity_file = f"{output_dir}/entity.jsonl"
    relation_file = f"{output_dir}/relation.jsonl"
    
    # 根据模式处理数据
    existing_entities = {}
    existing_relations = set()
    
    if append_mode:
        # 追加模式：读取已存在的数据用于去重
        # 读取已存在的实体文件
        if os.path.exists(entity_file):
            try:
                from tools.utils import read_jsonl
                existing_entity_list = read_jsonl(entity_file) or []
                for entity in existing_entity_list:
                    entity_name = entity.get('entity_name', '')
                    if entity_name:
                        existing_entities[entity_name] = entity
                print(f"📂 已读取 {len(existing_entities)} 个已存在的实体")
            except Exception as e:
                print(f"⚠️  读取已存在实体文件时出错: {e}，将视为空文件")
                existing_entities = {}
        
        # 读取已存在的关系文件
        if os.path.exists(relation_file):
            try:
                from tools.utils import read_jsonl
                existing_relation_list = read_jsonl(relation_file) or []
                for rel in existing_relation_list:
                    # 处理可能存在的列表格式（向后兼容）
                    if isinstance(rel, list):
                        rel = rel[0] if rel else {}
                    
                    # 根据数据格式选择正确的字段名
                    if 'src_id' in rel:
                        src_id = str(rel.get('src_id', '')).replace('"', '')
                        tgt_id = str(rel.get('tgt_id', '')).replace('"', '')
                    else:
                        src_id = str(rel.get('src_tgt', '')).replace('"', '')
                        tgt_id = str(rel.get('tgt_src', '')).replace('"', '')
                    
                    if src_id and tgt_id:
                        existing_relations.add((src_id, tgt_id))
                print(f"📂 已读取 {len(existing_relations)} 个已存在的关系")
            except Exception as e:
                print(f"⚠️  读取已存在关系文件时出错: {e}，将视为空文件")
                existing_relations = set()
        
        # 追加模式：过滤新数据，只保留不重复的
        # 追加模式：过滤新数据，只保留不重复的
        new_entities = []
        duplicate_entity_count = 0
        for entity in save_entity:
            entity_name = entity.get('entity_name', '')
            if entity_name and entity_name not in existing_entities:
                new_entities.append(entity)
                existing_entities[entity_name] = entity  # 添加到已存在集合中，避免同批次重复
            else:
                duplicate_entity_count += 1
        
        new_relations = []
        duplicate_relation_count = 0
        for rel in save_relation:
            # 处理可能存在的列表格式
            if isinstance(rel, list):
                rel = rel[0] if rel else {}
            
            # 根据数据格式选择正确的字段名
            if 'src_id' in rel:
                src_id = str(rel.get('src_id', '')).replace('"', '')
                tgt_id = str(rel.get('tgt_id', '')).replace('"', '')
            else:
                src_id = str(rel.get('src_tgt', '')).replace('"', '')
                tgt_id = str(rel.get('tgt_src', '')).replace('"', '')
            
            relation_key = (src_id, tgt_id)
            if src_id and tgt_id and relation_key not in existing_relations:
                new_relations.append(rel)
                existing_relations.add(relation_key)  # 添加到已存在集合中，避免同批次重复
            else:
                duplicate_relation_count += 1
        
        # 显示去重统计
        print(f"\n📊 去重统计:")
        print(f"   实体: 新增 {len(new_entities)} 条，跳过重复 {duplicate_entity_count} 条")
        print(f"   关系: 新增 {len(new_relations)} 条，跳过重复 {duplicate_relation_count} 条")
    else:
        # 覆盖模式，使用所有数据
        new_entities = save_entity
        new_relations = save_relation
        print(f"\n📊 覆盖模式: 将写入 {len(new_entities)} 个实体，{len(new_relations)} 个关系")
    
    # 根据模式写入文件
    write_mode = "a" if append_mode else "w"
    action_desc = "追加" if append_mode else "写入"
    
    try:
        if new_entities:
            write_jsonl(new_entities, entity_file, mode=write_mode)
            print(f"✅ 已{action_desc}实体到文件: {entity_file} ({len(new_entities)} 条)")
        else:
            print(f"ℹ️  没有新实体需要{action_desc}")
        
        # 显示总实体数（追加模式才统计总数）
        if append_mode:
            # 总实体数 = 已存在的 + 新增的（去重后）
            total_entities = len(existing_entities) + len(new_entities)
            print(f"   文件中共有实体: {total_entities} 条（原有 {len(existing_entities)} 条 + 新增 {len(new_entities)} 条）")
    except PermissionError as e:
        print(f"❌ 写入实体文件时权限错误: {e}")
        print(f"   文件路径: {entity_file}")
        print(f"   请检查：")
        print(f"   1. 文件是否被其他程序占用（如编辑器）")
        print(f"   2. 目录是否有写权限")
        print(f"   3. 磁盘空间是否充足")
        raise
    except Exception as e:
        print(f"❌ 写入实体文件时出错: {e}")
        raise
    
    try:
        if new_relations:
            write_jsonl(new_relations, relation_file, mode=write_mode)
            print(f"✅ 已{action_desc}关系到文件: {relation_file} ({len(new_relations)} 条)")
        else:
            print(f"ℹ️  没有新关系需要{action_desc}")
        
        # 显示总关系数（追加模式才统计总数）
        if append_mode:
            # 总关系数 = 已存在的 + 新增的（去重后）
            total_relations = len(existing_relations) + len(new_relations)
            print(f"   文件中共有关系: {total_relations} 条（原有 {len(existing_relations)} 条 + 新增 {len(new_relations)} 条）")
    except PermissionError as e:
        print(f"❌ 写入关系文件时权限错误: {e}")
        print(f"   文件路径: {relation_file}")
        print(f"   请检查：")
        print(f"   1. 文件是否被其他程序占用（如编辑器）")
        print(f"   2. 目录是否有写权限")
        print(f"   3. 磁盘空间是否充足")
        raise
    except Exception as e:
        print(f"❌ 写入关系文件时出错: {e}")
        raise
    
    save_duration = time.time() - save_start_time
    
    # 性能统计
    total_duration = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("📊 性能统计")
    print(f"{'='*60}")
    print(f"总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    print(f"  - 实体提取: {entity_duration:.2f} 秒 ({entity_duration/total_duration*100:.1f}%)")
    print(f"  - 关系提取: {relation_duration:.2f} 秒 ({relation_duration/total_duration*100:.1f}%)")
    print(f"  - 保存文件: {save_duration:.2f} 秒 ({save_duration/total_duration*100:.1f}%)")
    print(f"平均每个块总耗时: {total_duration/len(ordered_chunks):.2f} 秒")
    print(f"处理速度: {len(ordered_chunks)/total_duration*60:.2f} 个块/分钟")
    print(f"{'='*60}\n")
            
    
    
    
    
    
if __name__ == "__main__":
    MODEL = "deepseek-r1-32b:latest"
    num=5
    instanceManager=InstanceManager(
        url="http://10.61.2.49",
        ports=[11434 for i in range(num)],
        gpus=[i for i in range(num)],
        generate_model=MODEL,
        startup_delay=30
    )
    use_llm=instanceManager.generate_text_asy
    chunk_file="/newdataf/SJ/LeanRAG/datasets/mix/mix_chunk.json"
    chunks=get_chunk(chunk_file)
    output_dir="ttt"
    loop = asyncio.get_event_loop()
    loop.run_until_complete(triple_extraction(chunks, use_llm,output_dir))



    