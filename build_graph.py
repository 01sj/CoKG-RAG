import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import field
import json
import os
import logging
import numpy as np
from openai import OpenAI
import tiktoken
from tqdm import tqdm
import yaml
from sentence_transformers import SentenceTransformer
import torch
from openai import AsyncOpenAI, OpenAI
from _cluster_utils import Hierarchical_Clustering
from tools.utils import write_jsonl,InstanceManager
from database_utils import build_vector_search,create_db_table_mysql,insert_data_to_mysql
import requests
import multiprocessing
import gc
logger=logging.getLogger(__name__)

def clear_gpu_memory():
    """清理GPU内存"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        print(f"Warning: Failed to clear GPU memory: {e}")

def safe_embedding_init(entities: list[dict]) -> list[dict]:
    """安全的embedding初始化，包含错误处理和内存管理"""
    global _ST_EMB
    try:
        clear_gpu_memory()
        # 确保 embedding 模型在 CPU 上（避免与 vLLM 争抢 GPU 显存）
        if hasattr(_ST_EMB, 'device') and _ST_EMB.device.type != "cpu":
            print("⚠️  警告: embedding 模型不在 CPU 上，强制移动到 CPU")
            _ST_EMB = _ST_EMB.to("cpu")
        return embedding_init(entities)
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM in embedding_init: {e}")
        clear_gpu_memory()
        # 强制使用 CPU 进行 embedding
        print("🔄 切换到 CPU 进行 embedding 计算")
        # 如果还是内存不足，尝试更小的batch
        texts = [truncate_text(i['description']) for i in entities]
        batch_size = max(1, min(4, len(texts)))  # 进一步减少batch_size，但至少为1
        # 确保使用 CPU
        if hasattr(_ST_EMB, 'device') and _ST_EMB.device.type != "cpu":
            _ST_EMB = _ST_EMB.to("cpu")
        vectors = _ST_EMB.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for i, entity in enumerate(entities):
            entity['vector'] = np.array(vectors[i])
        return entities
    except Exception as e:
        print(f"Error in embedding_init: {e}")
        clear_gpu_memory()
        raise e

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
MODEL = config['deepseek']['model']
DEEPSEEK_API_KEY = config['deepseek']['api_key']
DEEPSEEK_URL = config['deepseek']['base_url']
EMBEDDING_MODEL = config['glm']['model']
EMBEDDING_URL = config['glm']['base_url']
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0

# Initialize local sentence-transformers embedding model once
_force_cpu = os.environ.get("FORCE_CPU", "1") == "1"  # 默认使用 CPU，避免与 vLLM 争抢 GPU 显存
_device = "cpu" if _force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
_st_model_name = EMBEDDING_MODEL if isinstance(EMBEDDING_MODEL, str) and len(EMBEDDING_MODEL) > 0 else "BAAI/bge-m3"

# 清理GPU缓存
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except:
    pass

try:
    _ST_EMB = SentenceTransformer(_st_model_name, device=_device)
    print(f"✅ 成功加载 Embedding 模型: {_st_model_name} (设备: {_device})")
except Exception as e:
    print(f"Failed to load {_st_model_name}: {e}")
    print("Falling back to CPU and BAAI/bge-m3")
    _device = "cpu"
    _ST_EMB = SentenceTransformer("BAAI/bge-m3", device=_device)
    print(f"✅ 使用备用 Embedding 模型: BAAI/bge-m3 (设备: {_device})")
_ST_EMB.max_seq_length = 4096
print(f"📏 Embedding 模型最大序列长度: {_ST_EMB.max_seq_length}")

def get_common_rag_res(WORKING_DIR):
    entity_path=f"{WORKING_DIR}/entity.jsonl"
    relation_path=f"{WORKING_DIR}/relation.jsonl"
    # i=0
    e_dic={}
    with open(entity_path,"r")as f:
        for xline in f:
            
            line=json.loads(xline)
            entity_name=str(line['entity_name'])
            description=line['description']
            source_id=line['source_id']
            if entity_name not in e_dic.keys():
                e_dic[entity_name]=dict(
                    entity_name=str(entity_name),
                    description=description,
                    source_id=source_id,
                    degree=0,
                )
            else:
                e_dic[entity_name]['description']+="|Here is another description : "+ description
                if e_dic[entity_name]['source_id']!= source_id:
                    e_dic[entity_name]['source_id']+= "|"+source_id
                    
    #         i+=1
    #         if i==1000:
    #             break
    # i=0
    r_dic={}
    with open(relation_path,"r")as f:
        for xline in f:
            
            line=json.loads(xline)

            # 处理数组格式的关系数据
            if isinstance(line, list):
                # 如果 line 是数组，遍历数组中的每个关系对象
                for relation in line:
                    src_tgt=str(relation['src_id'])
                    tgt_src=str(relation['tgt_id'])
                    description=relation['description']
                    weight=relation.get('weight', 1)
                    source_id=relation['source_id']
                    r_dic[(src_tgt,tgt_src)]={
                        'src_tgt':str(src_tgt),
                        'tgt_src':str(tgt_src),
                        'description':description,
                        'weight':weight,
                        'source_id':source_id
                    }
            else:
                # 如果 line 是单个对象，按原来的方式处理
                src_tgt=str(line['src_tgt'])
                tgt_src=str(line['tgt_src'])
                description=line['description']
                weight=1
                source_id=line['source_id']
                r_dic[(src_tgt,tgt_src)]={
                    'src_tgt':str(src_tgt),
                    'tgt_src':str(tgt_src),
                    'description':description,
                    'weight':weight,
                    'source_id':source_id
                }
            # e_dic[src_tgt]['degree']+=1
            # e_dic[tgt_src]['degree']+=1
            # i+=1
            # if i==1000:
            #     break
    
    
    return e_dic,r_dic


# Replace OpenAI embedding with local sentence-transformers

def embedding(texts: list[str]) -> np.ndarray:  # local embedding
    # 处理单个文本的情况
    if isinstance(texts, str):
        texts = [texts]
    
    # 确保batch_size至少为1
    batch_size = max(1, min(16, len(texts)))  # 从64减少到16，但至少为1
    vectors = _ST_EMB.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(vectors)

def embedding_init(entities:list[dict])-> list[dict]: 
    global _ST_EMB
    # 确保 embedding 模型在 CPU 上（避免与 vLLM 争抢 GPU 显存）
    if hasattr(_ST_EMB, 'device') and _ST_EMB.device.type != "cpu":
        print("⚠️  警告: embedding 模型不在 CPU 上，强制移动到 CPU")
        _ST_EMB = _ST_EMB.to("cpu")
    
    texts=[truncate_text(i['description']) for i in entities]
    # 减少batch_size以节省内存，确保至少为1
    batch_size = max(1, min(8, len(texts)))  # 进一步减少batch_size，避免内存问题
    vectors = _ST_EMB.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    for i, entity in enumerate(entities):
        entity['vector'] = np.array(vectors[i])
    return entities

tokenizer = tiktoken.get_encoding("cl100k_base")
def truncate_text(text, max_tokens=4096):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text

def embedding_data(entity_results):
    entities = [v for k, v in entity_results.items()]
    entity_with_embeddings=[]
    # 减少batch_size和max_workers以节省内存
    embeddings_batch_size = int(os.environ.get("EMB_BATCH", "16"))  # 默认16，进一步降低显存占用
    num_embeddings_batches = (len(entities) + embeddings_batch_size - 1) // embeddings_batch_size
    
    batches = [
        entities[i * embeddings_batch_size : min((i + 1) * embeddings_batch_size, len(entities))]
        for i in range(num_embeddings_batches)
    ]

    # 允许通过环境变量关闭多进程，或控制并发数
    emb_max_workers = int(os.environ.get("EMB_MAX_WORKERS", "1"))  # 默认1，避免多进程CUDA初始化问题
    if emb_max_workers <= 0:
        # 串行处理，最稳妥（CPU/低显存环境）
        for batch in tqdm(batches):
            try:
                result = safe_embedding_init(batch)
                entity_with_embeddings.extend(result)
                clear_gpu_memory()
            except Exception as e:
                print(f"Error processing batch: {e}")
                clear_gpu_memory()
                raise e
    else:
        # 受控并发
        with ProcessPoolExecutor(max_workers=emb_max_workers) as executor:
            futures = [executor.submit(safe_embedding_init, batch) for batch in batches]
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    entity_with_embeddings.extend(result)
                    clear_gpu_memory()
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    clear_gpu_memory()
                    raise e

    for i in entity_with_embeddings:
        entiy_name=i['entity_name']
        vector=i['vector']
        entity_results[entiy_name]['vector']=vector
    return entity_results



    
            

def hierarchical_clustering(global_config):
    entity_results,relation_results=get_common_rag_res(global_config['working_dir'])
    all_entities=embedding_data(entity_results)
    hierarchical_cluster = Hierarchical_Clustering()
    all_entities,generate_relations,community =hierarchical_cluster.perform_clustering(global_config=global_config,entities=all_entities,relations=relation_results,\
        WORKING_DIR=WORKING_DIR,max_workers=global_config['max_workers'])
    try :
        all_entities[-1]['vector']=embedding(all_entities[-1]['description'])
        build_vector_search(all_entities, f"{WORKING_DIR}")
    except Exception as e:
        print(f"Error in build_vector_search: {e}")
    for layer in all_entities:
        if type(layer) != list :
            if "vector" in layer.keys():
                del layer["vector"]
            continue
        for item in layer:
            if "vector" in item.keys():
                del item["vector"]
            if len(layer)==1:
                item['parent']='root'
    save_relation=[
    v for k, v in generate_relations.items()
]
    save_community=[
    v for k, v in community.items()
]
    
    # 删除旧文件（如果存在），确保每次都是全新构建
    relations_file = f"{global_config['working_dir']}/generate_relations.json"
    community_file = f"{global_config['working_dir']}/community.json"
    
    if os.path.exists(relations_file):
        os.remove(relations_file)
        print(f"🗑️  已删除旧的关系文件: {relations_file}")
    
    if os.path.exists(community_file):
        os.remove(community_file)
        print(f"🗑️  已删除旧的社区文件: {community_file}")
    
    # 写入新文件（使用追加模式，但因为文件已删除，实际是创建新文件）
    write_jsonl(save_relation, relations_file)
    write_jsonl(save_community, community_file)
    
    try:
        # 使用working_dir的basename作为数据库名称，确保一致性
        db_name = os.path.basename(global_config['working_dir'].rstrip('/'))
        create_db_table_mysql(global_config['working_dir'], target_database=db_name)
        insert_data_to_mysql(global_config['working_dir'], target_database=db_name)
    except Exception as e:
        print(f"Error in database operations: {e}")
        print("Continuing without database operations...")
    
if __name__=="__main__":
    # 程序开始时清理GPU内存
    clear_gpu_memory()
    
    try:
        multiprocessing.set_start_method("spawn", force=True)  # 强制设置
    except RuntimeError:
        pass  # 已经设置过，忽略
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="/newdataf/SJ/LeanRAG/GraphExtraction/ttt/")
    parser.add_argument("-n", "--num", type=int, default=2)
    args = parser.parse_args()

    WORKING_DIR = args.path
    num=args.num
    instanceManager=InstanceManager(
        url="http://10.61.2.49",  # 替换为你的 ollama 服务器地址
        ports=[11434 for i in range(num)],  # ollama 默认端口
        gpus=[i for i in range(num)],
        generate_model="deepseek-r1-32b:latest",  # 替换为你在 ollama 中部署的模型名
        startup_delay=30
    )
    global_config={}
    # 减少max_workers以避免GPU内存竞争
    global_config['max_workers']=min(2, num*2)  # 从num*4减少到num*2，最大不超过2
    global_config['working_dir']=WORKING_DIR
    global_config['use_llm_func']=instanceManager.generate_text
    global_config['embeddings_func']=embedding
    global_config["special_community_report_llm_kwargs"]=field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    hierarchical_clustering(global_config)