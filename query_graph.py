from collections import Counter, defaultdict
from dataclasses import field
import json
import os
import time
import logging
import numpy as np
from openai import OpenAI
import pymysql
import tiktoken
from tqdm import tqdm
import yaml
from tools.utils import InstanceManager
from openai import  OpenAI
from database_utils import build_vector_search,search_vector_search,find_tree_root,\
    search_nodes_link,search_nodes,search_community,search_chunks,get_text_units,find_path
from prompt import GRAPH_FIELD_SEP, PROMPTS
from itertools import combinations
from sentence_transformers import SentenceTransformer
import torch

logger=logging.getLogger(__name__)
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
_device = "cuda" if torch.cuda.is_available() else "cpu"
_st_model_name = EMBEDDING_MODEL if isinstance(EMBEDDING_MODEL, str) and len(EMBEDDING_MODEL) > 0 else "BAAI/bge-m3"
try:
    _ST_EMB = SentenceTransformer(_st_model_name, device=_device)
except Exception:
    _ST_EMB = SentenceTransformer("BAAI/bge-m3", device=_device)
_ST_EMB.max_seq_length = 4096

def embedding(texts) -> np.ndarray:
    # 处理单个字符串的情况
    if isinstance(texts, str):
        texts = [texts]
    
    # 确保batch_size至少为1
    batch_size = max(1, min(64, len(texts)))
    
    vectors = _ST_EMB.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    
    # 如果输入是单个字符串，返回单个向量列表（用于Milvus搜索）
    result = np.array(vectors)
    if len(texts) == 1:
        return [result[0].tolist()]  # 返回 [[向量]] 格式给Milvus
    return result

tokenizer = tiktoken.get_encoding("cl100k_base")
def truncate_text(text, max_tokens=4096):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text

def get_reasoning_chain(global_config,entities_set):
    maybe_edges=list(combinations(entities_set,2))
    reasoning_path=[]
    reasoning_path_information=[]
    db_name = os.path.basename(global_config['working_dir'].rstrip("/"))
    information_record=[]
    for edge in maybe_edges:
        a_path=[]
        b_path=[]
        node1=edge[0]
        node2=edge[1]
        node1_tree=find_tree_root(db_name,node1)
        node2_tree=find_tree_root(db_name,node2)
        
        # if node1_tree[1]!=node2_tree[1] :
        #     print("debug")
        for index,(i,j) in enumerate(zip(node1_tree,node2_tree)):
            if i==j:
                a_path.append(i)
                break
            if i in b_path or j in a_path:
                break
            if i!=j :
                a_path.append(i)
                b_path.append(j)
            
            
        reasoning_path.append(a_path+[b_path[len(b_path)-1-i] for  i in range(len(b_path))]) 
        a_path=list(set(a_path))
        b_path=list(set(b_path))
        for maybe_edge in list(combinations(a_path+b_path,2)):
            if maybe_edge[0]==maybe_edge[1]:
                continue
            information=search_nodes_link(maybe_edge[0],maybe_edge[1],global_config['working_dir'])
            if information==None:
                continue
            information_record.append(information)
            reasoning_path_information.append([maybe_edge[0],maybe_edge[1],information[2]])
    # columns=['src_tgt','tgt_src','path_description']
    # reasoning_path_information_description="\t\t".join(columns)+"\n"
    temp_relations_information=list(set([information[2] for information in reasoning_path_information]))
    reasoning_path_information_description="\n".join(temp_relations_information)  
    return  reasoning_path,reasoning_path_information_description

def get_entity_description(global_config,entities_set,mode=0):
    
    
    
    columns=['entity_name','parent','description']
    entity_descriptions="\t\t".join(columns)+"\n"
    entity_descriptions+="\n".join([information[0]+"\t\t"+information[1]+"\t\t"+information[2] for information in entities_set])

    return entity_descriptions
        
def get_aggregation_description(global_config,reasoning_path,if_findings=False):
    
    aggregation_results=[]
    
    communities=set([community for each_path in reasoning_path for community in each_path])
    for community in communities:
        temp=search_community(community,global_config['working_dir'])
        if temp=="":
            continue
        aggregation_results.append(temp)
    if if_findings:
        columns=['entity_name','entity_description','findings']
        aggregation_descriptions="\t\t".join(columns)+"\n"
        aggregation_descriptions+="\n".join([information[0]+"\t\t"+str(information[1])+"\t\t"+information[2] for information in aggregation_results])
    else:
        columns=['entity_name','entity_description']
        aggregation_descriptions="\t\t".join(columns)+"\n"
        aggregation_descriptions+="\n".join([information[0]+"\t\t"+str(information[1]) for information in aggregation_results])
    return aggregation_descriptions,communities

def query_graph(global_config,db,query):
    use_llm_func: callable = global_config["use_llm_func"]
    embedding: callable=global_config["embeddings_func"]
    b=time.time()
    level_mode=global_config['level_mode']
    topk=global_config['topk']
    chunks_file=global_config["chunks_file"]
    entity_results=search_vector_search(global_config['working_dir'],embedding(query),topk=topk,level_mode=level_mode)
    v=time.time()
    res_entity=[i[0]for i in entity_results]
    chunks=[i[-1]for i in entity_results]
    entity_descriptions=get_entity_description(global_config,entity_results)
    reasoning_path,reasoning_path_information_description=get_reasoning_chain(global_config,res_entity)
    # reasoning_path,reasoning_path_information_description=get_path_chain(global_config,res_entity)
    aggregation_descriptions,aggregation=get_aggregation_description(global_config,reasoning_path)
    # chunks=search_chunks(global_config['working_dir'],aggregation)
    text_units=get_text_units(global_config['working_dir'],chunks,chunks_file,k=5)
    describe=f"""
    entity_information:
    {entity_descriptions}
    aggregation_entity_information:
    {aggregation_descriptions}
    reasoning_path_information:
    {reasoning_path_information_description}
    text_units:
    {text_units}
    """
    e=time.time()
    
    # print(describe)
    sys_prompt =PROMPTS["rag_response"].format(context_data=describe)
    response=use_llm_func(query,system_prompt=sys_prompt)
    g=time.time()
    print(f"embedding time: {v-b:.2f}s")
    print(f"query time: {e-v:.2f}s")
    
    print(f"response time: {g-e:.2f}s")
    return describe,response
if __name__=="__main__":
    # ========== 需要修改的配置 ==========
    
    # 1. MySQL连接配置
    # 注意：如果MySQL在本机，建议使用socket连接
    try:
        # 方式1：使用socket连接（推荐，本机MySQL）
        db = pymysql.connect(
            host='localhost',
            user='root',
            passwd='123',
            unix_socket='/run/mysqld/mysqld.sock',
            charset='utf8mb4'
        )
        print("✓ MySQL connected via socket")
    except Exception as e:
        print(f"Socket connection failed: {e}")
        # 方式2：使用TCP连接（远程MySQL或socket失败时）
        try:
            db = pymysql.connect(
                host='10.61.2.49',
                user='root',
                passwd='123',
                port=3306,
                charset='utf8mb4'
            )
            print("✓ MySQL connected via TCP")
        except Exception as e:
            print(f"❌ MySQL connection failed: {e}")
            exit(1)
    
    # 2. 工作目录配置
    WORKING_DIR = "/newdataf/SJ/LeanRAG/GraphExtraction/ttt"
    
    # 3. Chunks文件路径
    # 注意：如果文件不存在，请修改为实际路径或先注释掉
    chunks_file = "/newdataf/SJ/LeanRAG/GraphExtraction/ttt/chunks.json"
    
    # 检查chunks文件是否存在
    if not os.path.exists(chunks_file):
        print(f"⚠ Warning: Chunks file not found: {chunks_file}")
        print("  Trying alternative path...")
        # chunks_file = "/newdataf/SJ/LeanRAG/ckg_data/mix_chunk/mix_chunk.json"
        chunks_file = "/newdataf/SJ/LeanRAG/datasets/mix/mix_chunk.json"
        if not os.path.exists(chunks_file):
            print(f"⚠ Warning: Alternative chunks file not found: {chunks_file}")
            print("  Will proceed without chunks (may affect answer quality)")
            chunks_file = None
        else:
            print(f"✓ Found chunks file: {chunks_file}")
    else:
        print(f"✓ Chunks file exists: {chunks_file}")
    
    # 4. LLM服务配置
    num = 2  # GPU数量
    instanceManager = InstanceManager(
        url="http://10.61.2.49",           # 您的ollama服务器地址
        ports=[11434 for i in range(num)], # ollama端口
        gpus=[i for i in range(num)],
        generate_model="deepseek-r1-32b:latest",  # 您使用的模型名
        startup_delay=30
    )
    
    # 5. 查询配置
    global_config = {
        'working_dir': WORKING_DIR,
        'chunks_file': chunks_file,
        'embeddings_func': embedding,
        'use_llm_func': instanceManager.generate_text,
        'topk': 10,           # 检索Top-K个实体
        'level_mode': 1       # 0:原始节点, 1:聚合节点, 2:全部
    }
    
    # 6. 您的查询问题（修改为您要问的问题）
    query = "What is the maturity date of the credit agreement?"
    
    # ========== 执行查询 ==========
    print(f"Query: {query}")
    print("="*50)
    
    ref, response = query_graph(global_config, db, query)
    
    print("\n[Retrieved Context]")
    print(ref)
    print("\n" + "#"*50)
    print("\n[LLM Response]")
    print(response)
    
    db.close()
    