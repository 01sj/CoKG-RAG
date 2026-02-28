import json
import os
import numpy as np
from pymilvus  import MilvusClient
import ollama
import pymysql
from collections import Counter
from mysql_config import MYSQL_CONFIG

def get_db_name(working_dir):
    """从 working_dir 提取数据库名，并用反引号包裹以支持特殊字符"""
    db_name = os.path.basename(working_dir.rstrip("/"))
    return f"`{db_name}`"

def get_mysql_connection(database=None):
    """获取 MySQL 连接，增加连接/读写超时与自动提交，避免长时间卡住。

    说明：
    - connect_timeout 防止网络不可达时长时间阻塞。
    - read_timeout/write_timeout 防止执行语句时无限等待。
    - autocommit=True 使 DDL/DML 立即生效，避免等待隐式提交。
    """
    config = MYSQL_CONFIG.copy()
    if database:
        config['database'] = database
    # 默认超时与自动提交设置（若用户已在 MYSQL_CONFIG 指定则不覆盖）
    config.setdefault('connect_timeout', 15)
    config.setdefault('read_timeout', 60)
    config.setdefault('write_timeout', 60)
    config.setdefault('autocommit', True)
    # 使用标准 Cursor 即可
    return pymysql.connect(**config)

def emb_text(text):
    response = ollama.embeddings(model="bge-m3:latest", prompt=text)
    return response["embedding"]
def build_vector_search(data,working_dir):
    try:
        milvus_client = MilvusClient(uri=f"{working_dir}/milvus_demo.db")
    except Exception as e:
        print(f"Error creating Milvus client: {e}")
        print("Please install milvus-lite: pip install pymilvus[milvus_lite]")
        return
    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        field_name="dense",
        index_name="dense_index",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128},
    )
    
    collection_name = "entity_collection"
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=1024,
        index_params=index_params,
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
    )
    id=0
    flatten=[]
    print("dealing data level")
    for level,sublist in enumerate(data):
        if type(sublist) is not list:
            item=sublist
            item['id']=id
            id+=1
            item['level']=level
            if len(item['vector'])==1:
                item['vector']=item['vector'][0]
            flatten.append(item)
        else:
            for item in sublist:
                item['id']=id
                id+=1
                item['level']=level
                if len(item['vector'])==1:
                    item['vector']=item['vector'][0]
                flatten.append(item)
        print(level)
        # embedding = emb_text(description)
   
    piece=10
    
    for indice in range(len(flatten)//piece +1):
        start = indice * piece
        end = min((indice + 1) * piece, len(flatten))
        data_batch = flatten[start:end]
        milvus_client.insert(
            collection_name="entity_collection",
            data=data_batch
        )
    # milvus_client.insert(
    #         collection_name=collection_name,
    #         data=data
    #     )

def search_vector_search(working_dir,query,topk=10,level_mode=2):
    '''
    level_mode: 0: 原始节点
                1: 聚合节点
                2: 所有节点
    '''
    # 标准化路径：移除末尾斜杠，确保路径格式一致
    working_dir = os.path.normpath(working_dir.rstrip("/"))
    
    if level_mode==0:
        filter_filed=" level == 0 "
    elif level_mode==1:
        filter_filed=" level > 0 "
    # elif level_mode==2:
    #     filter_filed=" level < 58736"
    else:
        filter_filed=""
    
    dataset=os.path.basename(working_dir)
    milvus_db_path = os.path.join(working_dir, "milvus_demo.db")
    
    # 调试信息：打印实际使用的路径
    print(f"[DEBUG] Working directory: {working_dir}")
    print(f"[DEBUG] Milvus database path: {milvus_db_path}")
    collection_name = "entity_collection"
    
    # 确保目录存在
    os.makedirs(working_dir, exist_ok=True)
    
    # 创建或打开 Milvus 客户端
    try:
        milvus_client = MilvusClient(uri=milvus_db_path)
        if os.path.exists(milvus_db_path):
            print(f"Using existing Milvus database at {milvus_db_path}")
        else:
            print(f"Created new Milvus database at {milvus_db_path}")
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to create or access Milvus database at {milvus_db_path}. "
            f"Error: {e}\n"
            f"Please ensure the directory {working_dir} exists and is writable."
        ) from e
    
    # 检查 collection 是否存在
    if not milvus_client.has_collection(collection_name):
        raise FileNotFoundError(
            f"Collection '{collection_name}' not found in Milvus database at {milvus_db_path}.\n"
            f"This means the vector index has not been built yet.\n"
            f"Please run build_law_graph.py or build_graph.py first to create the vector index:\n"
            f"  python build_law_graph.py -p {working_dir}\n"
            f"Or check if the working directory path is correct: {working_dir}"
        )
    # query_embedding = emb_text(query)
    search_results = milvus_client.search(
        collection_name=collection_name,
        data=query,
        limit=topk,
        params={"metric_type": "IP", "params": {}},
        filter=filter_filed,
        output_fields=["entity_name", "description","parent","level","source_id"],
    )
    # print(search_results)
    extract_results=[(i['entity']['entity_name'],i["entity"]["parent"],i["entity"]["description"],i["entity"]["source_id"])for i in search_results[0]]
    # print(extract_results)
    return extract_results
def create_db_table_mysql(working_dir, target_database="test_leanrag_operations"):
    try:
        con = get_mysql_connection()
        cur = con.cursor()
        # 使用指定的目标数据库名称
        dbname = target_database
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        return

    print(f"Using database name: {dbname}", flush=True)

    def _ensure_connection():
        try:
            # 保活：若连接断开则重连
            con.ping(reconnect=True)
        except Exception:
            pass

    # 会话级超时，避免等待锁与网络读写阻塞
    try:
        _ensure_connection()
        cur.execute("SET SESSION innodb_lock_wait_timeout = 10;")
        cur.execute("SET SESSION net_read_timeout = 60;")
        cur.execute("SET SESSION net_write_timeout = 60;")
    except Exception as e:
        print(f"Warn: set session timeouts failed: {e}")

    # 安全策略：不再 DROP DATABASE，改为 CREATE IF NOT EXISTS，降低长事务/权限导致的风险
    def _create_db(target_name: str) -> bool:
        for attempt in range(3):
            try:
                _ensure_connection()
                print(f"Creating database if not exists `{target_name}` (attempt {attempt+1}/3) ...", flush=True)
                cur.execute(f"create database if not exists `{target_name}` character set utf8mb4;")
                return True
            except Exception as e:
                print(f"Create `{target_name}` failed: {e}")
        return False

    ok = _create_db(dbname)
    if not ok:
        print(f"Error creating database: `{dbname}`; fallback to default.")
        dbname = "leanrag_db"
        if not _create_db(dbname):
            print("Fallback create default database failed, abort database operations.")
            try:
                cur.close(); con.close()
            except Exception:
                pass
            return

    # 切换数据库并建表
    print(f"Using database `{dbname}` ...", flush=True)
    _ensure_connection()
    cur.execute(f"use `{dbname}`;")

    print("Creating table `entities` ...", flush=True)
    cur.execute("create table if not exists entities\
        (entity_name varchar(500), description varchar(10000),source_id varchar(1000),\
            degree int,parent varchar(1000),level int ,INDEX en(entity_name))character set utf8mb4 COLLATE utf8mb4_unicode_ci;")

    print("Creating table `relations` ...", flush=True)
    cur.execute("create table if not exists relations\
        (src_tgt varchar(190),tgt_src varchar(190), description varchar(10000),\
            weight int,level int ,INDEX link(src_tgt,tgt_src))character set utf8mb4 COLLATE utf8mb4_unicode_ci;")

    print("Creating table `communities` ...", flush=True)
    cur.execute("create table if not exists communities\
        (entity_name varchar(500), entity_description varchar(10000),findings text,INDEX en(entity_name)\
             )character set utf8mb4 COLLATE utf8mb4_unicode_ci ;")

    try:
        con.commit()
    except Exception:
        # autocommit 下无须提交，但显式提交一次以兼容配置
        pass
    finally:
        cur.close()
        con.close()
    
def insert_data_to_mysql(working_dir, target_database="test_leanrag_operations"):
    dbname = target_database
    db = get_mysql_connection(database=dbname)
    cursor = db.cursor()
    
    # 显式选择数据库，确保连接到正确的数据库
    try:
        cursor.execute(f"USE `{dbname}`;")
    except Exception as e:
        print(f"Error selecting database `{dbname}`: {e}")
        db.close()
        return
    
    # 清空现有数据，避免重复
    print("🗑️  清空现有数据...")
    try:
        cursor.execute("TRUNCATE TABLE entities;")
        cursor.execute("TRUNCATE TABLE relations;")
        cursor.execute("TRUNCATE TABLE communities;")
        db.commit()
        print("✅ 现有数据已清空")
    except Exception as e:
        print(f"⚠️  清空数据时出现警告: {e}")
        # 继续执行，可能是表不存在
    
    entity_path=os.path.join(working_dir,"all_entities.json")
    with open(entity_path,"r")as f:
        val=[]
        batch_size = 1000  # 每次插入1000条，减少临时文件使用
        sql = "INSERT INTO entities(entity_name, description, source_id, degree,parent,level) VALUES (%s,%s,%s,%s,%s,%s)"
        
        for level,entitys in enumerate(f):
            local_entity=json.loads(entitys)
            if type(local_entity) is not dict:
                for entity in json.loads(entitys):
                    # entity=json.load(entity_l)
                    
                    entity_name=entity.get('entity_name', '')
                    description=entity.get('description', '')
                    # if "|Here" in description:
                    #     description=description.split("|Here")[0]
                    source_id="|".join(entity.get('source_id', '').split("|")[:5])
                   
                    degree=entity.get('degree', 0)
                    parent=entity.get('parent', '')
                    val.append((entity_name,description,source_id,degree,parent,level))
                    
                    # 批量插入，减少内存和临时文件使用
                    if len(val) >= batch_size:
                        try:
                            cursor.executemany(sql,tuple(val))
                            db.commit()
                            val = []  # 清空列表
                        except Exception as e:
                            db.rollback()
                            print(e)
                            print("insert entities error (batch)")
                            val = []
            else:
                entity=local_entity
                entity_name=entity.get('entity_name', '')
                description=entity.get('description', '')
                source_id="|".join(entity.get('source_id', '').split("|")[:5])
                degree=entity.get('degree', 0)
                parent=entity.get('parent', '')
                val.append((entity_name,description,source_id,degree,parent,level))
        
        # 插入剩余的数据
        if len(val) > 0:
            try:
                cursor.executemany(sql,tuple(val))
                db.commit()
            except Exception as e:
                db.rollback()
                print(e)
                print("insert entities error (final)")
         
    relation_path=os.path.join(working_dir,"generate_relations.json")
    with open(relation_path,"r")as f:
        val=[]
        batch_size = 1000  # 每次插入1000条
        sql = "INSERT INTO relations(src_tgt, tgt_src, description,  weight,level) VALUES (%s,%s,%s,%s,%s)"
        
        for relation_l in f:
            relation=json.loads(relation_l)
            src_tgt=relation.get('src_tgt', '')
            tgt_src=relation.get('tgt_src', '')
            description=relation.get('description', '')
            weight=relation.get('weight', 1)
            level=relation.get('level', 0)
            val.append((src_tgt,tgt_src,description,weight,level))
            
            # 批量插入
            if len(val) >= batch_size:
                try:
                    cursor.executemany(sql,tuple(val))
                    db.commit()
                    val = []
                except Exception as e:
                    db.rollback()
                    print(e)
                    print("insert relations error (batch)")
                    val = []
        
        # 插入剩余的数据
        if len(val) > 0:
            try:
                cursor.executemany(sql,tuple(val))
                db.commit()
            except Exception as e:
                db.rollback()
                print(e)
                print("insert relations error (final)")
        
    community_path=os.path.join(working_dir,"community.json")
    with open(community_path,"r")as f:
        val=[]
        batch_size = 1000  # 每次插入1000条
        sql = "INSERT INTO communities(entity_name, entity_description,  findings ) VALUES (%s,%s,%s)"
        
        for community_l in f:
            community=json.loads(community_l)
            title=community.get('entity_name', '')
            summary=community.get('entity_description', '')
            findings=str(community.get('findings', ''))  # 使用get方法，提供默认值
            val.append((title,summary,findings))
            
            # 批量插入
            if len(val) >= batch_size:
                try:
                    cursor.executemany(sql,tuple(val))
                    db.commit()
                    val = []
                except Exception as e:
                    db.rollback()
                    print(e)
                    print("insert communities error (batch)")
                    val = []
        
        # 插入剩余的数据
        if len(val) > 0:
            try:
                cursor.executemany(sql,tuple(val))
                db.commit()
            except Exception as e:
                db.rollback()
                print(e)
                print("insert communities error (final)")
def find_tree_root(working_dir,entity):
    db = get_mysql_connection()
    res=[entity]
    cursor = db.cursor()
    db_name = get_db_name(working_dir)
    depth_sql=f"select max(level) from {db_name}.entities"
    cursor.execute(depth_sql)
    depth=cursor.fetchall()[0][0]
    i=0
    
    while i< depth:
        sql=f"select parent from {db_name}.entities where entity_name=%s "
        
        cursor.execute(sql,(entity))
        ret=cursor.fetchall()
        # print(ret)
        i+=1
        if len(ret)==0:
            break
        entity=ret[0][0]
        res.append(entity)
    # res=list(set(res))
    # res = list(dict.fromkeys(res))

    return res

def find_path(entity1,entity2,working_dir,level,depth=5):
    db = get_mysql_connection()
    db_name = get_db_name(working_dir)
    cursor = db.cursor()

    query = f"""
        WITH RECURSIVE path_cte AS (
            SELECT 
                src_tgt,
                tgt_src,
                 CAST(CONCAT(src_tgt, '|', tgt_src) AS CHAR(5000)) AS path,
                1 AS depth
            FROM {db_name}.relations
            WHERE src_tgt = %s
              AND level = %s

            UNION ALL

            SELECT 
                p.src_tgt,
                t.tgt_src,
                CONCAT(p.path, '|', t.tgt_src),
                p.depth + 1
            FROM path_cte p
            JOIN {db_name}.relations t ON p.tgt_src = t.src_tgt
            WHERE NOT FIND_IN_SET(
                  CONVERT(t.tgt_src USING utf8mb4) COLLATE utf8mb4_unicode_ci,
                  CONVERT(p.path USING utf8mb4) COLLATE utf8mb4_unicode_ci
              )
              AND level = %s
              AND p.depth < %s
        )
        SELECT path
        FROM path_cte
        WHERE tgt_src = %s
        ORDER BY depth ASC
        LIMIT 1;
    """
    cursor.execute(query, (entity1,level,level,depth,entity2))
    result = cursor.fetchone()

    if result:
            return result[0].split('|')  # 返回节点列表
    else:
        return None

def search_nodes_link(entity1,entity2,working_dir,level=0):
    # cursor = db.cursor()
    # db_name=os.path.basename(working_dir)
    # sql=f"select * from {db_name}.relations where src_tgt=%s and tgt_src=%s and level=%s"
    # cursor.execute(sql,(entity1,entity2,level))
    # ret=cursor.fetchall()
    # if len(ret)==0:
    #     sql=f"select * from {db_name}.relations where src_tgt=%s and tgt_src=%s and level=%s"
    #     cursor.execute(sql,(entity2,entity1,level))
    #     ret=cursor.fetchall()
    # if len(ret)==0:
    #     return None
    # else:
    #     return ret[0]
    db = get_mysql_connection()
    cursor = db.cursor()
    db_name = get_db_name(working_dir)
    sql=f"select * from {db_name}.relations where src_tgt=%s and tgt_src=%s "
    cursor.execute(sql,(entity1,entity2))
    ret=cursor.fetchall()
    if len(ret)==0:
        sql=f"select * from {db_name}.relations where src_tgt=%s and tgt_src=%s "
        cursor.execute(sql,(entity2,entity1))
        ret=cursor.fetchall()
    if len(ret)==0:
        return None
    else:
        return ret[0]
def search_chunks(working_dir,entity_set):
    db = get_mysql_connection()
    res=[]
    db_name = get_db_name(working_dir)
    cursor = db.cursor()
    for entity in entity_set:
        if entity=='root':
            continue
        sql=f"select source_id from {db_name}.entities where entity_name=%s "
        cursor.execute(sql,(entity,))
        ret=cursor.fetchall()
        res.append(ret[0])
    return res
def search_nodes(entity_set,working_dir):
    db = get_mysql_connection()
    res=[]
    db_name = get_db_name(working_dir)
    cursor = db.cursor()
    for entity in entity_set:
        sql=f"select * from {db_name}.entities where entity_name=%s and level=0"
        cursor.execute(sql,(entity,))
        ret=cursor.fetchall()
        res.append(ret[0])
    return res
def get_text_units(working_dir,chunks_set,chunks_file,k=5):
    # 确保 k 不是 None
    if k is None:
        k = 5
    db_name = get_db_name(working_dir)
    chunks_list=[]
    for chunks in chunks_set:
        if "|" in chunks:
            temp_chunks=chunks.split("|")
        else:
            temp_chunks=[chunks]
        chunks_list+=temp_chunks
    counter = Counter(chunks_list)

    # 筛选出出现多次的元素
    # duplicates = [item for item, count in counter.items() if count > 2]
    duplicates = [item for item, _ in sorted(
    [(item, count) for item, count in counter.items() if count > 1],
    key=lambda x: x[1],
    reverse=True
        )[:k]]
    if len(duplicates)< k:
        used = set(duplicates)
        for item, _ in counter.items():
            if item not in used:
                duplicates.append(item)
                used.add(item)
            if len(duplicates) == k:
                break
    
    chunks_dict={}
    text_units=""
    
    # 如果chunks_file为None或不存在，返回空字符串
    if chunks_file is None or not os.path.exists(chunks_file):
        return ""
    
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        chunks_dict = {item["hash_code"]: item["text"] for item in chunks_data}
    except Exception as e:
        print(f"Warning: Failed to load chunks file: {e}")
        return ""
    
    # 遍历duplicates，只添加存在的chunks
    for chunks in duplicates:
        if chunks in chunks_dict:
            text_units += chunks_dict[chunks] + "\n"
        else:
            print(f"Warning: hash_code '{chunks}' not found in chunks file")
    
    return text_units
    
def search_community(entity_name,working_dir):
    db = get_mysql_connection()
    db_name = get_db_name(working_dir)
    cursor = db.cursor()
    sql=f"select * from {db_name}.communities where entity_name=%s"
    cursor.execute(sql,(entity_name,))
    ret=cursor.fetchall()
    if len(ret)!=0:
        return ret[0]
    else:
        return ""
            # return ret[0]
def insert_origin_relations(working_dir):
    dbname=os.path.basename(working_dir)
    db = get_mysql_connection(database=dbname)
    cursor = db.cursor()
    
    # 显式选择数据库
    try:
        cursor.execute(f"USE `{dbname}`;")
    except Exception as e:
        print(f"Error selecting database `{dbname}`: {e}")
        db.close()
        return
    
    # relation_path=os.path.join(f"datasets/{dbname}","relation.jsonl")
    # relation_path=os.path.join(f"/data/zyz/reproduce/HiRAG/eval/datasets/{dbname}/test")
    relation_path=os.path.join(f"hi_ex/{dbname}","relation.jsonl")
    # relation_path=os.path.join(f"32b/{dbname}","relation.jsonl")
    with open(relation_path,"r")as f:
        val=[]
        for relation_l in f:
            relation=json.loads(relation_l)
            src_tgt=relation['src_tgt']
            tgt_src=relation['tgt_src']
            if len(src_tgt)>190 or len(tgt_src)>190:
                print(f"src_tgt or tgt_src too long: {src_tgt} {tgt_src}")
                continue
            description=relation['description']
            weight=relation['weight']
            level=0
            val.append((src_tgt,tgt_src,description,weight,level))
        sql = "INSERT INTO relations(src_tgt, tgt_src, description,  weight,level) VALUES (%s,%s,%s,%s,%s)"
        try:
        # 执行sql语句
            cursor.executemany(sql,tuple(val))
            # 提交到数据库执行
            db.commit()
        except Exception as e:
            # 发生错误时回滚
            db.rollback()
            print(e)
            print("insert relations error")
if __name__ == "__main__":
    working_dir='exp/compare_hirag_opt1_commonkg_32b/mix'
    # build_vector_search()
    # search_vector_search()
    create_db_table_mysql(working_dir)
    insert_data_to_mysql(working_dir)
    insert_origin_relations(working_dir)
    # print(find_tree_root(working_dir,'Policies'))
    # print(search_nodes_link('Innovation Policy Network','document',working_dir,0))
    # from query_graph import embedding
    # topk=200
    # query=embedding("mary")
    # milvus_client = MilvusClient(uri=f"/cpfs04/user/zhangyaoze/workspace/trag/ttt/milvus_demo.db")
    # collection_name = "entity_collection"
    # # query_embedding = emb_text(query)
    # search_results = milvus_client.search(
    #     collection_name=collection_name,
    #     data=query,
    #     limit=topk,
    #     filter=' level ==1 ',
    #     params={"metric_type": "L2", "params": {}},
    #     output_fields=["entity_name", "description","vector","level"],
    # )
    # print(len(search_results[0]))
    # for entity in search_results[0]:
    #     if entity['entity']['level']!=1:
    #         print(entity)
        
    # search_results2 = milvus_client.search(
    #     collection_name=collection_name,
    #     data=[vec],
    #     limit=topk,
    #     params={"metric_type": "L2", "params": {}},
    #     output_fields=["entity_name", "description","vector"],
    # )
    # recall=search_results2[0][0]['entity']['vector']
    # print(recall==vec)