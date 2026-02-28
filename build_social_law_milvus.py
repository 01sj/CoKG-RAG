#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
社会法传统RAG向量库构建脚本
使用 Milvus Lite（本地文件存储，无需启动服务）
参考：src/lightrag-hku/lightrag/kg/milvus_impl.py
"""

import json
import os
import time
import asyncio
from typing import List, Dict
from tqdm import tqdm
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from sentence_transformers import SentenceTransformer
import numpy as np

# ==================== 配置参数 ====================
# 输入文件路径
CHUNKS_FILE = "/newdataf/SJ/LeanRAG/datasets/chunks/basic_laws_social_only_chunk.json"

# 向量库保存路径（Milvus Lite 本地文件）
VECTOR_DB_PATH = "/newdataf/SJ/LeanRAG/vectorDB/social_law_milvus.db"

# Collection 名称
COLLECTION_NAME = "social_law_chunks"

# Embedding 模型配置
EMBEDDING_MODEL = "BAAI/bge-m3"  # 使用更新的bge-m3模型
EMBEDDING_DIM = 1024  # bge-m3 的维度

# 批处理大小
BATCH_SIZE = 100


def load_chunks(file_path: str) -> List[Dict]:
    """加载分块数据"""
    print(f"正在加载分块文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"共加载 {len(chunks)} 个分块")
    return chunks


def init_embedding_model(model_name: str) -> SentenceTransformer:
    """初始化 Embedding 模型"""
    print(f"正在加载 Embedding 模型: {model_name}")
    model = SentenceTransformer(model_name)
    print("Embedding 模型加载完成")
    return model


def create_milvus_client(db_path: str) -> MilvusClient:
    """创建 Milvus Lite 客户端（本地文件存储）"""
    print(f"正在创建 Milvus Lite 客户端: {db_path}")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    client = MilvusClient(uri=db_path)
    print("Milvus Lite 客户端创建成功")
    return client


def create_collection_schema(dim: int) -> CollectionSchema:
    """创建 Collection Schema"""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="hash_code", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_id", dtype=DataType.INT64),
        FieldSchema(name="source_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="token_count", dtype=DataType.INT64),
        FieldSchema(name="created_at", dtype=DataType.INT64),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="社会法RAG向量库",
        enable_dynamic_field=False
    )
    return schema


def create_collection(client: MilvusClient, collection_name: str, dim: int):
    """创建 Collection"""
    # 如果 collection 已存在，先删除
    if client.has_collection(collection_name):
        print(f"Collection '{collection_name}' 已存在，正在删除...")
        client.drop_collection(collection_name)
    
    print(f"正在创建 Collection: {collection_name}")
    schema = create_collection_schema(dim)
    
    # 创建 collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=client.prepare_index_params()
    )
    
    # 创建向量索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128}
    )
    client.create_index(collection_name, index_params)
    
    print(f"Collection '{collection_name}' 创建成功")


def insert_data(client: MilvusClient, collection_name: str, chunks: List[Dict], model: SentenceTransformer, batch_size: int):
    """批量插入数据"""
    print(f"开始插入数据，共 {len(chunks)} 条，批次大小: {batch_size}")
    
    current_time = int(time.time())
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(chunks), batch_size), total=total_batches, desc="插入进度"):
        batch = chunks[i:i + batch_size]
        
        # 提取文本并生成 embedding
        texts = [chunk["text"] for chunk in batch]
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        # 准备插入数据
        data = []
        for j, chunk in enumerate(batch):
            data.append({
                "id": chunk["hash_code"],
                "vector": embeddings[j].tolist(),
                "text": chunk["text"][:65530],  # 截断过长文本
                "hash_code": chunk["hash_code"],
                "source_type": chunk["source_type"],
                "source_id": chunk["source_id"],
                "source_name": chunk["source_name"],
                "token_count": chunk["token_count"],
                "created_at": current_time
            })
        
        # 插入数据
        client.upsert(collection_name=collection_name, data=data)
    
    # 获取总数
    stats = client.get_collection_stats(collection_name)
    print(f"数据插入完成，共插入 {stats['row_count']} 条记录")


def save_metadata(db_path: str, collection_name: str, chunks_count: int, model_name: str):
    """保存元数据信息"""
    metadata = {
        "collection_name": collection_name,
        "chunks_count": chunks_count,
        "embedding_model": model_name,
        "embedding_dim": EMBEDDING_DIM,
        "vector_store_type": "Milvus Lite",
        "db_path": db_path,
        "source_file": CHUNKS_FILE
    }
    
    metadata_file = os.path.join(os.path.dirname(db_path), "social_law_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"元数据已保存至: {metadata_file}")


def main():
    """主函数"""
    print("=" * 60)
    print("社会法 RAG 向量库构建 (Milvus Lite)")
    print("=" * 60)
    
    # 1. 加载分块数据
    chunks = load_chunks(CHUNKS_FILE)
    
    # 2. 初始化 Embedding 模型
    model = init_embedding_model(EMBEDDING_MODEL)
    
    # 3. 创建 Milvus Lite 客户端
    client = create_milvus_client(VECTOR_DB_PATH)
    
    # 4. 创建 Collection
    create_collection(client, COLLECTION_NAME, EMBEDDING_DIM)
    
    # 5. 插入数据
    insert_data(client, COLLECTION_NAME, chunks, model, BATCH_SIZE)
    
    # 6. 保存元数据
    save_metadata(VECTOR_DB_PATH, COLLECTION_NAME, len(chunks), EMBEDDING_MODEL)
    
    print("=" * 60)
    print("向量库构建完成！")
    print(f"数据库文件: {VECTOR_DB_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"数据条数: {len(chunks)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
