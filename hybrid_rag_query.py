#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合RAG检索系统：语义检索 + 文本相似度重排序 + 知识图谱检索

流程：
1. 语义向量检索 Top10
2. BM25文本相似度重排序，使用混合检索
3. 比较两次排序的相关性（Spearman相关系数）
4. 如果相关性高（>=阈值），直接使用向量检索结果生成答案
5. 如果相关性低（<阈值），启用知识图谱检索，融合结果后生成答案

不仅仅是简单、复杂两种。策略分为四种
"""

import json
import os
import sys
import time
import logging
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import argparse

# ⚠️ 重要：必须在导入 torch 和其他库之前设置 CUDA_VISIBLE_DEVICES
# 可以通过环境变量 CUDA_VISIBLE_DEVICES 指定GPU，默认使用GPU 1（第2张卡）
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 默认使用GPU 1（第2张卡）
print(f"🔧 设置使用GPU卡: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

import numpy as np
from scipy.stats import spearmanr
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import jieba
import torch

# 导入知识图谱查询模块
from database_utils import (
    search_vector_search,
    find_tree_root,
    search_nodes_link,
    search_community,
    get_text_units,
)
from prompt import PROMPTS

# 导入查询简单度评估模块
from query_simplicity_module import (
    measure_query_simplicity,
    calculate_combined_score_with_simplicity
)

# ==================== 配置参数 ====================
# 向量数据库配置
VECTOR_DB_PATH = "/newdataf/SJ/LeanRAG/vectorDB/social_law_milvus.db"
COLLECTION_NAME = "social_law_chunks"

# 知识图谱配置
KG_WORKING_DIR = "/newdataf/SJ/LeanRAG/KG_output/social_law_7B_processed/"

# Embedding模型配置
# SentenceTransformer 会自动从 ~/.cache/huggingface/hub/ 加载缓存的模型
# 已设置离线模式，不会连接 Hugging Face
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # bge-m3的维度也是1024

# ⚠️ 重要：必须与创建向量库时使用的模型一致！
# 如果向量库使用 bge-large-zh-v1.5 创建，需要重建向量库或修改此处为 bge-large-zh-v1.5

# 检索参数
TOP_K = 10  # 检索Top-K
CORRELATION_THRESHOLD = 0.35  # 复杂度阈值（优化后）

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class HybridLegalRAG:
    """混合法律RAG检索系统"""
    
    def __init__(
        self,
        vector_db_path: str,
        collection_name: str,
        kg_working_dir: str,
        embedding_model_name: str,
        device: str = "cpu",
        llm_model_path: str = None,
        llm_params: dict = None,
        # 消融实验参数
        use_retrieval_only: bool = False,
        use_intrinsic_only: bool = False,
        fixed_topk: bool = False,
        flat_kg: bool = False
    ):
        """
        初始化混合RAG系统
        
        Args:
            vector_db_path: Milvus向量数据库路径
            collection_name: Collection名称
            kg_working_dir: 知识图谱工作目录
            embedding_model_name: Embedding模型名称
            device: 设备（cpu/cuda）
            llm_model_path: LLM模型路径
            llm_params: LLM参数
            use_retrieval_only: 消融1-只用检索一致性评估
            use_intrinsic_only: 消融2-只用问题本质评估
            fixed_topk: 消融3-固定Top-K文档数量
            flat_kg: 消融4-扁平KG结构
        """
        self.logger = logging.getLogger(__name__)
        
        # 保存消融实验标志
        self.use_retrieval_only = use_retrieval_only
        self.use_intrinsic_only = use_intrinsic_only
        self.fixed_topk = fixed_topk
        self.flat_kg = flat_kg
        
        # 记录消融实验模式
        if any([use_retrieval_only, use_intrinsic_only, fixed_topk, flat_kg]):
            self.logger.info("🔬 消融实验模式:")
            if use_retrieval_only:
                self.logger.info("   - 只用检索一致性评估")
            if use_intrinsic_only:
                self.logger.info("   - 只用问题本质评估")
            if fixed_topk:
                self.logger.info("   - 固定Top-K文档数量")
            if flat_kg:
                self.logger.info("   - 扁平KG结构")
        
        # 初始化向量数据库
        self.logger.info(f"正在连接向量数据库: {vector_db_path}")
        
        # 确保使用绝对路径
        import os
        vector_db_path = os.path.abspath(vector_db_path)
        
        # 确保目录存在
        db_dir = os.path.dirname(vector_db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            self.logger.info(f"   创建目录: {db_dir}")
        
        # 检查文件是否存在
        if os.path.exists(vector_db_path):
            self.logger.info(f"   使用已存在的数据库: {vector_db_path}")
        else:
            self.logger.warning(f"   ⚠️ 数据库文件不存在: {vector_db_path}")
            self.logger.warning(f"   将创建新的数据库文件")
        
        # 连接 Milvus（参考 database_utils.py 的方式）
        try:
            self.milvus_client = MilvusClient(uri=vector_db_path)
            self.logger.info(f"   ✅ 成功连接到向量数据库")
        except Exception as e:
            self.logger.error(f"   ❌ 连接失败: {e}")
            self.logger.error(f"   请确保:")
            self.logger.error(f"   1. 数据库文件存在: {vector_db_path}")
            self.logger.error(f"   2. pymilvus 版本兼容 (推荐 2.2.x)")
            self.logger.error(f"   3. 已安装 milvus-lite: pip install milvus-lite")
            raise
        
        self.collection_name = collection_name
        
        # 初始化Embedding模型
        # SentenceTransformer会自动使用Hugging Face的缓存机制
        # 首次运行会从Hugging Face下载模型到 ~/.cache/huggingface/hub/
        # 后续运行会直接从缓存加载，无需重复下载
        self.logger.info(f"正在加载Embedding模型: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.embedding_model.max_seq_length = 4096  # bge-m3支持更长的序列
        
        # 知识图谱配置
        self.kg_working_dir = kg_working_dir
        
        # 构建法律词典（改进分词）
        self._build_law_dictionary()
        
        # LLM配置
        self.llm = None
        self.sampling_params = None
        if llm_model_path:
            self._init_llm(llm_model_path, llm_params or {})
        
        self.logger.info("混合RAG系统初始化完成")
    
    def _build_law_dictionary(self):
        """构建法律专用词典，改进分词效果"""
        self.logger.info("正在构建法律词典...")
        
        try:
            # 尝试从chunks文件提取法律术语
            chunks_file = "datasets/chunks/basic_laws_social_only_chunk.json"
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                law_names = set()
                article_numbers = set()
                
                for chunk in chunks:
                    source_name = chunk.get('source_name', '')
                    
                    # 提取法律名称
                    match = re.search(r'《(.+?)》', source_name)
                    if match:
                        law_name = match.group(1)
                        law_names.add(law_name)
                        # 添加简称
                        if law_name.startswith("中华人民共和国"):
                            short_name = law_name.replace("中华人民共和国", "")
                            law_names.add(short_name)
                    
                    # 提取法条号
                    match = re.search(r'第(.+?)条', source_name)
                    if match:
                        article_num = match.group(1)
                        article_numbers.add(f"第{article_num}条")
                
                # 添加到jieba
                for law in law_names:
                    jieba.add_word(law, freq=10000, tag='law')
                
                for article in article_numbers:
                    jieba.add_word(article, freq=10000, tag='article')
                
                self.logger.info(f"   ✅ 添加 {len(law_names)} 个法律名称, {len(article_numbers)} 个法条号")
            else:
                self.logger.warning(f"   ⚠️ chunks文件不存在: {chunks_file}")
        
        except Exception as e:
            self.logger.warning(f"   ⚠️ 构建法律词典失败: {e}")
        
        # 添加常见法律术语
        common_terms = [
            "劳动报酬", "加班费", "经济补偿", "劳动合同", "劳动关系",
            "用人单位", "劳动者", "社会保险", "工伤", "职业病",
            "未成年人", "监护人", "安全生产", "法律责任", "行政处罚",
        ]
        
        for term in common_terms:
            jieba.add_word(term, freq=5000, tag='term')
        
        self.logger.info(f"   ✅ 添加 {len(common_terms)} 个常见法律术语")
    
    def _init_llm(self, model_path: str, params: dict):
        """初始化LLM"""
        from vllm import LLM, SamplingParams
        
        self.logger.info(f"正在加载LLM模型: {model_path}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=params.get("tp", 1),
            gpu_memory_utilization=params.get("gpu_mem_util", 0.75),
            max_model_len=params.get("max_model_len", 8192),
            dtype="auto",
        )
        
        self.sampling_params = SamplingParams(
            temperature=params.get("temperature", 0.3),
            top_p=params.get("top_p", 0.9),
            max_tokens=params.get("max_new_tokens", 1024),
            repetition_penalty=1.1,
        )
        
        self.logger.info("LLM模型加载完成")
    
    def rewrite_query_for_consistency(self, query: str, instruction: str = "") -> str:
        """
        混合查询重写：简单问题用规则，复杂问题用LLM
        
        核心思想：
        - 简单问题：基于规则重写，去除口语化表达 → 提高相关系数
        - 复杂问题：使用LLM重写，提取关键信息 → 提高相关系数
        
        Args:
            query: 原始查询
            instruction: 指令
            
        Returns:
            重写后的查询
        """
        original_query = query
        
        # ==================== 步骤1: 尝试基于规则的重写 ====================
        # 检测简单概念查询模式并重写
        simple_patterns = [
            # 基本概念查询
            (r'^什么是(.+?)[\?？]?$', r'\1的定义'),
            (r'^(.+?)的定义是什么[\?？]?$', r'\1的定义'),
            (r'^(.+?)的含义是什么[\?？]?$', r'\1的含义'),
            
            # 分类/列举查询
            (r'^(.+?)包括哪些(.+?)[\?？]?$', r'\1的\2种类'),
            (r'^(.+?)分为哪几种[\?？]?$', r'\1的分类'),
            (r'^(.+?)分为哪(.+?)[\?？]?$', r'\1的\2分类'),
            (r'^(.+?)有哪些[\?？]?$', r'\1的种类'),
            
            # 立法目的/宗旨查询
            (r'^(.+?法)的立法目的是什么[\?？]?$', r'\1第一条立法目的'),
            (r'^(.+?法)的立法宗旨是什么[\?？]?$', r'\1第一条立法宗旨'),
            (r'^(.+?法)的制定目的[\?？]?$', r'\1第一条立法目的'),
            
            # 适用范围查询
            (r'^(.+?法)适用于哪些(.+?)[\?？]?$', r'\1第二条适用范围 \2'),
            (r'^(.+?法)适用于哪[\?？]?$', r'\1第二条适用范围'),
            (r'^(.+?法)的适用范围[\?？]?$', r'\1第二条适用范围'),
            
            # 权利/义务查询
            (r'^(.+?)享有哪些(.+?)权利[\?？]?$', r'\1的\2权利'),
            (r'^(.+?)有哪些(.+?)权利[\?？]?$', r'\1的\2权利'),
            (r'^(.+?)应当承担哪些(.+?)义务[\?？]?$', r'\1的\2义务'),
            (r'^(.+?)的(.+?)权利有哪些[\?？]?$', r'\1的\2权利'),
        ]
        
        # 尝试匹配简单模式并重写
        for pattern, replacement in simple_patterns:
            match = re.match(pattern, query.strip())
            if match:
                rewritten = re.sub(pattern, replacement, query.strip())
                self.logger.info(f"   🔄 规则重写: '{original_query}' → '{rewritten}'")
                return rewritten
        
        # 检查指令中是否有"直接引用"等关键词
        if instruction and any(keyword in instruction for keyword in ['直接引用', '引用相关法律条文', '引用相关法律', '直接给出法条']):
            cleaned = query
            for word in ['是什么', '有哪些', '包括', '的', '？', '?', '吗', '呢']:
                cleaned = cleaned.replace(word, ' ')
            cleaned = ' '.join(cleaned.split())
            
            if cleaned != query and len(cleaned) > 2:
                self.logger.info(f"   🔄 规则简化: '{original_query}' → '{cleaned}'")
                return cleaned
        
        # ==================== 步骤2: 复杂问题使用LLM重写 ====================
        # 检测是否为复杂问题（场景描述、推理问题、咨询问题）
        is_complex = self._is_complex_query(query, instruction)
        
        if is_complex and self.llm is not None:
            self.logger.info(f"   🤖 检测到复杂查询，使用LLM重写...")
            rewritten = self._llm_rewrite_query(query, instruction)
            if rewritten and rewritten != query:
                self.logger.info(f"   🔄 LLM重写: '{original_query}' → '{rewritten}'")
                return rewritten
            else:
                self.logger.info(f"   ⚠️ LLM重写失败或无变化，保持原查询")
        
        # 如果不是简单问题也不是复杂问题，或者LLM不可用，保持原样
        return query
    
    def _is_complex_query(self, query: str, instruction: str = "") -> bool:
        """
        判断是否为复杂查询
        
        复杂查询特征：
        - 包含"场景"关键词
        - 查询长度超过50字符
        - 包含多个句子
        - 指令中包含"场景"、"咨询"、"类别"等关键词
        
        Args:
            query: 查询文本
            instruction: 指令
            
        Returns:
            是否为复杂查询
        """
        # 特征1: 包含"场景"关键词
        if '场景' in query or '场景:' in query or '场景：' in query:
            return True
        
        # 特征2: 查询很长（超过50字符）
        if len(query) > 50:
            return True
        
        # 特征3: 包含多个句子（多个句号、问号）
        sentence_count = query.count('。') + query.count('？') + query.count('?') + query.count('，')
        if sentence_count >= 2:
            return True
        
        # 特征4: 指令中包含特定关键词
        if instruction:
            complex_keywords = ['场景', '咨询', '类别', '确定', '分类', '判断']
            if any(keyword in instruction for keyword in complex_keywords):
                return True
        
        # 特征5: 包含具体的人名、地名、时间等（表示具体案例）
        # 简单检测：包含"我"、"他"、"她"、"公司"、"单位"等
        case_keywords = ['我', '他', '她', '公司', '单位', '工厂', '学校', '医院']
        if any(keyword in query for keyword in case_keywords):
            return True
        
        return False
    
    def _llm_rewrite_query(self, query: str, instruction: str = "") -> str:
        """
        使用LLM重写复杂查询
        
        目标：
        - 提取查询中的关键法律概念
        - 去除冗余的场景描述
        - 保留核心法律问题
        
        Args:
            query: 原始查询
            instruction: 指令
            
        Returns:
            重写后的查询
        """
        if self.llm is None:
            return query
        
        # 构建重写提示词
        rewrite_prompt = f"""你是一个法律查询优化专家。请将以下查询重写为更适合法律知识库检索的形式。

重写要求：
1. 提取核心法律概念和关键词
2. 去除冗余的场景描述和口语化表达
3. 保留关键的法律要素（如主体、行为、后果等）
4. 重写后的查询应该简洁、精确，便于检索
5. 如果是咨询类问题，提取其中涉及的法律问题

原始查询：{query}

重写后的查询（只输出重写结果，不要解释）："""
        
        try:
            # 使用LLM生成重写
            outputs = self.llm.generate(
                [rewrite_prompt], 
                self.sampling_params
            )
            
            if outputs and outputs[0].outputs:
                rewritten = outputs[0].outputs[0].text.strip()
                
                # 清理输出（去除可能的前缀）
                for prefix in ['重写后的查询：', '重写后：', '查询：', '答：', 'A:', 'Answer:']:
                    if rewritten.startswith(prefix):
                        rewritten = rewritten[len(prefix):].strip()
                
                # 验证重写结果的合理性
                if len(rewritten) > 5 and len(rewritten) < len(query) * 2:
                    return rewritten
                else:
                    self.logger.warning(f"   ⚠️ LLM重写结果不合理: '{rewritten}'")
                    return query
            else:
                return query
                
        except Exception as e:
            self.logger.error(f"   ❌ LLM重写失败: {e}")
            return query
    
    def _is_classification_task(self, instruction: str) -> bool:
        """
        判断是否为分类任务
        
        分类任务特征：
        - instruction中包含"确定以下咨询的类别"
        - instruction中包含"类别包括"
        - instruction中包含"将答案写在[类别]"
        
        Args:
            instruction: 指令文本
            
        Returns:
            是否为分类任务
        """
        if not instruction:
            return False
        
        classification_keywords = [
            '确定以下咨询的类别',
            '类别包括',
            '将答案写在[类别]',
            '确定咨询的类别',
            '判断以下咨询属于',
            '分类为以下类别'
        ]
        
        return any(keyword in instruction for keyword in classification_keywords)
    
    def _add_classification_examples(self, instruction: str, question: str) -> str:
        """
        为分类任务添加Few-shot示例
        
        Args:
            instruction: 原始指令
            question: 问题文本
            
        Returns:
            增强后的指令
        """
        # Few-shot示例 - 重点强调医疗纠纷的判断标准
        examples = """
【分类示例】以下是一些典型案例，帮助你更准确地判断：

示例1 - 医疗纠纷:
问题：医疗事故起诉期多长时间？两年前，孩子出生时腿骨断了，现在起诉还有效吗？
分析：明确提到"医疗事故"，涉及医疗机构的医疗行为导致的损害
类别：医疗纠纷

示例2 - 医疗纠纷:
问题：我在医院做结育手术被医生抅了大肠造成了三级伤疾，这能赔偿多少补偿
分析：医生在手术过程中的失误导致损害，属于医疗行为导致的损害
类别：医疗纠纷

示例3 - 劳动纠纷（非医疗纠纷）:
问题：我在医院工作了一年没有给我签合同，后面想走了叫我补签，这个怎么做合法，怎么要求赔偿
分析：虽然在医院工作，但核心问题是劳动合同、劳动关系，不涉及医疗行为
类别：劳动纠纷

示例4 - 人身损害（工伤，非医疗纠纷）:
问题：我在工地摔伤，进医院做了开颅手术，医药费工地老板垫付了，请问我做法医鉴定还要注意些什么？
分析：在工地受伤属于工伤，虽然在医院治疗，但不是医疗行为导致的损害
类别：人身损害

示例5 - 劳动纠纷（工伤）:
问题：我在工地受伤，老板不给赔偿，怎么办？
分析：工作场所受伤，涉及工伤赔偿、劳动关系
类别：劳动纠纷

示例6 - 消费权益:
问题：我在超市买的面包里面发霉了，我不小心吃了一口，还在保质期内该怎么办？
分析：涉及商品质量、消费者权益、食品安全
类别：消费权益

【核心判断标准】
1. 医疗纠纷的关键特征：
   ✓ 医疗机构的医疗行为（诊断、治疗、手术、护理）导致的损害
   ✓ 医生、护士等医务人员的医疗过错
   ✓ 明确提到"医疗事故"、"医疗纠纷"、"误诊"、"手术失误"
   
2. 不是医疗纠纷的情况：
   ✗ 在医院工作但涉及劳动关系问题 → 劳动纠纷
   ✗ 工伤后在医院治疗 → 人身损害或劳动纠纷
   ✗ 在医院发生的非医疗行为导致的伤害 → 人身损害

3. 人身损害 vs 劳动纠纷：
   - 人身损害：侧重伤害赔偿、伤残鉴定
   - 劳动纠纷：侧重劳动关系、工资、合同、工伤认定

现在请分类以下咨询：
"""
        
        # 在instruction后添加示例
        enhanced_instruction = instruction + "\n\n" + examples
        
        return enhanced_instruction
    
    def _detect_medical_dispute(self, question: str) -> Tuple[bool, float]:
        """
        检测是否为医疗纠纷（特殊处理）
        
        医疗纠纷的特征：
        - 包含"医院"、"医生"、"手术"、"治疗"等医疗关键词
        - 同时包含"误伤"、"事故"、"失误"等损害关键词
        - 或者明确提到"医疗事故"、"医疗纠纷"
        
        Args:
            question: 问题文本
            
        Returns:
            (is_medical, confidence): 是否为医疗纠纷及置信度
        """
        medical_keywords = ['医院', '医生', '手术', '治疗', '病人', '医疗', '诊断', '药品', '护士']
        damage_keywords = ['误伤', '事故', '失误', '伤残', '死亡', '损害', '赔偿']
        explicit_keywords = ['医疗事故', '医疗纠纷', '医疗损害', '医疗过错']
        
        # 检查是否包含明确的医疗纠纷关键词
        if any(keyword in question for keyword in explicit_keywords):
            return True, 0.9
        
        # 检查是否同时包含医疗和损害关键词
        has_medical = sum(1 for kw in medical_keywords if kw in question)
        has_damage = sum(1 for kw in damage_keywords if kw in question)
        
        if has_medical >= 1 and has_damage >= 1:
            # 排除工伤场景（在医院治疗工伤不算医疗纠纷）
            work_injury_keywords = ['工地', '工厂', '工作', '上班', '单位', '公司']
            has_work_injury = any(kw in question for kw in work_injury_keywords)
            
            if has_work_injury:
                # 进一步判断：如果明确提到医疗行为导致的损害，仍然是医疗纠纷
                if '医生' in question and any(kw in question for kw in ['误伤', '失误', '事故']):
                    return True, 0.7
                else:
                    return False, 0.3
            else:
                confidence = min(0.8, 0.4 + has_medical * 0.1 + has_damage * 0.2)
                return True, confidence
        
        return False, 0.0
    
    def _extract_law_article_info(self, query: str) -> Tuple[str, str]:
        """
        从查询中提取法律名称和条文号
        
        Args:
            query: 查询文本
            
        Returns:
            (law_name, article_num): 法律名称和条文号
        """
        law_name = ""
        article_num = ""
        
        # 提取法律名称（《xxx》格式）
        law_match = re.search(r'《(.+?)》', query)
        if law_match:
            law_name = law_match.group(1)
        else:
            # 尝试匹配不带书名号的法律名称
            # 常见法律名称列表
            law_patterns = [
                r'(中华人民共和国)?劳动合同法',
                r'(中华人民共和国)?劳动法',
                r'(中华人民共和国)?社会保险法',
                r'(中华人民共和国)?未成年人保护法',
                r'(中华人民共和国)?妇女权益保障法',
                r'(中华人民共和国)?老年人权益保障法',
                r'(中华人民共和国)?食品安全法',
                r'(中华人民共和国)?药品管理法',
                r'(中华人民共和国)?职业病防治法',
                r'(中华人民共和国)?教育法',
                r'(中华人民共和国)?安全生产法',
                r'劳动合同法实施条例',
            ]
            for pattern in law_patterns:
                match = re.search(pattern, query)
                if match:
                    law_name = match.group(0)
                    # 补全为完整名称
                    if not law_name.startswith("中华人民共和国") and "实施条例" not in law_name:
                        law_name = "中华人民共和国" + law_name
                    break
        
        # 提取条文号（第xx条）
        article_match = re.search(r'第([一二三四五六七八九十百千\d]+)条', query)
        if article_match:
            article_num = article_match.group(1)
        
        return law_name, article_num
    
    def semantic_search(self, query: str, top_k: int = 10, rewritten_query: str = None) -> List[Dict]:
        """
        语义向量检索
        
        Args:
            query: 原始查询文本
            top_k: 返回Top-K结果
            rewritten_query: 重写后的查询（如果提供，用于embedding）
            
        Returns:
            检索结果列表，每个结果包含text, semantic_score, semantic_rank, chunk_id
        """
        self.logger.info(f"步骤1: 语义向量检索 Top{top_k}...")
        
        # 使用重写后的查询（如果有）
        search_query = rewritten_query if rewritten_query else query
        
        if rewritten_query and rewritten_query != query:
            self.logger.info(f"   [调试] 使用重写查询进行embedding: '{search_query}'")
        
        # 生成query embedding（使用重写后的查询）
        query_embedding = self.embedding_model.encode(
            search_query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # 执行语义检索
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            anns_field="vector",
            limit=top_k,
            output_fields=["text", "hash_code", "source_name"],
            search_params=search_params
        )
        
        semantic_results = []
        for i, hit in enumerate(results[0]):
            semantic_results.append({
                'text': hit['entity'].get('text', ''),
                'semantic_score': hit['distance'],
                'semantic_rank': i,
                'hash_code': hit['entity'].get('hash_code', f'chunk_{i}'),
                'source_name': hit['entity'].get('source_name', ''),
            })
        
        self.logger.info(f"   ✅ 检索到 {len(semantic_results)} 个结果")
        return semantic_results
    
    def bm25_search(self, query: str, top_k: int = 10, rewritten_query: str = None) -> List[Dict]:
        """
        独立的BM25检索（不依赖语义检索结果）
        
        核心思想：
        - 从向量库中检索所有候选文档（Top-100）
        - 使用BM25对所有候选文档排序
        - 返回Top-K结果
        
        这样可以与语义检索完全独立，更好地判断是否需要KG
        
        Args:
            query: 原始查询文本
            top_k: 返回Top-K结果
            rewritten_query: 重写后的查询
            
        Returns:
            BM25检索结果列表
        """
        self.logger.info(f"步骤2: 独立BM25检索 Top{top_k}...")
        
        # 使用重写后的查询（如果有）
        search_query = rewritten_query if rewritten_query else query
        if rewritten_query and rewritten_query != query:
            self.logger.info(f"   [调试] 使用重写查询进行BM25: '{search_query}'")
        
        # 先用语义检索获取候选文档池（Top-100，扩大候选范围）
        query_embedding = self.embedding_model.encode(
            search_query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # 检索Top-200作为候选池（扩大候选范围）
        candidate_size = min(200, top_k * 20)
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            anns_field="vector",
            limit=candidate_size,
            output_fields=["text", "hash_code", "source_name"],
            search_params=search_params
        )
        
        self.logger.info(f"   获取 {len(results[0])} 个候选文档")
        
        # 提取文档文本
        docs = [hit['entity'].get('text', '') for hit in results[0]]
        
        # 分词
        tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
        tokenized_query = list(jieba.cut(search_query))
        
        self.logger.info(f"   查询分词: {tokenized_query[:10]}...")
        
        # BM25计算（使用固定参数）
        bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # 构建结果列表
        bm25_results = []
        for i, hit in enumerate(results[0]):
            bm25_results.append({
                'text': hit['entity'].get('text', ''),
                'bm25_score': float(bm25_scores[i]),
                'bm25_rank': 0,  # 稍后排序后更新
                'hash_code': hit['entity'].get('hash_code', f'chunk_{i}'),
                'source_name': hit['entity'].get('source_name', ''),
            })
        
        # 按BM25分数排序
        bm25_results = sorted(bm25_results, key=lambda x: x['bm25_score'], reverse=True)
        
        # 更新排名
        for i, result in enumerate(bm25_results):
            result['bm25_rank'] = i
        
        # 返回Top-K
        bm25_results = bm25_results[:top_k]
        
        self.logger.info(f"   ✅ BM25检索完成")
        # 显示Top3分数
        top3_bm25 = [f"{r['bm25_score']:.1f}" for r in bm25_results[:3]]
        self.logger.info(f"   Top3 BM25分数: {top3_bm25}")
        
        return bm25_results
    
    def create_hybrid_results(self, semantic_results: List[Dict], bm25_results: List[Dict], alpha: float = 0.7) -> List[Dict]:
        """
        创建混合检索结果（融合语义检索和BM25检索）
        
        Args:
            semantic_results: 语义检索结果
            bm25_results: BM25检索结果
            alpha: 语义权重（0-1），推荐0.7表示70%语义+30%BM25
            
        Returns:
            混合检索结果列表
        """
        self.logger.info(f"步骤4: 创建混合检索结果 (alpha={alpha})...")
        
        # 创建hash_code到结果的映射
        semantic_map = {r['hash_code']: r for r in semantic_results}
        bm25_map = {r['hash_code']: r for r in bm25_results}
        
        # 合并所有文档
        all_hash_codes = set(semantic_map.keys()) | set(bm25_map.keys())
        
        hybrid_results = []
        
        for hash_code in all_hash_codes:
            # 获取语义分数和BM25分数
            semantic_score = semantic_map[hash_code]['semantic_score'] if hash_code in semantic_map else 0.0
            bm25_score = bm25_map[hash_code]['bm25_score'] if hash_code in bm25_map else 0.0
            
            # 获取文档信息（优先从语义检索结果）
            if hash_code in semantic_map:
                doc_info = semantic_map[hash_code]
            else:
                doc_info = bm25_map[hash_code]
            
            hybrid_results.append({
                'text': doc_info['text'],
                'hash_code': hash_code,
                'source_name': doc_info['source_name'],
                'semantic_score': semantic_score,
                'bm25_score': bm25_score,
            })
        
        # 归一化分数
        semantic_scores = [r['semantic_score'] for r in hybrid_results]
        bm25_scores = [r['bm25_score'] for r in hybrid_results]
        
        sem_min, sem_max = min(semantic_scores), max(semantic_scores)
        if sem_max > sem_min:
            semantic_norm = [(s - sem_min) / (sem_max - sem_min) for s in semantic_scores]
        else:
            semantic_norm = [1.0] * len(semantic_scores)
        
        bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
        if bm25_max > bm25_min:
            bm25_norm = [(s - bm25_min) / (bm25_max - bm25_min) for s in bm25_scores]
        else:
            bm25_norm = [1.0] * len(bm25_scores)
        
        # 计算混合分数
        for i, result in enumerate(hybrid_results):
            result['semantic_norm'] = semantic_norm[i]
            result['bm25_norm'] = bm25_norm[i]
            result['hybrid_score'] = alpha * semantic_norm[i] + (1 - alpha) * bm25_norm[i]
        
        # 按混合分数排序
        hybrid_results = sorted(hybrid_results, key=lambda x: x['hybrid_score'], reverse=True)
        
        # 更新排名
        for i, result in enumerate(hybrid_results):
            result['hybrid_rank'] = i
        
        self.logger.info(f"   ✅ 混合检索完成，共 {len(hybrid_results)} 个文档")
        # 显示Top3分数
        top3_hybrid = [f"{r['hybrid_score']:.3f}" for r in hybrid_results[:3]]
        self.logger.info(f"   Top3 混合分数: {top3_hybrid}")
        
        return hybrid_results
    
    def compare_independent_rankings(self, semantic_results: List[Dict], bm25_results: List[Dict], query: str) -> Dict:
        """
        多维度比较两个独立检索结果的相似度（8个指标，4个维度）
        
        核心思想：
        - 如果两种方法检索到的文档重叠度高 → 简单问题，传统RAG足够
        - 如果两种方法检索到的文档差异大 → 复杂问题，需要KG
        
        Args:
            semantic_results: 语义检索结果
            bm25_results: BM25检索结果
            query: 查询文本
            
        Returns:
            包含核心评估指标的字典
        """
        self.logger.info("步骤3: 比较两种检索结果...")
        
        metrics = {}
        
        # 提取文档ID列表
        semantic_ids = [r['hash_code'] for r in semantic_results]
        bm25_ids = [r['hash_code'] for r in bm25_results]
        overlap_ids = set(semantic_ids) & set(bm25_ids)
        
        # ========== 核心指标1: BM25 Top1分数 ==========
        if len(bm25_results) > 0:
            metrics['bm25_top1_score'] = bm25_results[0].get('bm25_score', 10.0)
        else:
            metrics['bm25_top1_score'] = 10.0
        
        # ========== 核心指标2: 文档重叠率 ==========
        metrics['overlap_ratio'] = len(overlap_ids) / len(semantic_ids) if len(semantic_ids) > 0 else 0.0
        self.logger.info(f"   - 文档重叠率: {metrics['overlap_ratio']:.3f} ({len(overlap_ids)}/{len(semantic_ids)})")
        
        # ========== 核心指标3: Top-3重叠率 ==========
        top3_semantic = set(semantic_ids[:3])
        top3_bm25 = set(bm25_ids[:3])
        top3_overlap = len(top3_semantic & top3_bm25)
        metrics['top3_overlap'] = top3_overlap / 3
        self.logger.info(f"   - Top-3重叠率: {metrics['top3_overlap']:.3f} ({top3_overlap}/3)")
        
        # ========== 计算综合分数 ==========
        try:
            metrics['combined_score'] = self._calculate_combined_score(metrics, query=query)
            
            self.logger.info(f"\n   📊 综合相似度: {metrics['combined_score']:.3f}")
            self.logger.info(f"   解读: {'简单问题，传统RAG足够' if metrics['combined_score'] >= 0.7 else '复杂问题，需要KG辅助'}")
        except Exception as e:
            self.logger.error(f"   ❌ 计算综合相似度时出错: {e}")
            self.logger.error(f"   指标值: {metrics}")
            metrics['combined_score'] = 0.0
        
        return metrics
    

    
    def _calculate_combined_score(self, metrics: Dict, query: str = "") -> float:
        """
        计算综合复杂度分数（集成统一复杂度评估）
        
        新版本：使用统一复杂度评分体系
        - 三层评估架构：问题类型分类 → 复杂问题细分 → 五维度评估
        - 问题本质复杂度 = 直接计算（分数越高越复杂）
        - 检索不一致性 = 反转后的RCC计算（分数越高越不一致）
        - 最终复杂度 = 0.5 × 问题本质复杂度 + 0.5 × 检索不一致性
        - 决策规则：最终复杂度 ≥ 0.4 → KG；< 0.4 → 传统RAG
        
        Args:
            metrics: 所有指标的字典
            query: 查询文本
            
        Returns:
            最终复杂度分数（0-1），分数越高越复杂
        """
        # 安全获取指标值
        def safe_get(key, default=0.0):
            val = metrics.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)
        
        bm25_top1 = safe_get('bm25_top1_score', 10.0)
        overlap_ratio = safe_get('overlap_ratio')
        top3_overlap = safe_get('top3_overlap')
        
        # 🔬 消融实验1：只用检索一致性
        if self.use_retrieval_only:
            self.logger.info("   🔬 消融模式：只用检索一致性评估")
            # 只计算检索不一致性
            result = calculate_combined_score_with_simplicity(
                query=query,
                bm25_top1_score=bm25_top1,
                overlap_ratio=overlap_ratio,
                top3_overlap=top3_overlap,
                llm=None,  # 不使用LLM
                sampling_params=None,
                use_five_dimensions=False
            )
            # 只返回检索不一致性
            metrics.update(result)
            return result.get('retrieval_inconsistency', 0.5)
        
        # 🔬 消融实验2：只用问题本质
        if self.use_intrinsic_only:
            self.logger.info("   🔬 消融模式：只用问题本质评估")
            # 只计算问题本质复杂度
            result = calculate_combined_score_with_simplicity(
                query=query,
                bm25_top1_score=bm25_top1,
                overlap_ratio=overlap_ratio,
                top3_overlap=top3_overlap,
                llm=self.llm,
                sampling_params=self.sampling_params,
                use_five_dimensions=True
            )
            # 只返回问题本质复杂度
            metrics.update(result)
            return result.get('question_nature_complexity', 0.5)
        
        # 正常模式：使用完整评估
        result = calculate_combined_score_with_simplicity(
            query=query,
            bm25_top1_score=bm25_top1,
            overlap_ratio=overlap_ratio,
            top3_overlap=top3_overlap,
            llm=self.llm,  # 传入LLM实例
            sampling_params=self.sampling_params,
            use_five_dimensions=True  # 启用统一复杂度评估
        )
        
        # 记录详细信息
        self.logger.info(f"\n   📊 统一复杂度评估:")
        
        # 子问题拆分信息
        if result.get('decomposition'):
            decomp = result['decomposition']
            self.logger.info(f"      子问题拆分: {decomp['num_sub_questions']}个子问题")
            if decomp['num_sub_questions'] > 1:
                for i, sq in enumerate(decomp['sub_questions'][:3], 1):
                    self.logger.info(f"        {i}. {sq}")
                if len(decomp['sub_questions']) > 3:
                    self.logger.info(f"        ...")
        
        # 问题类型信息
        self.logger.info(f"      问题类型: {result['question_type']} (第{result['evaluation_layer']}层评估)")
        
        # 问题本质复杂度
        self.logger.info(f"      问题本质复杂度: {result['question_nature_complexity']:.3f}")
        if result.get('base_complexity') is not None:
            self.logger.info(f"        - 基础复杂度: {result['base_complexity']:.3f}")
        if result.get('five_dimension_score') is not None:
            self.logger.info(f"        - 五维度分数: {result['five_dimension_score']:.3f}")
        
        # 检索不一致性
        self.logger.info(f"      检索不一致性: {result['retrieval_inconsistency']:.3f}")
        
        # 最终复杂度
        self.logger.info(f"      最终复杂度: {result['final_complexity']:.3f} (阈值={result['threshold']})")
        self.logger.info(f"      决策: {result['decision']}")
        self.logger.info(f"      理由: {result['reason']}")
        
        # 将详细结果保存到metrics中
        metrics.update(result)
        
        # 返回最终复杂度（注意：新版本使用final_complexity）
        return result['final_complexity']
    
    def _rerank_and_select(
        self, 
        hybrid_results: List[Dict], 
        query: str, 
        combined_score: float,
        is_simple: bool,
        max_docs: int = 10
    ) -> Tuple[List[Dict], int]:
        """
        重排序并动态选择最相关的文档（支持消融实验）
        
        策略：
        1. 检测是否为法条查询（特殊处理）
        2. 根据问题类型（简单/复杂）确定基础文档数量
        3. 基于混合分数进行重排序（已经完成）
        4. 使用分数阈值过滤低相关文档
        5. 动态调整最终文档数量
        
        消融实验支持：
        - fixed_topk: 固定返回10个文档，跳过自适应选择
        
        Args:
            hybrid_results: 混合检索结果（已按hybrid_score排序）
            query: 查询文本
            combined_score: 综合相似度分数
            is_simple: 是否为简单问题
            max_docs: 最大文档数量（默认10，分类任务可以设置为15）
            
        Returns:
            (重排序后的文档列表, 选择的文档数量)
        """
        if not hybrid_results:
            return [], 0
        
        # 🔬 消融实验3：固定Top-K
        if self.fixed_topk:
            self.logger.info(f"   🔬 消融模式：固定Top-K文档数量")
            fixed_count = 10  # 固定使用10个文档
            selected_results = hybrid_results[:fixed_count]
            self.logger.info(f"   📊 固定选择: {fixed_count} 个文档")
            selected_scores = [f"{r['hybrid_score']:.3f}" for r in selected_results]
            self.logger.info(f"      - 选择文档分数: {selected_scores}")
            return selected_results, fixed_count
        
        # 检测是否为法条查询（包含"第XX条"的模式）
        import re
        is_law_article_query = bool(re.search(r'第\s*[一二三四五六七八九十百\d]+\s*条', query))
        
        # 策略1: 根据问题类型确定基础数量
        if is_simple:
            if is_law_article_query:
                # 法条查询：保留更多文档，确保不遗漏正确法条
                base_count = min(10, max_docs)
                min_count = 8
                max_count = max_docs
                self.logger.info(f"   🔍 检测到法条查询，使用保守策略")
            else:
                # 普通简单问题：法条查询，需要较多文档确保覆盖
                base_count = min(8, max_docs)
                min_count = 5
                max_count = max_docs
        else:
            # 复杂问题：场景推理，需要精准文档避免噪音
            base_count = min(5, max_docs)
            min_count = 3
            max_count = min(7, max_docs)
        
        # 策略2: 基于分数分布动态调整
        # 计算分数的相对下降，找到"断崖"位置
        scores = [r['hybrid_score'] for r in hybrid_results]
        
        # 找到分数显著下降的位置
        score_drops = []
        for i in range(len(scores) - 1):
            if scores[i] > 0:
                drop_ratio = (scores[i] - scores[i+1]) / scores[i]
                score_drops.append((i+1, drop_ratio))
        
        # 策略3: 使用相对阈值过滤
        # 法条查询使用更宽松的阈值（50%），普通查询使用60%
        if is_law_article_query:
            threshold_ratio = 0.5  # 更宽松，避免过滤掉正确法条
        else:
            threshold_ratio = 0.6
        
        if scores[0] > 0:
            relative_threshold = scores[0] * threshold_ratio
        else:
            relative_threshold = 0.5
        
        # 找到第一个低于阈值的位置
        cutoff_by_threshold = len(scores)
        for i, score in enumerate(scores):
            if score < relative_threshold:
                cutoff_by_threshold = i
                break
        
        # 找到分数显著下降的位置（下降超过20%）
        cutoff_by_drop = len(scores)
        for i, drop_ratio in score_drops:
            if drop_ratio > 0.2 and i >= min_count:
                cutoff_by_drop = i
                break
        
        # 综合决策：取多个策略的中间值
        # 法条查询时，优先保留更多文档
        if is_law_article_query:
            # 法条查询：取最大值而非中位数，确保不遗漏
            cutoff_candidates = [
                base_count,
                cutoff_by_threshold,
                cutoff_by_drop
            ]
            selected_count = max(cutoff_candidates)
        else:
            # 普通查询：取中位数
            cutoff_candidates = [
                base_count,
                cutoff_by_threshold,
                cutoff_by_drop
            ]
            cutoff_candidates.sort()
            selected_count = cutoff_candidates[len(cutoff_candidates) // 2]
        
        # 限制在[min_count, max_count]范围内
        selected_count = max(min_count, min(selected_count, max_count))
        
        # 确保不超过实际文档数量
        selected_count = min(selected_count, len(hybrid_results))
        
        # 记录决策过程
        self.logger.info(f"   📊 重排序决策:")
        self.logger.info(f"      - 问题类型: {'简单问题' if is_simple else '复杂问题'}")
        if is_law_article_query:
            self.logger.info(f"      - 法条查询: 是 (使用保守策略)")
        self.logger.info(f"      - 基础数量: {base_count}")
        self.logger.info(f"      - 阈值截断: {cutoff_by_threshold} (阈值={relative_threshold:.3f}, 比例={threshold_ratio})")
        self.logger.info(f"      - 分数断崖: {cutoff_by_drop}")
        self.logger.info(f"      - 最终选择: {selected_count} 个文档")
        
        # 返回选择的文档
        selected_results = hybrid_results[:selected_count]
        
        # 显示选择的文档分数
        selected_scores = [f"{r['hybrid_score']:.3f}" for r in selected_results]
        self.logger.info(f"      - 选择文档分数: {selected_scores}")
        
        return selected_results, selected_count

    
    def _kg_search_flat(self, query: str, top_k: int = 10) -> str:
        """
        一维图谱检索（扁平结构）
        
        只使用实体描述和文本单元，不包含层次结构（社区聚合、推理路径）
        适用于轻度复杂问题
        
        Args:
            query: 查询文本
            top_k: Top-K实体
            
        Returns:
            一维图谱检索的上下文描述
        """
        # 生成query embedding
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # 向量检索实体
        entity_results = search_vector_search(
            self.kg_working_dir,
            [query_embedding.tolist()],
            topk=top_k,
            level_mode=1
        )
        
        res_entity = [i[0] for i in entity_results]
        chunks = [i[-1] for i in entity_results]
        
        self.logger.info(f"   ✅ 检索到 {len(res_entity)} 个相关实体")
        
        # 生成实体描述
        entity_descriptions = self._get_entity_description(entity_results)
        
        # 提取文本单元
        text_units = get_text_units(self.kg_working_dir, chunks, None, k=5)
        
        # 组织上下文（一维结构）
        kg_context = f"""
entity_information:
{entity_descriptions}

text_units:
{text_units}
"""
        
        self.logger.info("   ✅ 一维图谱检索完成")
        return kg_context
    
    def kg_search(self, query: str, top_k: int = 10) -> str:
        """
        层次化知识图谱检索（完整结构）
        
        包含实体描述、社区聚合、推理路径（LCA算法）和文本单元
        适用于极复杂问题
        
        消融实验支持：
        - flat_kg: 只使用实体描述，跳过层次结构（社区聚合、推理路径）
        
        Args:
            query: 查询文本
            top_k: Top-K实体
            
        Returns:
            层次化图谱检索的上下文描述
        """
        # 生成query embedding
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # 向量检索实体
        entity_results = search_vector_search(
            self.kg_working_dir,
            [query_embedding.tolist()],
            topk=top_k,
            level_mode=1
        )
        
        res_entity = [i[0] for i in entity_results]
        chunks = [i[-1] for i in entity_results]
        
        self.logger.info(f"   ✅ 检索到 {len(res_entity)} 个相关实体")
        
        # 生成实体描述
        entity_descriptions = self._get_entity_description(entity_results)
        
        # 🔬 消融实验4：扁平KG
        if self.flat_kg:
            self.logger.info("   🔬 消融模式：扁平KG结构（跳过层次结构）")
            # 只使用实体描述，跳过层次结构
            text_units = get_text_units(self.kg_working_dir, chunks, None, k=5)
            
            kg_context = f"""
entity_information:
{entity_descriptions}

text_units:
{text_units}
"""
            self.logger.info("   ✅ 扁平KG检索完成")
            return kg_context
        
        # 正常模式：使用完整层次结构
        # 构建推理路径（使用LCA算法）
        self.logger.info("   构建推理路径（LCA算法）...")
        reasoning_path, reasoning_path_info = self._get_reasoning_chain(res_entity)
        
        # 聚合社区信息
        self.logger.info("   聚合社区信息...")
        aggregation_descriptions, aggregation = self._get_aggregation_description(reasoning_path)
        
        # 提取文本单元
        text_units = get_text_units(self.kg_working_dir, chunks, None, k=5)
        
        # 组织上下文（层次化结构）
        kg_context = f"""
entity_information:
{entity_descriptions}

aggregation_entity_information:
{aggregation_descriptions}

reasoning_path_information:
{reasoning_path_info}

text_units:
{text_units}
"""
        
        self.logger.info("   ✅ 层次化图谱检索完成")
        return kg_context
    
    def _get_entity_description(self, entity_results: List[tuple]) -> str:
        """生成实体描述"""
        columns = ["entity_name", "parent", "description"]
        entity_descriptions = "\t\t".join(columns) + "\n"
        entity_descriptions += "\n".join([
            info[0] + "\t\t" + info[1] + "\t\t" + info[2]
            for info in entity_results
        ])
        return entity_descriptions
    
    def _get_reasoning_chain(self, entities_set: List[str]) -> Tuple[List, str]:
        """构建推理路径（简化版）"""
        from itertools import combinations
        
        # 限制实体数量以提高速度
        if len(entities_set) > 5:
            entities_set = entities_set[:5]
        
        maybe_edges = list(combinations(entities_set, 2))
        reasoning_path = []
        reasoning_path_information = []
        db_name = os.path.basename(self.kg_working_dir.rstrip("/"))
        
        for node1, node2 in maybe_edges:
            # 查找树根
            node1_tree = find_tree_root(db_name, node1)
            node2_tree = find_tree_root(db_name, node2)
            
            a_path = []
            b_path = []
            for i, j in zip(node1_tree, node2_tree):
                if i == j:
                    a_path.append(i)
                    break
                if i in b_path or j in a_path:
                    break
                if i != j:
                    a_path.append(i)
                    b_path.append(j)
            
            path = a_path + [b_path[len(b_path) - 1 - i] for i in range(len(b_path))]
            reasoning_path.append(path)
            
            # 查询关系
            all_nodes = list(set(a_path + b_path))
            if len(all_nodes) > 5:
                all_nodes = all_nodes[:5]
            
            for maybe_edge in combinations(all_nodes, 2):
                if maybe_edge[0] != maybe_edge[1]:
                    info = search_nodes_link(maybe_edge[0], maybe_edge[1], self.kg_working_dir)
                    if info is not None:
                        reasoning_path_information.append([maybe_edge[0], maybe_edge[1], info[2]])
        
        temp_relations_information = list(set([info[2] for info in reasoning_path_information]))
        reasoning_path_information_description = "\n".join(temp_relations_information)
        
        return reasoning_path, reasoning_path_information_description
    
    def _get_aggregation_description(self, reasoning_path: List) -> Tuple[str, set]:
        """聚合社区信息"""
        aggregation_results = []
        communities = set([community for each_path in reasoning_path for community in each_path])
        
        for community in communities:
            temp = search_community(community, self.kg_working_dir)
            if temp == "":
                continue
            aggregation_results.append(temp)
        
        columns = ["entity_name", "entity_description"]
        aggregation_descriptions = "\t\t".join(columns) + "\n"
        aggregation_descriptions += "\n".join([
            info[0] + "\t\t" + str(info[1])
            for info in aggregation_results
        ])
        
        return aggregation_descriptions, communities
    
    def _clean_answer(self, answer: str, query: str) -> str:
        """
        清理答案中可能出现的问题重复（保守版本）
        
        Args:
            answer: 生成的答案
            query: 原始查询
            
        Returns:
            清理后的答案
        """
        # 🔧 修复：如果答案太短，直接返回，不做任何清理
        if not answer or len(answer.strip()) < 10:
            return answer
        
        original_answer = answer
        
        # 检查是否以"回答："、"答案："等开头（只清理这些明确的前缀）
        prefixes_to_remove = ['回答：', '答案：', '回答:', '答案:']
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                self.logger.info(f"   🧹 移除答案前缀: {prefix}")
                cleaned = answer[len(prefix):].strip()
                # 如果清理后答案太短，返回原答案
                if len(cleaned) >= 10:
                    return cleaned
                else:
                    self.logger.warning(f"   ⚠️ 清理后答案过短，保留原答案")
                    return original_answer
        
        # 🔧 不再移除问题重复，因为这可能会误删有效内容
        # 直接返回原答案
        return answer
    
    def _extract_law_article_info(self, query: str) -> Dict:
        """
        从查询中提取法条信息
        
        Args:
            query: 用户查询
            
        Returns:
            包含法律名称和条款编号的字典
        """
        import re
        
        # 提取法律名称（《xxx》）
        law_name_match = re.search(r'《([^》]+)》', query)
        law_name = law_name_match.group(1) if law_name_match else None
        
        # 提取条款编号（第XX条）
        article_match = re.search(r'第\s*([一二三四五六七八九十百千\d]+)\s*条', query)
        article_num = article_match.group(1) if article_match else None
        
        return {
            'law_name': law_name,
            'article_num': article_num,
            'has_law_info': law_name is not None and article_num is not None
        }
    
    def _find_exact_law_article(self, query: str, all_results: List[Dict]) -> Optional[Dict]:
        """
        在检索结果中精确匹配法条
        
        Args:
            query: 用户查询
            all_results: 所有检索结果（包括semantic和bm25）
            
        Returns:
            匹配的法条文档，如果没找到返回None
        """
        import re
        
        # 提取查询中的法条信息
        law_info = self._extract_law_article_info(query)
        
        if not law_info['has_law_info']:
            return None
        
        law_name = law_info['law_name']
        article_num = law_info['article_num']
        
        self.logger.info(f"   🔍 精确匹配法条: 《{law_name}》第{article_num}条")
        
        # 构建精确匹配的source_name模式
        target_source_name = f"《{law_name}》第{article_num}条"
        
        # 在所有结果中查找精确匹配
        # 优先级1: 精确匹配source_name（最可靠）
        for result in all_results:
            source_name = result.get('source_name', '')
            if source_name == target_source_name:
                self.logger.info(f"   ✅ 找到精确匹配（source_name）: {source_name}")
                return result
        
        # 优先级2: 检查source_name是否包含法律名称和条款号
        article_pattern = rf'第\s*{re.escape(article_num)}\s*条'
        for result in all_results:
            source_name = result.get('source_name', '')
            if law_name in source_name and re.search(article_pattern, source_name):
                self.logger.info(f"   ✅ 找到匹配（source_name模糊）: {source_name}")
                return result
        
        # 优先级3: 检查text字段（作为后备）
        for result in all_results:
            text = result.get('text', '')
            source_name = result.get('source_name', '')
            
            # 检查是否包含法律名称和条款编号
            if law_name in text and re.search(article_pattern, text):
                self.logger.info(f"   ✅ 找到匹配（text字段）: {source_name[:50]}...")
                return result
        
        self.logger.warning(f"   ⚠️ 未找到精确匹配的法条")
        return None
    
    def generate_answer(self, query: str, context: str, instruction: str = "", use_kg: bool = False, 
                       semantic_results: List[Dict] = None, bm25_results: List[Dict] = None) -> str:
        """
        生成答案
        
        Args:
            query: 用户查询
            context: 检索到的上下文
            instruction: 指令（来自数据集）
            use_kg: 是否使用了知识图谱
            semantic_results: 语义检索结果（用于法条精确匹配）
            bm25_results: BM25检索结果（用于法条精确匹配）
            
        Returns:
            生成的答案
        """
        self.logger.info("步骤5: LLM生成答案...")
        
        if self.llm is None:
            self.logger.warning("LLM未初始化，返回空答案")
            return ""
        
        # 检测是否为法条查询
        is_law_article_query = bool(re.search(r'第[一二三四五六七八九十百千\d]+条', query))
        is_simple_article_query = "只需直接给出法条内容" in instruction or "只需要给出具体法条内容" in instruction
        
        # 🔧 新增：法条精确匹配 - 直接返回法条原文
        if (is_law_article_query or is_simple_article_query) and semantic_results and bm25_results:
            # 合并所有检索结果进行精确匹配
            all_results = semantic_results + bm25_results
            exact_match = self._find_exact_law_article(query, all_results)
            
            if exact_match:
                # 找到精确匹配，直接返回法条原文，不经过LLM
                self.logger.info(f"   ✅ 找到精确匹配的法条，直接返回原文")
                law_text = exact_match['text'].strip()
                
                # 检查law_text是否已经包含格式化的法律名称和条款号
                # 如果已经包含（如：《法律名称》第XX条: 内容），则直接返回
                if re.match(r'《.+》第.+条:', law_text):
                    formatted_answer = law_text
                    self.logger.info(f"   📄 法条已格式化，直接返回（长度: {len(formatted_answer)}字符）")
                else:
                    # 如果没有格式化，则进行格式化
                    law_info = self._extract_law_article_info(query)
                    if law_info['has_law_info']:
                        formatted_answer = f"《{law_info['law_name']}》第{law_info['article_num']}条: {law_text}"
                    else:
                        formatted_answer = law_text
                    self.logger.info(f"   📄 格式化后返回法条（长度: {len(formatted_answer)}字符）")
                
                return formatted_answer
        
        if is_law_article_query or is_simple_article_query:
            # 🔧 法条查询：使用极简提示词，要求直接返回法条原文
            system_message = "你是一个法律助手。请严格按照用户问题，只返回指定的法条内容，不要返回其他法条。"
            
            # 🔧 简化上下文：只保留文本内容，去掉【文档X】标记
            # 检查context是否包含【文档】标记
            if "【文档" in context:
                # 移除【文档X】标记，只保留法条内容
                clean_context = re.sub(r'【文档\d+】[^\n]*\n', '', context)
            else:
                clean_context = context
            
            # 提取法条信息
            law_info = self._extract_law_article_info(query)
            
            # 如果能提取到法条信息，在提示词中明确指定
            if law_info['has_law_info']:
                target_article = f"《{law_info['law_name']}》第{law_info['article_num']}条"
                user_message = f"""以下是法律文本：

{clean_context}

问题：{query}

重要提示：请只返回 {target_article} 的内容，不要返回其他法条。
如果文本中没有该法条，请回答"未找到该法条"。

格式要求：
《法律名称》第XX条: [法条原文]"""
            else:
                user_message = f"""以下是法律文本：

{clean_context}

问题：{query}

请直接给出该法条的完整内容，不要添加任何解释、分析或额外说明。格式如下：
《法律名称》第XX条: [法条原文]"""
            
            # Qwen2对话格式
            composed = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
            
        else:
            # 使用完整的提示词模板
            system_prompt = PROMPTS.get("rag_response_zh", PROMPTS.get("local_rag_response", "")).format(
                context_data=context
            )
            
            # 如果有instruction，将其加入到query中
            if instruction:
                full_query = f"{instruction}\n{query}"
            else:
                full_query = query
            
            # 🔧 修复：使用Qwen2的对话格式
            if system_prompt:
                # 简化system_prompt，去掉复杂的markdown格式
                simple_system = "你是一个专业的法律知识问答助手。请根据提供的法律知识和上下文回答用户的问题。"
                user_message = f"""以下是相关的法律知识和上下文：

{context}

用户问题：{full_query}

请根据上述信息回答问题。"""
                
                composed = f"<|im_start|>system\n{simple_system}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
            else:
                system_message = "你是一个法律助手。"
                user_message = f"""以下是相关的上下文：

{context}

用户问题：{full_query}

请回答："""
                
                composed = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        # 🔧 调试：记录提示词长度
        self.logger.info(f"   提示词长度: {len(composed)} 字符")
        self.logger.info(f"   简化模式: {is_simple_article_query}")
        
        # 生成答案
        try:
            outputs = self.llm.generate([composed], self.sampling_params)
            if outputs and outputs[0].outputs:
                raw_answer = outputs[0].outputs[0].text
                answer = raw_answer.strip()
                
                # 🔧 调试：记录原始输出（前200字符）
                self.logger.info(f"   原始输出: {raw_answer[:200]}")
                self.logger.info(f"   ✅ 生成完成（长度: {len(answer)}字符）")
                
                # 🔧 修复：如果答案为空，记录详细信息但不要立即返回错误
                if not answer:
                    self.logger.warning(f"   ⚠️ 生成的答案为空！")
                    self.logger.warning(f"   原始输出完整内容: {repr(raw_answer)}")
                    self.logger.warning(f"   提示词前500字符: {composed[:500]}")
                    # 返回更有用的错误信息
                    return "抱歉，模型未能生成有效回答。请检查日志了解详情。"
                
                # 后处理：移除答案开头可能出现的问题重复
                cleaned_answer = self._clean_answer(answer, query)
                
                # 🔧 修复：如果清理后答案变空了，返回原始答案
                if not cleaned_answer or len(cleaned_answer.strip()) < 5:
                    self.logger.warning(f"   ⚠️ 清理后答案过短，使用原始答案")
                    self.logger.warning(f"   原始答案: {answer[:100]}")
                    self.logger.warning(f"   清理后答案: {cleaned_answer}")
                    return answer
                
                return cleaned_answer
            else:
                self.logger.warning("LLM返回空输出")
                return "抱歉，无法根据提供的信息生成回答。"
        except Exception as e:
            self.logger.error(f"LLM生成失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"生成失败: {str(e)}"
    
    def retrieve_and_answer(
        self,
        query: str,
        instruction: str = "",
        top_k: int = 10,
        correlation_threshold: float = 0.35,
        alpha: float = 0.7
    ) -> Dict:
        """
        完整的检索和回答流程（独立BM25检索版本）
        
        流程：
        1. 查询重写（提高检索效果）
        2. 语义检索 → Top-K文档A
        3. 独立BM25检索 → Top-K文档B
        4. 比较两种检索结果的相似度
        5. 如果相似度高（>=阈值）：使用混合检索结果
        6. 如果相似度低（<阈值）：调用知识图谱
        
        Args:
            query: 用户查询
            instruction: 指令（来自数据集）
            top_k: Top-K
            correlation_threshold: 复杂度阈值（默认0.35，优化后）
            alpha: 混合权重（0-1），推荐0.7
            
        Returns:
            结果字典，包含answer, similarity, used_kg等信息
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🔍 查询: {query[:50]}...")
        if instruction:
            self.logger.info(f"📋 指令: {instruction[:50]}...")
        self.logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        # 🆕 检测是否为分类任务
        is_classification = self._is_classification_task(instruction)
        
        # 🆕 分类任务使用不同的参数
        if is_classification:
            self.logger.info(f"   🏷️ 检测到分类任务，使用优化参数")
            original_top_k = top_k
            original_alpha = alpha
            
            # 调整参数：适度增加，避免过多噪音
            top_k = 15  # 从10增加到15（而不是20）
            # 更依赖语义检索（分类任务更需要语义理解）
            alpha = 0.75  # 从0.7增加到0.75（而不是0.8）
            # 提高阈值（分类任务对复杂度要求可以适当提高）
            adjusted_threshold = 0.45  # 使用0.45（反转后）
            
            self.logger.info(f"   - Top-K: {original_top_k} → {top_k}")
            self.logger.info(f"   - Alpha: {original_alpha} → {alpha}")
            self.logger.info(f"   - 阈值: 0.7 → {adjusted_threshold}")
            
            # 🆕 添加Few-shot示例
            instruction = self._add_classification_examples(instruction, query)
            self.logger.info(f"   ✅ 已添加分类示例到指令")
            
            # 🆕 检测医疗纠纷（特殊处理）
            is_medical, medical_confidence = self._detect_medical_dispute(query)
            if is_medical and medical_confidence > 0.6:
                self.logger.info(f"   ⚕️ 检测到可能的医疗纠纷（置信度: {medical_confidence:.2f}）")
                instruction += "\n\n【特别提示】该问题可能涉及医疗纠纷，请仔细判断是否为医疗机构的医疗行为导致的损害。如果是医生手术失误、误诊等医疗行为导致的损害，应判断为医疗纠纷；如果只是在医院治疗工伤，应判断为人身损害或劳动纠纷。"
        else:
            adjusted_threshold = 0.70  # 非分类任务使用原阈值
        
        # 步骤0: 查询重写
        rewritten_query = self.rewrite_query_for_consistency(query, instruction)
        
        # 步骤1: 语义向量检索
        semantic_results = self.semantic_search(query, top_k=top_k, rewritten_query=rewritten_query)
        
        # 步骤2: 独立BM25检索
        bm25_results = self.bm25_search(query, top_k=top_k, rewritten_query=rewritten_query)
        
        # 步骤3: 多维度比较两种独立检索结果
        metrics = self.compare_independent_rankings(
            semantic_results, bm25_results, query
        )
        
        # 🆕 分类任务使用调整后的阈值
        if not is_classification:
            # 非分类任务：使用统一阈值0.35（优化后）
            adjusted_threshold = 0.35
        
        self.logger.info(f"   使用{'分类任务' if is_classification else '统一'}阈值: {adjusted_threshold}")
        
        # 步骤4: 自适应策略路由与决策（三重阈值）
        combined_score = metrics['combined_score']
        
        # 定义三重阈值
        theta_low = 0.25      # 简单事实问题阈值
        theta_medium = 0.45   # 语义模糊问题阈值
        theta_high = 0.65     # 轻度复杂问题阈值
        
        self.logger.info(f"\n步骤4: 自适应策略路由与决策")
        self.logger.info(f"   最终复杂度: {combined_score:.3f}")
        self.logger.info(f"   三重阈值: θ_low={theta_low}, θ_medium={theta_medium}, θ_high={theta_high}")
        
        # ==================== 策略1: 简单事实问题 ====================
        if combined_score <= theta_low:
            self.logger.info(f"✓ C_final ({combined_score:.3f}) ≤ θ_low ({theta_low})")
            self.logger.info(f"   → 策略1: 简单事实问题，启用密集向量检索")
            
            # 只使用语义检索结果，不需要混合检索
            final_context = "\n\n".join([
                f"【文档{i+1}】{r['source_name']}\n{r['text']}"
                for i, r in enumerate(semantic_results[:8])  # 使用Top-8语义检索结果
            ])
            use_kg = False
            strategy = "dense_vector"
            
        # ==================== 策略2: 语义模糊问题 ====================
        elif theta_low < combined_score <= theta_medium:
            self.logger.info(f"✓ θ_low ({theta_low}) < C_final ({combined_score:.3f}) ≤ θ_medium ({theta_medium})")
            self.logger.info(f"   → 策略2: 语义模糊问题，启用混合检索+二次排序")
            
            # 创建混合检索结果
            self.logger.info(f"   4.1: 创建混合检索结果 (alpha={alpha})...")
            hybrid_results = self.create_hybrid_results(semantic_results, bm25_results, alpha=alpha)
            
            # 二次排序：去除尾部噪声
            self.logger.info(f"   4.2: 二次排序，去除尾部噪声...")
            reranked_results, selected_count = self._rerank_and_select(
                hybrid_results, 
                query, 
                combined_score,
                is_simple=False,  # 语义模糊问题需要精准定位
                max_docs=10
            )
            self.logger.info(f"   ✅ 选择 {selected_count} 个最相关文档")
            
            final_context = "\n\n".join([
                f"【文档{i+1}】{r['source_name']}\n{r['text']}"
                for i, r in enumerate(reranked_results)
            ])
            use_kg = False
            strategy = "hybrid_rerank"
            
        # ==================== 策略3: 轻度复杂问题 ====================
        elif theta_medium < combined_score <= theta_high:
            self.logger.info(f"✓ θ_medium ({theta_medium}) < C_final ({combined_score:.3f}) ≤ θ_high ({theta_high})")
            self.logger.info(f"   → 策略3: 轻度复杂问题，使用一维图谱检索")
            
            # 创建混合检索结果
            hybrid_results = self.create_hybrid_results(semantic_results, bm25_results, alpha=alpha)
            reranked_results, selected_count = self._rerank_and_select(
                hybrid_results, query, combined_score, is_simple=False, max_docs=8
            )
            
            # 一维图谱检索（扁平KG，只使用实体描述）
            self.logger.info(f"   4.1: 一维图谱检索（扁平结构）...")
            kg_context = self._kg_search_flat(query, top_k=top_k)
            
            # 融合向量检索和图谱检索
            vector_context = "\n\n".join([
                f"【文档{i+1}】{r['source_name']}\n{r['text']}"
                for i, r in enumerate(reranked_results)
            ])
            
            final_context = f"""
## 向量检索结果

{vector_context}

## 知识图谱检索结果（一维）

{kg_context}
"""
            use_kg = True
            strategy = "flat_kg"
            
        # ==================== 策略4: 极复杂问题 ====================
        else:  # combined_score > theta_high
            self.logger.info(f"✓ C_final ({combined_score:.3f}) ≥ θ_high ({theta_high})")
            self.logger.info(f"   → 策略4: 极复杂问题，启用层次化图谱检索+LCA推理路径")
            
            # 创建混合检索结果
            hybrid_results = self.create_hybrid_results(semantic_results, bm25_results, alpha=alpha)
            reranked_results, selected_count = self._rerank_and_select(
                hybrid_results, query, combined_score, is_simple=False, max_docs=6
            )
            
            # 层次化图谱检索（完整结构：实体+社区聚合+推理路径）
            self.logger.info(f"   4.1: 层次化图谱检索（完整结构+LCA推理）...")
            kg_context = self.kg_search(query, top_k=top_k)
            
            # 融合向量检索和图谱检索
            vector_context = "\n\n".join([
                f"【文档{i+1}】{r['source_name']}\n{r['text']}"
                for i, r in enumerate(reranked_results)
            ])
            
            final_context = f"""
## 向量检索结果

{vector_context}

## 知识图谱检索结果（层次化）

{kg_context}
"""
            use_kg = True
            strategy = "hierarchical_kg"
        
        # 步骤5: 生成答案（传入检索结果用于法条精确匹配）
        answer = self.generate_answer(
            query, 
            final_context, 
            instruction, 
            use_kg,
            semantic_results=semantic_results,
            bm25_results=bm25_results
        )
        
        elapsed_time = time.time() - start_time
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"✅ 完成！耗时: {elapsed_time:.2f}秒")
        self.logger.info(f"   使用策略: {strategy}")
        self.logger.info(f"{'='*60}\n")
        
        return {
            'query': query,
            'rewritten_query': rewritten_query,
            'instruction': instruction,
            'answer': answer,
            # 核心指标
            'bm25_top1_score': metrics['bm25_top1_score'],
            'overlap_ratio': metrics['overlap_ratio'],
            'top3_overlap': metrics['top3_overlap'],
            'combined_score': metrics['combined_score'],
            # 新增：统一复杂度评分相关指标
            'question_type': metrics.get('question_type', 'unknown'),
            'question_nature_complexity': metrics.get('question_nature_complexity', 0.0),
            'retrieval_inconsistency': metrics.get('retrieval_inconsistency', 0.0),
            'final_complexity': metrics.get('final_complexity', 0.0),
            'evaluation_layer': metrics.get('evaluation_layer', 0),
            'needs_kg': metrics.get('needs_kg', False),
            'used_kg': use_kg,
            'strategy': strategy,  # 新增：使用的策略
            'elapsed_time': elapsed_time,
            'semantic_results': semantic_results,
            'bm25_results': bm25_results,
            'hybrid_results': hybrid_results if strategy in ['hybrid_rerank', 'flat_kg', 'hierarchical_kg'] else [],
            'alpha': alpha,
        }


def setup_logging(log_dir: str = "logs") -> Tuple[logging.Logger, str]:
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"hybrid_rag_{timestamp}.log")
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 清除现有处理器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 日志文件: {log_file}")
    logger.info("="*60)
    
    return logger, log_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="混合RAG检索系统")
    
    # 数据路径
    parser.add_argument(
        "--input",
        type=str,
        default="E:/MyPrograms/LeanRAG/datasets/query_social.json",
        help="输入查询数据集路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出结果路径（默认为输入文件名_hybrid_pred.json）"
    )
    
    # 向量数据库配置
    parser.add_argument(
        "--vector-db",
        type=str,
        default=VECTOR_DB_PATH,
        help="Milvus向量数据库路径"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help="Collection名称"
    )
    
    # 知识图谱配置
    parser.add_argument(
        "--kg-dir",
        type=str,
        default=KG_WORKING_DIR,
        help="知识图谱工作目录"
    )
    
    # 模型配置
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help="Embedding模型名称"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="/newdatad/WHH/MyEmoHH/models/Qwen2-7B-Instruct",
        help="LLM模型路径"
    )
    
    # 检索参数
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="检索Top-K"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CORRELATION_THRESHOLD,
        help="复杂度阈值（默认0.35）"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="混合权重alpha（0-1），推荐0.7表示70%%语义+30%%BM25"
    )
    
    # LLM参数
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size（使用的GPU数量，默认1张卡）")
    parser.add_argument("--gpu-mem-util", type=float, default=0.75, help="GPU显存占用比例（单卡使用0.75）")
    parser.add_argument("--max-model-len", type=int, default=4096, help="最大模型序列长度")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.3, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="top_p采样")
    
    # 其他参数
    parser.add_argument("--device", type=str, default=DEVICE, help="设备（cpu/cuda）")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志目录")
    
    # ==================== 消融实验参数 ====================
    parser.add_argument(
        "--use-retrieval-only",
        action="store_true",
        help="消融1：只用检索一致性评估（移除问题本质评估）"
    )
    parser.add_argument(
        "--use-intrinsic-only",
        action="store_true",
        help="消融2：只用问题本质评估（移除检索一致性）"
    )
    parser.add_argument(
        "--fixed-topk",
        action="store_true",
        help="消融3：固定Top-K文档数量（移除自适应选择）"
    )
    parser.add_argument(
        "--flat-kg",
        action="store_true",
        help="消融4：扁平KG结构（移除层次结构）"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger, log_file = setup_logging(args.log_dir)
    logger.info("混合RAG检索系统启动")
    logger.info(f"🔧 GPU配置: 使用 {len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))} 张GPU卡")
    logger.info(f"   - CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"   - Tensor Parallel Size: {args.tp}")
    logger.info(f"输入文件: {args.input}")
    
    # 确定输出路径
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(input_dir, f"{input_name}_hybrid_pred.json")
    
    logger.info(f"输出文件: {args.output}")
    
    # 初始化混合RAG系统
    llm_params = {
        "tp": args.tp,
        "gpu_mem_util": args.gpu_mem_util,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    
    rag_system = HybridLegalRAG(
        vector_db_path=args.vector_db,
        collection_name=args.collection,
        kg_working_dir=args.kg_dir,
        embedding_model_name=args.embedding_model,
        device=args.device,
        llm_model_path=args.llm_model,
        llm_params=llm_params,
        # 消融实验参数
        use_retrieval_only=args.use_retrieval_only,
        use_intrinsic_only=args.use_intrinsic_only,
        fixed_topk=args.fixed_topk,
        flat_kg=args.flat_kg
    )
    
    # 加载查询数据集
    logger.info(f"正在加载查询数据集: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    logger.info(f"共加载 {len(queries)} 条查询")
    
    # 批量处理
    results = []
    question_times = []  # 记录每个问题的处理时间
    
    for i, item in enumerate(queries):
        logger.info(f"\n处理第 {i+1}/{len(queries)} 条查询")
        
        query = item.get("question", "").strip()
        instruction = item.get("instruction", "").strip()
        
        if not query:
            logger.warning("查询为空，跳过")
            new_item = dict(item)
            new_item["prediction"] = ""
            # 核心指标
            new_item["bm25_top1_score"] = 10.0
            new_item["overlap_ratio"] = 0.0
            new_item["top3_overlap"] = 0.0
            new_item["combined_score"] = 0.0
            new_item["used_kg"] = False
            new_item["answer_time"] = 0.0
            results.append(new_item)
            question_times.append({"question_id": i+1, "question": query[:50], "time": 0.0})
            continue
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行检索和回答
        try:
            result = rag_system.retrieve_and_answer(
                query=query,
                instruction=instruction,
                top_k=args.top_k,
                correlation_threshold=args.threshold,
                alpha=args.alpha
            )
            
            # 计算处理时间
            answer_time = time.time() - start_time
            
            # 计算处理时间
            answer_time = time.time() - start_time
            
            # 保存结果
            new_item = dict(item)
            new_item["prediction"] = result["answer"]
            # 核心指标
            new_item["bm25_top1_score"] = result["bm25_top1_score"]
            new_item["overlap_ratio"] = result["overlap_ratio"]
            new_item["top3_overlap"] = result["top3_overlap"]
            new_item["combined_score"] = result["combined_score"]
            
            # 新增：5维度指标
            if result.get('num_covered_dimensions') is not None:
                new_item["num_covered_dimensions"] = result.get("num_covered_dimensions", 0)
                new_item["question_complexity"] = result.get("question_complexity", 0.0)
                new_item["num_sub_questions"] = result.get("decomposition", {}).get("num_sub_questions", 1)
                new_item["covered_dimensions"] = result.get("covered_dimensions", [])
            else:
                # 兼容旧模式
                new_item["query_structure_simplicity"] = result.get("query_structure_simplicity", 0.0)
                new_item["retrieval_consistency_confidence"] = result.get("retrieval_consistency_confidence", 0.0)
                new_item["dimension_difference"] = result.get("dimension_difference", 0.0)
                new_item["confidence_level"] = result.get("confidence_level", "unknown")
            
            new_item["used_kg"] = result["used_kg"]
            new_item["elapsed_time"] = result["elapsed_time"]
            new_item["answer_time"] = answer_time  # 添加回答时间
            results.append(new_item)
            
            # 记录问题时间
            question_times.append({
                "question_id": i+1,
                "question": query[:50] + ("..." if len(query) > 50 else ""),
                "time": answer_time,
                "used_kg": result["used_kg"]
            })
            
            logger.info(f"⏱️  本问题处理时间: {answer_time:.2f}秒")
            
        except Exception as e:
            # 计算处理时间（即使失败也记录）
            answer_time = time.time() - start_time
            
            logger.error(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            new_item = dict(item)
            new_item["prediction"] = f"处理失败: {str(e)}"
            # 核心指标
            new_item["bm25_top1_score"] = 10.0
            new_item["overlap_ratio"] = 0.0
            new_item["top3_overlap"] = 0.0
            new_item["combined_score"] = 0.0
            new_item["used_kg"] = False
            new_item["answer_time"] = answer_time
            results.append(new_item)
            
            # 记录问题时间
            question_times.append({
                "question_id": i+1,
                "question": query[:50] + ("..." if len(query) > 50 else ""),
                "time": answer_time,
                "used_kg": False
            })
    
    # 保存结果
    logger.info(f"\n正在保存结果到: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    total_queries = len(results)
    kg_used_count = sum(1 for r in results if r.get("used_kg", False))
    
    # 统计各策略使用次数
    strategy_counts = {
        'dense_vector': 0,
        'hybrid_rerank': 0,
        'flat_kg': 0,
        'hierarchical_kg': 0
    }
    for r in results:
        strategy = r.get('strategy', 'unknown')
        if strategy in strategy_counts:
            strategy_counts[strategy] += 1
    
    # 计算核心指标的平均值
    avg_bm25_top1 = np.mean([r.get("bm25_top1_score", 10.0) for r in results])
    avg_overlap_ratio = np.mean([r.get("overlap_ratio", 0.0) for r in results])
    avg_top3_overlap = np.mean([r.get("top3_overlap", 0.0) for r in results])
    avg_combined_score = np.mean([r.get("combined_score", 0.0) for r in results])
    
    # 计算时间统计
    total_time = sum(qt["time"] for qt in question_times)
    avg_time = total_time / len(question_times) if question_times else 0
    
    logger.info(f"\n{'='*60}")
    logger.info("处理完成！")
    logger.info(f"总查询数: {total_queries}")
    logger.info(f"使用知识图谱: {kg_used_count} ({kg_used_count/total_queries*100:.1f}%)")
    
    logger.info(f"\n📊 策略使用统计:")
    logger.info(f"{'='*60}")
    logger.info(f"策略1 - 密集向量检索: {strategy_counts['dense_vector']} ({strategy_counts['dense_vector']/total_queries*100:.1f}%)")
    logger.info(f"策略2 - 混合检索+二次排序: {strategy_counts['hybrid_rerank']} ({strategy_counts['hybrid_rerank']/total_queries*100:.1f}%)")
    logger.info(f"策略3 - 一维图谱检索: {strategy_counts['flat_kg']} ({strategy_counts['flat_kg']/total_queries*100:.1f}%)")
    logger.info(f"策略4 - 层次化图谱检索: {strategy_counts['hierarchical_kg']} ({strategy_counts['hierarchical_kg']/total_queries*100:.1f}%)")
    
    logger.info(f"\n📊 核心指标统计:")
    logger.info(f"{'='*60}")
    logger.info(f"1️⃣ BM25 Top1分数:")
    logger.info(f"   - 平均值: {avg_bm25_top1:.3f}")
    logger.info(f"2️⃣ 文档重叠率:")
    logger.info(f"   - 平均值: {avg_overlap_ratio:.3f}")
    logger.info(f"3️⃣ Top-3重叠率:")
    logger.info(f"   - 平均值: {avg_top3_overlap:.3f}")
    logger.info(f"\n🎯 综合复杂度评分:")
    logger.info(f"   - 平均综合复杂度: {avg_combined_score:.3f}")
    
    logger.info(f"\n⏱️  时间统计:")
    logger.info(f"{'='*60}")
    logger.info(f"总处理时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    logger.info(f"平均每题时间: {avg_time:.2f}秒")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"结果已保存到: {args.output}")
    logger.info(f"日志已保存到: {log_file}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
