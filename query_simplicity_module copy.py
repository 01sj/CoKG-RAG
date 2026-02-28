"""
查询复杂度评估模块（5维度版本）

核心理念：
- 利用LLM将复杂问题拆分为子问题
- 基于子问题评估5个维度的复杂度
- 5个维度与检索不一致性互补，共同决策是否使用KG

五个核心维度（评估问题本质）：
1. 推理链长度 (Reasoning Chain Length, RCL)
2. 知识整合需求 (Knowledge Integration Requirement, KIR)
3. 关系推理复杂度 (Relational Reasoning Complexity, RRC)
4. 领域跨度 (Domain Span, DS)
5. 条件约束密度 (Conditional Constraint Density, CCD)

检索维度（评估检索难度）：
- 检索不一致性 (Retrieval Inconsistency, RI)


衡量的简单度
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple

# ==================== LLM子问题拆分 ====================

def decompose_query_with_llm(query: str, llm=None, sampling_params=None) -> Dict:
    """
    使用LLM将复杂问题拆分为子问题
    
    Args:
        query: 原始查询
        llm: LLM模型实例（可选）
        sampling_params: LLM采样参数（可选）
        
    Returns:
        包含子问题列表和拆分信息的字典
    """
    # 如果没有提供LLM，返回原问题作为唯一子问题
    if llm is None:
        return {
            'sub_questions': [query],
            'num_sub_questions': 1,
            'decomposition_success': False,
            'reason': 'LLM未提供'
        }
    
    # 构建拆分提示词
    decomposition_prompt = f"""你是一个法律问题分析专家。请将以下法律咨询问题拆分为多个子问题。

拆分规则：
1. 如果问题很简单（如查询法条内容、概念定义），直接返回"无需拆分"
2. 如果问题复杂（如场景推理、多条件判断），拆分为2-5个子问题
3. 子问题应该是独立的、可以单独回答的
4. 子问题应该覆盖原问题的所有关键要素

原始问题：{query}

请按以下格式输出（严格遵守格式）：
【是否需要拆分】是/否
【子问题列表】
1. 子问题1
2. 子问题2
...

如果不需要拆分，只输出：
【是否需要拆分】否
【原因】简单问题，无需拆分"""
    
    try:
        # 使用LLM生成拆分
        outputs = llm.generate([decomposition_prompt], sampling_params)
        
        if not outputs or not outputs[0].outputs:
            return {
                'sub_questions': [query],
                'num_sub_questions': 1,
                'decomposition_success': False,
                'reason': 'LLM返回空'
            }
        
        response = outputs[0].outputs[0].text.strip()
        
        # 解析响应
        if '【是否需要拆分】否' in response or '无需拆分' in response:
            return {
                'sub_questions': [query],
                'num_sub_questions': 1,
                'decomposition_success': True,
                'reason': '简单问题，无需拆分'
            }
        
        # 提取子问题
        sub_questions = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # 匹配 "1. xxx" 或 "1、xxx" 格式
            match = re.match(r'^\d+[.、]\s*(.+)$', line)
            if match:
                sub_q = match.group(1).strip()
                if sub_q and len(sub_q) > 3:  # 过滤太短的
                    sub_questions.append(sub_q)
        
        # 验证拆分结果
        if len(sub_questions) == 0:
            # 拆分失败，返回原问题
            return {
                'sub_questions': [query],
                'num_sub_questions': 1,
                'decomposition_success': False,
                'reason': '拆分失败，未提取到子问题'
            }
        elif len(sub_questions) == 1:
            # 只有一个子问题，可能是简单问题
            return {
                'sub_questions': sub_questions,
                'num_sub_questions': 1,
                'decomposition_success': True,
                'reason': '简单问题或单一问题'
            }
        else:
            # 成功拆分为多个子问题
            return {
                'sub_questions': sub_questions,
                'num_sub_questions': len(sub_questions),
                'decomposition_success': True,
                'reason': f'成功拆分为{len(sub_questions)}个子问题'
            }
    
    except Exception as e:
        # 异常情况，返回原问题
        return {
            'sub_questions': [query],
            'num_sub_questions': 1,
            'decomposition_success': False,
            'reason': f'拆分异常: {str(e)}'
        }


# ==================== 5个核心维度评估 ====================

def measure_reasoning_chain_length(sub_questions: List[str], original_query: str) -> Dict:
    """
    维度1: 推理链长度 (Reasoning Chain Length, RCL)
    
    基于子问题数量评估推理步骤的长度
    
    评分规则：
    - 1个子问题: 0.0 (单跳)
    - 2个子问题: 0.3 (两跳)
    - 3个子问题: 0.6 (三跳)
    - 4个子问题: 0.8 (四跳)
    - 5+个子问题: 1.0 (多跳)
    
    Args:
        sub_questions: 子问题列表
        original_query: 原始查询
        
    Returns:
        包含评估结果的字典
    """
    num_sub = len(sub_questions)
    
    # 计算分数
    if num_sub == 1:
        score = 0.0
        level = "单跳"
    elif num_sub == 2:
        score = 0.3
        level = "两跳"
    elif num_sub == 3:
        score = 0.6
        level = "三跳"
    elif num_sub == 4:
        score = 0.8
        level = "四跳"
    else:
        score = 1.0
        level = "多跳"
    
    return {
        'dimension': '推理链长度',
        'score': score,
        'level': level,
        'num_sub_questions': num_sub,
        'description': f'{num_sub}个子问题，{level}推理'
    }


def measure_knowledge_integration_requirement(sub_questions: List[str], original_query: str) -> Dict:
    """
    维度2: 知识整合需求 (Knowledge Integration Requirement, KIR)
    
    评估需要整合的知识点数量
    
    改进：考虑子问题数量作为知识整合的指标
    - 多个子问题本身就意味着需要整合多个知识点
    
    评分规则：
    - 检测法律概念/法条数量
    - 考虑子问题数量（每个子问题代表一个需要整合的知识点）
    
    Args:
        sub_questions: 子问题列表
        original_query: 原始查询
        
    Returns:
        包含评估结果的字典
    """
    # 法律概念关键词
    legal_concepts = [
        '劳动合同', '劳动关系', '工伤', '社会保险', '养老保险', '医疗保险',
        '失业保险', '工伤保险', '生育保险', '住房公积金',
        '劳动报酬', '加班费', '经济补偿', '赔偿', '补偿',
        '用人单位', '劳动者', '职工', '员工',
        '解除', '终止', '辞退', '辞职',
        '工作时间', '休息休假', '年休假', '病假', '产假',
        '劳动争议', '仲裁', '诉讼',
        '安全生产', '职业病', '工伤认定',
        '退休', '养老金', '退休金',
        '未成年人', '妇女', '老年人', '残疾人',
    ]
    
    # 法条模式
    law_patterns = [
        r'第\d+条',
        r'第[一二三四五六七八九十百]+条',
        r'《.+?》',
    ]
    
    # 统计知识点
    knowledge_points = set()
    
    # 从原始查询中提取
    for concept in legal_concepts:
        if concept in original_query:
            knowledge_points.add(concept)
    
    for pattern in law_patterns:
        matches = re.findall(pattern, original_query)
        knowledge_points.update(matches)
    
    # 从子问题中提取
    for sub_q in sub_questions:
        for concept in legal_concepts:
            if concept in sub_q:
                knowledge_points.add(concept)
        
        for pattern in law_patterns:
            matches = re.findall(pattern, sub_q)
            knowledge_points.update(matches)
    
    num_knowledge_points = len(knowledge_points)
    
    # 🔧 改进：如果有多个子问题，说明需要整合多个知识点
    # 每个子问题至少代表一个需要整合的知识点
    num_sub = len(sub_questions)
    if num_sub > 1:
        # 子问题数量也算作知识整合的指标
        effective_knowledge_points = max(num_knowledge_points, num_sub)
    else:
        effective_knowledge_points = num_knowledge_points
    
    # 计算分数（基于有效知识点数）
    if effective_knowledge_points <= 1:
        score = 0.0
        level = "单一知识点"
    elif effective_knowledge_points == 2:
        score = 0.4
        level = "少量知识点"
    elif effective_knowledge_points == 3:
        score = 0.7
        level = "中等知识点"
    elif effective_knowledge_points == 4:
        score = 0.9
        level = "较多知识点"
    else:
        score = 1.0
        level = "大量知识点"
    
    return {
        'dimension': '知识整合需求',
        'score': score,
        'level': level,
        'num_knowledge_points': num_knowledge_points,
        'num_sub_questions': num_sub,
        'effective_knowledge_points': effective_knowledge_points,
        'knowledge_points': list(knowledge_points)[:5],  # 只显示前5个
        'description': f'{effective_knowledge_points}个知识点（{num_knowledge_points}个概念+{num_sub}个子问题），{level}'
    }


def measure_relational_reasoning_complexity(sub_questions: List[str], original_query: str) -> Dict:
    """
    维度3: 关系推理复杂度 (Relational Reasoning Complexity, RRC)
    
    评估涉及的关系类型数量（从属、因果、条件、对比等）
    
    改进：场景推理问题通常涉及多种隐含关系
    
    评分规则：
    - 检测问题中的关系类型
    - 场景推理问题自动增加关系复杂度
    - 关系类型越多，推理越复杂
    
    Args:
        sub_questions: 子问题列表
        original_query: 原始查询
        
    Returns:
        包含评估结果的字典
    """
    # 关系类型关键词
    relation_types = {
        '从属关系': ['属于', '包含', '归属', '包括在'],
        '因果关系': ['因为', '由于', '导致', '造成', '引起', '所以', '结果'],
        '条件关系': ['如果', '满足', '符合', '条件', '前提', '必须', '需要'],
        '对比关系': ['区别', '对比', '比较', '不同', '相同'],
        '时序关系': ['之前', '之后', '先', '后', '接着', '然后'],
        '程序关系': ['流程', '步骤', '程序', '过程', '怎么办'],
        '判断关系': ['算不算', '是否', '能否', '可以吗', '认定为', '算', '是不是'],
    }
    
    # 统计关系类型
    detected_relations = set()
    all_text = original_query + ' ' + ' '.join(sub_questions)
    
    for relation_type, keywords in relation_types.items():
        for keyword in keywords:
            if keyword in all_text:
                detected_relations.add(relation_type)
                break
    
    # 🔧 改进：检测场景推理问题
    scenario_keywords = ['我', '他', '她', '公司', '单位', '工厂', '场景', '情况']
    has_scenario = any(kw in original_query for kw in scenario_keywords)
    
    # 场景推理问题通常涉及多种隐含关系
    if has_scenario and len(sub_questions) > 1:
        # 场景推理至少涉及：条件关系、判断关系
        detected_relations.add('条件关系（隐含）')
        if '判断关系' not in detected_relations:
            detected_relations.add('判断关系（隐含）')
    
    num_relations = len(detected_relations)
    
    # 计算分数
    if num_relations == 0:
        score = 0.0
        level = "无关系推理"
    elif num_relations == 1:
        score = 0.3
        level = "单一关系"
    elif num_relations == 2:
        score = 0.6
        level = "双重关系"
    elif num_relations == 3:
        score = 0.8
        level = "多重关系"
    else:
        score = 1.0
        level = "复杂关系网络"
    
    return {
        'dimension': '关系推理复杂度',
        'score': score,
        'level': level,
        'num_relations': num_relations,
        'relation_types': list(detected_relations),
        'has_scenario': has_scenario,
        'description': f'{num_relations}种关系类型，{level}'
    }


def measure_domain_span(sub_questions: List[str], original_query: str) -> Dict:
    """
    维度4: 领域跨度 (Domain Span, DS)
    
    评估涉及的法律领域数量
    
    改进：工伤相关问题通常涉及多个领域
    
    评分规则：
    - 检测涉及的法律领域
    - 工伤问题通常涉及劳动法+工伤保险+社会保险
    - 领域越多，跨度越大
    
    Args:
        sub_questions: 子问题列表
        original_query: 原始查询
        
    Returns:
        包含评估结果的字典
    """
    # 法律领域关键词
    legal_domains = {
        '劳动法': ['劳动合同', '劳动关系', '劳动报酬', '加班', '辞退', '劳动争议', '用人单位', '劳动者'],
        '社会保险法': ['社会保险', '养老保险', '医疗保险', '失业保险', '生育保险', '社保'],
        '工伤保险': ['工伤', '工伤认定', '工伤赔偿', '职业病', '工伤待遇'],
        '未成年人保护法': ['未成年人', '儿童', '学生', '监护人', '未满18'],
        '妇女权益保障法': ['妇女', '女职工', '产假', '生育', '性别歧视'],
        '老年人权益保障法': ['老年人', '养老', '赡养', '退休'],
        '安全生产法': ['安全生产', '生产安全', '安全事故', '安全责任'],
        '合同法': ['合同', '协议', '违约', '合同纠纷'],
    }
    
    # 统计领域
    detected_domains = set()
    all_text = original_query + ' ' + ' '.join(sub_questions)
    
    for domain, keywords in legal_domains.items():
        for keyword in keywords:
            if keyword in all_text:
                detected_domains.add(domain)
                break
    
    # 🔧 改进：工伤问题通常涉及多个领域
    if '工伤保险' in detected_domains:
        # 工伤问题通常也涉及劳动法和社会保险法
        # 检查是否有劳动相关内容
        labor_keywords = ['劳动', '用人单位', '劳动者', '职工', '工厂', '公司', '单位', '上班', '工作']
        if any(kw in all_text for kw in labor_keywords):
            detected_domains.add('劳动法')
        
        # 工伤保险本身就是社会保险的一部分
        # 只要有工伤，就涉及社会保险法
        detected_domains.add('社会保险法')
    
    num_domains = len(detected_domains)
    
    # 计算分数
    if num_domains <= 1:
        score = 0.0
        level = "单一领域"
    elif num_domains == 2:
        score = 0.5
        level = "跨两个领域"
    elif num_domains == 3:
        score = 0.8
        level = "跨三个领域"
    else:
        score = 1.0
        level = "跨多个领域"
    
    return {
        'dimension': '领域跨度',
        'score': score,
        'level': level,
        'num_domains': num_domains,
        'domains': list(detected_domains),
        'description': f'{num_domains}个法律领域，{level}'
    }


def measure_conditional_constraint_density(sub_questions: List[str], original_query: str) -> Dict:
    """
    维度5: 条件约束密度 (Conditional Constraint Density, CCD)
    
    评估包含的约束条件数量（年龄、时间、金额、地点等）
    
    评分规则：
    - 检测具体的约束条件
    - 条件越多，约束越密集
    
    Args:
        sub_questions: 子问题列表
        original_query: 原始查询
        
    Returns:
        包含评估结果的字典
    """
    all_text = original_query + ' ' + ' '.join(sub_questions)
    
    # 约束条件模式
    constraint_patterns = {
        '年龄': r'\d+岁|\d+周岁|未满\d+|满\d+',
        '时间': r'\d+年|\d+个月|\d+日|\d+天|\d+小时',
        '金额': r'\d+元|\d+万|\d+千',
        '数量': r'\d+次|\d+个|\d+人',
        '比例': r'\d+%|百分之\d+',
    }
    
    # 统计约束条件
    constraints = []
    for constraint_type, pattern in constraint_patterns.items():
        matches = re.findall(pattern, all_text)
        if matches:
            constraints.extend([(constraint_type, m) for m in matches])
    
    num_constraints = len(constraints)
    
    # 计算分数
    if num_constraints == 0:
        score = 0.0
        level = "无约束条件"
    elif num_constraints == 1:
        score = 0.3
        level = "单一约束"
    elif num_constraints == 2:
        score = 0.6
        level = "双重约束"
    elif num_constraints == 3:
        score = 0.8
        level = "多重约束"
    else:
        score = 1.0
        level = "密集约束"
    
    return {
        'dimension': '条件约束密度',
        'score': score,
        'level': level,
        'num_constraints': num_constraints,
        'constraints': constraints[:5],  # 只显示前5个
        'description': f'{num_constraints}个约束条件，{level}'
    }


# ==================== 检索一致性评估（保留原有逻辑）====================

def calculate_retrieval_consistency_confidence(
    bm25_top1_score: float,
    overlap_ratio: float,
    top3_overlap: float
) -> Dict:
    """
    计算检索一致性置信度 (Retrieval Consistency Confidence, RCC)
    
    这是第6个维度，评估检索系统的能力，与前5个维度互补
    
    Args:
        bm25_top1_score: BM25 Top1分数
        overlap_ratio: 文档重叠率
        top3_overlap: Top3重叠率
        
    Returns:
        包含评估结果的字典
    """
    # 查询简单性指数 (QSI) - 基于BM25
    query_simplicity_index = 1.0 / (1.0 + np.exp(0.5 * (bm25_top1_score - 12.5)))
    
    # 检索差异性指数 (RDI) - 基于重叠率
    retrieval_divergence = 1.0 - (0.7 * overlap_ratio + 0.3 * top3_overlap)
    
    # 简单问题置信度 (使用竞争函数)
    alpha = 1.5
    beta = 0.3
    epsilon = 1e-10
    
    simple_evidence = (query_simplicity_index ** alpha) * (retrieval_divergence ** beta) + epsilon
    complex_evidence = ((1 - query_simplicity_index) ** alpha) * ((1 - retrieval_divergence) ** beta) + epsilon
    
    retrieval_consistency_confidence = simple_evidence / (simple_evidence + complex_evidence)
    
    # 后处理：BM25较高且overlap较低时的惩罚
    if bm25_top1_score > 10.0 and overlap_ratio < 0.3:
        bm25_penalty = min(0.95, (bm25_top1_score - 10.0) / 6.0)
        overlap_bonus = overlap_ratio / 0.3
        final_penalty = bm25_penalty * (1.0 - overlap_bonus)
        retrieval_consistency_confidence = retrieval_consistency_confidence * (1.0 - final_penalty)
    
    retrieval_consistency_confidence = np.clip(retrieval_consistency_confidence, 0, 1)
    
    return {
        'dimension': '检索一致性置信度',
        'score': retrieval_consistency_confidence,
        'query_simplicity_index': query_simplicity_index,
        'retrieval_divergence': retrieval_divergence,
        'description': f'检索一致性: {retrieval_consistency_confidence:.3f}'
    }


# ==================== 综合决策 ====================

def calculate_five_dimension_score(
    query: str,
    sub_questions: List[str],
    bm25_top1_score: float,
    overlap_ratio: float,
    top3_overlap: float
) -> Dict:
    """
    基于5维度+检索一致性的综合评分
    
    决策逻辑：
    1. 评估5个维度（问题本质）
    2. 评估检索一致性（检索能力）
    3. 综合评分 = 50% * 问题本质 + 50% * 检索效果
    4. 维度覆盖判断：≥3个维度 → KG，≤2个维度 → 传统RAG
    
    Args:
        query: 原始查询
        sub_questions: 子问题列表
        bm25_top1_score: BM25 Top1分数
        overlap_ratio: 文档重叠率
        top3_overlap: Top3重叠率
        
    Returns:
        包含所有评估结果的字典
    """
    # ========== 评估5个维度 ==========
    dim1 = measure_reasoning_chain_length(sub_questions, query)
    dim2 = measure_knowledge_integration_requirement(sub_questions, query)
    dim3 = measure_relational_reasoning_complexity(sub_questions, query)
    dim4 = measure_domain_span(sub_questions, query)
    dim5 = measure_conditional_constraint_density(sub_questions, query)
    
    # 统计覆盖的维度数量（分数 > 0.3 视为覆盖）
    threshold = 0.3
    covered_dimensions = []
    if dim1['score'] > threshold:
        covered_dimensions.append(dim1['dimension'])
    if dim2['score'] > threshold:
        covered_dimensions.append(dim2['dimension'])
    if dim3['score'] > threshold:
        covered_dimensions.append(dim3['dimension'])
    if dim4['score'] > threshold:
        covered_dimensions.append(dim4['dimension'])
    if dim5['score'] > threshold:
        covered_dimensions.append(dim5['dimension'])
    
    num_covered = len(covered_dimensions)
    
    # 问题本质复杂度（5个维度的加权平均）
    # 权重：推理链(30%) + 知识整合(25%) + 关系推理(20%) + 领域跨度(15%) + 条件约束(10%)
    question_complexity = (
        0.30 * dim1['score'] +
        0.25 * dim2['score'] +
        0.20 * dim3['score'] +
        0.15 * dim4['score'] +
        0.10 * dim5['score']
    )
    
    question_simplicity = 1.0 - question_complexity
    
    # ========== 评估检索一致性 ==========
    rcc = calculate_retrieval_consistency_confidence(
        bm25_top1_score, overlap_ratio, top3_overlap
    )
    
    # ========== 综合评分计算 ==========
    # 问题复杂度分值：0-1，越高越复杂
    complexity_score = question_complexity
    
    # 检索一致性分值：0-1，越高越一致（说明检索效果好）
    retrieval_score = rcc['score']
    
    # 综合评分：问题复杂度(50%) + 检索不一致性(50%)
    # 注意：检索一致性高 → 简单问题，所以用 (1 - retrieval_score) 表示检索的复杂性
    # 综合评分越高 → 越需要KG
    final_score = 0.5 * complexity_score + 0.5 * (1.0 - retrieval_score)
    
    # ========== 决策规则（基于综合评分阈值）==========
    # 阈值设置：0.4
    # - 综合评分 ≥ 0.4 → 使用KG（问题复杂或检索效果差）
    # - 综合评分 < 0.4 → 使用传统RAG（问题简单且检索效果好）
    decision_threshold = 0.4
    
    if final_score >= decision_threshold:
        needs_kg = True
        reason = (
            f"需要KG：综合评分{final_score:.3f}≥{decision_threshold} "
            f"(问题复杂度{complexity_score:.3f} + 检索不一致性{(1.0-retrieval_score):.3f})"
        )
    else:
        needs_kg = False
        reason = (
            f"使用传统RAG：综合评分{final_score:.3f}<{decision_threshold} "
            f"(问题复杂度{complexity_score:.3f} + 检索不一致性{(1.0-retrieval_score):.3f})"
        )
    
    return {
        # 5个维度详情
        'dimension_1_reasoning_chain': dim1,
        'dimension_2_knowledge_integration': dim2,
        'dimension_3_relational_reasoning': dim3,
        'dimension_4_domain_span': dim4,
        'dimension_5_conditional_constraint': dim5,
        
        # 维度覆盖统计
        'num_covered_dimensions': num_covered,
        'covered_dimensions': covered_dimensions,
        'coverage_threshold': threshold,
        
        # 问题本质评分
        'question_complexity': round(question_complexity, 3),
        'question_simplicity': round(question_simplicity, 3),
        
        # 检索一致性
        'retrieval_consistency': rcc,
        
        # 综合评分组成
        'complexity_score': round(complexity_score, 3),
        'retrieval_score': round(retrieval_score, 3),
        'retrieval_inconsistency': round(1.0 - retrieval_score, 3),
        
        # 综合评分
        'final_score': round(final_score, 3),
        'decision_threshold': decision_threshold,
        
        # 决策结果
        'needs_kg': needs_kg,
        'reason': reason,
    }


# ==================== 新版：三层评估架构 ====================

def classify_simple_question_type(query: str) -> Dict:
    """
    第一层：快速分类简单问题类型
    
    简单问题类型及基础复杂度：
    - 法条查询 (0.1): 查询具体法条内容
    - 概念定义 (0.15): 查询法律概念定义
    - 简单列举 (0.2): 查询简单的列表信息
    - 元信息 (0.2): 查询立法目的、适用范围等
    
    Args:
        query: 查询文本
        
    Returns:
        包含问题类型和基础复杂度的字典
    """
    # 法条查询模式
    law_article_patterns = [
        r'第\d+条',
        r'第[一二三四五六七八九十百]+条',
        r'\d+条(?!件)',
        r'第\d+款',
    ]
    has_article = any(re.search(p, query) for p in law_article_patterns)
    content_words = ['内容', '是什么', '说的是', '讲了什么', '规定', '告诉我']
    has_content_query = any(word in query for word in content_words)
    
    if has_article and has_content_query and '场景' not in query:
        return {
            'type': '法条查询',
            'base_complexity': 0.1,
            'is_simple': True,
            'description': '查询具体法条内容'
        }
    
    # 概念定义模式
    concept_patterns = ['什么是', '的定义', '定义是', '概念是', '含义是']
    if any(p in query for p in concept_patterns):
        return {
            'type': '概念定义',
            'base_complexity': 0.15,
            'is_simple': True,
            'description': '查询法律概念定义'
        }
    
    # 简单列举模式（排除需要推理的列举）
    enumeration_patterns = ['包括哪些', '分为哪几种', '有哪些', '有几种']
    reasoning_words = ['算', '属于', '认定为', '判断']
    is_simple_enum = any(p in query for p in enumeration_patterns)
    needs_reasoning = any(w in query for w in reasoning_words)
    
    if is_simple_enum and not needs_reasoning:
        return {
            'type': '简单列举',
            'base_complexity': 0.2,
            'is_simple': True,
            'description': '查询简单的列表信息'
        }
    
    # 元信息查询模式
    meta_patterns = ['立法目的', '立法宗旨', '适用范围', '适用于哪些']
    if any(p in query for p in meta_patterns):
        return {
            'type': '元信息',
            'base_complexity': 0.2,
            'is_simple': True,
            'description': '查询立法目的、适用范围等'
        }
    
    # 不是简单问题
    return {
        'type': '复杂问题',
        'base_complexity': None,
        'is_simple': False,
        'description': '需要进一步分类'
    }


def classify_complex_question_type(query: str) -> Dict:
    """
    第二层：复杂问题细分类型
    
    复杂问题类型及基础复杂度：
    - 场景推理 (0.5): 具体场景的法律判断
    - 条件判断 (0.45): 判断是否满足某个法律条件
    - 多条件组合 (0.5): 涉及多个条件的综合判断
    - 权益维护 (0.55): 权益受损后的救济途径
    - 跨领域 (0.6): 涉及多个法律领域
    - 复杂列举 (0.4): 需要推理判断的列举
    
    Args:
        query: 查询文本
        
    Returns:
        包含问题类型和基础复杂度的字典
    """
    # 场景推理：包含具体场景描述
    scenario_keywords = [
        '我是', '我妈', '我爸', '本人', '下班', '路上', '途中',
        '公司', '老板', '工厂', '车间', '单位', '学生', '老师', '煤矿', '公园', '职工', '员工'
    ]
    abstract_patterns = ['适用于', '适用范围', '立法目的', '包括哪些']
    is_abstract = any(p in query for p in abstract_patterns)
    has_scenario = any(kw in query for kw in scenario_keywords)
    
    if has_scenario and not is_abstract:
        return {
            'type': '场景推理',
            'base_complexity': 0.5,
            'description': '具体场景的法律判断',
            'dimension_weights': {
                'rcl': 0.35,  # 推理链长度权重增加
                'kir': 0.25,
                'rrc': 0.25,  # 关系推理权重增加
                'ds': 0.10,
                'ccd': 0.05
            }
        }
    
    # 条件判断：判断是否满足某个法律条件
    judgment_keywords = [
        '算不算', '是否', '能否', '可以吗', '合理', '合法',
        '违法', '违背', '能起诉', '能报', '认定为'
    ]
    polite_phrases = ['能告诉', '可以告诉', '请告诉']
    is_polite = any(phrase in query for phrase in polite_phrases)
    has_judgment = any(kw in query for kw in judgment_keywords)
    
    if has_judgment and not is_polite:
        return {
            'type': '条件判断',
            'base_complexity': 0.45,
            'description': '判断是否满足某个法律条件',
            'dimension_weights': {
                'rcl': 0.25,
                'kir': 0.25,
                'rrc': 0.30,  # 关系推理权重增加
                'ds': 0.15,
                'ccd': 0.05
            }
        }
    
    # 多条件组合：涉及多个具体条件
    numeric_conditions = len(re.findall(r'\d+年|\d+岁|\d+个月|\d+日|\d+元', query))
    if numeric_conditions >= 2:
        return {
            'type': '多条件组合',
            'base_complexity': 0.5,
            'description': '涉及多个条件的综合判断',
            'dimension_weights': {
                'rcl': 0.30,
                'kir': 0.25,
                'rrc': 0.20,
                'ds': 0.10,
                'ccd': 0.15  # 条件约束权重增加
            }
        }
    
    # 权益维护：权益受损后的救济途径
    rights_keywords = ['拖欠', '辞退', '补偿', '赔偿', '欠薪', '不发', '维权']
    remedy_keywords = ['怎么办', '办法', '途径', '起诉', '仲裁', '投诉']
    has_rights_issue = any(kw in query for kw in rights_keywords)
    has_remedy_query = any(kw in query for kw in remedy_keywords)
    
    if has_rights_issue or has_remedy_query:
        return {
            'type': '权益维护',
            'base_complexity': 0.55,
            'description': '权益受损后的救济途径',
            'dimension_weights': {
                'rcl': 0.35,  # 推理链长度权重增加
                'kir': 0.30,  # 知识整合权重增加
                'rrc': 0.20,
                'ds': 0.10,
                'ccd': 0.05
            }
        }
    
    # 跨领域：涉及多个法律领域
    legal_domains = {
        '劳动法': ['劳动合同', '劳动关系', '劳动报酬', '加班', '辞退'],
        '社会保险法': ['社会保险', '养老保险', '医疗保险', '失业保险', '生育保险'],
        '工伤保险': ['工伤', '工伤认定', '工伤赔偿', '职业病'],
        '未成年人保护法': ['未成年人', '儿童', '学生', '监护人'],
        '妇女权益保障法': ['妇女', '女职工', '产假', '生育'],
    }
    
    detected_domains = 0
    for domain, keywords in legal_domains.items():
        if any(kw in query for kw in keywords):
            detected_domains += 1
    
    if detected_domains >= 2:
        return {
            'type': '跨领域',
            'base_complexity': 0.6,
            'description': '涉及多个法律领域',
            'dimension_weights': {
                'rcl': 0.25,
                'kir': 0.30,  # 知识整合权重增加
                'rrc': 0.20,
                'ds': 0.20,  # 领域跨度权重增加
                'ccd': 0.05
            }
        }
    
    # 复杂列举：需要推理判断的列举
    enumeration_patterns = ['包括哪些', '分为哪几种', '有哪些', '有几种']
    reasoning_words = ['算', '属于', '认定为', '判断']
    is_enum = any(p in query for p in enumeration_patterns)
    needs_reasoning = any(w in query for w in reasoning_words)
    
    if is_enum and needs_reasoning:
        return {
            'type': '复杂列举',
            'base_complexity': 0.4,
            'description': '需要推理判断的列举',
            'dimension_weights': {
                'rcl': 0.25,
                'kir': 0.30,  # 知识整合权重增加
                'rrc': 0.25,
                'ds': 0.15,
                'ccd': 0.05
            }
        }
    
    # 默认：一般复杂问题
    return {
        'type': '一般复杂问题',
        'base_complexity': 0.45,
        'description': '一般复杂问题',
        'dimension_weights': {
            'rcl': 0.30,
            'kir': 0.25,
            'rrc': 0.20,
            'ds': 0.15,
            'ccd': 0.10
        }
    }


def calculate_question_nature_complexity(
    query: str,
    sub_questions: List[str],
    llm=None,
    sampling_params=None
) -> Dict:
    """
    计算问题本质复杂度（新版）
    
    三层评估架构：
    1. 第一层：快速分类简单问题
    2. 第二层：复杂问题细分类型
    3. 第三层：五维度深度评估
    
    最终复杂度 = 基础分 × 0.3 + 五维度加权分 × 0.7
    问题本质复杂度 = 最终复杂度（分数越高越复杂）
    
    Args:
        query: 原始查询
        sub_questions: 子问题列表
        llm: LLM模型实例（可选）
        sampling_params: LLM采样参数（可选）
        
    Returns:
        包含评估结果的字典
    """
    # 第一层：快速分类简单问题
    simple_classification = classify_simple_question_type(query)
    
    if simple_classification['is_simple']:
        # 简单问题：直接使用基础复杂度
        base_complexity = simple_classification['base_complexity']
        question_nature_complexity = base_complexity
        
        return {
            'question_type': simple_classification['type'],
            'is_simple': True,
            'base_complexity': base_complexity,
            'question_nature_complexity': question_nature_complexity,
            'description': simple_classification['description'],
            'evaluation_layer': 1,
            'dimension_scores': None
        }
    
    # 第二层：复杂问题细分类型
    complex_classification = classify_complex_question_type(query)
    base_complexity = complex_classification['base_complexity']
    dimension_weights = complex_classification.get('dimension_weights', {
        'rcl': 0.30,
        'kir': 0.25,
        'rrc': 0.20,
        'ds': 0.15,
        'ccd': 0.10
    })
    
    # 第三层：五维度深度评估
    dim1 = measure_reasoning_chain_length(sub_questions, query)
    dim2 = measure_knowledge_integration_requirement(sub_questions, query)
    dim3 = measure_relational_reasoning_complexity(sub_questions, query)
    dim4 = measure_domain_span(sub_questions, query)
    dim5 = measure_conditional_constraint_density(sub_questions, query)
    
    # 使用动态权重计算五维度加权分
    five_dimension_score = (
        dimension_weights['rcl'] * dim1['score'] +
        dimension_weights['kir'] * dim2['score'] +
        dimension_weights['rrc'] * dim3['score'] +
        dimension_weights['ds'] * dim4['score'] +
        dimension_weights['ccd'] * dim5['score']
    )
    
    # 最终复杂度 = 基础分 × 0.3 + 五维度加权分 × 0.7
    question_nature_complexity = 0.3 * base_complexity + 0.7 * five_dimension_score
    
    return {
        'question_type': complex_classification['type'],
        'is_simple': False,
        'base_complexity': base_complexity,
        'five_dimension_score': five_dimension_score,
        'question_nature_complexity': question_nature_complexity,
        'description': complex_classification['description'],
        'evaluation_layer': 3,
        'dimension_weights': dimension_weights,
        'dimension_scores': {
            'reasoning_chain_length': dim1,
            'knowledge_integration': dim2,
            'relational_reasoning': dim3,
            'domain_span': dim4,
            'conditional_constraint': dim5
        }
    }


def calculate_retrieval_inconsistency(
    bm25_top1_score: float,
    overlap_ratio: float,
    top3_overlap: float
) -> Dict:
    """
    计算检索不一致性（反转逻辑）
    
    检索不一致性单度：评估检索系统的能力
    - 分数越高，说明检索结果越一致，问题越简单
    
    Args:
        bm25_top1_score: BM25 Top1分数
        overlap_ratio: 文档重叠率
        top3_overlap: Top3重叠率
        
    Returns:
        包含评估结果的字典
    """
    # 查询简单性指数 (QSI) - 基于BM25
    query_simplicity_index = 1.0 / (1.0 + np.exp(0.5 * (bm25_top1_score - 12.5)))
    
    # 检索差异性指数 (RDI) - 基于重叠率
    retrieval_divergence = 1.0 - (0.7 * overlap_ratio + 0.3 * top3_overlap)
    
    # 简单问题置信度 (使用竞争函数)
    alpha = 1.5
    beta = 0.3
    epsilon = 1e-10
    
    simple_evidence = (query_simplicity_index ** alpha) * (retrieval_divergence ** beta) + epsilon
    complex_evidence = ((1 - query_simplicity_index) ** alpha) * ((1 - retrieval_divergence) ** beta) + epsilon
    
    retrieval_consistency_simplicity = simple_evidence / (simple_evidence + complex_evidence)
    
    # 后处理：BM25较高且overlap较低时的惩罚
    if bm25_top1_score > 10.0 and overlap_ratio < 0.3:
        bm25_penalty = min(0.95, (bm25_top1_score - 10.0) / 6.0)
        overlap_bonus = overlap_ratio / 0.3
        final_penalty = bm25_penalty * (1.0 - overlap_bonus)
        retrieval_consistency_simplicity = retrieval_consistency_simplicity * (1.0 - final_penalty)
    
    retrieval_consistency_simplicity = np.clip(retrieval_consistency_simplicity, 0, 1)
    
    return {
        'retrieval_consistency_simplicity': retrieval_consistency_simplicity,
        'query_simplicity_index': query_simplicity_index,
        'retrieval_divergence': retrieval_divergence,
        'description': f'检索一致性简单度: {retrieval_consistency_simplicity:.3f}'
    }


def calculate_final_simplicity_score(
    query: str,
    sub_questions: List[str],
    bm25_top1_score: float,
    overlap_ratio: float,
    top3_overlap: float,
    w1: float = 0.5,
    w2: float = 0.5,
    threshold: float = 0.6,
    llm=None,
    sampling_params=None
) -> Dict:
    """
    计算最终简单度评分（主函数）
    
    统一简单度评分体系：
    1. 问题本质简单度 = 1.0 - 问题本质复杂度
    2. 检索一致性简单度 = 现有RCC计算
    3. 最终简单度 = w1 × 问题本质简单度 + w2 × 检索一致性简单度
    4. 决策规则：最终简单度 ≥ threshold → 传统RAG；< threshold → KG
    
    Args:
        query: 原始查询
        sub_questions: 子问题列表
        bm25_top1_score: BM25 Top1分数
        overlap_ratio: 文档重叠率
        top3_overlap: Top3重叠率
        w1: 问题本质权重（默认0.5）
        w2: 检索一致性权重（默认0.5）
        threshold: 决策阈值（默认0.6）
        llm: LLM模型实例（可选）
        sampling_params: LLM采样参数（可选）
        
    Returns:
        包含所有评估结果的字典
    """
    # 计算问题本质简单度
    question_nature_result = calculate_question_nature_simplicity(
        query, sub_questions, llm, sampling_params
    )
    question_nature_simplicity = question_nature_result['question_nature_simplicity']
    
    # 计算检索一致性简单度
    retrieval_result = calculate_retrieval_consistency_simplicity(
        bm25_top1_score, overlap_ratio, top3_overlap
    )
    retrieval_consistency_simplicity = retrieval_result['retrieval_consistency_simplicity']
    
    # 计算最终简单度
    final_simplicity = w1 * question_nature_simplicity + w2 * retrieval_consistency_simplicity
    final_simplicity = np.clip(final_simplicity, 0, 1)
    
    # 决策规则
    use_traditional_rag = final_simplicity >= threshold
    
    if use_traditional_rag:
        decision = "使用传统RAG"
        reason = f"最终简单度{final_simplicity:.3f}≥{threshold}（问题本质{question_nature_simplicity:.3f} + 检索一致性{retrieval_consistency_simplicity:.3f}）"
    else:
        decision = "使用知识图谱"
        reason = f"最终简单度{final_simplicity:.3f}<{threshold}（问题本质{question_nature_simplicity:.3f} + 检索一致性{retrieval_consistency_simplicity:.3f}）"
    
    return {
        # 问题类型
        'question_type': question_nature_result['question_type'],
        'is_simple_question': question_nature_result['is_simple'],
        'evaluation_layer': question_nature_result['evaluation_layer'],
        
        # 问题本质评估
        'question_nature_simplicity': round(question_nature_simplicity, 3),
        'question_nature_complexity': round(question_nature_result['question_nature_complexity'], 3),
        'base_complexity': question_nature_result.get('base_complexity'),
        'five_dimension_score': question_nature_result.get('five_dimension_score'),
        'dimension_weights': question_nature_result.get('dimension_weights'),
        'dimension_scores': question_nature_result.get('dimension_scores'),
        
        # 检索一致性评估
        'retrieval_consistency_simplicity': round(retrieval_consistency_simplicity, 3),
        'query_simplicity_index': round(retrieval_result['query_simplicity_index'], 3),
        'retrieval_divergence': round(retrieval_result['retrieval_divergence'], 3),
        
        # 最终评分
        'final_simplicity': round(final_simplicity, 3),
        'weights': {'w1': w1, 'w2': w2},
        'threshold': threshold,
        
        # 决策结果
        'use_traditional_rag': use_traditional_rag,
        'needs_kg': not use_traditional_rag,
        'decision': decision,
        'reason': reason,
    }


# ==================== 兼容旧接口 ====================

def measure_multi_hop_requirement(query: str) -> dict:
    """
    评估查询的多跳推理需求
    
    单跳查询特征：
    - 直接定位到具体法条/概念
    - 答案在单一文档中
    - 不需要跨文档推理
    
    多跳查询特征：
    - 需要从场景提取要素
    - 需要匹配多个法律条件
    - 需要综合多个法条推理
    - 需要因果关系推理
    
    Args:
        query: 查询文本
        
    Returns:
        包含多跳评估结果的字典
    """
    hop_score = 0.0  # 0=单跳, 1=多跳
    indicators = []
    
    # ========== 单跳指标（减分）==========
    
    # 指标1: 精确法条定位 (-0.5)
    law_article_patterns = [
        r'第\d+条',  # 第85条
        r'第[一二三四五六七八九十百]+条',  # 第八十五条
        r'\d+条(?!件)',  # 85条（排除"条件"）
        r'第\d+款',  # 第2款
    ]
    has_article = any(re.search(p, query) for p in law_article_patterns)
    content_words = ['内容', '是什么', '说的是', '讲了什么', '规定', '告诉我']
    has_content_query = any(word in query for word in content_words)
    
    if has_article and has_content_query and '场景' not in query:
        hop_score -= 0.5
        indicators.append("精确法条定位（单跳）")
    
    # 指标2: 简单概念查询 (-0.4)
    concept_patterns = ['什么是', '的定义', '定义是', '概念是', '含义是']
    if any(p in query for p in concept_patterns):
        hop_score -= 0.4
        indicators.append("概念定义查询（单跳）")
    
    # 指标3: 简单列举查询 (-0.3)
    enumeration_patterns = ['包括哪些', '分为哪几种', '有哪些', '有几种']
    # 排除需要推理的列举，如"哪些情况算工伤"
    reasoning_words = ['算', '属于', '认定为', '判断']
    is_simple_enum = any(p in query for p in enumeration_patterns)
    needs_reasoning = any(w in query for w in reasoning_words)
    
    if is_simple_enum and not needs_reasoning:
        hop_score -= 0.3
        indicators.append("简单列举查询（单跳）")
    
    # 指标4: 立法目的/适用范围查询 (-0.3)
    meta_patterns = ['立法目的', '立法宗旨', '适用范围', '适用于哪些']
    if any(p in query for p in meta_patterns):
        hop_score -= 0.3
        indicators.append("元信息查询（单跳）")
    
    # ========== 多跳指标（加分）==========
    
    # 指标1: 具体场景描述 (+0.5) - 需要场景→要素→法条→判断
    scenario_keywords = [
        '场景:', '公司', '老板', '工厂', '车间', '单位',
        '我是', '我妈', '我爸', '本人', '下班', '路上', '途中',
        '学生', '老师', '煤矿', '公园', '职工', '员工'
    ]
    # 排除抽象查询中的"单位"、"职工"等
    abstract_patterns = ['适用于', '适用范围', '立法目的', '包括哪些']
    is_abstract = any(p in query for p in abstract_patterns)
    
    has_scenario = any(kw in query for kw in scenario_keywords)
    if has_scenario and not is_abstract:
        hop_score += 0.5
        indicators.append("具体场景（多跳：场景→要素→法条→判断）")
    
    # 指标2: 条件判断需求 (+0.4) - 需要条件→法条→匹配→结论
    judgment_keywords = [
        '算不算', '是否', '能否', '可以吗', '合理', '合法',
        '违法', '违背', '能起诉', '能报', '认定为'
    ]
    # 排除礼貌用语
    polite_phrases = ['能告诉', '可以告诉', '请告诉']
    is_polite = any(phrase in query for phrase in polite_phrases)
    
    has_judgment = any(kw in query for kw in judgment_keywords)
    if has_judgment and not is_polite:
        hop_score += 0.4
        indicators.append("条件判断（多跳：条件→法条→匹配→结论）")
    
    # 指标3: 权益维护/救济途径 (+0.4) - 需要违法行为→责任→救济途径
    rights_keywords = ['拖欠', '辞退', '补偿', '赔偿', '欠薪', '不发', '维权']
    remedy_keywords = ['怎么办', '办法', '途径', '起诉', '仲裁', '投诉']
    
    has_rights_issue = any(kw in query for kw in rights_keywords)
    has_remedy_query = any(kw in query for kw in remedy_keywords)
    
    if has_rights_issue or has_remedy_query:
        hop_score += 0.4
        indicators.append("权益维护（多跳：违法行为→责任→救济）")
    
    # 指标4: 多个条件组合 (+0.3) - 需要条件1→条件2→综合判断
    numeric_conditions = len(re.findall(r'\d+年|\d+岁|\d+个月|\d+日|\d+元', query))
    if numeric_conditions >= 2:
        hop_score += 0.3
        indicators.append(f"多条件组合（多跳：{numeric_conditions}个条件需综合判断）")
    
    # 指标5: 因果关系推理 (+0.3) - 需要原因→法条→后果
    causal_keywords = ['因为', '由于', '导致', '造成', '引起', '所以', '结果']
    if any(kw in query for kw in causal_keywords):
        hop_score += 0.3
        indicators.append("因果推理（多跳：原因→法条→后果）")
    
    # 指标6: 复杂列举（需要推理的列举）(+0.3)
    if is_simple_enum and needs_reasoning:
        hop_score += 0.3
        indicators.append("复杂列举（多跳：需要推理判断的列举）")
    
    # 归一化到[0, 1]
    hop_score = max(0.0, min(1.0, hop_score))
    
    return {
        'hop_score': hop_score,
        'is_multi_hop': hop_score > 0.0,
        'indicators': indicators
    }


def measure_hierarchical_dependency(query: str) -> dict:
    """
    评估查询对层级结构的依赖程度
    
    平铺信息即可回答：
    - 单一法条内容
    - 独立的概念定义
    - 简单的列表信息
    
    需要层级结构辅助：
    - 需要理解法条之间的从属关系（总则→分则）
    - 需要理解概念之间的包含关系（工伤→工伤保险→社会保险）
    - 需要理解条件之间的逻辑关系（必要条件→充分条件）
    - 需要理解程序之间的先后关系（协商→仲裁→诉讼）
    
    Args:
        query: 查询文本
        
    Returns:
        包含层级依赖评估结果的字典
    """
    hierarchy_score = 0.0  # 0=平铺, 1=层级
    indicators = []
    
    # ========== 平铺信息指标（减分）==========
    
    # 指标1: 单一法条查询 (-0.5)
    law_article_patterns = [
        r'第\d+条',
        r'第[一二三四五六七八九十百]+条',
        r'\d+条(?!件)',
        r'第\d+款',
    ]
    has_article = any(re.search(p, query) for p in law_article_patterns)
    content_words = ['内容', '是什么', '说的是', '讲了什么', '规定']
    has_content_query = any(word in query for word in content_words)
    
    if has_article and has_content_query:
        hierarchy_score -= 0.5
        indicators.append("单一法条（平铺信息）")
    
    # 指标2: 独立概念定义 (-0.4)
    concept_patterns = ['什么是', '的定义', '定义是', '概念是', '含义是']
    if any(p in query for p in concept_patterns):
        hierarchy_score -= 0.4
        indicators.append("独立概念（平铺信息）")
    
    # 指标3: 简单列举（无层级关系）(-0.3)
    simple_enum_patterns = ['包括哪些', '有哪些', '有几种']
    if any(p in query for p in simple_enum_patterns):
        hierarchy_score -= 0.3
        indicators.append("简单列举（平铺信息）")
    
    # ========== 层级结构指标（加分）==========
    
    # 指标1: 从属关系查询 (+0.5)
    # 例如："工伤保险属于社会保险吗？"
    subordinate_keywords = ['属于', '包含在', '归属']
    if any(kw in query for kw in subordinate_keywords):
        hierarchy_score += 0.5
        indicators.append("从属关系（需要层级：子概念→父概念）")
    
    # 指标2: 条件判断查询 (+0.5)
    # 例如："这算工伤吗？"、"能退休吗？"
    # 这类问题需要理解条件层级：必要条件→充分条件→综合判断
    judgment_keywords = ['算不算', '是不是', '算', '能否', '可以吗', '能不能']
    # 排除礼貌用语
    polite_phrases = ['能告诉', '可以告诉', '请告诉', '能不能告诉']
    is_polite = any(phrase in query for phrase in polite_phrases)
    
    has_judgment = any(kw in query for kw in judgment_keywords)
    if has_judgment and not is_polite:
        hierarchy_score += 0.5
        indicators.append("条件判断（需要层级：条件A∧条件B→结论）")
    
    # 指标3: 条件层级查询 (+0.4)
    # 例如："满足哪些条件才能认定为工伤？"
    condition_keywords = ['满足', '符合', '条件', '要求', '前提', '必须']
    recognition_keywords = ['认定', '判断', '确定', '算作']
    
    has_condition = any(kw in query for kw in condition_keywords)
    has_recognition = any(kw in query for kw in recognition_keywords)
    
    if has_condition or has_recognition:
        hierarchy_score += 0.4
        indicators.append("条件层级（需要层级：必要条件→充分条件）")
    
    # 指标3: 程序/流程查询 (+0.4)
    # 例如："劳动争议的处理流程是什么？"
    procedure_keywords = ['流程', '程序', '步骤', '过程', '怎么办', '如何处理']
    if any(kw in query for kw in procedure_keywords):
        hierarchy_score += 0.4
        indicators.append("程序层级（需要层级：步骤1→步骤2→步骤3）")
    
    # 指标4: 关系推理查询 (+0.4)
    # 例如："用人单位和劳动者的关系是什么？"
    relation_keywords = ['关系', '区别', '联系', '对比', '比较']
    if any(kw in query for kw in relation_keywords):
        hierarchy_score += 0.4
        indicators.append("关系推理（需要层级：实体A↔关系↔实体B）")
    
    # 指标5: 范围/边界查询 (+0.3)
    # 例如："哪些情况算工伤？"（需要理解工伤的边界）
    scope_keywords = ['哪些情况', '什么情况', '范围', '边界', '界定']
    if any(kw in query for kw in scope_keywords):
        hierarchy_score += 0.3
        indicators.append("范围界定（需要层级：核心情况→边界情况）")
    
    # 指标6: 责任/后果查询 (+0.4)
    # 例如："公司违法需要承担什么责任？"（需要理解违法行为→法律责任的层级）
    responsibility_keywords = ['责任', '后果', '处罚', '赔偿', '补偿']
    if any(kw in query for kw in responsibility_keywords):
        hierarchy_score += 0.4
        indicators.append("责任后果（需要层级：违法行为→法律责任→具体后果）")
    
    # 指标7: 多主体关系 (+0.3)
    # 例如："用人单位、劳动者、工会的关系"
    # 检测是否涉及多个法律主体
    subjects = ['用人单位', '劳动者', '职工', '工会', '政府', '部门', '机构']
    subject_count = sum(1 for s in subjects if s in query)
    if subject_count >= 2:
        hierarchy_score += 0.3
        indicators.append(f"多主体关系（需要层级：{subject_count}个主体的关系网络）")
    
    # 归一化到[0, 1]
    hierarchy_score = max(0.0, min(1.0, hierarchy_score))
    
    return {
        'hierarchy_score': hierarchy_score,
        'needs_hierarchy': hierarchy_score > 0.0,
        'indicators': indicators
    }


def measure_query_simplicity(query: str) -> float:
    """
    综合评估查询简单度（兼容旧接口）
    
    简单度 = 1 - (多跳需求 + 层级依赖) / 2
    
    Args:
        query: 查询文本
        
    Returns:
        简单度分数 (0-1)，越高越简单
    """
    hop_result = measure_multi_hop_requirement(query)
    hierarchy_result = measure_hierarchical_dependency(query)
    
    # 复杂度 = (多跳分数 + 层级分数) / 2
    complexity = (hop_result['hop_score'] + hierarchy_result['hierarchy_score']) / 2
    
    # 简单度 = 1 - 复杂度
    simplicity = 1.0 - complexity
    
    return simplicity


def calculate_combined_score_with_simplicity(
    query: str,
    bm25_top1_score: float,
    overlap_ratio: float,
    top3_overlap: float,
    llm=None,
    sampling_params=None,
    use_five_dimensions: bool = True
) -> dict:
    """
    综合计算问题复杂度（支持新旧两种模式）
    
    新模式（use_five_dimensions=True）：
    - 使用统一简单度评分体系
    - 三层评估架构：问题类型分类 → 复杂问题细分 → 五维度评估
    - 最终简单度 = 0.5 × 问题本质简单度 + 0.5 × 检索一致性简单度
    - 决策规则：最终简单度 ≥ 0.6 → 传统RAG；< 0.6 → KG
    
    旧模式（use_five_dimensions=False）：
    - 使用原有的多跳推理+层级结构评估
    - 保持向后兼容
    
    Args:
        query: 查询文本
        bm25_top1_score: BM25 Top1分数
        overlap_ratio: 文档重叠率
        top3_overlap: Top3重叠率
        llm: LLM模型实例（可选，用于子问题拆分）
        sampling_params: LLM采样参数（可选）
        use_five_dimensions: 是否使用新的统一简单度评分（默认True）
        
    Returns:
        包含各项指标的字典
    """
    if use_five_dimensions:
        # ========== 新模式：统一简单度评分体系 ==========
        
        # 步骤1: LLM拆分子问题
        decomposition = decompose_query_with_llm(query, llm, sampling_params)
        sub_questions = decomposition['sub_questions']
        
        # 步骤2: 计算最终简单度（三层评估架构）
        result = calculate_final_simplicity_score(
            query=query,
            sub_questions=sub_questions,
            bm25_top1_score=bm25_top1_score,
            overlap_ratio=overlap_ratio,
            top3_overlap=top3_overlap,
            w1=0.5,  # 问题本质权重
            w2=0.5,  # 检索一致性权重
            threshold=0.6,  # 决策阈值
            llm=llm,
            sampling_params=sampling_params
        )
        
        # 添加拆分信息
        result['decomposition'] = decomposition
        result['sub_questions'] = sub_questions
        result['num_sub_questions'] = len(sub_questions)
        result['mode'] = 'unified_simplicity'
        
        return result
    
    else:
        # ========== 旧模式：多跳+层级评估（保持兼容）==========
        
        # 维度1: 多跳推理需求
        hop_result = measure_multi_hop_requirement(query)
        multi_hop_score = hop_result['hop_score']
        
        # 维度2: 层级结构依赖
        hierarchy_result = measure_hierarchical_dependency(query)
        hierarchy_score = hierarchy_result['hierarchy_score']
        
        # 维度3: 检索一致性置信度
        rcc = calculate_retrieval_consistency_confidence(
            bm25_top1_score, overlap_ratio, top3_overlap
        )
        retrieval_consistency_confidence = rcc['score']
        
        # 综合评分
        query_structure_complexity = (multi_hop_score + hierarchy_score) / 2
        query_structure_simplicity = 1.0 - query_structure_complexity
        
        weight_structure = 0.5
        weight_retrieval = 0.5
        
        final_simplicity_score = (
            weight_structure * query_structure_simplicity +
            weight_retrieval * retrieval_consistency_confidence
        )
        
        # 一致性检查
        dimension_diff = abs(query_structure_simplicity - retrieval_consistency_confidence)
        if dimension_diff > 0.4:
            consistency_penalty = 0.1
            final_simplicity_score = final_simplicity_score * (1.0 - consistency_penalty)
        
        return {
            # 维度1: 多跳推理需求
            'multi_hop_score': round(multi_hop_score, 3),
            'is_multi_hop': hop_result['is_multi_hop'],
            'hop_indicators': hop_result['indicators'],
            
            # 维度2: 层级结构依赖
            'hierarchy_score': round(hierarchy_score, 3),
            'needs_hierarchy': hierarchy_result['needs_hierarchy'],
            'hierarchy_indicators': hierarchy_result['indicators'],
            
            # 查询结构综合
            'query_structure_complexity': round(query_structure_complexity, 3),
            'query_structure_simplicity': round(query_structure_simplicity, 3),
            
            # 维度3: 检索一致性
            'query_simplicity_index': round(rcc['query_simplicity_index'], 3),
            'retrieval_divergence': round(rcc['retrieval_divergence'], 3),
            'retrieval_consistency_confidence': round(retrieval_consistency_confidence, 3),
            
            # 综合评分
            'final_simplicity_score': round(final_simplicity_score, 3),
            'dimension_difference': round(dimension_diff, 3),
            
            # 判断结果
            'is_simple': final_simplicity_score >= 0.5,
            'confidence': 'high' if dimension_diff < 0.2 else 'medium' if dimension_diff < 0.4 else 'low',
            
            # 决策建议
            'needs_kg': multi_hop_score > 0.3 or hierarchy_score > 0.3,
            'reason': _generate_decision_reason(multi_hop_score, hierarchy_score, final_simplicity_score),
            'mode': 'legacy'
        }


def calculate_combined_score_with_simplicity_old(
    query: str,
    bm25_top1_score: float,
    overlap_ratio: float,
    top3_overlap: float
) -> dict:
    """
    综合计算问题复杂度（重构版）
    
    三个维度：
    1. 多跳推理需求 (Multi-hop Reasoning Requirement, MRR)
       - 评估是否需要多步推理
       - 范围：[0, 1]，0=单跳，1=多跳
    
    2. 层级结构依赖 (Hierarchical Structure Dependency, HSD)
       - 评估是否需要层级结构辅助理解
       - 范围：[0, 1]，0=平铺，1=层级
    
    3. 检索一致性置信度 (Retrieval Consistency Confidence, RCC)
       - 基于BM25分数和重叠率判断检索难度
       - 范围：[0, 1]，越高越简单
    
    最终简单度 = 1 - (MRR + HSD) / 2 的加权组合
    
    Args:
        query: 查询文本
        bm25_top1_score: BM25 Top1分数
        overlap_ratio: 文档重叠率
        top3_overlap: Top3重叠率
        
    Returns:
        包含各项指标的字典
    """
    # ========== 维度1: 多跳推理需求 (MRR) ==========
    hop_result = measure_multi_hop_requirement(query)
    multi_hop_score = hop_result['hop_score']
    
    # ========== 维度2: 层级结构依赖 (HSD) ==========
    hierarchy_result = measure_hierarchical_dependency(query)
    hierarchy_score = hierarchy_result['hierarchy_score']
    
    # ========== 维度3: 检索一致性置信度 (RCC) ==========
    # 3.1 查询简单性指数 (QSI) - 基于BM25
    query_simplicity_index = 1.0 / (1.0 + np.exp(0.5 * (bm25_top1_score - 12.5)))
    
    # 3.2 检索差异性指数 (RDI) - 基于重叠率
    retrieval_divergence = 1.0 - (0.7 * overlap_ratio + 0.3 * top3_overlap)
    
    # 3.3 简单问题置信度 (使用竞争函数)
    alpha = 1.5
    beta = 0.3
    epsilon = 1e-10
    
    simple_evidence = (query_simplicity_index ** alpha) * (retrieval_divergence ** beta) + epsilon
    complex_evidence = ((1 - query_simplicity_index) ** alpha) * ((1 - retrieval_divergence) ** beta) + epsilon
    
    retrieval_consistency_confidence = simple_evidence / (simple_evidence + complex_evidence)
    
    # 后处理：BM25较高且overlap较低时的惩罚
    if bm25_top1_score > 10.0 and overlap_ratio < 0.3:
        bm25_penalty = min(0.95, (bm25_top1_score - 10.0) / 6.0)
        overlap_bonus = overlap_ratio / 0.3
        final_penalty = bm25_penalty * (1.0 - overlap_bonus)
        retrieval_consistency_confidence = retrieval_consistency_confidence * (1.0 - final_penalty)
    
    retrieval_consistency_confidence = np.clip(retrieval_consistency_confidence, 0, 1)
    
    # ========== 综合评分 ==========
    # 查询结构复杂度 = (多跳需求 + 层级依赖) / 2
    query_structure_complexity = (multi_hop_score + hierarchy_score) / 2
    query_structure_simplicity = 1.0 - query_structure_complexity
    
    # 权重分配：
    # - 查询结构权重 50%（问题本质：多跳+层级）
    # - 检索一致性权重 50%（检索难度）
    weight_structure = 0.5
    weight_retrieval = 0.5
    
    final_simplicity_score = (
        weight_structure * query_structure_simplicity +
        weight_retrieval * retrieval_consistency_confidence
    )
    
    # ========== 一致性检查 ==========
    # 如果查询结构和检索一致性差异很大，说明存在冲突
    dimension_diff = abs(query_structure_simplicity - retrieval_consistency_confidence)
    if dimension_diff > 0.4:
        # 降低10%的置信度
        consistency_penalty = 0.1
        final_simplicity_score = final_simplicity_score * (1.0 - consistency_penalty)
    
    return {
        # 维度1: 多跳推理需求
        'multi_hop_score': round(multi_hop_score, 3),
        'is_multi_hop': hop_result['is_multi_hop'],
        'hop_indicators': hop_result['indicators'],
        
        # 维度2: 层级结构依赖
        'hierarchy_score': round(hierarchy_score, 3),
        'needs_hierarchy': hierarchy_result['needs_hierarchy'],
        'hierarchy_indicators': hierarchy_result['indicators'],
        
        # 查询结构综合
        'query_structure_complexity': round(query_structure_complexity, 3),
        'query_structure_simplicity': round(query_structure_simplicity, 3),
        
        # 维度3: 检索一致性
        'query_simplicity_index': round(query_simplicity_index, 3),
        'retrieval_divergence': round(retrieval_divergence, 3),
        'retrieval_consistency_confidence': round(retrieval_consistency_confidence, 3),
        
        # 综合评分
        'final_simplicity_score': round(final_simplicity_score, 3),
        'dimension_difference': round(dimension_diff, 3),
        
        # 判断结果
        'is_simple': final_simplicity_score >= 0.5,
        'confidence': 'high' if dimension_diff < 0.2 else 'medium' if dimension_diff < 0.4 else 'low',
        
        # 决策建议
        'needs_kg': multi_hop_score > 0.3 or hierarchy_score > 0.3,
        'reason': _generate_decision_reason(multi_hop_score, hierarchy_score, final_simplicity_score)
    }


def _generate_decision_reason(multi_hop: float, hierarchy: float, simplicity: float) -> str:
    """生成决策理由"""
    if simplicity >= 0.7:
        return "简单问题：单跳查询，平铺信息即可，使用混合检索"
    elif simplicity >= 0.5:
        return "中等问题：轻度推理需求，混合检索可能足够"
    elif multi_hop > 0.5 and hierarchy > 0.5:
        return "复杂问题：需要多跳推理且依赖层级结构，强烈建议使用知识图谱"
    elif multi_hop > 0.5:
        return "复杂问题：需要多跳推理，建议使用知识图谱辅助"
    elif hierarchy > 0.5:
        return "复杂问题：需要层级结构理解，建议使用知识图谱辅助"
    else:
        return "复杂问题：检索一致性低，建议使用知识图谱"


# 测试代码
if __name__ == "__main__":
    print("=" * 100)
    print("查询简单度评估模块测试（重构版）")
    print("=" * 100)
    
    # 测试用例
    test_cases = [
        # 简单问题（单跳 + 平铺）
        {
            'query': '《劳动合同法》第85条的内容是什么？',
            'type': '简单问题',
            'expected': '单跳 + 平铺'
        },
        {
            'query': '什么是劳动合同？',
            'type': '简单问题',
            'expected': '单跳 + 平铺'
        },
        {
            'query': '劳动合同法的立法目的是什么？',
            'type': '简单问题',
            'expected': '单跳 + 平铺'
        },
        
        # 中等问题（单跳 + 层级 或 多跳 + 平铺）
        {
            'query': '工伤保险属于社会保险吗？',
            'type': '中等问题',
            'expected': '单跳 + 层级（从属关系）'
        },
        {
            'query': '哪些情况算工伤？',
            'type': '中等问题',
            'expected': '单跳 + 层级（范围界定）'
        },
        
        # 复杂问题（多跳 + 层级）
        {
            'query': '我在工厂上班，下班路上摔伤了，这算工伤吗？',
            'type': '复杂问题',
            'expected': '多跳（场景推理）+ 层级（条件判断）'
        },
        {
            'query': '公司拖欠我工资，我该怎么办？',
            'type': '复杂问题',
            'expected': '多跳（权益维护）+ 层级（程序流程）'
        },
        {
            'query': '我是75年的女职工，交了5年社保，能退休吗？',
            'type': '复杂问题',
            'expected': '多跳（多条件）+ 层级（条件判断）'
        },
        {
            'query': '满足哪些条件才能认定为工伤？',
            'type': '复杂问题',
            'expected': '多跳（复杂列举）+ 层级（条件层级）'
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        query = case['query']
        print(f"\n{'='*100}")
        print(f"测试 {i}: {case['type']}")
        print(f"{'='*100}")
        print(f"查询: {query}")
        print(f"预期: {case['expected']}")
        print(f"\n{'-'*100}")
        
        # 评估多跳推理需求
        hop_result = measure_multi_hop_requirement(query)
        print(f"\n【维度1: 多跳推理需求】")
        print(f"  分数: {hop_result['hop_score']:.3f} ({'多跳' if hop_result['is_multi_hop'] else '单跳'})")
        if hop_result['indicators']:
            print(f"  指标:")
            for indicator in hop_result['indicators']:
                print(f"    - {indicator}")
        
        # 评估层级结构依赖
        hierarchy_result = measure_hierarchical_dependency(query)
        print(f"\n【维度2: 层级结构依赖】")
        print(f"  分数: {hierarchy_result['hierarchy_score']:.3f} ({'需要层级' if hierarchy_result['needs_hierarchy'] else '平铺信息'})")
        if hierarchy_result['indicators']:
            print(f"  指标:")
            for indicator in hierarchy_result['indicators']:
                print(f"    - {indicator}")
        
        # 综合简单度
        simplicity = measure_query_simplicity(query)
        print(f"\n【综合简单度】")
        print(f"  分数: {simplicity:.3f} ({'简单' if simplicity >= 0.5 else '复杂'})")
        
        # 模拟检索指标进行完整评估
        if case['type'] == '简单问题':
            bm25, overlap, top3 = 8.0, 0.8, 1.0
        elif case['type'] == '中等问题':
            bm25, overlap, top3 = 12.0, 0.5, 0.67
        else:
            bm25, overlap, top3 = 15.0, 0.2, 0.0
        
        result = calculate_combined_score_with_simplicity(
            query=query,
            bm25_top1_score=bm25,
            overlap_ratio=overlap,
            top3_overlap=top3,
            use_five_dimensions=False  # 使用旧模式
        )
        
        print(f"\n【完整评估（含检索指标）】")
        print(f"  多跳推理: {result['multi_hop_score']:.3f}")
        print(f"  层级依赖: {result['hierarchy_score']:.3f}")
        print(f"  查询结构复杂度: {result['query_structure_complexity']:.3f}")
        print(f"  检索一致性置信度: {result['retrieval_consistency_confidence']:.3f}")
        print(f"  最终简单度: {result['final_simplicity_score']:.3f}")
        print(f"  判断: {'简单' if result['is_simple'] else '复杂'} (置信度: {result['confidence']})")
        print(f"  是否需要KG: {'是' if result['needs_kg'] else '否'}")
        print(f"  决策理由: {result['reason']}")
    
    print(f"\n{'='*100}")
    print("测试完成")
    print("="*100)


# ==================== 新增测试代码 ====================

def test_five_dimensions():
    """测试5维度评估（不使用LLM）"""
    print("=" * 100)
    print("查询简单度评估模块测试（5维度版本）")
    print("=" * 100)
    
    # 测试用例
    test_cases = [
        # 简单问题
        {
            'query': '《劳动合同法》第85条的内容是什么？',
            'type': '简单问题',
            'sub_questions': ['《劳动合同法》第85条的内容是什么？'],
            'bm25': 8.0, 'overlap': 0.8, 'top3': 1.0
        },
        {
            'query': '什么是劳动合同？',
            'type': '简单问题',
            'sub_questions': ['什么是劳动合同？'],
            'bm25': 8.0, 'overlap': 0.8, 'top3': 1.0
        },
        
        # 中等问题
        {
            'query': '工伤保险属于社会保险吗？',
            'type': '中等问题',
            'sub_questions': ['工伤保险属于社会保险吗？', '工伤保险的定义是什么？'],
            'bm25': 12.0, 'overlap': 0.5, 'top3': 0.67
        },
        
        # 复杂问题
        {
            'query': '我在工厂上班，下班路上摔伤了，这算工伤吗？',
            'type': '复杂问题',
            'sub_questions': [
                "什么情况算工伤？",
                "下班路上受伤算工伤吗？",
                "工伤认定的条件是什么？"
            ],
            'bm25': 15.0, 'overlap': 0.2, 'top3': 0.0
        },
        {
            'query': '我是75年的女职工，交了5年社保，能退休吗？',
            'type': '复杂问题',
            'sub_questions': [
                "女职工退休年龄是多少？",
                "社保缴费年限要求是多少？",
                "75年出生现在多少岁？"
            ],
            'bm25': 15.0, 'overlap': 0.2, 'top3': 0.0
        },
    ]
    
    print("\n【测试模式: 模拟子问题（不使用LLM）】")
    print("=" * 100)
    
    for i, case in enumerate(test_cases, 1):
        query = case['query']
        sub_questions = case['sub_questions']
        
        print(f"\n{'-'*100}")
        print(f"测试 {i}: {case['type']}")
        print(f"{'-'*100}")
        print(f"查询: {query}")
        
        # 评估5个维度
        result = calculate_five_dimension_score(
            query=query,
            sub_questions=sub_questions,
            bm25_top1_score=case['bm25'],
            overlap_ratio=case['overlap'],
            top3_overlap=case['top3']
        )
        
        print(f"\n【子问题】({len(sub_questions)}个)")
        for j, sq in enumerate(sub_questions, 1):
            print(f"  {j}. {sq}")
        
        print(f"\n【5个维度评估】")
        print(f"  1. 推理链长度: {result['dimension_1_reasoning_chain']['score']:.3f} - {result['dimension_1_reasoning_chain']['level']}")
        print(f"  2. 知识整合需求: {result['dimension_2_knowledge_integration']['score']:.3f} - {result['dimension_2_knowledge_integration']['level']}")
        print(f"  3. 关系推理复杂度: {result['dimension_3_relational_reasoning']['score']:.3f} - {result['dimension_3_relational_reasoning']['level']}")
        print(f"  4. 领域跨度: {result['dimension_4_domain_span']['score']:.3f} - {result['dimension_4_domain_span']['level']}")
        print(f"  5. 条件约束密度: {result['dimension_5_conditional_constraint']['score']:.3f} - {result['dimension_5_conditional_constraint']['level']}")
        
        print(f"\n【维度覆盖】")
        print(f"  覆盖维度数: {result['num_covered_dimensions']}/5")
        print(f"  覆盖维度: {', '.join(result['covered_dimensions']) if result['covered_dimensions'] else '无'}")
        
        print(f"\n【综合评分】")
        print(f"  问题本质复杂度: {result['question_complexity']:.3f}")
        print(f"  检索一致性置信度: {result['retrieval_consistency']['score']:.3f}")
        print(f"  最终评分: {result['final_score']:.3f}")
        
        print(f"\n【决策结果】")
        print(f"  是否需要KG: {'是' if result['needs_kg'] else '否'}")
        print(f"  决策理由: {result['reason']}")
    
    print(f"\n{'='*100}")
    print("测试完成")
    print("="*100)
    print("\n提示：实际使用时，子问题应该由LLM自动生成。")
    print("请在主脚本中调用 calculate_combined_score_with_simplicity() 并传入LLM实例。")


if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--old':
        # 运行旧的测试（多跳+层级）
        print("运行旧版测试（多跳推理+层级结构）...\n")
        # 旧测试代码已经在上面定义
    else:
        # 默认运行新的测试（5维度）
        test_five_dimensions()
