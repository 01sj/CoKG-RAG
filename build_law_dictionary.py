#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建法律专用词典，提高BM25效果
"""

import json
import re
import jieba
from collections import Counter

def extract_legal_terms_from_chunks(chunks_file):
    """从chunks文件中提取法律术语"""
    
    print("="*80)
    print("从向量库数据提取法律术语")
    print("="*80)
    
    # 加载数据
    print(f"\n正在加载: {chunks_file}")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"共加载 {len(chunks)} 条记录")
    
    # 提取法律名称
    law_names = set()
    law_names_full = set()
    article_numbers = set()
    
    print("\n正在提取法律术语...")
    for chunk in chunks:
        source_name = chunk['source_name']
        
        # 提取法律名称（完整）
        match = re.search(r'《(.+?)》', source_name)
        if match:
            law_name_full = match.group(1)
            law_names_full.add(law_name_full)
            
            # 提取简称（去掉"中华人民共和国"）
            if law_name_full.startswith("中华人民共和国"):
                law_name_short = law_name_full.replace("中华人民共和国", "")
                law_names.add(law_name_short)
            else:
                law_names.add(law_name_full)
        
        # 提取法条号
        match = re.search(r'第(.+?)条', source_name)
        if match:
            article_num = match.group(1)
            article_numbers.add(f"第{article_num}条")
    
    print(f"\n提取完成:")
    print(f"  完整法律名称: {len(law_names_full)} 个")
    print(f"  简称法律名称: {len(law_names)} 个")
    print(f"  法条号: {len(article_numbers)} 个")
    
    return law_names_full, law_names, article_numbers


def build_law_dictionary(chunks_file, output_file="law_dictionary.txt"):
    """构建法律词典"""
    
    # 1. 提取术语
    law_names_full, law_names, article_numbers = extract_legal_terms_from_chunks(chunks_file)
    
    # 2. 合并所有术语
    all_terms = set()
    all_terms.update(law_names_full)
    all_terms.update(law_names)
    all_terms.update(article_numbers)
    
    # 3. 添加常见法律术语
    common_terms = [
        # 劳动法相关
        "劳动报酬", "加班费", "经济补偿", "劳动合同", "劳动关系",
        "用人单位", "劳动者", "工资", "最低工资", "工作时间",
        "休息休假", "社会保险", "工伤", "职业病", "劳动争议",
        
        # 社会保险相关
        "养老保险", "医疗保险", "失业保险", "工伤保险", "生育保险",
        "社会保险费", "缴费基数", "缴费年限", "退休", "退休金",
        
        # 未成年人保护相关
        "未成年人", "监护人", "法定代理人", "父母", "学校",
        "教育机构", "人身安全", "身心健康", "合法权益",
        
        # 安全生产相关
        "安全生产", "生产安全事故", "安全生产监督", "安全生产条件",
        "安全生产责任", "安全生产管理", "安全生产教育",
        
        # 其他常见术语
        "法律责任", "行政处罚", "刑事责任", "民事责任",
        "人民法院", "人民检察院", "公安机关", "劳动行政部门",
    ]
    
    all_terms.update(common_terms)
    
    # 4. 保存到文件
    print(f"\n正在保存词典到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for term in sorted(all_terms):
            f.write(f"{term}\n")
    
    print(f"词典保存完成，共 {len(all_terms)} 个词条")
    
    # 5. 添加到jieba
    print("\n正在添加到jieba词典...")
    for term in all_terms:
        if len(term) >= 2:  # 只添加2个字符以上的词
            jieba.add_word(term, freq=10000)
    
    print("jieba词典更新完成")
    
    return all_terms


def test_tokenization(law_dictionary_file=None):
    """测试分词效果"""
    
    print("\n" + "="*80)
    print("测试分词效果")
    print("="*80)
    
    # 如果提供了词典文件，加载它
    if law_dictionary_file:
        print(f"\n加载词典: {law_dictionary_file}")
        with open(law_dictionary_file, 'r', encoding='utf-8') as f:
            for line in f:
                term = line.strip()
                if term:
                    jieba.add_word(term, freq=10000)
        print("词典加载完成")
    
    # 测试查询
    test_queries = [
        "劳动合同法第八十五条的内容是什么？",
        "中华人民共和国劳动法第六十八条",
        "社会保险法关于医疗保险的规定",
        "未成年人保护法第五十四条",
        "用人单位应当支付劳动报酬和加班费",
    ]
    
    print("\n分词测试:")
    print("-"*80)
    
    for query in test_queries:
        tokens = list(jieba.cut(query))
        print(f"\n查询: {query}")
        print(f"分词: {tokens}")
        
        # 统计关键词
        keywords = [t for t in tokens if len(t) >= 2 and t not in ['的', '是', '什么', '内容', '关于', '规定']]
        print(f"关键词: {keywords}")


def compare_bm25_scores():
    """对比改进前后的BM25分数"""
    
    print("\n" + "="*80)
    print("对比BM25分数")
    print("="*80)
    
    from rank_bm25 import BM25Okapi
    
    # 模拟文档
    documents = [
        "《中华人民共和国劳动合同法》第八十五条: 用人单位有下列情形之一的，由劳动行政部门责令限期支付劳动报酬...",
        "《中华人民共和国劳动法》第八十五条: 县级以上各级人民政府劳动行政部门依法对用人单位遵守劳动法律...",
        "《中华人民共和国工会法》第八十五条: 违反本法规定，对依法履行职责的工会工作人员无正当理由调动工作岗位...",
    ]
    
    doc_names = [
        "劳动合同法第85条 ✅",
        "劳动法第85条 ❌",
        "工会法第85条 ❌",
    ]
    
    query = "劳动合同法第八十五条的内容是什么"
    
    # 改进前（不加载词典）
    print("\n【改进前】")
    jieba.del_word("劳动合同法")
    jieba.del_word("第八十五条")
    
    tokenized_docs_before = [list(jieba.cut(doc)) for doc in documents]
    tokenized_query_before = list(jieba.cut(query))
    
    print(f"查询分词: {tokenized_query_before}")
    
    bm25_before = BM25Okapi(tokenized_docs_before)
    scores_before = bm25_before.get_scores(tokenized_query_before)
    
    print("\nBM25分数:")
    for name, score in zip(doc_names, scores_before):
        print(f"  {name:<30} {score:.3f}")
    
    # 改进后（加载词典）
    print("\n【改进后】")
    jieba.add_word("劳动合同法", freq=10000)
    jieba.add_word("劳动法", freq=10000)
    jieba.add_word("工会法", freq=10000)
    jieba.add_word("第八十五条", freq=10000)
    
    tokenized_docs_after = [list(jieba.cut(doc)) for doc in documents]
    tokenized_query_after = list(jieba.cut(query))
    
    print(f"查询分词: {tokenized_query_after}")
    
    bm25_after = BM25Okapi(tokenized_docs_after)
    scores_after = bm25_after.get_scores(tokenized_query_after)
    
    print("\nBM25分数:")
    for name, score in zip(doc_names, scores_after):
        print(f"  {name:<30} {score:.3f}")
    
    # 对比
    print("\n【对比】")
    print(f"{'文档':<30} {'改进前':<10} {'改进后':<10} {'变化':<10}")
    print("-"*80)
    for name, before, after in zip(doc_names, scores_before, scores_after):
        change = after - before
        symbol = "⬆️" if change > 0 else "⬇️" if change < 0 else "="
        print(f"{name:<30} {before:<10.3f} {after:<10.3f} {change:+.3f} {symbol}")


def main():
    """主函数"""
    
    print("\n" + "="*80)
    print("法律词典构建工具")
    print("="*80)
    
    chunks_file = "datasets/chunks/basic_laws_social_only_chunk.json"
    output_file = "law_dictionary.txt"
    
    # 1. 构建词典
    all_terms = build_law_dictionary(chunks_file, output_file)
    
    # 2. 测试分词
    test_tokenization(output_file)
    
    # 3. 对比BM25分数
    compare_bm25_scores()
    
    print("\n" + "="*80)
    print("完成")
    print("="*80)
    print(f"\n词典文件: {output_file}")
    print(f"词条数量: {len(all_terms)}")
    print("\n使用方法:")
    print("1. 在程序启动时加载词典")
    print("2. 使用jieba.add_word()添加到分词器")
    print("3. 进行BM25重排序时会自动使用改进的分词")


if __name__ == "__main__":
    main()
