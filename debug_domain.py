#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试领域跨度检测
"""

query = "我在工厂上班，下班路上摔伤了，这算工伤吗？"
sub_questions = [
    "什么情况算工伤？",
    "下班路上受伤算工伤吗？",
    "工伤认定的条件是什么？"
]

all_text = query + ' ' + ' '.join(sub_questions)

print("=" * 80)
print("调试领域跨度检测")
print("=" * 80)
print(f"\n原始查询: {query}")
print(f"\n子问题:")
for i, sq in enumerate(sub_questions, 1):
    print(f"  {i}. {sq}")

print(f"\n合并文本: {all_text}")

# 法律领域关键词
legal_domains = {
    '劳动法': ['劳动合同', '劳动关系', '劳动报酬', '加班', '辞退', '劳动争议', '用人单位', '劳动者'],
    '社会保险法': ['社会保险', '养老保险', '医疗保险', '失业保险', '生育保险', '社保'],
    '工伤保险': ['工伤', '工伤认定', '工伤赔偿', '职业病', '工伤待遇'],
}

print(f"\n{'='*80}")
print("检测各领域关键词")
print('='*80)

detected_domains = set()

for domain, keywords in legal_domains.items():
    print(f"\n{domain}:")
    found = False
    for keyword in keywords:
        if keyword in all_text:
            print(f"  ✓ 找到: {keyword}")
            detected_domains.add(domain)
            found = True
            break
    if not found:
        print(f"  ✗ 未找到任何关键词")

print(f"\n{'='*80}")
print("检测到的领域")
print('='*80)
print(f"领域: {detected_domains}")
print(f"数量: {len(detected_domains)}")

# 测试优化逻辑
print(f"\n{'='*80}")
print("测试优化逻辑")
print('='*80)

if '工伤保险' in detected_domains:
    print("✓ 检测到工伤保险领域")
    
    # 检查劳动法关键词
    labor_keywords = ['劳动', '用人单位', '劳动者', '职工']
    print(f"\n检查劳动法关键词: {labor_keywords}")
    for kw in labor_keywords:
        if kw in all_text:
            print(f"  ✓ 找到: {kw}")
            detected_domains.add('劳动法')
            break
    else:
        print(f"  ✗ 未找到任何劳动法关键词")
    
    # 检查社会保险法关键词
    insurance_keywords = ['保险', '待遇', '赔偿']
    print(f"\n检查社会保险法关键词: {insurance_keywords}")
    for kw in insurance_keywords:
        if kw in all_text:
            print(f"  ✓ 找到: {kw}")
            detected_domains.add('社会保险法')
            break
    else:
        print(f"  ✗ 未找到任何社会保险法关键词")

print(f"\n{'='*80}")
print("最终结果")
print('='*80)
print(f"检测到的领域: {detected_domains}")
print(f"领域数量: {len(detected_domains)}")
