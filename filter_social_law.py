#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
过滤社会法法条脚本

功能：从 basic_laws_minshang_shehui.jsonl 中删除民商法法条，只保留社会法法条
"""

import json
import os
from collections import Counter

def filter_social_law(input_file, output_file):
    """
    过滤出社会法法条
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    print(f"📂 输入文件: {input_file}")
    print(f"📂 输出文件: {output_file}")
    
    # 统计信息
    total_count = 0
    category_count = Counter()
    social_law_count = 0
    social_law_titles = set()
    
    # 读取并过滤数据
    social_law_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                total_count += 1
                
                # 统计类别
                category = data.get('basic_category', '未知')
                category_count[category] += 1
                
                # 只保留社会法
                if category == '社会法':
                    # 重新分配ID（从0开始）
                    data['id'] = social_law_count
                    social_law_data.append(data)
                    social_law_count += 1
                    social_law_titles.add(data.get('basic_law_title', ''))
                
            except json.JSONDecodeError as e:
                print(f"⚠️  第 {line_num} 行JSON解析错误: {e}")
                continue
    
    # 写入过滤后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in social_law_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    # 输出统计信息
    print(f"\n📊 处理统计:")
    print(f"   总法条数: {total_count}")
    print(f"   各类别分布:")
    for category, count in category_count.most_common():
        print(f"     - {category}: {count} 条")
    
    print(f"\n✅ 过滤结果:")
    print(f"   保留社会法法条: {social_law_count} 条")
    print(f"   删除民商法法条: {category_count.get('民商法', 0)} 条")
    
    print(f"\n📚 社会法包含的法律:")
    for title in sorted(social_law_titles):
        if title:
            print(f"     - {title}")
    
    print(f"\n💾 已保存到: {output_file}")

def main():
    """主函数"""
    
    # 文件路径
    input_file = "datasets/basic_laws_minshang_shehui.jsonl"
    output_file = "datasets/basic_laws_social_only.jsonl"
    
    print("🔍 社会法法条过滤器")
    print("=" * 50)
    
    # 执行过滤
    filter_social_law(input_file, output_file)
    
    # 验证结果
    print(f"\n🔍 验证结果:")
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"   输出文件行数: {len(lines)}")
            
            # 检查前几条数据
            print(f"   前3条法条:")
            for i, line in enumerate(lines[:3]):
                data = json.loads(line)
                print(f"     {i+1}. {data['name']} ({data['basic_category']})")
    
    print(f"\n✨ 过滤完成！现在可以使用 {output_file} 构建纯社会法知识图谱")

if __name__ == "__main__":
    main()