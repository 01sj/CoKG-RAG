#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从社会法数据集中筛选只包含基础16部社会法的数据
"""

import json
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 基础16部社会法列表（来自 basic_laws_list.txt）
BASIC_SOCIAL_LAWS = [
    "义务教育法",
    "人口与计划生育法",
    "劳动合同法",
    "劳动法",
    "妇女权益保障法",
    "教育法",
    "安全生产法",
    "工会法",
    "未成年人保护法",
    "社会保险法",
    "老年人权益保障法",
    "职业病防治法",
    "药品管理法",
    "食品安全法",
    "高等教育法",
    "残疾人保障法",
]

# 创建更宽松的匹配模式（考虑不同的表述方式）
BASIC_SOCIAL_LAWS_PATTERNS = []
for law in BASIC_SOCIAL_LAWS:
    # 添加原始名称
    BASIC_SOCIAL_LAWS_PATTERNS.append(law)
    # 添加带"中华人民共和国"前缀的版本
    BASIC_SOCIAL_LAWS_PATTERNS.append(f"中华人民共和国{law}")
    # 添加简化版本（去掉"中华人民共和国"）
    if law.startswith("中华人民共和国"):
        BASIC_SOCIAL_LAWS_PATTERNS.append(law.replace("中华人民共和国", ""))


def is_basic_social_law(item):
    """
    判断一条数据是否只涉及基础16部社会法
    
    策略：
    1. 提取所有引用的法律
    2. 检查是否都在基础16部社会法列表中
    """
    # 提取文本内容
    references = item.get("reference", [])
    question = item.get("question", "")
    answer = item.get("answer", "")
    
    # 合并所有文本
    all_text = " ".join(references) + " " + question + " " + answer
    
    # 检查是否包含基础社会法
    matched_laws = []
    for law in BASIC_SOCIAL_LAWS:
        # 检查多种可能的表述
        if (law in all_text or 
            f"中华人民共和国{law}" in all_text or
            f"《{law}》" in all_text or
            f"《中华人民共和国{law}》" in all_text):
            matched_laws.append(law)
    
    # 如果匹配到基础社会法，返回True
    if matched_laws:
        return True, matched_laws
    
    return False, []


def filter_basic_social_laws(input_file, output_file):
    """
    筛选基础16部社会法的数据
    """
    logger.info(f"开始加载数据: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✅ 成功加载 {len(data)} 条数据")
    except Exception as e:
        logger.error(f"❌ 加载数据失败: {e}")
        return
    
    # 筛选基础社会法数据
    filtered_data = []
    law_counter = {law: 0 for law in BASIC_SOCIAL_LAWS}
    
    logger.info("开始筛选基础社会法数据...")
    logger.info(f"目标法律: {', '.join(BASIC_SOCIAL_LAWS)}\n")
    
    for idx, item in enumerate(tqdm(data, desc="筛选进度")):
        is_basic, matched_laws = is_basic_social_law(item)
        if is_basic:
            filtered_data.append(item)
            # 统计每部法律的数量
            for law in matched_laws:
                law_counter[law] += 1
            
            # 每1000条打印一次进度
            if len(filtered_data) % 1000 == 0:
                logger.info(f"已筛选出 {len(filtered_data)} 条基础社会法数据")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"筛选完成！")
    logger.info(f"{'='*60}")
    logger.info(f"原始数据: {len(data)} 条")
    logger.info(f"基础社会法数据: {len(filtered_data)} 条")
    logger.info(f"占比: {len(filtered_data)/len(data)*100:.2f}%")
    
    # 显示每部法律的统计
    logger.info(f"\n各法律数据量统计:")
    sorted_laws = sorted(law_counter.items(), key=lambda x: x[1], reverse=True)
    
    total_count = sum(law_counter.values())
    for i, (law, count) in enumerate(sorted_laws, 1):
        if count > 0:
            percentage = count / total_count * 100 if total_count > 0 else 0
            logger.info(f"  {i:>2}. {law:<20} {count:>6} 条 ({percentage:>5.2f}%)")
    
    # 统计未匹配的法律
    unmatched_laws = [law for law, count in law_counter.items() if count == 0]
    if unmatched_laws:
        logger.info(f"\n⚠️  未匹配到数据的法律 ({len(unmatched_laws)} 部):")
        for law in unmatched_laws:
            logger.info(f"    - {law}")
    
    # 保存结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ 结果已保存至: {output_file}")
        
        # 计算文件大小
        import os
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        logger.info(f"文件大小: {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"❌ 保存结果失败: {e}")
        return
    
    # 显示一些示例
    logger.info(f"\n数据示例（前5条）:")
    for i, item in enumerate(filtered_data[:5], 1):
        question = item.get("question", "")[:60]
        logger.info(f"  [{i}] {question}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="筛选基础16部社会法的数据")
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/训练数据_社会法.json",
        help="输入文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/训练数据_基础社会法16部.json",
        help="输出文件路径"
    )
    
    args = parser.parse_args()
    
    logger.info(f"{'='*60}")
    logger.info(f"基础社会法数据筛选工具")
    logger.info(f"{'='*60}")
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出文件: {args.output}")
    logger.info(f"目标: 筛选16部基础社会法")
    logger.info(f"{'='*60}\n")
    
    filter_basic_social_laws(args.input, args.output)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ 处理完成！")
    logger.info(f"{'='*60}")
