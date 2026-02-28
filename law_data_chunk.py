#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
法律数据处理脚本：支持JSON、TXT、DOCX格式的法律文档分块
适用于中文法律知识图谱构建
整合了ragflow的法律文档解析逻辑
"""

import json
import tiktoken
from hashlib import md5
import os
import re
import sys
import shlex
from pathlib import Path

# 添加ragflow相关路径
current_file = Path(__file__).resolve()
project_root = current_file.parent
sys.path.insert(0, str(project_root))

# 导入ragflow的解析模块
try:
    from rag.app.laws import Docx, chunk as ragflow_chunk
    from deepdoc.parser.utils import get_text
    from rag.nlp import bullets_category, remove_contents_table, make_colon_as_title, tree_merge
    from rag.nlp import rag_tokenizer, Node
    RAGFLOW_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入ragflow模块: {e}")
    print("将仅支持JSON格式")
    RAGFLOW_AVAILABLE = False


def compute_mdhash_id(content, prefix: str = ""):
    """计算内容的MD5哈希值"""
    return prefix + md5(content.encode()).hexdigest()


def extract_legal_data(input_jsonl, output_json, start_idx=0, end_idx=50):
    """
    从JSONL文件中提取指定范围的法律数据
    
    参数:
        input_jsonl: 输入的JSONL文件路径
        output_json: 输出的JSON文件路径
        start_idx: 起始索引
        end_idx: 结束索引（不包含）
    """
    print(f"正在从 {input_jsonl} 提取数据...")
    print(f"提取范围: 第 {start_idx} 到 {end_idx-1} 条")
    
    data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= end_idx:
                break
            if i >= start_idx and line.strip():
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"警告: 第 {i} 行JSON解析失败: {e}")
                    continue
    
    # 保存提取的数据
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"成功提取 {len(data)} 条数据到 {output_json}")
    return data


def chunk_legal_documents(
    legal_items,
    model_name="cl100k_base",
    max_token_size=512,
    overlap_token_size=64,
    merge_small_items=True
):
    """
    对法律文档进行分块处理
    
    参数:
        legal_items: 法律条文列表，每项包含 id, name, content
        model_name: tiktoken编码器模型名称
        max_token_size: 每个块的最大token数
        overlap_token_size: 块之间的重叠token数
        merge_small_items: 是否合并小的法条
    
    返回:
        分块结果列表
    """
    ENCODER = tiktoken.get_encoding(model_name)
    results = []
    
    if merge_small_items:
        # 策略：合并小法条到合适大小
        current_batch = []
        current_tokens = 0
        current_ids = []
        
        for item in legal_items:
            # 组合法条标题和内容
            full_text = f"{item['name']}: {item['content']}"
            tokens = ENCODER.encode(full_text)
            token_count = len(tokens)
            
            # 如果单个法条就超过max_token_size，需要拆分
            if token_count > max_token_size:
                # 先保存当前批次
                if current_batch:
                    merged_text = " ".join(current_batch)
                    results.append({
                        "hash_code": compute_mdhash_id(merged_text),
                        "text": merged_text.strip(),
                        "source_type": "merged_articles",
                        "source_ids": current_ids,
                        "token_count": current_tokens
                    })
                    current_batch = []
                    current_tokens = 0
                    current_ids = []
                
                # 对大法条进行分块
                for start in range(0, token_count, max_token_size - overlap_token_size):
                    chunk_tokens = tokens[start : start + max_token_size]
                    chunk_text = ENCODER.decode(chunk_tokens)
                    results.append({
                        "hash_code": compute_mdhash_id(chunk_text),
                        "text": chunk_text.strip(),
                        "source_type": "split_article",
                        "source_id": item['id'],
                        "source_name": item['name'],
                        "token_count": len(chunk_tokens)
                    })
            
            # 如果加上当前法条会超过max_token_size，先保存当前批次
            elif current_tokens + token_count > max_token_size:
                if current_batch:
                    merged_text = " ".join(current_batch)
                    results.append({
                        "hash_code": compute_mdhash_id(merged_text),
                        "text": merged_text.strip(),
                        "source_type": "merged_articles",
                        "source_ids": current_ids,
                        "token_count": current_tokens
                    })
                current_batch = [full_text]
                current_tokens = token_count
                current_ids = [item['id']]
            
            # 否则加入当前批次
            else:
                current_batch.append(full_text)
                current_tokens += token_count
                current_ids.append(item['id'])
        
        # 保存最后一个批次
        if current_batch:
            merged_text = " ".join(current_batch)
            results.append({
                "hash_code": compute_mdhash_id(merged_text),
                "text": merged_text.strip(),
                "source_type": "merged_articles",
                "source_ids": current_ids,
                "token_count": current_tokens
            })
    
    else:
        # 策略：每个法条独立处理
        for item in legal_items:
            full_text = f"{item['name']}: {item['content']}"
            tokens = ENCODER.encode(full_text)
            token_count = len(tokens)
            
            # 如果法条较长，需要分块
            if token_count > max_token_size:
                for start in range(0, token_count, max_token_size - overlap_token_size):
                    chunk_tokens = tokens[start : start + max_token_size]
                    chunk_text = ENCODER.decode(chunk_tokens)
                    results.append({
                        "hash_code": compute_mdhash_id(chunk_text),
                        "text": chunk_text.strip(),
                        "source_type": "split_article",
                        "source_id": item['id'],
                        "source_name": item['name'],
                        "token_count": len(chunk_tokens)
                    })
            else:
                # 短法条直接保存
                results.append({
                    "hash_code": compute_mdhash_id(full_text),
                    "text": full_text.strip(),
                    "source_type": "single_article",
                    "source_id": item['id'],
                    "source_name": item['name'],
                    "token_count": token_count
                })
    
    return results


def chunk_txt_file(filename, max_token_size=1024, overlap_token_size=128):
    """
    使用ragflow逻辑处理TXT文件
    
    参数:
        filename: TXT文件路径
        max_token_size: 每个块最大token数
        overlap_token_size: 重叠token数
    
    返回:
        分块结果列表，格式与chunk_legal_documents一致
    """
    if not RAGFLOW_AVAILABLE:
        raise ImportError("ragflow模块不可用，无法处理TXT文件")
    
    print(f"使用ragflow解析TXT文件: {filename}")
    
    # 使用ragflow的chunk函数解析
    def dummy_callback(prog=None, msg=""):
        if msg:
            print(f"  {msg}")
    
    # 解析文件
    ragflow_chunks = ragflow_chunk(
        filename=filename,
        binary=None,
        from_page=0,
        to_page=100000,
        lang="Chinese",
        callback=dummy_callback,
        parser_config={
            "chunk_token_num": max_token_size,
            "delimiter": "\n!?。；！？",
            "layout_recognize": "DeepDOC"
        }
    )
    
    # 转换格式：ragflow返回的是包含content_with_weight等字段的字典列表
    # 我们需要转换为统一的格式
    ENCODER = tiktoken.get_encoding("cl100k_base")
    results = []
    
    for chunk_doc in ragflow_chunks:
        # 获取文本内容
        text = chunk_doc.get("content_with_weight", chunk_doc.get("content_ltks", ""))
        if not text or not text.strip():
            continue
        
        # 计算token数
        tokens = ENCODER.encode(text)
        token_count = len(tokens)
        
        # 如果超过max_token_size，需要进一步分块
        if token_count > max_token_size:
            for start in range(0, token_count, max_token_size - overlap_token_size):
                chunk_tokens = tokens[start : start + max_token_size]
                chunk_text = ENCODER.decode(chunk_tokens)
                results.append({
                    "hash_code": compute_mdhash_id(chunk_text),
                    "text": chunk_text.strip(),
                    "source_type": "ragflow_txt_split",
                    "source_file": os.path.basename(filename),
                    "token_count": len(chunk_tokens)
                })
        else:
            results.append({
                "hash_code": compute_mdhash_id(text),
                "text": text.strip(),
                "source_type": "ragflow_txt",
                "source_file": os.path.basename(filename),
                "token_count": token_count
            })
    
    return results


def chunk_docx_file(filename, max_token_size=1024, overlap_token_size=128, preserve_section_structure=True):
    """
    使用ragflow逻辑处理DOCX文件
    
    参数:
        filename: DOCX文件路径
        max_token_size: 每个块最大token数（仅在preserve_section_structure=False时使用）
        overlap_token_size: 重叠token数（仅在preserve_section_structure=False时使用）
        preserve_section_structure: 是否保持章节结构完整性（True=保持，False=按token分割）
    
    返回:
        分块结果列表，格式与chunk_legal_documents一致
    """
    if not RAGFLOW_AVAILABLE:
        raise ImportError("ragflow模块不可用，无法处理DOCX文件")
    
    print(f"使用ragflow解析DOCX文件: {filename}")
    if preserve_section_structure:
        print("  保持章节结构完整性（不进行token级别的二次分割）")
    else:
        print(f"  允许按token分割（max_token={max_token_size}, overlap={overlap_token_size}）")
    
    # 使用ragflow的Docx类解析
    def dummy_callback(prog=None, msg=""):
        if msg:
            print(f"  {msg}")
    
    # 解析docx文件
    docx_parser = Docx()
    chunks = docx_parser(filename, binary=None)
    
    # 转换格式
    ENCODER = tiktoken.get_encoding("cl100k_base")
    results = []
    
    for chunk_text in chunks:
        if not chunk_text or not chunk_text.strip():
            continue
        
        # 计算token数
        tokens = ENCODER.encode(chunk_text)
        token_count = len(tokens)
        
        if preserve_section_structure:
            # 保持章节结构完整性，不进行二次分割
            # 即使超过max_token_size也保持原样（与ragflow在线演示一致）
            results.append({
                "hash_code": compute_mdhash_id(chunk_text),
                "text": chunk_text.strip(),
                "source_type": "ragflow_docx",
                "source_file": os.path.basename(filename),
                "token_count": token_count
            })
        else:
            # 如果超过max_token_size，需要进一步分块
            if token_count > max_token_size:
                for start in range(0, token_count, max_token_size - overlap_token_size):
                    chunk_tokens = tokens[start : start + max_token_size]
                    chunk_text_split = ENCODER.decode(chunk_tokens)
                    results.append({
                        "hash_code": compute_mdhash_id(chunk_text_split),
                        "text": chunk_text_split.strip(),
                        "source_type": "ragflow_docx_split",
                        "source_file": os.path.basename(filename),
                        "token_count": len(chunk_tokens)
                    })
            else:
                results.append({
                    "hash_code": compute_mdhash_id(chunk_text),
                    "text": chunk_text.strip(),
                    "source_type": "ragflow_docx",
                    "source_file": os.path.basename(filename),
                    "token_count": token_count
                })
    
    return results


def process_single_file(input_file, max_token_size=1024, overlap_token_size=128, 
                        merge_small_items=False, output_dir="datasets", verbose=True):
    """
    处理单个文件
    
    参数:
        input_file: 输入文件路径
        max_token_size: 每个块最大token数
        overlap_token_size: 重叠token数
        merge_small_items: 是否合并小法条
        output_dir: 输出目录
        verbose: 是否显示详细信息
    
    返回:
        (success: bool, output_file: str, chunk_count: int, error_msg: str)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"处理文件: {input_file}")
        print(f"{'='*60}")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        error_msg = f"文件不存在: {input_file}"
        if verbose:
            print(f"错误: {error_msg}")
        return False, None, 0, error_msg
    
    # 根据输入文件生成输出文件名
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_chunk_file = os.path.join(output_dir, f"{base_name}_chunk.json")
    
    # 确保输出目录存在且有写入权限
    try:
        os.makedirs(output_dir, exist_ok=True)
        # 测试写入权限
        test_file = os.path.join(output_dir, ".write_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
    except (OSError, PermissionError) as e:
        # 如果没有权限，使用当前目录
        if verbose:
            print(f"警告: 无法写入 {output_dir}: {e}")
            print("改为使用当前目录")
        output_dir = "."
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_chunk_file = f"{base_name}_chunk.json"
    
    # 根据文件扩展名选择处理方式
    file_ext = os.path.splitext(input_file)[1].lower()
    
    if verbose:
        print(f"文件格式: {file_ext}")
    
    results = []
    
    try:
        if file_ext in ['.json', '.jsonl']:
            # 处理JSON/JSONL格式
            if verbose:
                print(f"使用{file_ext.upper()}格式解析...")
            
            legal_data = []
            if file_ext == '.json':
                # JSON格式：整个文件是一个数组
                with open(input_file, 'r', encoding='utf-8') as f:
                    legal_data = json.load(f)
            else:
                # JSONL格式：每行是一个JSON对象
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                legal_data.append(item)
                            except json.JSONDecodeError as e:
                                if verbose:
                                    print(f"  警告: 第{line_num}行JSON解析失败: {e}")
                                continue
            
            if not legal_data:
                error_msg = "没有提取到任何数据"
                if verbose:
                    print(f"错误: {error_msg}")
                return False, None, 0, error_msg
            
            if verbose:
                print(f"读取到 {len(legal_data)} 条法律条文")
                print(f"分块参数: max_token={max_token_size}, overlap={overlap_token_size}")
            
            results = chunk_legal_documents(
                legal_data,
                max_token_size=max_token_size,
                overlap_token_size=overlap_token_size,
                merge_small_items=merge_small_items
            )
        
        elif file_ext in ['.txt', '.md', '.markdown']:
            # 处理TXT格式（使用ragflow逻辑）
            if not RAGFLOW_AVAILABLE:
                error_msg = "ragflow模块不可用，无法处理TXT文件"
                if verbose:
                    print(f"错误: {error_msg}")
                return False, None, 0, error_msg
            
            if verbose:
                print("使用ragflow解析TXT文件...")
                print(f"分块参数: max_token={max_token_size}, overlap={overlap_token_size}")
            
            results = chunk_txt_file(
                input_file,
                max_token_size=max_token_size,
                overlap_token_size=overlap_token_size
            )
        
        elif file_ext == '.docx':
            # 处理DOCX格式（使用ragflow逻辑）
            if not RAGFLOW_AVAILABLE:
                error_msg = "ragflow模块不可用，无法处理DOCX文件"
                if verbose:
                    print(f"错误: {error_msg}")
                return False, None, 0, error_msg
            
            if verbose:
                print("使用ragflow解析DOCX文件...")
                print("  保持章节结构完整性（与ragflow在线演示一致）")
            
            # DOCX文件保持章节结构完整性，不进行token级别的二次分割
            results = chunk_docx_file(
                input_file,
                max_token_size=max_token_size,
                overlap_token_size=overlap_token_size,
                preserve_section_structure=True  # 保持章节结构
            )
        
        else:
            error_msg = f"不支持的文件格式: {file_ext}"
            if verbose:
                print(f"错误: {error_msg}")
                print("支持的格式: .json, .jsonl, .txt, .md, .markdown, .docx")
            return False, None, 0, error_msg
        
        if not results:
            error_msg = "没有生成任何分块"
            if verbose:
                print(f"错误: {error_msg}")
            return False, None, 0, error_msg
        
        # 保存分块结果
        if verbose:
            print(f"保存分块结果到: {output_chunk_file}")
        with open(output_chunk_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        if verbose:
            # 显示统计信息
            source_types = {}
            total_tokens = 0
            for item in results:
                source_type = item.get('source_type', 'unknown')
                source_types[source_type] = source_types.get(source_type, 0) + 1
                total_tokens += item.get('token_count', 0)
            
            print(f"✓ 处理完成: {len(results)} 个分块, 总token数: {total_tokens:,}")
        
        return True, output_chunk_file, len(results), None
    
    except Exception as e:
        error_msg = f"处理文件时出错: {str(e)}"
        if verbose:
            print(f"错误: {error_msg}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False, None, 0, error_msg


def find_supported_files(input_path):
    """
    查找支持的文件
    
    参数:
        input_path: 文件路径或目录路径
    
    返回:
        支持的文件列表
    """
    supported_extensions = ['.json', '.jsonl', '.txt', '.md', '.markdown', '.docx']
    files = []
    
    # 处理编码问题：确保路径使用正确的编码
    try:
        # 如果路径包含中文字符，尝试不同的编码方式
        if isinstance(input_path, str):
            # 尝试规范化路径
            input_path = os.path.normpath(input_path)
        
        if os.path.isfile(input_path):
            # 单个文件
            ext = os.path.splitext(input_path)[1].lower()
            if ext in supported_extensions:
                files.append(input_path)
        elif os.path.isdir(input_path):
            # 目录，递归查找所有支持的文件
            for root, dirs, filenames in os.walk(input_path):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in supported_extensions:
                        full_path = os.path.join(root, filename)
                        files.append(full_path)
        else:
            # 路径不存在，可能是编码问题，尝试通配符匹配
            import glob
            # 如果输入路径包含通配符或者看起来像是被编码影响的路径
            if '*' in input_path or '?' in input_path:
                # 使用glob匹配
                matched_files = glob.glob(input_path)
                for matched_file in matched_files:
                    if os.path.isfile(matched_file):
                        ext = os.path.splitext(matched_file)[1].lower()
                        if ext in supported_extensions:
                            files.append(matched_file)
            else:
                # 尝试在父目录中查找类似的文件名
                parent_dir = os.path.dirname(input_path) or '.'
                if os.path.isdir(parent_dir):
                    target_filename = os.path.basename(input_path)
                    for root, dirs, filenames in os.walk(parent_dir):
                        for filename in filenames:
                            # 检查文件名是否包含目标文件名的关键部分
                            if target_filename in filename or filename in target_filename:
                                ext = os.path.splitext(filename)[1].lower()
                                if ext in supported_extensions:
                                    full_path = os.path.join(root, filename)
                                    files.append(full_path)
                                    break
                        if files:  # 找到文件就停止搜索
                            break
    except Exception as e:
        print(f"路径处理出错: {e}")
        return []
    
    return files


def main():
    """主函数"""
    print("=" * 60)
    print("法律数据处理与分块脚本")
    print("支持格式: JSON, JSONL, TXT, DOCX")
    print("支持批量处理: 可指定多个文件或目录")
    print("默认处理: /newdataf/SJ/LeanRAG/datasets/basic_laws_social_only.jsonl (社会法数据)")
    print("输出目录: /newdataf/SJ/LeanRAG/datasets/chunks_v2/")
    print("=" * 60)
    
    # 分块参数
    max_token_size = 1024          # 每个块最大token数
    overlap_token_size = 128       # 重叠token数
    merge_small_items = False       # 是否合并小法条（KG抽取阶段关闭合并，保持法条原子性）
    output_dir = "/newdataf/SJ/LeanRAG/datasets/chunks_v2"        # 输出目录
    
    # 解析命令行参数（支持带空格的文件名）
    if len(sys.argv) > 1:
        # 智能合并参数：如果参数被空格分割，尝试重新组合
        input_paths = []
        i = 1
        while i < len(sys.argv):
            current = sys.argv[i]
            
            # 如果当前参数是一个完整的文件路径（存在），直接使用
            if os.path.exists(current) or os.path.isdir(current):
                input_paths.append(current)
                i += 1
                continue
            
            # 如果当前参数不存在，可能是被分割的文件名，尝试合并后续参数
            # 检查是否包含路径分隔符或看起来像路径的一部分
            if '/' in current or '\\' in current or current.startswith('.'):
                # 尝试逐步合并后续参数，直到找到一个存在的路径
                combined = current
                j = i + 1
                found = False
                
                while j < len(sys.argv):
                    test_path = combined + ' ' + sys.argv[j]
                    if os.path.exists(test_path) or os.path.isdir(test_path):
                        input_paths.append(test_path)
                        i = j + 1
                        found = True
                        break
                    # 检查是否已经有扩展名，如果有，停止合并
                    if os.path.splitext(test_path)[1] in ['.json', '.txt', '.md', '.markdown', '.docx']:
                        # 即使不存在，也认为这是一个完整的文件名
                        input_paths.append(test_path)
                        i = j + 1
                        found = True
                        break
                    combined = test_path
                    j += 1
                
                if not found:
                    # 如果合并后仍然不存在，使用合并后的路径（让后续代码处理错误）
                    input_paths.append(combined)
                    i = j
            else:
                # 不像是路径，直接使用
                input_paths.append(current)
                i += 1
    else:
        # 默认处理社会法数据（已过滤）
        input_paths = ["/newdataf/SJ/LeanRAG/datasets/basic_laws_social_only.jsonl"]
    
    # 收集所有需要处理的文件
    all_files = []
    for path in input_paths:
        files = find_supported_files(path)
        if files:
            all_files.extend(files)
        else:
            print(f"警告: 未找到支持的文件: {path}")
            # 如果是目录，尝试直接列出其中的DOCX文件
            if os.path.isdir(path):
                try:
                    for filename in os.listdir(path):
                        if filename.lower().endswith(('.json', '.jsonl', '.docx')):
                            full_path = os.path.join(path, filename)
                            if os.path.isfile(full_path):
                                all_files.append(full_path)
                                print(f"  发现DOCX文件: {filename}")
                except Exception as e:
                    print(f"  列出目录文件时出错: {e}")
    
    if not all_files:
        print("错误: 没有找到任何支持的文件!")
        print("支持的格式: .json, .jsonl, .txt, .md, .markdown, .docx")
        return
    
    # 去重并排序
    all_files = sorted(list(set(all_files)))
    
    print(f"\n找到 {len(all_files)} 个文件需要处理:")
    for i, f in enumerate(all_files, 1):
        print(f"  {i}. {f}")
    
    # 批量处理
    print(f"\n{'='*60}")
    print("开始批量处理")
    print(f"{'='*60}")
    
    success_count = 0
    fail_count = 0
    total_chunks = 0
    results_summary = []
    
    for i, input_file in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] ", end="")
        success, output_file, chunk_count, error_msg = process_single_file(
            input_file,
            max_token_size=max_token_size,
            overlap_token_size=overlap_token_size,
            merge_small_items=merge_small_items,
            output_dir=output_dir,
            verbose=True
        )
        
        if success:
            success_count += 1
            total_chunks += chunk_count
            results_summary.append({
                "file": input_file,
                "output": output_file,
                "chunks": chunk_count,
                "status": "成功"
            })
        else:
            fail_count += 1
            results_summary.append({
                "file": input_file,
                "output": None,
                "chunks": 0,
                "status": "失败",
                "error": error_msg
            })
    
    # 显示最终统计
    print(f"\n{'='*60}")
    print("批量处理完成!")
    print(f"{'='*60}")
    print(f"总文件数: {len(all_files)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总分块数: {total_chunks:,}")
    
    if fail_count > 0:
        print(f"\n失败的文件:")
        for r in results_summary:
            if r["status"] == "失败":
                print(f"  - {r['file']}: {r.get('error', '未知错误')}")
    
    print(f"\n所有输出文件保存在: {output_dir}/")


if __name__ == "__main__":
    main()


