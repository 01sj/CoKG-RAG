import re

from typing import Any, Union
import html
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]
def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]
async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    # 注意：对于中文法律实体，保持原始大小写（不转换为大写）
    # 因为中文法律术语的大小写有特定含义（如"第一条"不应变成"第一条"）
    entity_name_raw = clean_str(record_attributes[1])
    if not entity_name_raw.strip():
        return None
    # 对于中文实体，保持原始大小写；对于英文实体，可以转换为大写
    # 简单判断：如果包含中文字符，保持原样；否则转换为大写
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in entity_name_raw)
    entity_name = entity_name_raw if has_chinese else entity_name_raw.upper()
    
    # 实体类型：保持原始大小写（中文类型如"法条"不应转换为大写）
    entity_type_raw = clean_str(record_attributes[2])
    has_chinese_type = any('\u4e00' <= char <= '\u9fff' for char in entity_type_raw)
    entity_type = entity_type_raw if has_chinese_type else entity_type_raw.upper()
    
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key  # source_id 直接使用 chunk_key（文本块的 hash_code）
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )
def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))
async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )