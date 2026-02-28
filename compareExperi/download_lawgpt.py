#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 LawGPT_zh 模型
"""

import os
from huggingface_hub import snapshot_download

LOCAL_DIR = "/newdatae/model/LawGPT_zh"

print("="*70)
print("下载 LawGPT_zh 中文法律模型")
print("="*70)
print(f"\n模型: Dorado607/LawGPT_zh")
print(f"保存路径: {LOCAL_DIR}")
print(f"\n⏳ 开始下载...\n")

try:
    model_dir = snapshot_download(
        repo_id="Dorado607/LawGPT_zh",
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    print("\n" + "="*70)
    print("✅ 下载完成！")
    print("="*70)
    print(f"模型路径: {model_dir}")
    
    print(f"\n📝 测试命令:")
    print(f"python compareExperi/test_lawgpt.py --model-path {LOCAL_DIR}")
    
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n💡 如果网络问题，可以使用镜像:")
    print("export HF_ENDPOINT=https://hf-mirror.com")
