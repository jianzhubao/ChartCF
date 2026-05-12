"""
sitecustomize.py - Swift RLHF Patches
"""

import sys

# 导入 patch 函数
from src.chartcf.patches import apply_all_patches

# 应用所有补丁
patches_applied = apply_all_patches()

if patches_applied:
    print(f"[sitecustomize.py] ✓ Swift RLHF patches applied: {', '.join(patches_applied)}",
          file=sys.stderr, flush=True)
