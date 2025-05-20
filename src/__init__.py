"""
Package chính cho dự án phân loại bài tập
"""

import os
import sys
import importlib.util

# Import module cfg từ thư mục cha
spec = importlib.util.spec_from_file_location('cfg', os.path.join(os.path.dirname(__file__), '..', 'cfg.py'))
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
