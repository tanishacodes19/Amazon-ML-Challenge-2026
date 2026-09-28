import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'src')
import os, numpy as np, polars as pl
from eval_framework import load_benchmark, compute_macro_f05

VAL_DIR = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026\validation_benchmark"
s1_val, gt_val, cands_val, s23_val = load_benchmark()

# Let's inspect what files exist in VAL_DIR
print("Files in VAL_DIR:", os.listdir(VAL_DIR))
