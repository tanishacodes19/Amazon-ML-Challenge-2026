import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import time
import numpy as np
import polars as pl
import xgboost as xgb
from eval_framework import load_benchmark, compute_macro_f05
from feature_engine_v2 import V5_FEATURES_EXPANDED

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
MODEL_PATH = os.path.join(BASE, "model", "xgboost_v8_52features.json")
V9_CANDS_PATH = os.path.join(VAL_DIR, "val_v9_cands.parquet")
FEAT_CACHE_X = os.path.join(VAL_DIR, "val_v9_features52_X.npy")

print("=" * 70)
print("EVALUATING V9 CANDIDATE SET (87.28% BLOCKING RECALL) WITH MODEL V8")
print("=" * 70)

t0 = time.time()
s1_val, gt_val, _, _ = load_benchmark()
cands = pl.read_parquet(V9_CANDS_PATH)
print(f"Loaded {len(cands):,} V9 candidates in {time.time()-t0:.2f}s")

print(f"Loading cached 52 features from {FEAT_CACHE_X}...")
X_val = np.load(FEAT_CACHE_X)
print(f"X_val shape: {X_val.shape}")

# Feature indices
idx_name_ratio = V5_FEATURES_EXPANDED.index("name_ratio")
idx_addr_ratio = V5_FEATURES_EXPANDED.index("address_ratio")
idx_house_match = V5_FEATURES_EXPANDED.index("house_match")

name_ratios = X_val[:, idx_name_ratio]
addr_ratios = X_val[:, idx_addr_ratio]
house_matches = X_val[:, idx_house_match]

s1_ids = cands["source1_entity_id"].to_numpy()
m_ids = cands["matched_entity_id"].to_numpy()

# Load Model V8
print(f"Loading Model V8 from {MODEL_PATH}...")
bst = xgb.Booster()
bst.load_model(MODEL_PATH)

t_pred = time.time()
dval = xgb.DMatrix(X_val)
probs = bst.predict(dval)
print(f"Predictions generated in {time.time()-t_pred:.2f}s. Range: [{probs.min():.4f}, {probs.max():.4f}]")

all_s1 = set(s1_val["source1_entity_id"])

print("\n" + "=" * 70)
print(f"{'Threshold':>10} | {'Macro F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'TP':>7} | {'FP':>6} | {'Singl FP':>8}")
print("=" * 70)

best_score = 0.0
best_tau = 0.0

for tau in [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]:
    # Gating rule: High prob + physical corroboration
    gate = (probs >= tau) & ((name_ratios >= 0.25) | (addr_ratios >= 0.50) | (house_matches == 1))
    
    pred_s1 = s1_ids[gate]
    pred_m = m_ids[gate]
    
    pred_df = pl.DataFrame({
        "source1_entity_id": pred_s1,
        "matched_entity_id": pred_m
    })
    
    metrics = compute_macro_f05(pred_df, gt_val, s1_val)
    f05 = metrics["macro_f05"]
    p = metrics["precision"] * 100.0
    r = metrics["recall"] * 100.0
    tp = metrics["tp"]
    fp = metrics["fp"]
    singl_fp = metrics["singleton_fp"]
    
    if f05 > best_score:
        best_score = f05
        best_tau = tau
        
    print(f"{tau:>10.3f} | {f05:>11.4f} | {p:>9.2f}% | {r:>7.2f}% | {tp:>7,} | {fp:>6,} | {singl_fp:>8,}")

print("=" * 70)
print(f"PEAK MACRO F0.5 ON V9 CANDIDATES: {best_score:.4f} at tau={best_tau:.3f}")
print(f"Comparison: V7 Peak = 0.8691 | V8 Peak = 0.8833 | V9 Peak = {best_score:.4f}")
print("=" * 70)
