import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import duckdb
import numpy as np
import polars as pl
from eval_framework import load_benchmark
from feature_engine_v2 import V5_FEATURES_EXPANDED

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

s1_val, gt_val, _, s23_val = load_benchmark()
v8_cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v8_cands.parquet"))
X_v8 = np.load(os.path.join(VAL_DIR, "val_v8_features52_X.npy"))

import xgboost as xgb
clf = xgb.XGBClassifier()
clf.load_model(os.path.join(BASE, "model", "xgboost_v8_52features.json"))
probs = clf.predict_proba(X_v8)[:, 1]

gt_set = set(zip(gt_val["source1_entity_id"], gt_val["matched_entity_id"]))
s1_with_gt = set(gt_val["source1_entity_id"])
all_s1 = set(s1_val["source1_entity_id"])
singletons = all_s1 - s1_with_gt

v8_scored = v8_cands.with_columns([
    pl.Series("prob", probs),
    pl.Series("idx", np.arange(len(probs)))
]).filter(pl.col("prob") >= 0.900)

singleton_fps = v8_scored.filter(pl.col("source1_entity_id").is_in(list(singletons)))
print(f"Total singleton false positives: {len(singleton_fps)} across {singleton_fps['source1_entity_id'].n_unique()} entities")

# Inspect feature values for singleton FPs
fp_indices = singleton_fps["idx"].to_numpy()
fp_features = X_v8[fp_indices]

# Compare to TP features
tp_cands = v8_scored.filter(~pl.col("source1_entity_id").is_in(list(singletons)))
tp_indices = tp_cands["idx"].to_numpy()[:len(fp_indices)*5]
tp_features = X_v8[tp_indices]

# Print feature means
print("\n--- Feature Mean Comparison: Singleton FPs vs True Positives ---")
for feat in ["name_ratio", "address_ratio", "name_token_jaccard", "address_token_jaccard", "house_match", "postal_exact", "name_exact", "name_char4_jaccard", "address_char4_jaccard", "strong_name_address", "strong_name_house"]:
    fi = V5_FEATURES_EXPANDED.index(feat)
    fp_m = np.mean(fp_features[:, fi])
    tp_m = np.mean(tp_features[:, fi])
    print(f"{feat:<26} | FP Mean: {fp_m:.3f} | TP Mean: {tp_m:.3f}")

# Test simple suppression rules on singleton FPs
print("\n--- Testing Gated Decision Rules on Macro F0.5 ---")
from eval_framework import compute_macro_f05

# Rule 1: Baseline at 0.900
res_base = compute_macro_f05(v8_scored.select(["source1_entity_id", "matched_entity_id"]), gt_val, s1_val)
print(f"Base @ 0.900: Macro F0.5 = {res_base['macro_f05']:.4f} | Prec: {res_base['precision']*100:.2f}% | Rec: {res_base['recall']*100:.2f}% | S_FP: {res_base['singleton_fp']}")

# Rule 2: Gated match - require prob >= 0.90 AND (name_ratio >= 0.40 OR addr_ratio >= 0.40 OR house_match == 1)
name_ratio_col = X_v8[:, V5_FEATURES_EXPANDED.index("name_ratio")]
addr_ratio_col = X_v8[:, V5_FEATURES_EXPANDED.index("address_ratio")]
h_match_col = X_v8[:, V5_FEATURES_EXPANDED.index("house_match")]
p_match_col = X_v8[:, V5_FEATURES_EXPANDED.index("postal_exact")]
tok_j_col = X_v8[:, V5_FEATURES_EXPANDED.index("name_token_jaccard")]

for min_name in [0.25, 0.30, 0.35, 0.40]:
    mask = (probs >= 0.90) & ((name_ratio_col >= min_name) | (addr_ratio_col >= 0.50) | (h_match_col == 1))
    res = compute_macro_f05(v8_cands.filter(mask).select(["source1_entity_id", "matched_entity_id"]), gt_val, s1_val)
    print(f"Gated (name >= {min_name:.2f} or addr >= 0.50): Macro F0.5 = {res['macro_f05']:.4f} | Prec: {res['precision']*100:.2f}% | Rec: {res['recall']*100:.2f}% | S_FP: {res['singleton_fp']}")
