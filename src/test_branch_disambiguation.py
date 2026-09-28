import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import numpy as np
import polars as pl
from eval_framework import load_benchmark

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

gt_val = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
s1_val = pl.read_parquet(os.path.join(VAL_DIR, "val_s1.parquet"))
s23_val = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet"))

v14_cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v14_cands.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v14_features52_X.npy"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v14_features52_df.parquet"))

import xgboost as xgb
import lightgbm as lgb

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))

p_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_lgb = lgb_model.predict(X_val)
p_ens = 0.50 * p_xgb + 0.50 * p_lgb

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
name_ratios = pairs_df["name_ratio"].to_numpy()
addr_ratios = pairs_df["address_ratio"].to_numpy()
house_matches = pairs_df["house_match"].to_numpy()

from feature_engine_v2 import V5_FEATURES_EXPANDED
idx_postal_exact = V5_FEATURES_EXPANDED.index("postal_exact")
postal_exact = X_val[:, idx_postal_exact]

gt_set = set(zip(gt_val["source1_entity_id"], gt_val["matched_entity_id"]))
total_gt = len(gt_set)

# Base gate with anti-merge
base_g = (p_ens >= 0.972) & ((name_ratios >= 0.40) | (addr_ratios >= 0.40) | (house_matches == 1))
rescue = (p_ens >= 0.90) & (((name_ratios >= 0.60) & (addr_ratios >= 0.60)) | ((name_ratios >= 0.90) & (house_matches == 1)) | (addr_ratios >= 0.95))
raw_gate = (base_g | rescue) & (name_ratios >= 0.40)

pred_set_raw = set(zip(s1_ids[raw_gate], m_ids[raw_gate]))
tp_raw = len(pred_set_raw & gt_set)
fp_raw = len(pred_set_raw - gt_set)
p_raw = tp_raw / len(pred_set_raw)
r_raw = tp_raw / total_gt
f05_raw = (1.25 * p_raw * r_raw) / (0.25 * p_raw + r_raw)
print(f"Current Anti-Merge Baseline: F0.5 = {f05_raw:.4f} | Prec: {p_raw*100:.2f}% | Rec: {r_raw*100:.2f}% | TP: {tp_raw:,} | FP: {fp_raw:,}")

# Look at false positives where house match == 0
fp_pairs = [p for p in pred_set_raw if p not in gt_set]
print(f"\nAnalyzing remaining {len(fp_pairs):,} False Positives...")

# Check house number conflict:
# When house_match == 0 AND house_number was present in both S1 and S23
# (i.e. different house numbers on the same or different street)
print("\nTesting House Conflict Penalty & Postal Discrepancy Filter...")
for drop_conflicting_house in [True, False]:
    for min_addr_when_name_high in [0.35, 0.45, 0.55]:
        # Filter: if name is very high (>=0.90) but address is near zero (<min_addr_when_name_high), it's likely a different branch in another city!
        filt = raw_gate & ~((name_ratios >= 0.85) & (addr_ratios < min_addr_when_name_high))
        
        pred_set = set(zip(s1_ids[filt], m_ids[filt]))
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
        rec = tp / total_gt if total_gt > 0 else 0.0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0
        print(f"min_addr={min_addr_when_name_high:.2f} | F0.5 = {f05:.4f} | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | TP: {tp:,} | FP: {fp:,}")
