import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import numpy as np
import polars as pl
from eval_framework import load_benchmark, compute_macro_f05

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

gt_val = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
s1_val = pl.read_parquet(os.path.join(VAL_DIR, "val_s1.parquet"))
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
idx_acronym = V5_FEATURES_EXPANDED.index("name_acronym_match")
is_acronym = X_val[:, idx_acronym]

gt_set = set(zip(gt_val["source1_entity_id"], gt_val["matched_entity_id"]))
total_gt = len(gt_set)

print("=" * 85)
print("TESTING CO-LOCATION ANTI-MERGE FILTER (Eliminating shared-building false merges)")
print("=" * 85)
print(f"{'Filter Setting':<45} | {'Global F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'TP':>7} | {'FP':>6}")
print("-" * 95)

base_g = (p_ens >= 0.972) & ((name_ratios >= 0.20) | (addr_ratios >= 0.40) | (house_matches == 1))
rescue = (p_ens >= 0.90) & (((name_ratios >= 0.60) & (addr_ratios >= 0.60)) | ((name_ratios >= 0.90) & (house_matches == 1)) | (addr_ratios >= 0.95))
raw_gate = base_g | rescue

for min_name_ratio in [0.20, 0.40, 0.50, 0.55, 0.60, 0.65]:
    # Reject if name_ratio < min_name_ratio unless it's a verified acronym or extreme score
    anti_merge_mask = (name_ratios >= min_name_ratio) | (is_acronym == 1) | (p_ens >= 0.998)
    gate = raw_gate & anti_merge_mask
    
    pred_set = set(zip(s1_ids[gate], m_ids[gate]))
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    rec = tp / total_gt if total_gt > 0 else 0.0
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0
    
    tag = f"min_name_ratio >= {min_name_ratio:.2f}"
    print(f"{tag:<45} | {f05:>11.4f} | {prec*100:>9.2f}% | {rec*100:>7.2f}% | {tp:>7,} | {fp:>6,}")

# Also test combining with stricter address requirements when name similarity is medium
for min_name in [0.55, 0.60]:
    for min_med_addr in [0.70, 0.80, 0.85]:
        # If name is between min_name and 0.80, require address >= min_med_addr
        anti_merge = (
            (name_ratios >= 0.80) |
            ((name_ratios >= min_name) & (addr_ratios >= min_med_addr)) |
            (is_acronym == 1)
        )
        gate = raw_gate & anti_merge
        pred_set = set(zip(s1_ids[gate], m_ids[gate]))
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
        rec = tp / total_gt if total_gt > 0 else 0.0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0
        tag = f"name>={min_name:.2f} (med addr>={min_med_addr:.2f})"
        print(f"{tag:<45} | {f05:>11.4f} | {prec*100:>9.2f}% | {rec*100:>7.2f}% | {tp:>7,} | {fp:>6,}")
