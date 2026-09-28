import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
sys.path.append(os.path.join(BASE, "src"))

import numpy as np
import polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

VAL_DIR = os.path.join(BASE, "validation_benchmark")
gt_val = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v14_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v14_features52_X.npy"))

import xgboost as xgb
import lightgbm as lgb

print("Loading models...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))

print("Computing ensemble probabilities...")
p_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_lgb = lgb_model.predict(X_val)
p_ens = 0.50 * p_xgb + 0.50 * p_lgb

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt_val["source1_entity_id"], gt_val["matched_entity_id"]))
total_gt = len(gt_set)

# Map all 52 features to convenient numpy arrays
feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}

# Identify ground truth label for every candidate in pool
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

print(f"Total candidate pairs: {len(is_gt):,}")
print(f"Total GT pairs in candidate pool: {np.sum(is_gt):,} / {total_gt:,} ({np.sum(is_gt)/total_gt*100:.2f}%)")

# Base filter (p_ens >= 0.95 and standard gates)
# Current baseline:
base_gate = (p_ens >= 0.972) & (
    (feat_dict["name_ratio"] >= 0.40) | 
    (feat_dict["name_acronym_match"] == 1)
) & (
    (feat_dict["name_ratio"] >= 0.20) | 
    (feat_dict["address_ratio"] >= 0.40) | 
    (feat_dict["house_match"] == 1)
)

cur_tp = np.sum(base_gate & is_gt)
cur_fp = np.sum(base_gate & ~is_gt)
cur_p = cur_tp / (cur_tp + cur_fp)
cur_r = cur_tp / total_gt
cur_f05 = (1.25 * cur_p * cur_r) / (0.25 * cur_p + cur_r)
print(f"Current Base: TP={cur_tp:,}, FP={cur_fp:,}, P={cur_p*100:.2f}%, R={cur_r*100:.2f}%, F0.5={cur_f05:.4f}")

# Analyze what characterizes False Positives in base_gate
fp_indices = np.where(base_gate & ~is_gt)[0]
print(f"\nAnalyzing {len(fp_indices):,} False Positives...")

# Potential anti-FP rules:
# Rule A: Distinct Brand Token Conflict
# If name_first_token_match == 0 AND name_ratio < 0.70 AND name_token_set < 0.70 AND acronym == 0
fp_cond_a = (feat_dict["name_first_token_match"] == 0) & (feat_dict["name_ratio"] < 0.65) & (feat_dict["name_token_set"] < 0.70) & (feat_dict["name_acronym_match"] == 0)
print(f"Rule A (Brand Prefix Mismatch with low set ratio): drops {np.sum(fp_cond_a & base_gate & ~is_gt)} FP, but drops {np.sum(fp_cond_a & base_gate & is_gt)} TP")

# Rule B: Missing address mismatch
# If address_ratio < 0.20 and house_match == 0 and postal_exact == 0 and name_ratio < 0.90
fp_cond_b = (feat_dict["address_ratio"] < 0.20) & (feat_dict["house_match"] == 0) & (feat_dict["postal_exact"] == 0) & (feat_dict["name_ratio"] < 0.90)
print(f"Rule B (Zero Address Corroboration): drops {np.sum(fp_cond_b & base_gate & ~is_gt)} FP, drops {np.sum(fp_cond_b & base_gate & is_gt)} TP")

# Rule C: Co-located business (Address very high >= 0.85, but Name is low < 0.50)
fp_cond_c = (feat_dict["address_ratio"] >= 0.85) & (feat_dict["name_ratio"] < 0.50) & (feat_dict["name_acronym_match"] == 0)
print(f"Rule C (Co-located company): drops {np.sum(fp_cond_c & base_gate & ~is_gt)} FP, drops {np.sum(fp_cond_c & base_gate & is_gt)} TP")

# Rule D: Different House Numbers when both have house number
# (i.e. house_match == 0, address_len > 10, but address contains digits that don't match)
fp_cond_d = (feat_dict["address_numeric_overlap"] == 0) & (feat_dict["address_ratio"] < 0.60) & (feat_dict["name_ratio"] < 0.80)
print(f"Rule D (Conflicting Numeric/House): drops {np.sum(fp_cond_d & base_gate & ~is_gt)} FP, drops {np.sum(fp_cond_d & base_gate & is_gt)} TP")

# Analyze Missed Ground Truth (FN) that are in the candidate pool:
fn_indices = np.where(~base_gate & is_gt)[0]
print(f"\nAnalyzing {len(fn_indices):,} Missed True Matches in candidate pool...")

# Test Rescue Rules:
# Rescue 1: High Name + High Postal
r1 = (~base_gate) & (feat_dict["name_ratio"] >= 0.80) & (feat_dict["postal_exact"] == 1) & (p_ens >= 0.60)
print(f"Rescue 1 (Name >= 0.80 + Postal Exact): adds {np.sum(r1 & is_gt)} TP, adds {np.sum(r1 & ~is_gt)} FP")

# Rescue 2: High Name + House Match
r2 = (~base_gate) & (feat_dict["name_ratio"] >= 0.75) & (feat_dict["house_match"] == 1) & (p_ens >= 0.60)
print(f"Rescue 2 (Name >= 0.75 + House Match): adds {np.sum(r2 & is_gt)} TP, adds {np.sum(r2 & ~is_gt)} FP")

# Rescue 3: High Token Set + High Address Ratio
r3 = (~base_gate) & (feat_dict["name_token_set"] >= 0.85) & (feat_dict["address_ratio"] >= 0.75) & (p_ens >= 0.60)
print(f"Rescue 3 (Token Set >= 0.85 + Addr >= 0.75): adds {np.sum(r3 & is_gt)} TP, adds {np.sum(r3 & ~is_gt)} FP")

# Rescue 4: Strong Cross feature
r4 = (~base_gate) & (feat_dict["strong_name_address"] == 1) & (p_ens >= 0.70)
print(f"Rescue 4 (Strong Name-Address Cross): adds {np.sum(r4 & is_gt)} TP, adds {np.sum(r4 & ~is_gt)} FP")

# Rescue 5: Acronym Match with Address
r5 = (~base_gate) & (feat_dict["name_acronym_match"] == 1) & (feat_dict["address_ratio"] >= 0.60) & (p_ens >= 0.60)
print(f"Rescue 5 (Acronym Match + Addr >= 0.60): adds {np.sum(r5 & is_gt)} TP, adds {np.sum(r5 & ~is_gt)} FP")
