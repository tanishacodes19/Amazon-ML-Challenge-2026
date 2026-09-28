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

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))
p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt_val["source1_entity_id"], gt_val["matched_entity_id"]))
total_gt = len(gt_set)
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}

name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house = feat_dict["house_match"]
postal = feat_dict["postal_exact"]
name_ex = feat_dict["name_exact"]
name_contains = feat_dict["name_contains"]
acronym = feat_dict["name_acronym_match"]
token_set = feat_dict["name_token_set"]
addr_set = feat_dict["address_token_set"]

# Clean baseline with anti-merge
base_g = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house == 1))
rescue_old = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house == 1)) | (addr_r >= 0.95))
raw_gate = (base_g | rescue_old) & (name_r >= 0.40)

def test_rule(name, rule_mask):
    tp = np.sum(rule_mask & is_gt)
    fp = np.sum(rule_mask & ~is_gt)
    purity = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"{name:<55} | TP={tp:>5,} | FP={fp:>4,} | Purity={purity*100:>6.2f}%")

print("=" * 85)
print("TESTING CANDIDATE RESCUES OUTSIDE THE BASELINE GATE")
print("=" * 85)
outside = ~raw_gate

test_rule("1. Single Brand Name Containment (name_contains & addr>=0.65 & p>=0.95)", outside & (name_contains == 1) & (addr_r >= 0.65) & (p_ens >= 0.95))
test_rule("2. Acronym Match (acronym == 1 & addr>=0.40 & p>=0.90)", outside & (acronym == 1) & (addr_r >= 0.40) & (p_ens >= 0.90))
test_rule("3. Name Exact Match (name_exact == 1 & addr>=0.30 & p>=0.80)", outside & (name_ex == 1) & (addr_r >= 0.30) & (p_ens >= 0.80))
test_rule("4. Name Exact Match + House Match (name_exact & house & p>=0.50)", outside & (name_ex == 1) & (house == 1) & (p_ens >= 0.50))
test_rule("5. Name Exact Match + Postal Match (name_exact & postal & p>=0.50)", outside & (name_ex == 1) & (postal == 1) & (p_ens >= 0.50))
test_rule("6. Token Set 1.0 + Addr Set >= 0.80 (p>=0.85)", outside & (token_set >= 0.98) & (addr_set >= 0.80) & (p_ens >= 0.85))
test_rule("7. High Name (>=0.85) + House Match (house==1 & p>=0.85)", outside & (name_r >= 0.85) & (house == 1) & (p_ens >= 0.85))
test_rule("8. High Name (>=0.85) + High Addr (>=0.85) & p>=0.85", outside & (name_r >= 0.85) & (addr_r >= 0.85) & (p_ens >= 0.85))

# Combine all pure rules (purity >= 95%)
pure_rescues = (
    ((name_contains == 1) & (addr_r >= 0.65) & (p_ens >= 0.95)) |
    ((acronym == 1) & (addr_r >= 0.40) & (p_ens >= 0.90)) |
    ((name_ex == 1) & (addr_r >= 0.30) & (p_ens >= 0.80)) |
    ((name_ex == 1) & (house == 1) & (p_ens >= 0.50)) |
    ((name_ex == 1) & (postal == 1) & (p_ens >= 0.50)) |
    ((token_set >= 0.98) & (addr_set >= 0.80) & (p_ens >= 0.85)) |
    ((name_r >= 0.85) & (addr_r >= 0.85) & (p_ens >= 0.85))
)

all_gate = raw_gate | pure_rescues
pred_set = set(zip(s1_ids[all_gate], m_ids[all_gate]))
tp = len(pred_set & gt_set)
fp = len(pred_set - gt_set)
prec = tp / len(pred_set)
rec = tp / total_gt
f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
print("-" * 85)
print(f"COMBINED PEAK RESULT: F0.5 = {f05:.6f} | Prec = {prec*100:.3f}% | Rec = {rec*100:.3f}% | TP = {tp:,} | FP = {fp:,}")
