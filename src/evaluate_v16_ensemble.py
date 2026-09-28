import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import numpy as np
import polars as pl
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
sys.path.append(os.path.join(BASE, "src"))

from feature_engine_v2 import V5_FEATURES_EXPANDED
from eval_framework import compute_macro_f05

VAL_DIR = os.path.join(BASE, "validation_benchmark")

print("=" * 80)
print("EVALUATING V16 ENSEMBLE ON EXPANDED CANDIDATE POOL (623k PAIRS)")
print("=" * 80)

gt_val = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
s1_val = pl.read_parquet(os.path.join(VAL_DIR, "val_s1.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))

import xgboost as xgb
import lightgbm as lgb

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))

print("Computing ensemble probabilities across 623k pairs...")
p_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_lgb = lgb_model.predict(X_val)
p_ens = 0.50 * p_xgb + 0.50 * p_lgb

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt_val["source1_entity_id"], gt_val["matched_entity_id"]))
total_gt = len(gt_set)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]

print("\n" + "=" * 95)
print(f"{'Decision Policy':<45} | {'Global F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'TP':>7} | {'FP':>5}")
print("-" * 95)

# Sweep threshold combinations
for base_tau in [0.940, 0.950, 0.960, 0.970, 0.972]:
    for r_tau in [0.88, 0.90]:
        base_gate = (p_ens >= base_tau) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
        rescue1 = (p_ens >= r_tau) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
        
        # Co-location Anti-Merge Guard
        co_location = (name_r >= 0.40)
        
        # High purity rescues
        r_contain = (p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)
        r_acronym = (p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40)
        
        gate = ((base_gate | rescue1) & co_location) | r_contain | r_acronym
        
        pred_set = set(zip(s1_ids[gate], m_ids[gate]))
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        
        prec = tp / len(pred_set) if len(pred_set) > 0 else 0
        rec = tp / total_gt
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0
        
        tag = f"base={base_tau:.3f} r1={r_tau:.2f} + physical_rescue"
        print(f"{tag:<45} | {f05:>11.4f} | {prec*100:>9.2f}% | {rec*100:>7.2f}% | {tp:>7,} | {fp:>5,}")

print("=" * 95)
