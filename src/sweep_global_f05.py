import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import numpy as np
import polars as pl

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

gt_set = set(zip(gt_val["source1_entity_id"], gt_val["matched_entity_id"]))
total_gt = len(gt_set)

print("=" * 80)
print(f"{'Threshold / Gate':<45} | {'Global F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'TP':>7} | {'FP':>6}")
print("-" * 80)

best_f = 0.0
best_tau = 0.0

for tau in [0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.965, 0.97, 0.975, 0.98]:
    gate = (p_ens >= tau) & ((name_ratios >= 0.20) | (addr_ratios >= 0.40) | (house_matches == 1))
    s_s1 = s1_ids[gate]
    s_m = m_ids[gate]
    pred_set = set(zip(s_s1, s_m))
    
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    
    prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    rec = tp / total_gt if total_gt > 0 else 0.0
    
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0
    
    if f05 > best_f:
        best_f = f05
        best_tau = tau
        
    print(f"tau = {tau:>5.3f}                                     | {f05:>11.4f} | {prec*100:>9.2f}% | {rec*100:>7.2f}% | {tp:>7,} | {fp:>6,}")

# Test with multi-tier rescue
for base_tau in [0.96, 0.965, 0.97, 0.972]:
    for r_tau in [0.88, 0.90]:
        base_g = (p_ens >= base_tau) & ((name_ratios >= 0.20) | (addr_ratios >= 0.40) | (house_matches == 1))
        rescue = (p_ens >= r_tau) & (((name_ratios >= 0.60) & (addr_ratios >= 0.60)) | ((name_ratios >= 0.90) & (house_matches == 1)) | (addr_ratios >= 0.95))
        gate = base_g | rescue
        pred_set = set(zip(s1_ids[gate], m_ids[gate]))
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
        rec = tp / total_gt if total_gt > 0 else 0.0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0
        tag = f"base={base_tau:.3f} + rescue={r_tau:.2f}"
        if f05 > best_f:
            best_f = f05
            best_tau = tag
        print(f"{tag:<45} | {f05:>11.4f} | {prec*100:>9.2f}% | {rec*100:>7.2f}% | {tp:>7,} | {fp:>6,}")

print(f"\nPEAK GLOBAL F_0.5: {best_f:.6f} with {best_tau}")
