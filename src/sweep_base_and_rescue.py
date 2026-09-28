import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
sys.path.insert(0, 'src')
import os
import numpy as np
import polars as pl
from eval_framework import load_benchmark, compute_macro_f05

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

s1_val, gt_val, _, _ = load_benchmark()
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

print("Sweeping Base tau + Rescue Gate combinations...")
print(f"{'Config':<45} | {'Macro F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'TP':>7} | {'FP':>6}")
print("-" * 95)

best_f = 0.0
best_cfg = ""

for base_tau in [0.93, 0.94, 0.95, 0.955, 0.96, 0.965]:
    for rescue_tau in [0.85, 0.88, 0.90]:
        for min_both in [0.65, 0.70, 0.75]:
            base_gate = (p_ens >= base_tau) & ((name_ratios >= 0.20) | (addr_ratios >= 0.40) | (house_matches == 1))
            rescue_mask = (
                (p_ens >= rescue_tau) & 
                (
                    ((name_ratios >= min_both) & (addr_ratios >= min_both)) |
                    ((name_ratios >= 0.90) & (house_matches == 1)) |
                    (addr_ratios >= 0.95)
                )
            )
            combined = base_gate | rescue_mask
            pred_df = pl.DataFrame({"source1_entity_id": s1_ids[combined], "matched_entity_id": m_ids[combined]})
            res = compute_macro_f05(pred_df, gt_val, s1_val)
            cfg = f"base={base_tau:.3f} | rescue={rescue_tau:.2f} | both={min_both:.2f}"
            if res['macro_f05'] > best_f:
                best_f = res['macro_f05']
                best_cfg = cfg
            if res['macro_f05'] >= 0.9500:
                print(f"{cfg:<45} | {res['macro_f05']:>11.4f} | {res['precision']*100:>9.2f}% | {res['recall']*100:>7.2f}% | {res['tp']:>7,} | {res['fp']:>6,}")

print(f"\nPEAK MACRO F0.5: {best_f:.4f} with {best_cfg}")
