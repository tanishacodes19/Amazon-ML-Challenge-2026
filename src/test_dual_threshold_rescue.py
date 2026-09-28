import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
sys.path.insert(0, 'src')
import os
import duckdb
import numpy as np
import polars as pl
from eval_framework import load_benchmark, compute_macro_f05

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

s1_val, gt_val, _, s23_val = load_benchmark()
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

print("=" * 80)
print("TESTING DUAL-THRESHOLD PHYSICAL CORROBORATION RESCUE GATE")
print("=" * 80)

# Base V14: tau = 0.950
base_gate = (p_ens >= 0.950) & ((name_ratios >= 0.20) | (addr_ratios >= 0.40) | (house_matches == 1))
res_base = compute_macro_f05(pl.DataFrame({"source1_entity_id": s1_ids[base_gate], "matched_entity_id": m_ids[base_gate]}), gt_val, s1_val)
print(f"Base V14 (tau=0.950): Macro F0.5 = {res_base['macro_f05']:.4f} | Prec: {res_base['precision']*100:.2f}% | Rec: {res_base['recall']*100:.2f}% | TP: {res_base['tp']:,} | FP: {res_base['fp']:,} | Singl FP: {res_base['singleton_fp']}")

# Sweep rescue conditions
print("\nTesting Rescue Conditions on Top of Base (tau=0.950):")
print(f"{'Condition':<50} | {'Macro F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'TP':>7} | {'FP':>6} | {'Singl FP':>8}")
print("-" * 115)

best_f = res_base['macro_f05']
best_cond = "Base"

for rescue_tau in [0.88, 0.90, 0.92]:
    for min_both in [0.70, 0.75, 0.80]:
        for min_addr in [0.90, 0.95]:
            rescue_mask = (
                (p_ens >= rescue_tau) & 
                (
                    ((name_ratios >= min_both) & (addr_ratios >= min_both)) |
                    ((name_ratios >= 0.90) & (house_matches == 1)) |
                    (addr_ratios >= min_addr)
                )
            )
            combined_gate = base_gate | rescue_mask
            pred_df = pl.DataFrame({
                "source1_entity_id": s1_ids[combined_gate],
                "matched_entity_id": m_ids[combined_gate]
            })
            res = compute_macro_f05(pred_df, gt_val, s1_val)
            tag = f"tau>={rescue_tau:.2f} & (both>={min_both:.2f} | a>={min_addr:.2f})"
            if res['macro_f05'] > best_f:
                best_f = res['macro_f05']
                best_cond = tag
            print(f"{tag:<50} | {res['macro_f05']:>11.4f} | {res['precision']*100:>9.2f}% | {res['recall']*100:>7.2f}% | {res['tp']:>7,} | {res['fp']:>6,} | {res['singleton_fp']:>8,}")

print(f"\n*** BEST MACRO F0.5: {best_f:.4f} with {best_cond} ***")
