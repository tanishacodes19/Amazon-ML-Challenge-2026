import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import time
import numpy as np
import polars as pl
from eval_framework import load_benchmark, compute_macro_f05
import xgboost as xgb
import lightgbm as lgb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

s1_val, gt_val, _, s23_val = load_benchmark()
cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v12_cands.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v12_features52_X.npy"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v12_features52_df.parquet"))

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v11_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v11_52features.txt"))

p_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_lgb = lgb_model.predict(X_val)
p_ens = 0.60 * p_xgb + 0.40 * p_lgb

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
name_ratios = pairs_df["name_ratio"].to_numpy()
addr_ratios = pairs_df["address_ratio"].to_numpy()
house_matches = pairs_df["house_match"].to_numpy()

print("=" * 70)
print("TESTING SINGLETON SHIELD & ENTITY-LEVEL MARGINAL CALIBRATION")
print("=" * 70)

# Baseline at tau=0.920
base_mask = (p_ens >= 0.920) & ((name_ratios >= 0.25) | (addr_ratios >= 0.50) | (house_matches == 1))
res_base = compute_macro_f05(pl.DataFrame({"source1_entity_id": s1_ids[base_mask], "matched_entity_id": m_ids[base_mask]}), gt_val, s1_val)
print(f"Baseline (tau=0.920): Macro F0.5 = {res_base['macro_f05']:.4f} | Prec = {res_base['precision']*100:.2f}% | Rec = {res_base['recall']*100:.2f}% | Singl FP = {res_base['singleton_fp']}")

# Test 1: Singleton Shield
# If an S1 entity has only 1 predicted match with 0.920 <= P < tau_margin, require stronger corroboration
for tau_margin in [0.93, 0.94, 0.95]:
    for min_single_ratio in [0.40, 0.50, 0.60]:
        # Count candidate matches per S1 at base_mask
        df_base = pl.DataFrame({"source1_entity_id": s1_ids[base_mask], "matched_entity_id": m_ids[base_mask], "prob": p_ens[base_mask], "name_r": name_ratios[base_mask], "addr_r": addr_ratios[base_mask], "house": house_matches[base_mask]})
        s1_counts = df_base.group_by("source1_entity_id").agg(pl.count().alias("pred_count"))
        df_joined = df_base.join(s1_counts, on="source1_entity_id")
        
        # Shield condition: if pred_count == 1 and prob < tau_margin, must satisfy (name_r >= min_single_ratio or house == 1)
        shield_drop = (df_joined["pred_count"] == 1) & (df_joined["prob"] < tau_margin) & (df_joined["name_r"] < min_single_ratio) & (df_joined["house"] != 1)
        df_filtered = df_joined.filter(~shield_drop)
        
        res = compute_macro_f05(df_filtered.select(["source1_entity_id", "matched_entity_id"]), gt_val, s1_val)
        if res["macro_f05"] >= res_base["macro_f05"]:
            print(f"Shield (margin={tau_margin:.2f}, min_ratio={min_single_ratio:.2f}) -> Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['precision']*100:.2f}% | Singl FP: {res['singleton_fp']} (dropped {shield_drop.sum():,} risky singles)")

print("\n" + "=" * 70)
