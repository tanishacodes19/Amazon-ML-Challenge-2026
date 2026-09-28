import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import duckdb
import numpy as np
import polars as pl

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

# Load ground truth and benchmark data
gt_val = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
s1_val = pl.read_parquet(os.path.join(VAL_DIR, "val_s1.parquet"))

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

# Apply V15 Peak Decision Policy:
# base=0.972 | r1=0.90 (both>=0.60 | a>=0.95 | name>=0.90 & house=1) | r2=0.80 (name>=0.90 & addr>=0.85)
base_gate = (p_ens >= 0.972) & ((name_ratios >= 0.20) | (addr_ratios >= 0.40) | (house_matches == 1))
rescue1 = (p_ens >= 0.90) & (((name_ratios >= 0.60) & (addr_ratios >= 0.60)) | ((name_ratios >= 0.90) & (house_matches == 1)) | (addr_ratios >= 0.95))
rescue2 = (p_ens >= 0.80) & (name_ratios >= 0.90) & (addr_ratios >= 0.85)

combined_mask = base_gate | rescue1 | rescue2

pred_df = pl.DataFrame({
    "source1_entity_id": s1_ids[combined_mask],
    "matched_entity_id": m_ids[combined_mask]
})

# Calculate Pairwise TP, FP, FN
gt_set = set(zip(gt_val["source1_entity_id"], gt_val["matched_entity_id"]))
pred_set = set(zip(pred_df["source1_entity_id"], pred_df["matched_entity_id"]))

tp = len(pred_set & gt_set)
fp = len(pred_set - gt_set)
fn = len(gt_set - pred_set)
total_gt = len(gt_set)
total_pred = len(pred_set)

precision = tp / total_pred if total_pred > 0 else 0.0
recall = tp / total_gt if total_gt > 0 else 0.0

# Exact Formula provided by user:
# F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
beta_sq = 0.5 ** 2 # 0.25
f_05_formula = (1.25 * precision * recall) / (0.25 * precision + recall)

print("=" * 70)
print("EVALUATION WITH OFFICIAL F_0.5 FORMULA:")
print("  Formula: F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)")
print("=" * 70)
print(f"Total Reference GT Pairs: {total_gt:,}")
print(f"Total Predicted Pairs:    {total_pred:,}")
print(f"True Positives (TP):      {tp:,}")
print(f"False Positives (FP):     {fp:,}")
print(f"False Negatives (FN):     {fn:,}")
print("-" * 70)
print(f"Global Precision:         {precision*100:.4f}% ({precision:.6f})")
print(f"Global Recall:            {recall*100:.4f}% ({recall:.6f})")
print(f"GLOBAL F_0.5 SCORE:       {f_05_formula:.6f} ({f_05_formula*100:.2f}%)")
print("=" * 70)

# Also check per-entity Macro F0.5
from eval_framework import compute_macro_f05
res_macro = compute_macro_f05(pred_df, gt_val, s1_val)
print(f"Entity-Averaged Macro F_0.5: {res_macro['macro_f05']:.6f}")
print("=" * 70)
