import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import duckdb
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

# Predictions at tau=0.972 + rescue=0.90
base_g = (p_ens >= 0.972) & ((name_ratios >= 0.20) | (addr_ratios >= 0.40) | (house_matches == 1))
rescue = (p_ens >= 0.90) & (((name_ratios >= 0.60) & (addr_ratios >= 0.60)) | ((name_ratios >= 0.90) & (house_matches == 1)) | (addr_ratios >= 0.95))
mask = base_g | rescue

pred_df = pl.DataFrame({
    "source1_entity_id": s1_ids[mask],
    "matched_entity_id": m_ids[mask],
    "score": p_ens[mask],
    "name_ratio": name_ratios[mask],
    "addr_ratio": addr_ratios[mask],
    "house_match": house_matches[mask]
})

gt_set = set(zip(gt_val["source1_entity_id"], gt_val["matched_entity_id"]))

# Find False Positives
fps = []
for row in pred_df.iter_rows(named=True):
    pair = (row["source1_entity_id"], row["matched_entity_id"])
    if pair not in gt_set:
        fps.append(row)

fps_df = pl.DataFrame(fps)
print(f"Total False Positives: {len(fps_df):,}")

# Join with text to inspect
s1_lookup = {r[0]: (r[1], r[2], r[3]) for r in s1_val.select(["source1_entity_id", "business_name", "business_address", "country"]).iter_rows()}
s23_lookup = {r[0]: (r[1], r[2], r[3]) for r in s23_val.select(["matched_entity_id", "business_name", "business_address", "country"]).iter_rows()}

print("\n--- SAMPLE 20 FALSE POSITIVES (Why did the model merge them?) ---")
for i, r in enumerate(fps_df.head(20).iter_rows(named=True), 1):
    s1_id = r["source1_entity_id"]
    m_id = r["matched_entity_id"]
    s1_info = s1_lookup.get(s1_id, ("?", "?", "?"))
    m_info = s23_lookup.get(m_id, ("?", "?", "?"))
    print(f"\n[FP #{i}] Score: {r['score']:.4f} | NameRat: {r['name_ratio']:.2f} | AddrRat: {r['addr_ratio']:.2f} | HouseMatch: {r['house_match']}")
    print(f"  S1:  {s1_info[0]} | {s1_info[1]}")
    print(f"  S23: {m_info[0]} | {m_info[1]}")
