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
s1_val = pl.read_parquet(os.path.join(VAL_DIR, "val_s1.parquet"))
s23_val = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet"))
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
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# Build text lookups
s1_dict = {row["source1_entity_id"]: row for row in s1_val.select(["source1_entity_id", "business_name", "business_address"]).to_dicts()}
s23_dict = {row["matched_entity_id"]: row for row in s23_val.select(["matched_entity_id", "business_name", "business_address"]).to_dicts()}

# Baseline gate with anti-merge
feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
base_g = (p_ens >= 0.972) & ((feat_dict["name_ratio"] >= 0.20) | (feat_dict["address_ratio"] >= 0.40) | (feat_dict["house_match"] == 1))
rescue = (p_ens >= 0.90) & (((feat_dict["name_ratio"] >= 0.60) & (feat_dict["address_ratio"] >= 0.60)) | ((feat_dict["name_ratio"] >= 0.90) & (feat_dict["house_match"] == 1)) | (feat_dict["address_ratio"] >= 0.95))
raw_gate = (base_g | rescue) & (feat_dict["name_ratio"] >= 0.40)

fp_indices = np.where(raw_gate & ~is_gt)[0]
print(f"=== SAMPLE OF 15 FALSE POSITIVES (out of {len(fp_indices):,}) ===")
for idx in fp_indices[:15]:
    s1_id = s1_ids[idx]
    m_id = m_ids[idx]
    s1_row = s1_dict.get(s1_id, {})
    s23_row = s23_dict.get(m_id, {})
    print(f"\n[FP Score: {p_ens[idx]:.4f} | NameRat: {feat_dict['name_ratio'][idx]:.2f} | AddrRat: {feat_dict['address_ratio'][idx]:.2f}]")
    print(f"  S1 : {s1_row.get('business_name')}  ||  {s1_row.get('business_address')}")
    print(f"  S23: {s23_row.get('business_name')}  ||  {s23_row.get('business_address')}")

fn_indices = np.where(~raw_gate & is_gt & (p_ens >= 0.60))[0]
print(f"\n=== SAMPLE OF 15 MISSED GROUND TRUTH (FN with score >= 0.60, out of {len(fn_indices):,}) ===")
for idx in fn_indices[:15]:
    s1_id = s1_ids[idx]
    m_id = m_ids[idx]
    s1_row = s1_dict.get(s1_id, {})
    s23_row = s23_dict.get(m_id, {})
    print(f"\n[FN Score: {p_ens[idx]:.4f} | NameRat: {feat_dict['name_ratio'][idx]:.2f} | AddrRat: {feat_dict['address_ratio'][idx]:.2f}]")
    print(f"  S1 : {s1_row.get('business_name')}  ||  {s1_row.get('business_address')}")
    print(f"  S23: {s23_row.get('business_name')}  ||  {s23_row.get('business_address')}")
