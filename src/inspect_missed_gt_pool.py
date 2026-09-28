import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import os, numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))
s1 = pl.read_parquet(os.path.join(VAL_DIR, "val_s1_v13_clean.parquet"))
s23 = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet"))

import xgboost as xgb, lightgbm as lgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))
p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
postal_e = feat_dict["postal_exact"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]

# V15 gate
base_g = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
v15_gate = ((base_g | rescue1) & (name_r >= 0.40)) | ((p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)) | ((p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40))

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# Missed GT pairs that ARE in the candidate pool!
missed_gt_indices = np.where(is_gt & ~v15_gate)[0]
print(f"Total True Positives in candidate pool: {is_gt.sum():,}")
print(f"Captured by V15 gate: {(is_gt & v15_gate).sum():,}")
print(f"MISSED True Positives in candidate pool: {len(missed_gt_indices):,}")

# What are their probabilities?
missed_probs = p_ens[missed_gt_indices]
print(f"\nProbability distribution of missed GT:")
for q in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  p{q:02d}: {np.percentile(missed_probs, q):.4f}")

# How many have p_ens >= 0.80? >= 0.50? >= 0.10?
for thresh in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.30, 0.10]:
    c = (missed_probs >= thresh).sum()
    print(f"  Missed GT with p_ens >= {thresh:.2f}: {c:,} / {len(missed_gt_indices):,} ({c/len(missed_gt_indices):.2%})")

# Look at feature stats of missed GT:
print(f"\nFeature averages of missed GT:")
print(f"  name_ratio mean: {name_r[missed_gt_indices].mean():.3f} (median: {np.median(name_r[missed_gt_indices]):.3f})")
print(f"  addr_ratio mean: {addr_r[missed_gt_indices].mean():.3f} (median: {np.median(addr_r[missed_gt_indices]):.3f})")
print(f"  house_match == 1: {(house_m[missed_gt_indices] == 1).sum():,} ({((house_m[missed_gt_indices] == 1).mean()):.2%})")
print(f"  postal_exact == 1: {(postal_e[missed_gt_indices] == 1).sum():,} ({((postal_e[missed_gt_indices] == 1).mean()):.2%})")

# Sample 15 missed GT
s1_dict = {r["source1_entity_id"]: r for r in s1.to_dicts()}
s23_dict = {r["matched_entity_id"]: r for r in s23.to_dicts()}

print("\nSample 15 Missed GT Pairs:")
for idx in missed_gt_indices[:15]:
    sid = s1_ids[idx]; mid = m_ids[idx]
    r1 = s1_dict.get(sid, {})
    r2 = s23_dict.get(mid, {})
    print(f"\n[p_ens: {p_ens[idx]:.4f} | NameRat: {name_r[idx]:.2f} | AddrRat: {addr_r[idx]:.2f} | House: {house_m[idx]} | Postal: {postal_e[idx]}]")
    print(f"  S1 : {r1.get('business_name')}  ||  {r1.get('business_address')}")
    print(f"  S23: {r2.get('business_name')}  ||  {r2.get('business_address')}")
