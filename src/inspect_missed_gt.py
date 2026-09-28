import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')
s1 = pl.read_parquet('validation_benchmark/val_s1_v13_clean.parquet')
s23 = pl.read_parquet('validation_benchmark/val_s23_v13_clean.parquet')

import xgboost as xgb, lightgbm as lgb
xgb_v13 = xgb.XGBClassifier()
xgb_v13.load_model('model/xgboost_v13_52features.json')
lgb_v13 = lgb.Booster(model_file='model/lightgbm_v13_52features.txt')
p_ens_v13 = 0.50 * xgb_v13.predict_proba(X_val)[:, 1] + 0.50 * lgb_v13.predict(X_val)

# Also load V17
xgb_v17 = xgb.XGBClassifier()
xgb_v17.load_model('model/xgboost_v17_52features.json')
lgb_v17 = lgb.Booster(model_file='model/lightgbm_v17_52features.txt')
p_ens_v17 = 0.50 * xgb_v17.predict_proba(X_val)[:, 1] + 0.50 * lgb_v17.predict(X_val)

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

total_gt = len(gt_set)
gt_in_pool = is_gt.sum()
print(f"Total GT pairs: {total_gt:,}")
print(f"GT pairs in candidate pool: {gt_in_pool:,} ({gt_in_pool/total_gt*100:.2f}%)")

# Currently selected in V16 champion:
# p_v13 >= 0.90 (or similar) with injective matching
# Let's inspect the GT pairs that are in the candidate pool but have p_v13 < 0.90
gt_pool_idx = np.where(is_gt)[0]
missed_at_90 = gt_pool_idx[p_ens_v13[gt_pool_idx] < 0.90]
print(f"GT pairs in pool with p_v13 < 0.90: {len(missed_at_90):,}")

# Distribution of p_v13 and p_v17 for these missed GTs:
p_v13_missed = p_ens_v13[missed_at_90]
p_v17_missed = p_ens_v17[missed_at_90]

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
name_token = feat_dict["name_token_set"]
addr_token = feat_dict["address_token_set"]

s1_dict = {r["source1_entity_id"]: r for r in s1.to_dicts()}
s23_dict = {r["matched_entity_id"]: r for r in s23.to_dicts()}

print("\n--- Breakdown of Missed GTs (p_v13 < 0.90) ---")
print(f"Address token set == 0.0: {(addr_token[missed_at_90] == 0.0).sum():,} / {len(missed_at_90):,}")
print(f"Address ratio < 0.50: {(addr_r[missed_at_90] < 0.50).sum():,} / {len(missed_at_90):,}")
print(f"Name token set >= 0.80: {(name_token[missed_at_90] >= 0.80).sum():,} / {len(missed_at_90):,}")
print(f"Name ratio >= 0.80: {(name_r[missed_at_90] >= 0.80).sum():,} / {len(missed_at_90):,}")

print("\nSample 20 Missed GT Pairs (where name is strong but p < 0.90):")
strong_name_missed = missed_at_90[name_token[missed_at_90] >= 0.80]
for idx in strong_name_missed[:20]:
    sid = s1_ids[idx]; mid = m_ids[idx]
    r1 = s1_dict.get(sid, {})
    r2 = s23_dict.get(mid, {})
    print(f"\n[p_v13: {p_ens_v13[idx]:.4f} | p_v17: {p_ens_v17[idx]:.4f} | NameTok: {name_token[idx]:.2f} | AddrTok: {addr_token[idx]:.2f} | AddrRat: {addr_r[idx]:.2f}]")
    print(f"  S1 : [{sid}] {r1.get('business_name')}  ||  {r1.get('business_address')}")
    print(f"  S23: [{mid}] {r2.get('business_name')}  ||  {r2.get('business_address')}")
