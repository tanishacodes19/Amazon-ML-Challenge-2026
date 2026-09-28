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
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('model/xgboost_v13_52features.json')
lgb_model = lgb.Booster(model_file='model/lightgbm_v13_52features.txt')
p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# Baseline acceptance
feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]

base_gate = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
co_location = (name_r >= 0.40)
r_contain = (p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)
r_acronym = (p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40)
v16_mask = ((base_gate | rescue1) & co_location) | r_contain | r_acronym

# Check S23 address presence
s23_addr_map = dict(zip(s23['matched_entity_id'], s23['clean_addr']))
s23_addrs = [s23_addr_map.get(m, '') for m in m_ids]
has_s23_addr = np.array([a is not None and len(str(a).strip()) > 3 and str(a).strip() != 'nan' for a in s23_addrs], dtype=bool)

missed_with_addr = is_gt & (~v16_mask) & has_s23_addr
print(f"Total Missed GT with VALID address: {missed_with_addr.sum():,}")

s1_dict = {r["source1_entity_id"]: r for r in s1.to_dicts()}
s23_dict = {r["matched_entity_id"]: r for r in s23.to_dicts()}

missed_idx = np.where(missed_with_addr)[0]

# Distribution of p_ens
p_missed = p_ens[missed_idx]
print(f"p_ens distribution for missed GT with valid address:")
for thresh in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]:
    print(f"  p_ens >= {thresh:.2f}: {(p_missed >= thresh).sum():,} / {len(missed_idx):,}")

print("\nSample 20 Missed GT Pairs with VALID Address:")
# Sort by highest p_ens first
high_p_first = missed_idx[np.argsort(-p_ens[missed_idx])]
for idx in high_p_first[:20]:
    sid = s1_ids[idx]; mid = m_ids[idx]
    r1 = s1_dict.get(sid, {})
    r2 = s23_dict.get(mid, {})
    print(f"\n[p_ens: {p_ens[idx]:.4f} | NameRat: {name_r[idx]:.2f} | AddrRat: {addr_r[idx]:.2f} | House: {house_m[idx]}]")
    print(f"  S1 : [{sid}] {r1.get('business_name')}  ||  {r1.get('business_address')}")
    print(f"  S23: [{mid}] {r2.get('business_name')}  ||  {r2.get('business_address')}")
