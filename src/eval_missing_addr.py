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

xgb_v17 = xgb.XGBClassifier()
xgb_v17.load_model('model/xgboost_v17_52features.json')
lgb_v17 = lgb.Booster(model_file='model/lightgbm_v17_52features.txt')
p_ens_v17 = 0.50 * xgb_v17.predict_proba(X_val)[:, 1] + 0.50 * lgb_v17.predict(X_val)

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# Let's check s23 address presence
s23_addr_map = dict(zip(s23['matched_entity_id'], s23['clean_addr']))
s23_names = dict(zip(s23['matched_entity_id'], s23['clean_name']))
s1_names = dict(zip(s1['source1_entity_id'], s1['clean_name']))

# Count how many times each clean_name appears in S1
from collections import Counter
s1_name_counts = Counter(s1['clean_name'])

# Check for each pair if s23 has missing address
s23_addrs = [s23_addr_map.get(m, '') for m in m_ids]
has_s23_addr = np.array([a is not None and len(str(a).strip()) > 3 and str(a).strip() != 'nan' for a in s23_addrs], dtype=bool)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_tok = feat_dict["name_token_set"]
name_rat = feat_dict["name_ratio"]
addr_rat = feat_dict["address_ratio"]

print(f"Total pairs in candidate pool: {len(is_gt):,}")
print(f"Pairs with valid S23 address: {has_s23_addr.sum():,} (GT: {(is_gt & has_s23_addr).sum():,})")
print(f"Pairs with MISSING S23 address: {(~has_s23_addr).sum():,} (GT: {(is_gt & (~has_s23_addr)).sum():,})")

# Let's inspect the missing S23 address pool:
missing_mask = ~has_s23_addr
gt_missing = is_gt & missing_mask
print(f"\nIn Missing S23 Address subset ({missing_mask.sum():,} pairs, {gt_missing.sum():,} GTs):")

for name_thresh in [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]:
    cand = missing_mask & (name_tok >= name_thresh)
    tp = (cand & is_gt).sum()
    fp = (cand & (~is_gt)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"  NameTok >= {name_thresh:.2f}: TP={tp:,}, FP={fp:,}, Precision={prec*100:.2f}% (Total Cand: {cand.sum():,})")

print("\nWhat if we ALSO require the S1 business name to be UNIQUE in S1?")
is_s1_unique = np.array([s1_name_counts.get(s1_names.get(s, ''), 0) == 1 for s in s1_ids], dtype=bool)
for name_thresh in [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]:
    cand = missing_mask & (name_tok >= name_thresh) & is_s1_unique
    tp = (cand & is_gt).sum()
    fp = (cand & (~is_gt)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"  Unique S1 + NameTok >= {name_thresh:.2f}: TP={tp:,}, FP={fp:,}, Precision={prec*100:.2f}% (Total Cand: {cand.sum():,})")
