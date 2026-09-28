import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

print("1. Loading validation benchmark data...")
gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')

import xgboost as xgb, lightgbm as lgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('model/xgboost_v13_52features.json')
lgb_model = lgb.Booster(model_file='model/lightgbm_v13_52features.txt')
p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# Baseline acceptance mask (from V16 champion)
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
v16_accepted = ((base_gate | rescue1) & co_location) | r_contain | r_acronym

print(f"Total pairs: {len(X_val):,}")
print(f"Total GT in pool: {is_gt.sum():,}")
print(f"Currently accepted by V16 gate: {v16_accepted.sum():,} (TP: {(v16_accepted & is_gt).sum():,}, FP: {(v16_accepted & ~is_gt).sum():,})")

# Unaccepted pool
unaccepted = ~v16_accepted
unacc_gt = unaccepted & is_gt
unacc_non_gt = unaccepted & (~is_gt)
print(f"Unaccepted pairs: {unaccepted.sum():,} (GT: {unacc_gt.sum():,}, Non-GT: {unacc_non_gt.sum():,})")

# Let's inspect features for unacc_gt vs unacc_non_gt
print("\nFeature Comparison on UNACCEPTED pairs (Mean values):")
print(f"{'Feature Name':<30} | {'Unacc GT (n=' + str(unacc_gt.sum()) + ')':>20} | {'Unacc Non-GT (n=' + str(unacc_non_gt.sum()) + ')':>25}")
print("-" * 80)

# Sort features by discrimination power (difference between unacc_gt and unacc_non_gt)
diffs = []
for i, f in enumerate(V5_FEATURES_EXPANDED):
    val_gt = X_val[unacc_gt, i].mean()
    val_nongt = X_val[unacc_non_gt, i].mean()
    diff = val_gt - val_nongt
    diffs.append((diff, f, val_gt, val_nongt))

diffs.sort(reverse=True)
for diff, f, val_gt, val_nongt in diffs[:20]:
    print(f"{f:<30} | {val_gt:>20.4f} | {val_nongt:>25.4f} | Diff: +{diff:.4f}")

