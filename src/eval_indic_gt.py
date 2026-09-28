import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')
s23_val = pl.read_parquet('validation_benchmark/val_s23_v13_clean.parquet')

def has_non_ascii(s):
    if not s: return False
    return any(ord(c) > 127 for c in str(s))

s23_non_ascii_ids = set(s23_val.filter(pl.col('business_name').map_elements(has_non_ascii, return_dtype=pl.Boolean))['matched_entity_id'])

# Check GT pairs with non-ascii
gt_non_ascii = gt.filter(pl.col('matched_entity_id').is_in(s23_non_ascii_ids))
print(f"Total non-ASCII GT pairs: {len(gt_non_ascii):,}")

# Check how many are in candidate pool
pool_m_ids = set(pairs_df['matched_entity_id'])
gt_in_pool = gt_non_ascii.filter(pl.col('matched_entity_id').is_in(pool_m_ids))
print(f"Non-ASCII GT pairs in candidate pool: {len(gt_in_pool):,} ({len(gt_in_pool)/len(gt_non_ascii)*100:.2f}%)")

# Check how many are accepted by V16 Champion
import xgboost as xgb, lightgbm as lgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('model/xgboost_v13_52features.json')
lgb_model = lgb.Booster(model_file='model/lightgbm_v13_52features.txt')
p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

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
v16_gate = ((base_gate | rescue1) & co_location) | r_contain | r_acronym

# Filter pairs
s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# Injective
cands = np.where(v16_gate)[0]
s_ord = cands[np.argsort(-p_ens[cands])]
s23_seen = set()
accepted = []
for i in s_ord:
    mid = m_ids[i]
    if mid not in s23_seen:
        s23_seen.add(mid)
        accepted.append(i)

m_eval = np.zeros(len(v16_gate), dtype=bool)
m_eval[accepted] = True

# Check how many non-ascii GT pairs were accepted
accepted_m_ids = set(m_ids[m_eval])
non_ascii_gt_accepted = sum(1 for mid in gt_non_ascii['matched_entity_id'] if mid in accepted_m_ids)
print(f"Non-ASCII GT pairs accepted by V16: {non_ascii_gt_accepted:,} ({non_ascii_gt_accepted/len(gt_non_ascii)*100:.2f}%)")
print(f"Non-ASCII GT pairs MISSED: {len(gt_non_ascii) - non_ascii_gt_accepted:,}")
