import sys
sys.path.insert(0, 'src')
import numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')
s23 = pl.read_parquet('validation_benchmark/val_s23_v13_clean.parquet')
s1 = pl.read_parquet('validation_benchmark/val_s1_v13_clean.parquet')

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

s23_addr_dict = dict(zip(s23['matched_entity_id'], s23['clean_addr']))
s23_addrs = np.array([s23_addr_dict.get(mid, '') for mid in m_ids])
nan_mask = (s23_addrs == 'nan')

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict['name_ratio']
name_tok_s = feat_dict['name_token_set']
c_match = feat_dict['country_match']
name_ex = feat_dict['name_exact']
name_tok_j = feat_dict['name_token_jaccard']

print(f"Total pairs with S23 address == 'nan': {nan_mask.sum():,} (True GT: {(nan_mask & is_gt).sum():,})")

print("\nTesting name agreement on nan address pairs (with Country Match):")
for nr in [1.0, 0.95, 0.90, 0.85, 0.80, 0.70]:
    m = nan_mask & (c_match == 1) & (name_r >= nr)
    tp = (m & is_gt).sum()
    fp = (m & ~is_gt).sum()
    pur = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"  name_r >= {nr:.2f}: TP={tp:4,d}, FP={fp:4,d}, Purity={pur:.2%}")

print("\nTesting with S23-Injective Constraint (at most 1 S1 per S23):")
# Sort by name_ratio descending
cand_nan = np.where(nan_mask & (c_match == 1))[0]
sorted_nan = cand_nan[np.argsort(-name_r[cand_nan])]
seen_m = set()
inj_nan_mask = np.zeros(len(nan_mask), dtype=bool)
for idx in sorted_nan:
    mid = m_ids[idx]
    if mid not in seen_m:
        seen_m.add(mid)
        inj_nan_mask[idx] = True

for nr in [1.0, 0.95, 0.90, 0.85, 0.80, 0.70]:
    m = inj_nan_mask & (name_r >= nr)
    tp = (m & is_gt).sum()
    fp = (m & ~is_gt).sum()
    pur = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"  [Injective] name_r >= {nr:.2f}: TP={tp:4,d}, FP={fp:4,d}, Purity={pur:.2%}")
