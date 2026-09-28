import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED
from collections import Counter

gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')
s1 = pl.read_parquet('validation_benchmark/val_s1_v13_clean.parquet')
s23 = pl.read_parquet('validation_benchmark/val_s23_v13_clean.parquet')

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

s23_addr_map = dict(zip(s23['matched_entity_id'], s23['clean_addr']))
s23_addrs = [s23_addr_map.get(m, '') for m in m_ids]
has_s23_addr = np.array([a is not None and len(str(a).strip()) > 3 and str(a).strip() != 'nan' for a in s23_addrs], dtype=bool)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_tok = feat_dict["name_token_set"]
name_rat = feat_dict["name_ratio"]
addr_rat = feat_dict["address_ratio"]

# What if we look at the False Positives among the Unique S1 + NameTok == 1.00?
missing_mask = ~has_s23_addr
s1_names = dict(zip(s1['source1_entity_id'], s1['clean_name']))
s23_names = dict(zip(s23['matched_entity_id'], s23['clean_name']))
s1_name_counts = Counter(s1['clean_name'])
is_s1_unique = np.array([s1_name_counts.get(s1_names.get(s, ''), 0) == 1 for s in s1_ids], dtype=bool)

cand = missing_mask & (name_tok >= 0.95) & is_s1_unique
cand_indices = np.where(cand & (~is_gt))[0]

print(f"Sample 15 False Positives with Missing Address & NameTok >= 0.95 & Unique S1:")
s1_dict = {r["source1_entity_id"]: r for r in s1.to_dicts()}
s23_dict = {r["matched_entity_id"]: r for r in s23.to_dicts()}

for idx in cand_indices[:15]:
    sid = s1_ids[idx]; mid = m_ids[idx]
    r1 = s1_dict.get(sid, {})
    r2 = s23_dict.get(mid, {})
    true_s1 = [s for s, m in gt_set if m == mid]
    print(f"\n[NameTok: {name_tok[idx]:.2f} | NameRat: {name_rat[idx]:.2f}]")
    print(f"  S1 (Proposed) : [{sid}] {r1.get('business_name')}  ||  {r1.get('business_address')}")
    print(f"  S23           : [{mid}] {r2.get('business_name')}  ||  {r2.get('business_address')}")
    if true_s1:
        r_true = s1_dict.get(true_s1[0], {})
        print(f"  TRUE S1 in GT : [{true_s1[0]}] {r_true.get('business_name')}  ||  {r_true.get('business_address')}")
    else:
        print(f"  TRUE S1 in GT : NONE (Singleton)")
