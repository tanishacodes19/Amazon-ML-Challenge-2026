import sys, os
sys.path.insert(0, 'src')
import numpy as np, polars as pl
from collections import Counter

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))
s1 = pl.read_parquet(os.path.join(VAL_DIR, "val_s1_v13_clean.parquet"))
s23 = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet"))

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

import xgboost as xgb, lightgbm as lgb
xgb_v13 = xgb.XGBClassifier()
xgb_v13.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_v13 = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))
p_ens = 0.50 * xgb_v13.predict_proba(X_val)[:, 1] + 0.50 * lgb_v13.predict(X_val)

s1_name_dict = dict(zip(s1["source1_entity_id"], s1["clean_name"]))
s23_name_dict = dict(zip(s23["matched_entity_id"], s23["clean_name"]))

s1_names = [s1_name_dict.get(sid, "") for sid in s1_ids]
m_names = [s23_name_dict.get(mid, "") for mid in m_ids]

# Stop words and generic industry/legal keywords
GENERIC_WORDS = {
    'the', 'and', 'for', 'of', 'in', 'on', 'at', 'to', 'a', 'an',
    'ltd', 'pvt', 'inc', 'corp', 'llc', 'llp', 'co', 'company', 'limited', 'private',
    'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'holdings',
    'associates', 'partners', 'technologies', 'technology', 'consulting', 'consultancy',
    'management', 'international', 'global', 'national', 'systems', 'industries', 'industry',
    'india', 'us', 'usa', 'de', 'la', 'le', 'les', 'des', 'du', 'et', 'en'
}

def get_core_tokens(name_str):
    tokens = [w for w in name_str.lower().split() if len(w) >= 3 and w not in GENERIC_WORDS]
    return set(tokens)

# Current gate
from feature_engine_v2 import V5_FEATURES_EXPANDED
feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
postal_e = feat_dict["postal_exact"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]

base_g = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
gate = ((base_g | rescue1) & (name_r >= 0.40)) | ((p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)) | ((p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40))

# False positives in current gate
fp_indices = np.where(gate & ~is_gt)[0]
tp_indices = np.where(gate & is_gt)[0]
print(f"Current Gate TP: {len(tp_indices):,}, FP: {len(fp_indices):,}")

# Check core token overlap on FP vs TP
fp_core_overlaps = []
for idx in fp_indices:
    c1 = get_core_tokens(s1_names[idx])
    c2 = get_core_tokens(m_names[idx])
    overlap = len(c1 & c2)
    jacc = overlap / len(c1 | c2) if len(c1 | c2) > 0 else 0
    fp_core_overlaps.append((overlap, jacc))

tp_core_overlaps = []
for idx in tp_indices:
    c1 = get_core_tokens(s1_names[idx])
    c2 = get_core_tokens(m_names[idx])
    overlap = len(c1 & c2)
    jacc = overlap / len(c1 | c2) if len(c1 | c2) > 0 else 0
    tp_core_overlaps.append((overlap, jacc))

fp_zero_overlap = sum(1 for ov, j in fp_core_overlaps if ov == 0)
tp_zero_overlap = sum(1 for ov, j in tp_core_overlaps if ov == 0)

print(f"\nCore Token Overlap == 0:")
print(f"  In False Positives: {fp_zero_overlap:,} / {len(fp_indices):,} ({fp_zero_overlap/len(fp_indices):.2%})")
print(f"  In True Positives:  {tp_zero_overlap:,} / {len(tp_indices):,} ({tp_zero_overlap/len(tp_indices):.2%})")
