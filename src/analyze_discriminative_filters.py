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

base_g = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
v15_base = ((base_g | rescue1) & (name_r >= 0.40)) | ((p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)) | ((p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40))

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# Apply S23-injective
cands = np.where(v15_base)[0]
s_ord = cands[np.argsort(-p_ens[cands])]
s23_seen = set()
injective_mask = np.zeros(len(v15_base), dtype=bool)
for i in s_ord:
    mid = m_ids[i]
    if mid not in s23_seen:
        s23_seen.add(mid)
        injective_mask[i] = True

tp_mask = injective_mask & is_gt
fp_mask = injective_mask & ~is_gt

print(f"Total Selected: {injective_mask.sum():,} (TP: {tp_mask.sum():,}, FP: {fp_mask.sum():,})")

# Feature comparison between TP and FP in the accepted pool
print(f"\n{'Feature':30s} | {'TP Mean':>10s} | {'FP Mean':>10s} | {'Diff':>10s}")
print("-" * 68)
feature_diffs = []
for f_name in V5_FEATURES_EXPANDED:
    v = feat_dict[f_name]
    tp_val = v[tp_mask].mean()
    fp_val = v[fp_mask].mean()
    diff = tp_val - fp_val
    feature_diffs.append((diff, f_name, tp_val, fp_val))

feature_diffs.sort(key=lambda x: abs(x[0]), reverse=True)
for diff, f_name, tp_val, fp_val in feature_diffs[:25]:
    print(f"{f_name:30s} | {tp_val:10.4f} | {fp_val:10.4f} | {diff:+10.4f}")

# Test simple filters based on the biggest differentiators
print("\n=== TESTING DISCRIMINATIVE FILTERS ===")
base_tp = tp_mask.sum()
base_fp = fp_mask.sum()
base_p = base_tp / (base_tp + base_fp)
base_r = base_tp / 86275
base_f05 = 1.25 * base_p * base_r / (0.25 * base_p + base_r)
print(f"BASE: TP={base_tp:,}, FP={base_fp:,}, P={base_p:.4%}, R={base_r:.4%}, F0.5={base_f05:.6f}\n")

# Try each of the top features as a filter
candidate_filters = [
    ("name_token_jaccard >= 0.15", feat_dict["name_token_jaccard"] >= 0.15),
    ("name_token_jaccard >= 0.20", feat_dict["name_token_jaccard"] >= 0.20),
    ("name_first_token_match == 1 or name_ratio >= 0.70", (feat_dict["name_first_token_match"] == 1) | (name_r >= 0.70)),
    ("name_char3_jaccard >= 0.20", feat_dict["name_char3_jaccard"] >= 0.20),
    ("name_char3_jaccard >= 0.25", feat_dict["name_char3_jaccard"] >= 0.25),
    ("name_token_sort >= 0.50", feat_dict["name_token_sort"] >= 0.50),
    ("name_token_sort >= 0.55", feat_dict["name_token_sort"] >= 0.55),
    ("name_len_rel_diff <= 0.60", feat_dict["name_len_rel_diff"] <= 0.60),
    ("name_len_rel_diff <= 0.50", feat_dict["name_len_rel_diff"] <= 0.50),
    ("name_address_ratio_mean >= 0.60", feat_dict["name_address_ratio_mean"] >= 0.60),
    ("name_address_harmonic >= 0.50", feat_dict["name_address_harmonic"] >= 0.50),
    ("name_address_harmonic >= 0.55", feat_dict["name_address_harmonic"] >= 0.55),
]

for label, filt_cond in candidate_filters:
    # Filter the accepted mask
    new_mask = injective_mask & filt_cond
    tp = (new_mask & is_gt).sum()
    fp = (new_mask & ~is_gt).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / 86275
    f05 = 1.25 * p * r / (0.25 * p + r)
    dtp = tp - base_tp
    dfp = fp - base_fp
    df = f05 - base_f05
    print(f"{label:45s}: TP={tp:,} ({dtp:+4d}) | FP={fp:,} ({dfp:+3d}) | P={p:.4%} | R={r:.4%} | F0.5={f05:.6f} ({df:+.6f})")
