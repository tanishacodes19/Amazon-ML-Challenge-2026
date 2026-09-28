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
name_first_m = feat_dict["name_first_token_match"]
name_last_m = feat_dict["name_last_token_match"]
name_tok_j = feat_dict["name_token_jaccard"]
name_tok_s = feat_dict["name_token_set"]
addr_tok_s = feat_dict["address_token_set"]

# V15 Base gate
base_g = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
v15_gate = ((base_g | rescue1) & (name_r >= 0.40)) | ((p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)) | ((p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40))

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# S23 Injective selection
cand_indices = np.where(v15_gate)[0]
sorted_order = cand_indices[np.argsort(-p_ens[cand_indices])]
seen_s23 = set()
injective_mask = np.zeros(len(v15_gate), dtype=bool)
for idx in sorted_order:
    mid = m_ids[idx]
    if mid not in seen_s23:
        seen_s23.add(mid)
        injective_mask[idx] = True

fp_indices = np.where(injective_mask & ~is_gt)[0]
print(f"Total False Positives in S23-Injective V15: {len(fp_indices):,}")

# Feature distribution of these 718 FPs:
print(f"FP name_ratio < 0.50: {(name_r[fp_indices] < 0.50).sum():,} / {len(fp_indices):,}")
print(f"FP name_ratio < 0.60: {(name_r[fp_indices] < 0.60).sum():,} / {len(fp_indices):,}")
print(f"FP name_token_jaccard == 0: {(name_tok_j[fp_indices] == 0).sum():,} / {len(fp_indices):,}")
print(f"FP name_first_token_match == 0: {(name_first_m[fp_indices] == 0).sum():,} / {len(fp_indices):,}")

# Let's test a fine-grained sweep of thresholds for base_tau and min_name_ratio WITH S23 injective
print("\n=== SWEEPING BASE_TAU & MIN_NAME WITH S23-INJECTIVE ===")
for base_tau in [0.970, 0.972, 0.975, 0.980, 0.982, 0.985, 0.988]:
    for min_nr in [0.40, 0.45, 0.50, 0.52, 0.55]:
        bg = (p_ens >= base_tau) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
        r1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
        gate = ((bg | r1) & (name_r >= min_nr)) | ((p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)) | ((p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40))
        
        cands = np.where(gate)[0]
        s_ord = cands[np.argsort(-p_ens[cands])]
        s23_seen = set()
        m_eval = np.zeros(len(gate), dtype=bool)
        for i in s_ord:
            mid = m_ids[i]
            if mid not in s23_seen:
                s23_seen.add(mid)
                m_eval[i] = True
                
        tp = (m_eval & is_gt).sum()
        fp = (m_eval & ~is_gt).sum()
        p = tp / (tp + fp)
        r = tp / 86275
        f05 = 1.25 * p * r / (0.25 * p + r)
        print(f"base_tau={base_tau:.3f}, min_nr={min_nr:.2f} -> TP={tp:5,d}, FP={fp:3,d}, P={p:.4%}, R={r:.4%}, F0.5={f05:.6f}")
