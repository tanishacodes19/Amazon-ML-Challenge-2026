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
total_gt = len(gt_set)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]
name_tok = feat_dict["name_token_set"]
name_fst = feat_dict["name_first_token_match"]
name_pfx = feat_dict["name_prefix5"]

# Injective evaluator
def eval_policy(gate, scores):
    cands = np.where(gate)[0]
    s_ord = cands[np.argsort(-scores[cands])]
    s23_seen = set()
    accepted = []
    for i in s_ord:
        mid = m_ids[i]
        if mid not in s23_seen:
            s23_seen.add(mid)
            accepted.append(i)
            
    m_eval = np.zeros(len(gate), dtype=bool)
    m_eval[accepted] = True
    tp = (m_eval & is_gt).sum()
    fp = (m_eval & ~is_gt).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / total_gt
    f05 = 1.25 * p * r / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
    return tp, fp, p, r, f05

print("\n--- Sweeping Smart Co-location and Token Gates ---")
print(f"{'Description':<50} | {'Global F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'TP':>7} | {'FP':>5}")
print("-" * 102)

# Baseline V16
base_gate = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
r_contain = (p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)
r_acronym = (p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40)

# V16 baseline
v16_gate = ((base_gate | rescue1) & (name_r >= 0.40)) | r_contain | r_acronym
tp, fp, p, r, f05 = eval_policy(v16_gate, p_ens)
print(f"{'V16 Baseline (name_r >= 0.40)':<50} | {f05:>11.4f} | {p*100:>9.2f}% | {r*100:>7.2f}% | {tp:>7,} | {fp:>5,}")

# What if name condition allows first-token match or token-set or prefix?
for min_tau in [0.972, 0.965, 0.960, 0.950]:
    for nr_thresh in [0.35, 0.30, 0.25, 0.20]:
        # Smart name gate:
        # Either name_r >= nr_thresh OR name_token_set >= 0.70 OR name_first_token_match == 1 OR name_contains == 1
        smart_name = (name_r >= nr_thresh) | ((name_tok >= 0.70) & (addr_r >= 0.40)) | ((name_fst == 1) & (name_r >= 0.15) & (addr_r >= 0.40))
        
        bg = (p_ens >= min_tau) & smart_name & ((addr_r >= 0.35) | (house_m == 1))
        r1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.85) & (house_m == 1)) | (addr_r >= 0.95))
        r_c = (p_ens >= 0.92) & (name_c == 1) & (addr_r >= 0.50)
        r_a = (p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40)
        
        g = bg | r1 | r_c | r_a
        tp, fp, p, r, f05 = eval_policy(g, p_ens)
        tag = f"tau={min_tau:.3f} nr={nr_thresh:.2f} + smart_name"
        print(f"{tag:<50} | {f05:>11.4f} | {p*100:>9.2f}% | {r*100:>7.2f}% | {tp:>7,} | {fp:>5,}")
