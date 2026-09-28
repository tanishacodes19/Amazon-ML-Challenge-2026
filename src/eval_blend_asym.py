import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, numpy as np, polars as pl
import xgboost as xgb, lightgbm as lgb
from feature_engine_v2 import V5_FEATURES_EXPANDED

print("1. Loading models...")
xgb_v13 = xgb.XGBClassifier()
xgb_v13.load_model('model/xgboost_v13_52features.json')
lgb_v13 = lgb.Booster(model_file='model/lightgbm_v13_52features.txt')

lgb_asym = lgb.Booster(model_file='model/lightgbm_asym_52features.txt')

print("2. Loading validation benchmark data (623k pairs)...")
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)
total_gt = len(gt_set)

p_v13 = 0.50 * xgb_v13.predict_proba(X_val)[:, 1] + 0.50 * lgb_v13.predict(X_val)
p_asym = lgb_asym.predict(X_val)

# Feature dictionaries
feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]

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

print("\n--- Sweeping Blend of V13 + Asymmetric V18 with Full Gating ---")
for w in [0.0, 0.2, 0.3, 0.5]:
    p_blend = (1 - w) * p_v13 + w * p_asym
    for base_tau in [0.965, 0.970, 0.972, 0.975]:
        base_gate = (p_blend >= base_tau) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
        rescue1 = (p_blend >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
        co_location = (name_r >= 0.40)
        r_contain = (p_blend >= 0.95) & (name_c == 1) & (addr_r >= 0.65)
        r_acronym = (p_blend >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40)
        
        gate = ((base_gate | rescue1) & co_location) | r_contain | r_acronym
        tp, fp, p, r, f05 = eval_policy(gate, p_blend)
        print(f"w={w:.1f} tau={base_tau:.3f} | TP={tp:6,d} | FP={fp:4,d} | P={p*100:6.2f}% | R={r*100:6.2f}% | F0.5={f05:.6f}")
