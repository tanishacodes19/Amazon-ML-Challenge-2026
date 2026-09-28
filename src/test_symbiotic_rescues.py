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
addr_tok = feat_dict["address_token_set"]
addr_jacc = feat_dict["address_token_jaccard"]

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

# Baseline V16
base_gate = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
r_contain = (p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)
r_acronym = (p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40)
v16_gate = ((base_gate | rescue1) & (name_r >= 0.40)) | r_contain | r_acronym

tp, fp, p, r, f05 = eval_policy(v16_gate, p_ens)
print(f"V16 Baseline: TP={tp:,} | FP={fp:,} | P={p*100:.2f}% | R={r*100:.2f}% | F0.5={f05:.4f}")

# Unaccepted pairs
unacc = ~eval_policy(v16_gate, p_ens)[0] # we want unaccepted mask

print("\n--- Testing High-Purity Symbiotic Rescues (Both Name AND Address Strong) ---")
best_f05 = f05
best_cfg = None
for n_tok_min in [0.75, 0.80, 0.85]:
    for a_tok_min in [0.75, 0.80, 0.85]:
        for p_min in [0.50, 0.70, 0.85]:
            r_both = (p_ens >= p_min) & (name_tok >= n_tok_min) & (addr_tok >= a_tok_min) & (name_r >= 0.40)
            combined_gate = v16_gate | r_both
            tp_c, fp_c, p_c, r_c, f05_c = eval_policy(combined_gate, p_ens)
            print(f"n_tok>={n_tok_min:.2f} a_tok>={a_tok_min:.2f} p>={p_min:.2f} | TP={tp_c:,} (+{tp_c-tp}) | FP={fp_c:,} (+{fp_c-fp}) | P={p_c*100:.2f}% | R={r_c*100:.2f}% | F0.5={f05_c:.6f}")
            if f05_c > best_f05:
                best_f05 = f05_c
                best_cfg = (n_tok_min, a_tok_min, p_min, tp_c, fp_c, p_c, r_c, f05_c)
print(f"\nBest config: {best_cfg}")
