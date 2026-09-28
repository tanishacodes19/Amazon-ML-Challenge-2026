import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')

import xgboost as xgb, lightgbm as lgb
xgb_v13 = xgb.XGBClassifier()
xgb_v13.load_model('model/xgboost_v13_52features.json')
lgb_v13 = lgb.Booster(model_file='model/lightgbm_v13_52features.txt')
p_v13 = 0.50 * xgb_v13.predict_proba(X_val)[:, 1] + 0.50 * lgb_v13.predict(X_val)

xgb_v17 = xgb.XGBClassifier()
xgb_v17.load_model('model/xgboost_v17_52features.json')
lgb_v17 = lgb.Booster(model_file='model/lightgbm_v17_52features.txt')
p_v17 = 0.50 * xgb_v17.predict_proba(X_val)[:, 1] + 0.50 * lgb_v17.predict(X_val)

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

total_gt = len(gt_set)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}


# Function to run greedy injective matching on candidate pool
def evaluate_injective(scores, mask):
    # Filter candidates by mask
    sub_idx = np.where(mask)[0]
    if len(sub_idx) == 0:
        return 0, 0, 0, 0, 0
    sub_scores = scores[sub_idx]
    sub_s1 = s1_ids[sub_idx]
    sub_m = m_ids[sub_idx]
    sub_gt = is_gt[sub_idx]
    
    # Sort descending by score
    sort_order = np.argsort(-sub_scores)
    sorted_s1 = sub_s1[sort_order]
    sorted_m = sub_m[sort_order]
    sorted_gt = sub_gt[sort_order]
    
    # Injective: each matched_entity_id (S2/S3) can only match AT MOST ONE S1
    # Keep highest scoring match for each matched_entity_id
    seen_m = set()
    tp = 0
    fp = 0
    for i in range(len(sort_order)):
        m = sorted_m[i]
        if m in seen_m:
            continue
        seen_m.add(m)
        if sorted_gt[i]:
            tp += 1
        else:
            fp += 1
            
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / total_gt
    f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
    return tp, fp, p, r, f05

print("--- Testing V13 across probability thresholds (Injective Matching) ---")
for t in [0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80, 0.75, 0.70]:
    tp, fp, p, r, f05 = evaluate_injective(p_v13, p_v13 >= t)
    print(f"V13 t={t:.2f} | TP={tp:,} | FP={fp:,} | P={p*100:.2f}% | R={r*100:.2f}% | F0.5={f05:.4f}")

print("\n--- Testing V17 across probability thresholds (Injective Matching) ---")
for t in [0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80, 0.75, 0.70]:
    tp, fp, p, r, f05 = evaluate_injective(p_v17, p_v17 >= t)
    print(f"V17 t={t:.2f} | TP={tp:,} | FP={fp:,} | P={p*100:.2f}% | R={r*100:.2f}% | F0.5={f05:.4f}")

print("\n--- Testing Blend: 0.50*V13 + 0.50*V17 (Injective Matching) ---")
p_blend = 0.50 * p_v13 + 0.50 * p_v17
for t in [0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80, 0.75, 0.70]:
    tp, fp, p, r, f05 = evaluate_injective(p_blend, p_blend >= t)
    print(f"Blend t={t:.2f} | TP={tp:,} | FP={fp:,} | P={p*100:.2f}% | R={r*100:.2f}% | F0.5={f05:.4f}")
