import sys
sys.path.insert(0, 'src')
import numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')

import xgboost as xgb, lightgbm as lgb
xgb_v13 = xgb.XGBClassifier()
xgb_v13.load_model('model/xgboost_v13_52features.json')
lgb_v13 = lgb.Booster(model_file='model/lightgbm_v13_52features.txt')
p_ens = 0.50 * xgb_v13.predict_proba(X_val)[:, 1] + 0.50 * lgb_v13.predict(X_val)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict['name_ratio']
addr_r = feat_dict['address_ratio']
house_m = feat_dict['house_match']

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

print(f"{'Threshold':10s} | {'min_nr':6s} | {'TP':6s} | {'FP':5s} | {'Precision':10s} | {'Recall':8s} | {'F0.5':8s}")
print("-" * 65)

for t in [0.900, 0.910, 0.920, 0.930, 0.940, 0.950, 0.960, 0.970]:
    for nr in [0.20, 0.30, 0.35, 0.40, 0.45]:
        mask = (p_ens >= t) & (name_r >= nr)
        
        # Apply S23 Injective Filter
        cands = np.where(mask)[0]
        s_ord = cands[np.argsort(-p_ens[cands])]
        s23_seen = set()
        m_eval = np.zeros(len(mask), dtype=bool)
        for i in s_ord:
            mid = m_ids[i]
            if mid not in s23_seen:
                s23_seen.add(mid)
                m_eval[i] = True
                
        tp = (m_eval & is_gt).sum()
        fp = (m_eval & ~is_gt).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / 86275
        f05 = 1.25 * p * r / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
        if f05 >= 0.965:
            print(f"p >= {t:.3f}   | {nr:.2f}   | {tp:6,d} | {fp:5,d} | {p:10.4%} | {r:8.4%} | {f05:8.6f}")
