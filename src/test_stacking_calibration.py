import sys
sys.path.insert(0, 'src')
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')

import xgboost as xgb, lightgbm as lgb
xgb_v13 = xgb.XGBClassifier()
xgb_v13.load_model('model/xgboost_v13_52features.json')
lgb_v13 = lgb.Booster(model_file='model/lightgbm_v13_52features.txt')
p_xgb = xgb_v13.predict_proba(X_val)[:, 1]
p_lgb = lgb_v13.predict(X_val)

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
y_val = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=int)
TOTAL_GT = 86275

from feature_engine_v2 import V5_FEATURES_EXPANDED
feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict['name_ratio']

# 5-fold cross-validation calibration
print("=== 5-FOLD CALIBRATION OF XGBOOST + LIGHTGBM ===")
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

p_calibrated = np.zeros(len(y_val))

# Features for stacking: p_xgb, p_lgb, name_ratio, address_ratio, harmonic mean
X_stack = np.column_stack([
    p_xgb, p_lgb,
    feat_dict['name_ratio'],
    feat_dict['address_ratio'],
    feat_dict['house_match'],
    feat_dict['postal_exact'],
    feat_dict['name_address_harmonic'],
    feat_dict['strong_name_address']
])

for train_idx, val_idx in kf.split(X_stack):
    stacker = LogisticRegression(C=1.0, max_iter=200)
    stacker.fit(X_stack[train_idx], y_val[train_idx])
    p_calibrated[val_idx] = stacker.predict_proba(X_stack[val_idx])[:, 1]

print("Calibration complete. Sweeping calibrated thresholds with S23 injective constraint...")

best_f05 = 0.0
for tau in np.arange(0.80, 0.99, 0.02):
    for min_nr in [0.20, 0.30, 0.40]:
        mask = (p_calibrated >= tau) & (name_r >= min_nr)
        
        cands = np.where(mask)[0]
        s_ord = cands[np.argsort(-p_calibrated[cands])]
        s23_seen = set()
        m_eval = np.zeros(len(mask), dtype=bool)
        for i in s_ord:
            mid = m_ids[i]
            if mid not in s23_seen:
                s23_seen.add(mid)
                m_eval[i] = True
                
        tp = (m_eval & (y_val == 1)).sum()
        fp = (m_eval & (y_val == 0)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / TOTAL_GT
        f05 = 1.25 * p * r / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
        if f05 > best_f05:
            best_f05 = f05
            print(f"tau={tau:.2f}, min_nr={min_nr:.2f} -> TP={tp:5,d}, FP={fp:4,d}, P={p:.4%}, R={r:.4%}, F0.5={f05:.6f}")
