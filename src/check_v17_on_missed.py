import sys
sys.path.insert(0, 'src')
import numpy as np, polars as pl
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

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

missed_v13 = is_gt & (p_v13 < 0.90)
print(f"Missed GT by V13 (p < 0.90): {missed_v13.sum():,}")
print(f"Their V17 mean prob: {p_v17[missed_v13].mean():.4f}")
for t in [0.90, 0.80, 0.70, 0.50]:
    cnt = (p_v17[missed_v13] >= t).sum()
    print(f"  V17 >= {t}: {cnt:,} / {missed_v13.sum():,} ({cnt/missed_v13.sum():.2%})")
