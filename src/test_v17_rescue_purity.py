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
p_v13 = 0.50 * xgb_v13.predict_proba(X_val)[:, 1] + 0.50 * lgb_v13.predict(X_val)

xgb_v17 = xgb.XGBClassifier()
xgb_v17.load_model('model/xgboost_v17_52features.json')
lgb_v17 = lgb.Booster(model_file='model/lightgbm_v17_52features.txt')
p_v17 = 0.50 * xgb_v17.predict_proba(X_val)[:, 1] + 0.50 * lgb_v17.predict(X_val)

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict['name_ratio']
addr_r = feat_dict['address_ratio']
house_m = feat_dict['house_match']

print("=== V17 RESCUE PURITY (WHERE V13 < 0.90) ===")
for t in [0.70, 0.75, 0.80, 0.85, 0.90]:
    for nr in [0.50, 0.60, 0.70, 0.80]:
        m = (p_v13 < 0.90) & (p_v17 >= t) & (name_r >= nr)
        tp = (m & is_gt).sum()
        fp = (m & ~is_gt).sum()
        pur = tp / (tp + fp) if (tp + fp) > 0 else 0
        if tp >= 20:
            print(f"V17 >= {t:.2f} & name_r >= {nr:.2f}: TP={tp:4,d}, FP={fp:3,d}, Purity={pur:.2%}")
