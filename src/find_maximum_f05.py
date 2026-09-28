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

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)
TOTAL_GT = 86275

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict['name_ratio']
addr_r = feat_dict['address_ratio']
house_m = feat_dict['house_match']
postal_e = feat_dict['postal_exact']
is_acronym = feat_dict['name_acronym_match']
name_c = feat_dict['name_contains']
name_ex = feat_dict['name_exact']

print("=== GRID SEARCH TO FIND ABSOLUTE MAXIMUM GLOBAL F0.5 ===")
best_f05 = 0.0
best_params = None

for base_tau in np.arange(0.950, 0.980, 0.005):
    for min_nr in [0.25, 0.30, 0.35, 0.40]:
        for rescue_tau in [0.85, 0.88, 0.90, 0.92]:
            bg = (p_ens >= base_tau) & (name_r >= min_nr) & ((addr_r >= 0.25) | (house_m == 1) | (postal_e == 1))
            r1 = (p_ens >= rescue_tau) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95)) & (name_r >= min_nr)
            r_acr = (is_acronym == 1) & (addr_r >= 0.70)
            r_ex = (name_ex == 1) & (postal_e == 1)
            
            gate = bg | r1 | r_acr | r_ex
            
            # S23 Injective Filter
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
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / TOTAL_GT
            f05 = 1.25 * p * r / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
            
            if f05 > best_f05:
                best_f05 = f05
                best_params = (base_tau, min_nr, rescue_tau, tp, fp, p, r, f05)
                print(f"NEW MAX! base_tau={base_tau:.3f}, min_nr={min_nr:.2f}, rescue_tau={rescue_tau:.2f} -> TP={tp:5,d}, FP={fp:4,d}, P={p:.4%}, R={r:.4%}, F0.5={f05:.6f}")

print("\n" + "="*70)
print(f"ABSOLUTE MAXIMUM GLOBAL F0.5 FOUND:")
print(f"  base_tau:   {best_params[0]:.3f}")
print(f"  min_nr:     {best_params[1]:.2f}")
print(f"  rescue_tau: {best_params[2]:.2f}")
print(f"  TP:         {best_params[3]:,}")
print(f"  FP:         {best_params[4]:,}")
print(f"  Precision:  {best_params[5]:.4%}")
print(f"  Recall:     {best_params[6]:.4%}")
print(f"  Global F0.5:{best_params[7]:.6f}")
print("="*70)
