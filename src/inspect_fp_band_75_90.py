import sys
sys.path.insert(0, 'src')
import numpy as np, polars as pl

gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')
s1 = pl.read_parquet('validation_benchmark/val_s1_v13_clean.parquet')
s23 = pl.read_parquet('validation_benchmark/val_s23_v13_clean.parquet')

import xgboost as xgb, lightgbm as lgb
xgb_v13 = xgb.XGBClassifier()
xgb_v13.load_model('model/xgboost_v13_52features.json')
lgb_v13 = lgb.Booster(model_file='model/lightgbm_v13_52features.txt')
p_ens = 0.50 * xgb_v13.predict_proba(X_val)[:, 1] + 0.50 * lgb_v13.predict(X_val)

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# False positives in range [0.75, 0.90]
fp_band = (~is_gt) & (p_ens >= 0.75) & (p_ens < 0.90)
print(f"Total False Positives in [0.75, 0.90): {fp_band.sum():,}")

s1_dict = {r["source1_entity_id"]: r for r in s1.to_dicts()}
s23_dict = {r["matched_entity_id"]: r for r in s23.to_dicts()}

fp_indices = np.where(fp_band)[0]
from feature_engine_v2 import V5_FEATURES_EXPANDED
feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("\nSample 15 False Positives with p in [0.75, 0.90):")
for idx in fp_indices[:15]:
    sid = s1_ids[idx]; mid = m_ids[idx]
    r1 = s1_dict.get(sid, {})
    r2 = s23_dict.get(mid, {})
    # Check if mid actually matches SOME other S1 in ground truth
    true_s1 = [s for s, m in gt_set if m == mid]
    print(f"\n[p_ens: {p_ens[idx]:.4f} | NameRat: {name_r[idx]:.2f} | AddrRat: {addr_r[idx]:.2f} | House: {house_m[idx]}]")
    print(f"  S1 (Proposed) : [{sid}] {r1.get('business_name')}  ||  {r1.get('business_address')}")
    print(f"  S23           : [{mid}] {r2.get('business_name')}  ||  {r2.get('business_address')}")
    if true_s1:
        r_true = s1_dict.get(true_s1[0], {})
        print(f"  TRUE S1 in GT : [{true_s1[0]}] {r_true.get('business_name')}  ||  {r_true.get('business_address')}")
    else:
        print(f"  TRUE S1 in GT : NONE (Singleton / not in GT)")

