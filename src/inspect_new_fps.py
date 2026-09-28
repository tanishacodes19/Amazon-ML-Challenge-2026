import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')
s1 = pl.read_parquet('validation_benchmark/val_s1_v13_clean.parquet')
s23 = pl.read_parquet('validation_benchmark/val_s23_v13_clean.parquet')

import xgboost as xgb, lightgbm as lgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('model/xgboost_v13_52features.json')
lgb_model = lgb.Booster(model_file='model/lightgbm_v13_52features.txt')
p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]
name_tok = feat_dict["name_token_set"]
name_fst = feat_dict["name_first_token_match"]

base_gate = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
r_contain = (p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)
r_acronym = (p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40)
v16_gate = ((base_gate | rescue1) & (name_r >= 0.40)) | r_contain | r_acronym

smart_name = (name_r >= 0.35) | ((name_tok >= 0.70) & (addr_r >= 0.40)) | ((name_fst == 1) & (name_r >= 0.15) & (addr_r >= 0.40))
bg = (p_ens >= 0.972) & smart_name & ((addr_r >= 0.35) | (house_m == 1))
r1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.85) & (house_m == 1)) | (addr_r >= 0.95))
r_c = (p_ens >= 0.92) & (name_c == 1) & (addr_r >= 0.50)
r_a = (p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40)
new_gate = bg | r1 | r_c | r_a

# Newly admitted pairs:
newly_admitted = new_gate & (~v16_gate)
new_fp = newly_admitted & (~is_gt)
print(f"Newly admitted pairs: {newly_admitted.sum():,} (TP: {(newly_admitted & is_gt).sum():,}, FP: {new_fp.sum():,})")

s1_dict = {r["source1_entity_id"]: r for r in s1.to_dicts()}
s23_dict = {r["matched_entity_id"]: r for r in s23.to_dicts()}

fp_idx = np.where(new_fp)[0]
print(f"\nSample 15 Newly Admitted False Positives:")
for idx in fp_idx[:15]:
    sid = s1_ids[idx]; mid = m_ids[idx]
    r1 = s1_dict.get(sid, {})
    r2 = s23_dict.get(mid, {})
    true_s1 = [s for s, m in gt_set if m == mid]
    print(f"\n[p_ens: {p_ens[idx]:.4f} | NameRat: {name_r[idx]:.2f} | AddrRat: {addr_r[idx]:.2f} | House: {house_m[idx]}]")
    print(f"  S1 (Proposed) : [{sid}] {r1.get('business_name')}  ||  {r1.get('business_address')}")
    print(f"  S23           : [{mid}] {r2.get('business_name')}  ||  {r2.get('business_address')}")
    if true_s1:
        r_true = s1_dict.get(true_s1[0], {})
        print(f"  TRUE S1 in GT : [{true_s1[0]}] {r_true.get('business_name')}  ||  {r_true.get('business_address')}")
    else:
        print(f"  TRUE S1 in GT : NONE (Singleton)")
