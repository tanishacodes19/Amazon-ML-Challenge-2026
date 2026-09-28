import sys
sys.path.insert(0, 'src')
import numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

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

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict['name_ratio']
addr_r = feat_dict['address_ratio']
house_m = feat_dict['house_match']
postal_e = feat_dict['postal_exact']
is_acronym = feat_dict['name_acronym_match']
name_c = feat_dict['name_contains']
name_ex = feat_dict['name_exact']
addr_ex = feat_dict['address_exact']
c_match = feat_dict['country_match']
addr_tok_s = feat_dict['address_token_set']
name_tok_s = feat_dict['name_token_set']
name_tok_j = feat_dict['name_token_jaccard']
addr_tok_j = feat_dict['address_token_jaccard']

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# Current V15 gate
base_g = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
current_gate = ((base_g | rescue1) & (name_r >= 0.40)) | ((p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)) | ((p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40))

# True Positives in pool missed by current gate
missed_gt = is_gt & ~current_gate
print(f"Total GT in V16 candidate pool: {is_gt.sum():,}")
print(f"Accepted by current gate: {(is_gt & current_gate).sum():,}")
print(f"MISSED GT in candidate pool: {missed_gt.sum():,}")

missed_indices = np.where(missed_gt)[0]

# Let's inspect probability bands of missed GT
print("\nProbability bands of the 6,804 missed GT:")
for lo, hi in [(0.90, 0.972), (0.80, 0.90), (0.70, 0.80), (0.50, 0.70), (0.20, 0.50), (0.0, 0.20)]:
    m = missed_gt & (p_ens >= lo) & (p_ens < hi)
    print(f"  prob [{lo:.2f}, {hi:.2f}): {m.sum():5,d} pairs")

# Now let's test specific clean slices among the missed GT to find where the high purity clusters hide!
print("\n=== CLUSTERING THE MISSED GROUND TRUTH BY SIGNALS ===")
slices = [
    # 1. Very High Address + Moderate Name (prob was depressed by moderate name)
    ("HighAddr90_Name50_Prob70", (addr_r >= 0.90) & (name_r >= 0.50) & (p_ens >= 0.70)),
    ("HighAddr95_Name45_Prob60", (addr_r >= 0.95) & (name_r >= 0.45) & (p_ens >= 0.60)),
    ("HighAddr85_Name60_Prob70", (addr_r >= 0.85) & (name_r >= 0.60) & (p_ens >= 0.70)),
    
    # 2. Very High Name + Moderate Address
    ("HighName90_Addr50_Prob70", (name_r >= 0.90) & (addr_r >= 0.50) & (p_ens >= 0.70)),
    ("HighName95_Addr40_Prob60", (name_r >= 0.95) & (addr_r >= 0.40) & (p_ens >= 0.60)),
    ("HighName85_Addr60_Prob70", (name_r >= 0.85) & (addr_r >= 0.60) & (p_ens >= 0.70)),
    
    # 3. House Match + High Agreement
    ("HouseMatch_Name70_Addr70_Prob60", (house_m == 1) & (name_r >= 0.70) & (addr_r >= 0.70) & (p_ens >= 0.60)),
    ("HouseMatch_Name60_Addr80_Prob60", (house_m == 1) & (name_r >= 0.60) & (addr_r >= 0.80) & (p_ens >= 0.60)),
    ("HouseMatch_Name80_Addr60_Prob60", (house_m == 1) & (name_r >= 0.80) & (addr_r >= 0.60) & (p_ens >= 0.60)),
    
    # 4. Token Set Agreement (Order Invariant)
    ("NameTokSet90_AddrTokSet90_Prob70", (name_tok_s >= 0.90) & (addr_tok_s >= 0.90) & (p_ens >= 0.70)),
    ("NameTokSet85_AddrTokSet85_Prob60", (name_tok_s >= 0.85) & (addr_tok_s >= 0.85) & (p_ens >= 0.60)),
    
    # 5. Exact Postal + High Token Agreement
    ("PostalExact_Name70_Addr60", (postal_e == 1) & (name_r >= 0.70) & (addr_r >= 0.60) & (p_ens >= 0.70)),
    ("PostalExact_Name80_Prob60", (postal_e == 1) & (name_r >= 0.80) & (p_ens >= 0.60)),
]

for label, cond in slices:
    # Look at pairs NOT yet in current gate
    new_m = cond & ~current_gate
    tp = (new_m & is_gt).sum()
    fp = (new_m & ~is_gt).sum()
    pur = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"{label:35s}: New TP={tp:4,d}, FP={fp:4,d}, Purity={pur:.2%}")
