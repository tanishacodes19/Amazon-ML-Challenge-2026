import sys
sys.path.insert(0, 'src')
import os
import polars as pl
import numpy as np
from eval_framework import load_benchmark, compute_macro_f05
from feature_engine_v2 import V5_FEATURES_EXPANDED
import xgboost as xgb
import lightgbm as lgb

s1_val, gt_val, _, s23_val = load_benchmark()
cands = pl.read_parquet('validation_benchmark/val_v11_cands.parquet')
X_val = np.load('validation_benchmark/val_v11_features52_X.npy')

xgb_model = xgb.XGBClassifier()
xgb_model.load_model('model/xgboost_v11_52features.json')
lgb_model = lgb.Booster(model_file='model/lightgbm_v11_52features.txt')

p_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_lgb = lgb_model.predict(X_val)
p_ens = 0.60 * p_xgb + 0.40 * p_lgb

idx_name_ratio = V5_FEATURES_EXPANDED.index('name_ratio')
idx_addr_ratio = V5_FEATURES_EXPANDED.index('address_ratio')
idx_house_match = V5_FEATURES_EXPANDED.index('house_match')
idx_name_exact = V5_FEATURES_EXPANDED.index('name_exact')
idx_addr_exact = V5_FEATURES_EXPANDED.index('address_exact')
idx_name_token_jaccard = V5_FEATURES_EXPANDED.index('name_token_jaccard')
idx_addr_token_jaccard = V5_FEATURES_EXPANDED.index('address_token_jaccard')

name_ratios = X_val[:, idx_name_ratio]
addr_ratios = X_val[:, idx_addr_ratio]
house_matches = X_val[:, idx_house_match]
name_exacts = X_val[:, idx_name_exact]
addr_exacts = X_val[:, idx_addr_exact]
name_jaccards = X_val[:, idx_name_token_jaccard]
addr_jaccards = X_val[:, idx_addr_token_jaccard]

s1_ids = cands['source1_entity_id'].to_numpy()
m_ids = cands['matched_entity_id'].to_numpy()

gt_s1 = set(gt_val['source1_entity_id'].to_list())
all_s1 = set(s1_val['source1_entity_id'].to_list())
true_singletons = all_s1 - gt_s1

print(f"Total True Singletons in Validation Set: {len(true_singletons):,}")

# Baseline check
base_gate = (p_ens >= 0.90) & ((name_ratios >= 0.25) | (addr_ratios >= 0.50) | (house_matches == 1))
res_base = compute_macro_f05(pl.DataFrame({"source1_entity_id": s1_ids[base_gate], "matched_entity_id": m_ids[base_gate]}), gt_val, s1_val)
print(f"Baseline Macro F0.5 @ tau=0.90: {res_base['macro_f05']:.4f} (TP={res_base['tp']:,}, FP={res_base['fp']:,}, Singl FP={res_base['singleton_fp']})")

# Let's test different decision gates
print("\n" + "=" * 70)
print("TESTING ADVANCED GATE LOGIC")
print("=" * 70)

for min_name_ratio in [0.20, 0.25, 0.30, 0.35]:
    for min_addr_ratio in [0.30, 0.40, 0.50]:
        for tau in [0.89, 0.90, 0.91, 0.92]:
            gate = (p_ens >= tau) & (
                ((name_ratios >= min_name_ratio) & (addr_ratios >= min_addr_ratio)) |
                (house_matches == 1) |
                (name_exacts == 1) |
                (addr_exacts == 1)
            )
            res = compute_macro_f05(pl.DataFrame({"source1_entity_id": s1_ids[gate], "matched_entity_id": m_ids[gate]}), gt_val, s1_val)
            if res['macro_f05'] >= 0.9265:
                print(f"tau={tau:.2f}, name_min={min_name_ratio:.2f}, addr_min={min_addr_ratio:.2f} -> Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['precision']*100:.2f}% | Rec: {res['recall']*100:.2f}% | Singl FP: {res['singleton_fp']}")
