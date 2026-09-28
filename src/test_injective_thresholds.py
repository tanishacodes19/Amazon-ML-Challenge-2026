import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'src')
import os, numpy as np, polars as pl
from collections import defaultdict
from eval_framework import load_benchmark, compute_macro_f05

VAL_DIR = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026\validation_benchmark"
s1_val, gt_val, cands_val, s23_val = load_benchmark()

X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))

import xgboost as xgb, lightgbm as lgb
xgb_m = xgb.XGBClassifier()
xgb_m.load_model("model/xgboost_v13_52features.json")
lgb_m = lgb.Booster(model_file="model/lightgbm_v13_52features.txt")

p_xgb = xgb_m.predict_proba(X_val)[:, 1]
p_lgb = lgb_m.predict(X_val)
p_blend = 0.50 * p_xgb + 0.50 * p_lgb

from feature_engine_v2 import V5_FEATURES_EXPANDED
idx_name_ratio = V5_FEATURES_EXPANDED.index("name_ratio")
name_r = X_val[:, idx_name_ratio]

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()

print(f"{'Threshold':>10} | {'Macro F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'Injective Pairs':>16} | {'Matched S1':>12}")
print("-" * 85)

for tau in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]:
    # Gate at tau with basic name guard
    mask = (p_blend >= tau) & (name_r >= 0.25)
    
    sub_s1 = s1_ids[mask]
    sub_m = m_ids[mask]
    sub_p = p_blend[mask]
    
    # Sort descending by probability
    order = np.argsort(-sub_p)
    sorted_s1 = sub_s1[order]
    sorted_m = sub_m[order]
    
    # Injective resolution
    seen_m = set()
    inj_s1 = []
    inj_m = []
    for sid, mid in zip(sorted_s1, sorted_m):
        if mid not in seen_m:
            seen_m.add(mid)
            inj_s1.append(sid)
            inj_m.append(mid)
            
    pred_df = pl.DataFrame({
        "source1_entity_id": inj_s1,
        "matched_entity_id": inj_m
    })
    
    res = compute_macro_f05(pred_df, gt_val, s1_val)
    n_s1_matched = len(set(inj_s1))
    print(f"{tau:>10.2f} | {res['macro_f05']:>11.4f} | {res['precision']*100:>9.2f}% | {res['recall']*100:>7.2f}% | {len(inj_s1):>16,} | {n_s1_matched:>12,}")
