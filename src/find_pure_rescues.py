import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
sys.path.append(os.path.join(BASE, "src"))

import numpy as np
import polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

VAL_DIR = os.path.join(BASE, "validation_benchmark")
gt_val = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v14_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v14_features52_X.npy"))

import xgboost as xgb
import lightgbm as lgb

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))

p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt_val["source1_entity_id"], gt_val["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}

# Base candidate set
base_gate = (p_ens >= 0.972) & (
    (feat_dict["name_ratio"] >= 0.40) | 
    (feat_dict["name_acronym_match"] == 1)
) & (
    (feat_dict["name_ratio"] >= 0.20) | 
    (feat_dict["address_ratio"] >= 0.40) | 
    (feat_dict["house_match"] == 1)
)

print("Searching for high-purity rescue pockets in p_ens in [0.70, 0.972)...")

unselected = ~base_gate

# Grid search combinations
results = []
for p_min in [0.80, 0.85, 0.88, 0.90, 0.92]:
    for name_min in [0.70, 0.80, 0.85, 0.90]:
        for addr_min in [0.70, 0.80, 0.85, 0.90]:
            cond = unselected & (p_ens >= p_min) & (feat_dict["name_ratio"] >= name_min) & (feat_dict["address_ratio"] >= addr_min)
            tp_add = np.sum(cond & is_gt)
            fp_add = np.sum(cond & ~is_gt)
            if tp_add > 20:
                purity = tp_add / (tp_add + fp_add)
                results.append((p_min, name_min, addr_min, tp_add, fp_add, purity))

results.sort(key=lambda x: (x[5], x[3]), reverse=True)

print(f"{'p_min':<8} | {'name_min':<8} | {'addr_min':<8} | {'TP added':<10} | {'FP added':<10} | {'Purity %':<10}")
print("-" * 65)
for r in results[:20]:
    print(f"{r[0]:<8.2f} | {r[1]:<8.2f} | {r[2]:<8.2f} | {r[3]:<10,} | {r[4]:<10,} | {r[5]*100:<10.2f}%")
