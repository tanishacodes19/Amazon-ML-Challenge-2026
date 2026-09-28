import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
sys.path.append(os.path.join(BASE, "src"))

import numpy as np
import polars as pl

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
name_r = pairs_df["name_ratio"].to_numpy()
addr_r = pairs_df["address_ratio"].to_numpy()
house = pairs_df["house_match"].to_numpy()

from feature_engine_v2 import V5_FEATURES_EXPANDED
name_contains = X_val[:, V5_FEATURES_EXPANDED.index("name_contains")]
first_tok = X_val[:, V5_FEATURES_EXPANDED.index("name_first_token_match")]
postal_ex = X_val[:, V5_FEATURES_EXPANDED.index("postal_exact")]
acronym = X_val[:, V5_FEATURES_EXPANDED.index("name_acronym_match")]
name_ex = X_val[:, V5_FEATURES_EXPANDED.index("name_exact")]
strong_na = X_val[:, V5_FEATURES_EXPANDED.index("strong_name_address")]

gt_set = set(zip(gt_val["source1_entity_id"], gt_val["matched_entity_id"]))
total_gt = len(gt_set)

print("=" * 95)
print(f"{'Gate Strategy':<55} | {'Global F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'TP':>7} | {'FP':>5}")
print("-" * 95)

# Strategy 1: Ultra precision gate
for tau in [0.980, 0.985, 0.990, 0.992, 0.995]:
    # Require either first_token match, acronym, or high name ratio
    gate = (p_ens >= tau) & ((name_r >= 0.40) | (acronym == 1))
    pred_set = set(zip(s1_ids[gate], m_ids[gate]))
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    prec = tp / len(pred_set) if len(pred_set) > 0 else 0
    rec = tp / total_gt
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0
    print(f"Base tau={tau:.3f} + name>=0.40                               | {f05:>11.4f} | {prec*100:>9.2f}% | {rec*100:>7.2f}% | {tp:>7,} | {fp:>5,}")

# Strategy 2: Ultra precision base + multi-tier infallible rescues
for tau in [0.980, 0.985, 0.990]:
    for r_score in [0.85, 0.90]:
        base = (p_ens >= tau) & ((name_r >= 0.45) | (acronym == 1))
        # Rescue 1: Exact Name + House Match
        r_exact = (name_ex == 1) & (house == 1) & (p_ens >= 0.50)
        # Rescue 2: Exact Name + Exact Postal
        r_np = (name_ex == 1) & (postal_ex == 1) & (p_ens >= 0.50)
        # Rescue 3: High name + High address + model >= r_score
        r_addr = (p_ens >= r_score) & (name_r >= 0.80) & (addr_r >= 0.80)
        # Rescue 4: First token match + House match + model >= r_score
        r_house = (p_ens >= r_score) & (first_tok == 1) & (house == 1) & (name_r >= 0.60)
        
        comb = base | r_exact | r_np | r_addr | r_house
        pred_set = set(zip(s1_ids[comb], m_ids[comb]))
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        prec = tp / len(pred_set) if len(pred_set) > 0 else 0
        rec = tp / total_gt
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0
        tag = f"tau={tau:.3f} + infallible rescues (r_score={r_score:.2f})"
        print(f"{tag:<55} | {f05:>11.4f} | {prec*100:>9.2f}% | {rec*100:>7.2f}% | {tp:>7,} | {fp:>5,}")
