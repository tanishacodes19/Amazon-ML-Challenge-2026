import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import os, numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))

import xgboost as xgb, lightgbm as lgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v17_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v17_52features.txt"))
p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
postal_e = feat_dict["postal_exact"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)
TOTAL_GT = 86275

print("=== SWEEPING V17 WITHOUT LOOSE RESCUES ===")
for base_tau in [0.950, 0.960, 0.970, 0.975, 0.980, 0.985, 0.990, 0.992, 0.995]:
    for min_nr in [0.20, 0.30, 0.40, 0.50]:
        gate = (p_ens >= base_tau) & (name_r >= min_nr)
        
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
        if f05 >= 0.965:
            print(f"tau={base_tau:.3f}, min_nr={min_nr:.2f} -> TP={tp:5,d}, FP={fp:3,d}, P={p:.4%}, R={r:.4%}, F0.5={f05:.6f}")
