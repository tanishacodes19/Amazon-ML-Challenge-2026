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
xgb_v13 = xgb.XGBClassifier()
xgb_v13.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_v13 = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))
p_v13 = 0.50 * xgb_v13.predict_proba(X_val)[:, 1] + 0.50 * lgb_v13.predict(X_val)

xgb_v17 = xgb.XGBClassifier()
xgb_v17.load_model(os.path.join(BASE, "model", "xgboost_v17_52features.json"))
lgb_v17 = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v17_52features.txt"))
p_v17 = 0.50 * xgb_v17.predict_proba(X_val)[:, 1] + 0.50 * lgb_v17.predict(X_val)

p_blend = 0.80 * p_v13 + 0.20 * p_v17

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

print("=== FINE-GRAINED HIGH PRECISION SWEEP (TARGETING P >= 99.3 - 99.6%) ===")
results = []
for tau in np.arange(0.960, 0.990, 0.002):
    for min_nr in [0.35, 0.40, 0.45, 0.50]:
        bg = (p_blend >= tau) & (name_r >= min_nr) & ((addr_r >= 0.30) | (house_m == 1) | (postal_e == 1))
        rescue1 = (p_blend >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95)) & (name_r >= min_nr)
        r_acr = (is_acronym == 1) & (addr_r >= 0.70)
        gate = bg | rescue1 | r_acr
        
        cands = np.where(gate)[0]
        s_ord = cands[np.argsort(-p_blend[cands])]
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
        results.append((f05, p, r, tp, fp, tau, min_nr))

results.sort(key=lambda x: x[0], reverse=True)
print(f"{'Rank':4s} | {'F0.5':8s} | {'Precision':10s} | {'Recall':8s} | {'TP':6s} | {'FP':5s} | {'tau':6s} | {'min_nr':6s}")
print("-" * 65)
for i, (f05, p, r, tp, fp, tau, min_nr) in enumerate(results[:20]):
    print(f"{i+1:4d} | {f05:8.6f} | {p:10.4%} | {r:8.4%} | {tp:6,d} | {fp:5,d} | {tau:6.3f} | {min_nr:6.2f}")
