import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import os, numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

print("1. Loading validation benchmark data...")
gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))

import xgboost as xgb, lightgbm as lgb

print("2. Predicting with V13 Models...")
xgb_v13 = xgb.XGBClassifier()
xgb_v13.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_v13 = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))
p_v13 = 0.50 * xgb_v13.predict_proba(X_val)[:, 1] + 0.50 * lgb_v13.predict(X_val)

print("3. Predicting with V17 Models...")
xgb_v17 = xgb.XGBClassifier()
xgb_v17.load_model(os.path.join(BASE, "model", "xgboost_v17_52features.json"))
lgb_v17 = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v17_52features.txt"))
p_v17 = 0.50 * xgb_v17.predict_proba(X_val)[:, 1] + 0.50 * lgb_v17.predict(X_val)

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

print("\n=== SWEEPING ENSEMBLE BLEND (ALPHA * V13 + (1-ALPHA) * V17) ===")
best_f05 = 0.0
best_blend = None

for alpha in [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0]:
    p_blend = alpha * p_v13 + (1.0 - alpha) * p_v17
    
    for tau in [0.950, 0.960, 0.970, 0.972, 0.975, 0.980]:
        for min_nr in [0.35, 0.40, 0.45]:
            bg = (p_blend >= tau) & (name_r >= min_nr) & ((addr_r >= 0.30) | (house_m == 1) | (postal_e == 1))
            rescue1 = (p_blend >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
            gate = ((bg | rescue1) & (name_r >= min_nr)) | ((p_blend >= 0.95) & (name_c == 1) & (addr_r >= 0.65)) | ((p_blend >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40))
            
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
            
            if f05 > best_f05:
                best_f05 = f05
                best_blend = (alpha, tau, min_nr, tp, fp, p, r, f05)
                print(f"NEW BEST! alpha={alpha:.2f}, tau={tau:.3f}, min_nr={min_nr:.2f} -> TP={tp:5,d}, FP={fp:3,d}, P={p:.4%}, R={r:.4%}, F0.5={f05:.6f}")

print("\n" + "="*70)
print(f"OVERALL BEST BLEND RESULT:")
print(f"  alpha (V13 weight): {best_blend[0]:.2f}")
print(f"  tau:                {best_blend[1]:.3f}")
print(f"  min_nr:             {best_blend[2]:.2f}")
print(f"  TP:                 {best_blend[3]:,}")
print(f"  FP:                 {best_blend[4]:,}")
print(f"  Precision:          {best_blend[5]:.4%}")
print(f"  Recall:             {best_blend[6]:.4%}")
print(f"  Global F0.5:        {best_blend[7]:.6f}")
print("="*70)
