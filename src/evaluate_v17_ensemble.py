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

print("2. Predicting with V17 XGBoost + LightGBM models...")
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

print("\n=== SWEEPING V17 THRESHOLDS WITH S23-INJECTIVE CONSTRAINT ===")
best_f05 = 0.0
best_row = None

for base_tau in [0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98]:
    for min_nr in [0.20, 0.30, 0.40, 0.45]:
        bg = (p_ens >= base_tau) & (name_r >= min_nr) & ((addr_r >= 0.30) | (house_m == 1) | (postal_e == 1))
        # Rescue channels
        r1 = (p_ens >= 0.85) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
        r2 = (is_acronym == 1) & (addr_r >= 0.70)
        gate = bg | r1 | r2
        
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
        
        if f05 > best_f05:
            best_f05 = f05
            best_row = (base_tau, min_nr, tp, fp, p, r, f05)
            print(f"NEW BEST! tau={base_tau:.3f}, min_nr={min_nr:.2f} -> TP={tp:5,d}, FP={fp:3,d}, P={p:.4%}, R={r:.4%}, F0.5={f05:.6f}")

print("\n" + "="*70)
print(f"V17 OPTIMAL BENCHMARK RESULT:")
print(f"  base_tau: {best_row[0]}")
print(f"  min_nr:   {best_row[1]}")
print(f"  TP:       {best_row[2]:,}")
print(f"  FP:       {best_row[3]:,}")
print(f"  Precision:{best_row[4]:.4%}")
print(f"  Recall:   {best_row[5]:.4%}")
print(f"  F0.5:     {best_row[6]:.6f}")
print("="*70)
