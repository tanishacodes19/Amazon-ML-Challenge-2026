import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'src')
import os, numpy as np, polars as pl
import xgboost as xgb, lightgbm as lgb
from eval_framework import load_benchmark, compute_macro_f05

VAL_DIR = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026\validation_benchmark"
s1_val, gt_val, cands_val, s23_val = load_benchmark()

X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))

print(f"Loaded val_v16: {len(pairs_df)} pairs, {X_val.shape} features")

# Check recall of val_v16 candidates against GT
s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()

cand_pairs_set = set(zip(s1_ids, m_ids))
gt_pairs_set = set(zip(gt_val["source1_entity_id"].to_list(), gt_val["matched_entity_id"].to_list()))
rec_cands = len(cand_pairs_set & gt_pairs_set)
print(f"Candidate Blocking Recall: {rec_cands:,} / {len(gt_pairs_set):,} ({rec_cands/len(gt_pairs_set)*100:.2f}%)")

# Predict with V13 ensemble
xgb_m = xgb.XGBClassifier()
xgb_m.load_model("model/xgboost_v13_52features.json")
lgb_m = lgb.Booster(model_file="model/lightgbm_v13_52features.txt")

p_xgb = xgb_m.predict_proba(X_val)[:, 1]
p_lgb = lgb_m.predict(X_val)
p_blend = 0.50 * p_xgb + 0.50 * p_lgb

from feature_engine_v2 import V5_FEATURES_EXPANDED
idx_name_ratio = V5_FEATURES_EXPANDED.index("name_ratio")
idx_addr_ratio = V5_FEATURES_EXPANDED.index("address_ratio")
idx_house_match = V5_FEATURES_EXPANDED.index("house_match")
idx_name_contains = V5_FEATURES_EXPANDED.index("name_contains")
idx_acronym = V5_FEATURES_EXPANDED.index("name_acronym_match")

name_r = X_val[:, idx_name_ratio]
addr_r = X_val[:, idx_addr_ratio]
house_m = X_val[:, idx_house_match]
name_c = X_val[:, idx_name_contains]
is_acronym = X_val[:, idx_acronym]

# Evaluate different gates:
print("\n--- Testing Gates on Validation Set ---")
gates_to_test = [
    ("V19/V20 Gate", ((p_blend >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1)) | 
                     ((p_blend >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95)))) & 
                     (name_r >= 0.40) | 
                     ((p_blend >= 0.95) & (name_c == 1) & (addr_r >= 0.65)) | 
                     ((p_blend >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40))),
    ("Relaxed p >= 0.85 & name_r >= 0.3", (p_blend >= 0.85) & (name_r >= 0.30)),
    ("Relaxed p >= 0.80 & name_r >= 0.3", (p_blend >= 0.80) & (name_r >= 0.30)),
    ("Relaxed p >= 0.70 & name_r >= 0.3", (p_blend >= 0.70) & (name_r >= 0.30)),
    ("Relaxed p >= 0.60 & name_r >= 0.3", (p_blend >= 0.60) & (name_r >= 0.30)),
    ("Relaxed p >= 0.50 & name_r >= 0.3", (p_blend >= 0.50) & (name_r >= 0.30)),
    ("Injective Match Only (Top-1 p >= 0.5)", None)
]

for name, gate in gates_to_test:
    if gate is not None:
        pred_df = pl.DataFrame({
            "source1_entity_id": s1_ids[gate],
            "matched_entity_id": m_ids[gate]
        })
        res = compute_macro_f05(pred_df, gt_val, s1_val)
        print(f"{name:<35}: F0.5={res['macro_f05']:.4f} | Prec={res['precision']*100:.2f}% | Rec={res['recall']*100:.2f}% | TP={res['tp']:,} | FP={res['fp']:,} | Pred={res['predicted_matches']:,}")
