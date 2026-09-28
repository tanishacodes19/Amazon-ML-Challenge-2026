import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, numpy as np, polars as pl
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
sys.path.append(os.path.join(BASE, "src"))

VAL_DIR = os.path.join(BASE, "validation_benchmark")
gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))

import xgboost as xgb, lightgbm as lgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))
p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

from feature_engine_v2 import V5_FEATURES_EXPANDED
feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# Baseline gate
base_g = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
baseline_gate = ((base_g | rescue1) & (name_r >= 0.40)) | ((p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)) | ((p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40))

outside = ~baseline_gate

def evaluate_rescue(name, mask):
    tp = np.sum(mask & is_gt)
    fp = np.sum(mask & ~is_gt)
    purity = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"{name:<60} | TP={tp:>5,} | FP={fp:>4,} | Purity={purity*100:>6.2f}%")

print("=" * 85)
print("TESTING CANDIDATE RESCUES ON UNSELECTED CANDIDATES")
print("=" * 85)

evaluate_rescue("Rule 1A: (name>=0.80 & addr>=0.80 & p>=0.80)", outside & (name_r >= 0.80) & (addr_r >= 0.80) & (p_ens >= 0.80))
evaluate_rescue("Rule 1B: (name>=0.80 & addr>=0.80 & p>=0.60)", outside & (name_r >= 0.80) & (addr_r >= 0.80) & (p_ens >= 0.60))
evaluate_rescue("Rule 1C: (name>=0.85 & addr>=0.85 & p>=0.50)", outside & (name_r >= 0.85) & (addr_r >= 0.85) & (p_ens >= 0.50))
evaluate_rescue("Rule 2A: (Acronym & addr>=0.80 & p>=0.70)", outside & (is_acronym == 1) & (addr_r >= 0.80) & (p_ens >= 0.70))
evaluate_rescue("Rule 2B: (Acronym & addr>=0.90 & p>=0.50)", outside & (is_acronym == 1) & (addr_r >= 0.90) & (p_ens >= 0.50))
evaluate_rescue("Rule 3A: (name>=0.95 & addr<0.20 & p>=0.80)", outside & (name_r >= 0.95) & (addr_r < 0.20) & (p_ens >= 0.80))
evaluate_rescue("Rule 3B: (name>=0.90 & addr<0.20 & p>=0.80)", outside & (name_r >= 0.90) & (addr_r < 0.20) & (p_ens >= 0.80))
evaluate_rescue("Rule 4:  (Single Brand Word Containment & addr>=0.85 & p>=0.80)", outside & (name_c == 1) & (addr_r >= 0.85) & (p_ens >= 0.80))

print("=" * 85)
