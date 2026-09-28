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
total_gt = len(gt_set)
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}

name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house = feat_dict["house_match"]
first_tok = feat_dict["name_first_token_match"]
last_tok = feat_dict["name_last_token_match"]
shared_tok = feat_dict["name_shared_token_count"]
token_jacc = feat_dict["name_token_jaccard"]
acronym = feat_dict["name_acronym_match"]
name_contains = feat_dict["name_contains"]

print("=" * 95)
print("TESTING TARGETED RESCUES & ADVANCED ANTI-MERGE RULES")
print("=" * 95)

# Current Baseline:
base_g = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house == 1))
rescue_old = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house == 1)) | (addr_r >= 0.95))
raw_gate = (base_g | rescue_old) & (name_r >= 0.40)

def score_mask(mask, name):
    pred_set = set(zip(s1_ids[mask], m_ids[mask]))
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    prec = tp / len(pred_set) if len(pred_set) > 0 else 0
    rec = tp / total_gt
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0
    print(f"{name:<55} | F0.5={f05:.4f} | Prec={prec*100:.2f}% | Rec={rec*100:.2f}% | TP={tp:,} | FP={fp:,}")
    return f05

score_mask(raw_gate, "Baseline (V14 Ensemble + min_name>=0.40)")

# Enhancement 1: Single Brand Token Containment Rescue
# When S23 is contained in S1 (like 'Sanghi' in 'Sanghi Association' or 'ME' for 'Mount Emanuel')
# and Address is high (addr_r >= 0.65) and model score >= 0.95:
containment_rescue = (p_ens >= 0.95) & (name_contains == 1) & (addr_r >= 0.65)
score_mask(raw_gate | containment_rescue, "+ Containment Rescue (Single Brand Name + Addr>=0.65)")

# Enhancement 2: Acronym Rescue (acronym == 1 and addr_r >= 0.40 and p_ens >= 0.90)
acronym_rescue = (p_ens >= 0.90) & (acronym == 1) & (addr_r >= 0.40)
score_mask(raw_gate | containment_rescue | acronym_rescue, "+ Acronym Rescue (Acronym Match + Addr>=0.40)")

# Enhancement 3: High-Confidence Branch Conflict Filter
# If name_ratio is between 0.60 and 0.88, address_ratio is high (>=0.80),
# BUT name_token_jaccard is low (< 0.40) and name_shared_token_count <= 1
# (e.g. 'Green Products' vs 'Green Constructions', 'Thiruvanmiyur Pack' vs 'Thiruvanmiyur Films')
branch_conflict = (name_r >= 0.50) & (name_r < 0.88) & (addr_r >= 0.75) & (token_jacc < 0.40) & (shared_tok <= 1) & (acronym == 0)

clean_gate = (raw_gate | containment_rescue | acronym_rescue) & ~branch_conflict
score_mask(clean_gate, "+ Branch Disambiguation Filter (Eliminates Green Products/Constructions)")

# Enhancement 4: Ultra Threshold Sweep on clean_gate
for base_t in [0.965, 0.970, 0.972, 0.975, 0.980]:
    b = (p_ens >= base_t) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house == 1))
    r = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house == 1)) | (addr_r >= 0.95))
    g = ((b | r) & (name_r >= 0.40)) | containment_rescue | acronym_rescue
    g = g & ~branch_conflict
    score_mask(g, f"Tuned Base tau={base_t:.3f} + Branch Guard + Containment")
