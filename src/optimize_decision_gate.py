import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import time
import numpy as np
import polars as pl
import xgboost as xgb
from eval_framework import load_benchmark, compute_macro_f05
from feature_engine_v2 import V5_FEATURES_EXPANDED

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
MODEL_PATH = os.path.join(BASE, "model", "xgboost_v8_52features.json")
V9_CANDS_PATH = os.path.join(VAL_DIR, "val_v9_cands.parquet")
FEAT_CACHE_X = os.path.join(VAL_DIR, "val_v9_features52_X.npy")

s1_val, gt_val, _, _ = load_benchmark()
cands = pl.read_parquet(V9_CANDS_PATH)
X_val = np.load(FEAT_CACHE_X)

bst = xgb.Booster()
bst.load_model(MODEL_PATH)
probs = bst.predict(xgb.DMatrix(X_val))

idx_name_ratio = V5_FEATURES_EXPANDED.index("name_ratio")
idx_addr_ratio = V5_FEATURES_EXPANDED.index("address_ratio")
idx_house_match = V5_FEATURES_EXPANDED.index("house_match")
idx_postal_exact = V5_FEATURES_EXPANDED.index("postal_exact")
idx_name_tok_jacc = V5_FEATURES_EXPANDED.index("name_token_jaccard")

name_ratios = X_val[:, idx_name_ratio]
addr_ratios = X_val[:, idx_addr_ratio]
house_matches = X_val[:, idx_house_match]
postal_exacts = X_val[:, idx_postal_exact]
name_tok_jaccs = X_val[:, idx_name_tok_jacc]

s1_ids = cands["source1_entity_id"].to_numpy()
m_ids = cands["matched_entity_id"].to_numpy()

print("=" * 70)
print("TESTING ADVANCED PRECISION GATING RULES TO PUSH BEYOND 0.9159")
print("=" * 70)

rules = [
    ("Baseline: prob >= 0.900", lambda p, tau: p >= tau),
    ("Rule 1: prob >= tau & (name >= 0.25 | addr >= 0.50 | house == 1)", 
     lambda p, tau: (p >= tau) & ((name_ratios >= 0.25) | (addr_ratios >= 0.50) | (house_matches == 1))),
    ("Rule 2: prob >= tau & (name >= 0.30 | addr >= 0.40 | house == 1 | postal == 1)", 
     lambda p, tau: (p >= tau) & ((name_ratios >= 0.30) | (addr_ratios >= 0.40) | (house_matches == 1) | (postal_exacts == 1))),
    ("Rule 3: prob >= tau & (name >= 0.20 & addr >= 0.20 | house == 1 | postal == 1)", 
     lambda p, tau: (p >= tau) & (((name_ratios >= 0.20) & (addr_ratios >= 0.20)) | (house_matches == 1) | (postal_exacts == 1))),
    ("Rule 4: Dynamic threshold (tau=0.88 if strong else tau=0.94)",
     lambda p, tau: np.where((name_ratios >= 0.60) | (addr_ratios >= 0.70) | ((house_matches == 1) & (postal_exacts == 1)), p >= 0.88, p >= 0.94)),
    ("Rule 5: Dual Anchor (name_jacc >= 0.30 | addr >= 0.50 | (house==1 & postal==1))",
     lambda p, tau: (p >= tau) & ((name_tok_jaccs >= 0.30) | (addr_ratios >= 0.50) | ((house_matches == 1) & (postal_exacts == 1))))
]

for name, rule_fn in rules:
    for tau in [0.88, 0.90, 0.92]:
        mask = rule_fn(probs, tau)
        pred_df = pl.DataFrame({
            "source1_entity_id": s1_ids[mask],
            "matched_entity_id": m_ids[mask]
        })
        m = compute_macro_f05(pred_df, gt_val, s1_val)
        print(f"{name[:45]:<45} | tau={tau:.2f} | F0.5={m['macro_f05']:.4f} | P={m['precision']*100:.2f}% | R={m['recall']*100:.2f}% | Singl FP={m['singleton_fp']}")
