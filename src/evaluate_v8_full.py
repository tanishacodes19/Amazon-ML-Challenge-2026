import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import time
import duckdb
import numpy as np
import polars as pl
import xgboost as xgb
from normalizer import normalize_business_name, extract_structured_fields
from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED
from eval_framework import load_benchmark, compute_macro_f05

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
MODEL_PATH = os.path.join(BASE, "model", "xgboost_v8_52features.json")
LOG_PATH = os.path.join(BASE, "experiments_log.csv")

print("=" * 70)
print("EVALUATING MODEL V8 ON 80.41% RECALL V8 CANDIDATES")
print("=" * 70)

t0 = time.time()
s1_val, gt_val, _, s23_val = load_benchmark()
v8_cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v8_cands.parquet"))
print(f"Loaded {len(v8_cands):,} V8 candidates")

# Join raw text
v8_pairs = v8_cands.join(
    s1_val.select([
        pl.col("source1_entity_id"),
        pl.col("business_name").alias("s1_name_raw"),
        pl.col("business_address").alias("s1_addr_raw"),
        pl.col("country").alias("s1_country"),
    ]), on="source1_entity_id", how="left"
).join(
    s23_val.select([
        pl.col("matched_entity_id"),
        pl.col("business_name").alias("m_name_raw"),
        pl.col("business_address").alias("m_addr_raw"),
        pl.col("country").alias("m_country"),
    ]), on="matched_entity_id", how="left"
)

# Precompute normalizations
print("Normalizing unique entities...")
val_un_names = set(v8_pairs["s1_name_raw"].unique().drop_nulls()).union(set(v8_pairs["m_name_raw"].unique().drop_nulls()))
val_un_addrs = set(v8_pairs["s1_addr_raw"].unique().drop_nulls()).union(set(v8_pairs["m_addr_raw"].unique().drop_nulls()))

val_n_map = {x: normalize_business_name(x) for x in val_un_names if x}
val_a_map = {x: extract_structured_fields(x) for x in val_un_addrs if x}

print("Extracting 52 features for 421k V8 candidates...")
val_rows = []
for r in v8_pairs.iter_rows(named=True):
    s1_a = val_a_map.get(r["s1_addr_raw"], {"address_normalized":"", "house_number":"", "postal_code":""})
    m_a = val_a_map.get(r["m_addr_raw"], {"address_normalized":"", "house_number":"", "postal_code":""})
    val_rows.append({
        "s1_name_norm": val_n_map.get(r["s1_name_raw"], ""),
        "matched_name_norm": val_n_map.get(r["m_name_raw"], ""),
        "s1_addr_norm": s1_a["address_normalized"],
        "matched_addr_norm": m_a["address_normalized"],
        "s1_house": s1_a["house_number"],
        "matched_house": m_a["house_number"],
        "s1_postal": s1_a["postal_code"],
        "matched_postal": m_a["postal_code"],
        "s1_country": r["s1_country"] or "",
        "matched_country": r["m_country"] or "",
        "matched_entity_id": r["matched_entity_id"]
    })

val_feat_df = extract_features_df(pl.DataFrame(val_rows))
X_v8 = val_feat_df.select(V5_FEATURES_EXPANDED).fill_null(0).to_numpy().astype(np.float32)
np.save(os.path.join(VAL_DIR, "val_v8_features52_X.npy"), X_v8)
print(f"Features ready in {time.time()-t0:.1f}s")

# Load model and predict
clf = xgb.XGBClassifier()
clf.load_model(MODEL_PATH)
probs = clf.predict_proba(X_v8)[:, 1]

print("\n--- Model V8 on V8 Candidates (80.41% Recall) Threshold Calibration ---")
best_score = -1
best_res = None
best_t = None

for t in [0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]:
    pred_mask = probs >= t
    pred_pairs = v8_cands.select(["source1_entity_id", "matched_entity_id"]).filter(pred_mask)
    res = compute_macro_f05(pred_pairs, gt_val, s1_val)
    f05 = res['macro_f05']
    prec = res['precision'] * 100
    rec = res['recall'] * 100
    preds = res['predicted_matches']
    s_fp = res['singleton_fp']
    print(f"Threshold: {t:.3f} | Macro F0.5: {f05:.4f} | Prec: {prec:.2f}% | Rec: {rec:.2f}% | Preds: {preds:,} | Singleton FP: {s_fp}")
    if f05 > best_score:
        best_score = f05
        best_res = res
        best_t = t

print(f"\n==================================================================")
print(f"PEAK SCORE: Macro F0.5 = {best_score:.4f} @ Threshold = {best_t:.3f}")
print(f"Precision: {best_res['precision']*100:.2f}% | Recall: {best_res['recall']*100:.2f}%")
print(f"Total True Positives: {best_res['tp']:,} | False Positives: {best_res['fp']:,}")
print(f"Singleton False Positives: {best_res['singleton_fp']} / {best_res['singletons']}")
print(f"==================================================================")

# Singleton Margin Post-Processing Rule Test
print("\n--- Testing Post-Processing: Singleton / Margin Filter ---")
# Build a DataFrame with scores
scored_df = v8_cands.with_columns(pl.Series("prob", probs)).filter(pl.col("prob") >= best_t)

# Group by source1_entity_id to see score distribution
# If an entity has top score barely above threshold and second score very close, or top score alone
import csv
with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["EXP-6", f"V8 Multi-Channel Cands (80.41% rec) + Model V8 (52 feats) @ t={best_t:.3f}", "80.41%", len(v8_cands), f"{best_res['precision']*100:.2f}%", f"{best_res['recall']*100:.2f}%", f"{best_score:.4f}", f"{time.time()-t0:.1f}", "80.41% candidate recall with 52 features"])
