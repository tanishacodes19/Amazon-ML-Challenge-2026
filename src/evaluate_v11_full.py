import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import time
import numpy as np
import polars as pl
import xgboost as xgb
from normalizer import normalize_business_name, extract_structured_fields
from eval_framework import load_benchmark, compute_macro_f05
from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
MODEL_PATH = os.path.join(BASE, "model", "xgboost_v8_52features.json")
V11_CANDS_PATH = os.path.join(VAL_DIR, "val_v11_cands.parquet")
FEAT_CACHE_X = os.path.join(VAL_DIR, "val_v11_features52_X.npy")
FEAT_CACHE_DF = os.path.join(VAL_DIR, "val_v11_features52_df.parquet")

print("=" * 70)
print("EVALUATING V11 CANDIDATE SET (89.98% BLOCKING RECALL) WITH MODEL V8")
print("=" * 70)

t0 = time.time()
s1_val, gt_val, _, s23_val = load_benchmark()
cands = pl.read_parquet(V11_CANDS_PATH)
print(f"Loaded {len(cands):,} V11 candidates in {time.time()-t0:.2f}s")

if os.path.exists(FEAT_CACHE_X) and os.path.exists(FEAT_CACHE_DF):
    print(f"Loading cached 52 features from {FEAT_CACHE_X}...")
    X_val = np.load(FEAT_CACHE_X)
    pairs_df = pl.read_parquet(FEAT_CACHE_DF)
else:
    print("Joining candidates with text...")
    v11_pairs = cands.join(
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
            pl.col("country").alias("matched_country"),
        ]), on="matched_entity_id", how="left"
    )
    
    print("Normalizing unique entities...")
    val_un_names = set(v11_pairs["s1_name_raw"].unique().drop_nulls()).union(set(v11_pairs["m_name_raw"].unique().drop_nulls()))
    val_un_addrs = set(v11_pairs["s1_addr_raw"].unique().drop_nulls()).union(set(v11_pairs["m_addr_raw"].unique().drop_nulls()))
    
    val_n_map = {x: normalize_business_name(x) for x in val_un_names if x}
    val_a_map = {x: extract_structured_fields(x) for x in val_un_addrs if x}
    
    print(f"Extracting 52 features for {len(v11_pairs):,} V11 candidates...")
    t_feat = time.time()
    val_rows = []
    for r in v11_pairs.iter_rows(named=True):
        s1_a = val_a_map.get(r["s1_addr_raw"], {"address_normalized":"", "house_number":"", "postal_code":""})
        m_a = val_a_map.get(r["m_addr_raw"], {"address_normalized":"", "house_number":"", "postal_code":""})
        val_rows.append({
            "source1_entity_id": r["source1_entity_id"],
            "matched_entity_id": r["matched_entity_id"],
            "s1_name_norm": val_n_map.get(r["s1_name_raw"], ""),
            "matched_name_norm": val_n_map.get(r["m_name_raw"], ""),
            "s1_addr_norm": s1_a["address_normalized"],
            "matched_addr_norm": m_a["address_normalized"],
            "s1_house": s1_a["house_number"],
            "matched_house": m_a["house_number"],
            "s1_postal": s1_a["postal_code"],
            "matched_postal": m_a["postal_code"],
            "s1_country": r["s1_country"] or "",
            "matched_country": r["matched_country"] or "",
        })
    
    pairs_pl = pl.DataFrame(val_rows)
    feat_pl = extract_features_df(pairs_pl)
    print(f"Features extracted in {time.time()-t_feat:.1f}s")
    
    X_val = feat_pl.select(V5_FEATURES_EXPANDED).to_numpy()
    np.save(FEAT_CACHE_X, X_val)
    
    pairs_df = pl.DataFrame({
        "source1_entity_id": cands["source1_entity_id"],
        "matched_entity_id": cands["matched_entity_id"],
        "name_ratio": feat_pl["name_ratio"],
        "address_ratio": feat_pl["address_ratio"],
        "house_match": feat_pl["house_match"]
    })
    pairs_df.write_parquet(FEAT_CACHE_DF)
    print(f"Saved feature cache to {FEAT_CACHE_X}")

# Load Model V8
print(f"\nLoading Model V8 from {MODEL_PATH}...")
bst = xgb.Booster()
bst.load_model(MODEL_PATH)

dval = xgb.DMatrix(X_val)
probs = bst.predict(dval)
print(f"Predictions generated for {len(probs):,} pairs. Prob range: [{probs.min():.4f}, {probs.max():.4f}]")

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
name_ratios = pairs_df["name_ratio"].to_numpy()
addr_ratios = pairs_df["address_ratio"].to_numpy()
house_matches = pairs_df["house_match"].to_numpy()

print("\n" + "=" * 70)
print(f"{'Threshold':>10} | {'Macro F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'TP':>7} | {'FP':>6} | {'Singl FP':>8}")
print("=" * 70)

best_score = 0.0
best_tau = 0.0

for tau in [0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]:
    gate = (probs >= tau) & ((name_ratios >= 0.25) | (addr_ratios >= 0.50) | (house_matches == 1))
    
    pred_s1 = s1_ids[gate]
    pred_m = m_ids[gate]
    
    pred_df = pl.DataFrame({
        "source1_entity_id": pred_s1,
        "matched_entity_id": pred_m
    })
    
    metrics = compute_macro_f05(pred_df, gt_val, s1_val)
    f05 = metrics["macro_f05"]
    p = metrics["precision"] * 100.0
    r = metrics["recall"] * 100.0
    tp = metrics["tp"]
    fp = metrics["fp"]
    singl_fp = metrics["singleton_fp"]
    
    if f05 > best_score:
        best_score = f05
        best_tau = tau
        
    print(f"{tau:>10.3f} | {f05:>11.4f} | {p:>9.2f}% | {r:>7.2f}% | {tp:>7,} | {fp:>6,} | {singl_fp:>8,}")

print("=" * 70)
print(f"PEAK MACRO F0.5 ON V11 CANDIDATES: {best_score:.4f} at tau={best_tau:.3f}")
print(f"Comparison: V8 Peak = 0.8833 | V9 Peak = 0.9159 | V10 Peak = 0.9196 | V11 Peak = {best_score:.4f}")
print("=" * 70)
