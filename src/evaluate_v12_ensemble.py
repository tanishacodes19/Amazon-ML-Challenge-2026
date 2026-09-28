import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import time
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb
from normalizer import normalize_business_name, extract_structured_fields
from eval_framework import load_benchmark, compute_macro_f05
from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
MODEL_DIR = os.path.join(BASE, "model")

XGB_PATH = os.path.join(MODEL_DIR, "xgboost_v11_52features.json")
LGB_PATH = os.path.join(MODEL_DIR, "lightgbm_v11_52features.txt")

V12_CANDS_PATH = os.path.join(VAL_DIR, "val_v12_cands.parquet")
FEAT_CACHE_X = os.path.join(VAL_DIR, "val_v12_features52_X.npy")
FEAT_CACHE_DF = os.path.join(VAL_DIR, "val_v12_features52_df.parquet")

print("=" * 70)
print("EVALUATING V12 CANDIDATE SET (91.09% BLOCKING RECALL) WITH ENSEMBLE")
print("=" * 70)

t0 = time.time()
s1_val, gt_val, _, s23_val = load_benchmark()
cands = pl.read_parquet(V12_CANDS_PATH)
print(f"Loaded {len(cands):,} V12 candidates in {time.time()-t0:.2f}s")

if os.path.exists(FEAT_CACHE_X) and os.path.exists(FEAT_CACHE_DF):
    print(f"Loading cached 52 features from {FEAT_CACHE_X}...")
    X_val = np.load(FEAT_CACHE_X)
    pairs_df = pl.read_parquet(FEAT_CACHE_DF)
else:
    print("Joining candidates with text...")
    v12_pairs = cands.join(
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
    val_un_names = set(v12_pairs["s1_name_raw"].unique().drop_nulls()).union(set(v12_pairs["m_name_raw"].unique().drop_nulls()))
    val_un_addrs = set(v12_pairs["s1_addr_raw"].unique().drop_nulls()).union(set(v12_pairs["m_addr_raw"].unique().drop_nulls()))
    
    val_n_map = {x: normalize_business_name(x) for x in val_un_names if x}
    val_a_map = {x: extract_structured_fields(x) for x in val_un_addrs if x}
    
    print(f"Extracting 52 features for {len(v12_pairs):,} V12 candidates...")
    t_feat = time.time()
    val_rows = []
    for r in v12_pairs.iter_rows(named=True):
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

# Load Models
print("\nLoading Trained Ensemble Models...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_PATH)

lgb_model = lgb.Booster(model_file=LGB_PATH)

print("Scoring V12 candidates...")
p_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_lgb = lgb_model.predict(X_val)
p_ens = 0.60 * p_xgb + 0.40 * p_lgb

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
name_ratios = pairs_df["name_ratio"].to_numpy()
addr_ratios = pairs_df["address_ratio"].to_numpy()
house_matches = pairs_df["house_match"].to_numpy()

models_eval = [
    ("Model A (XGBoost V11)", p_xgb),
    ("Model B (LightGBM V11)", p_lgb),
    ("Ensemble Blend (60% XGB + 40% LGB)", p_ens)
]

for m_name, probs in models_eval:
    print(f"\n--- {m_name} on V12 Candidates ---")
    print(f"{'Threshold':>10} | {'Macro F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'TP':>7} | {'FP':>6} | {'Singl FP':>8}")
    print("-" * 70)
    best_f = 0.0
    best_t = 0.0
    for tau in [0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95]:
        gate = (probs >= tau) & ((name_ratios >= 0.25) | (addr_ratios >= 0.50) | (house_matches == 1))
        pred_df = pl.DataFrame({
            "source1_entity_id": s1_ids[gate],
            "matched_entity_id": m_ids[gate]
        })
        res = compute_macro_f05(pred_df, gt_val, s1_val)
        if res["macro_f05"] > best_f:
            best_f = res["macro_f05"]
            best_t = tau
        print(f"{tau:>10.3f} | {res['macro_f05']:>11.4f} | {res['precision']*100:>9.2f}% | {res['recall']*100:>7.2f}% | {res['tp']:>7,} | {res['fp']:>6,} | {res['singleton_fp']:>8,}")
    print(f"Peak for {m_name}: {best_f:.4f} @ tau={best_t:.3f}")

print("\n" + "=" * 70)
print(f"Evaluation Completed in {time.time()-t0:.1f}s")
print("=" * 70)
