import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
sys.path.append(os.path.join(BASE, "src"))
import time
import duckdb
import pandas as pd
import numpy as np
import polars as pl
import xgboost as xgb
from normalizer import normalize_business_name, normalize_address, extract_structured_fields
from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED
from eval_framework import load_benchmark, compute_macro_f05

VAL_DIR = os.path.join(BASE, "validation_benchmark")
MODEL_DIR = os.path.join(BASE, "model")
LOG_PATH = os.path.join(BASE, "experiments_log.csv")

print("=" * 70)
print("PHASE 7, 8, 9: TRAINING 52-FEATURE BALANCED XGBOOST & THRESHOLD TUNING")
print("=" * 70)

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

GT_PAIRS = os.path.join(BASE, "ground_truth_pairs.tsv").replace("\\", "/")
CANDIDATES = os.path.join(BASE, "training_candidate_pairs.tsv").replace("\\", "/")
S1_FILE = os.path.join(BASE, "normalized_data", "train_source1_normalized.tsv").replace("\\", "/")
S2_FILE = os.path.join(BASE, "normalized_data", "train_source2_normalized.tsv").replace("\\", "/")
S3_FILE = os.path.join(BASE, "normalized_data", "train_source3_normalized.tsv").replace("\\", "/")

# Sample 150,000 training positives from the training partition (% 4 != 0)
print("\nSampling training pairs from training split (% 4 != 0)...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE train_pos AS
SELECT g.source1_entity_id, g.matched_entity_id, 1 AS label
FROM read_csv('{GT_PAIRS}', delim='\\t', header=true) g
WHERE ABS(HASH(g.source1_entity_id)) % 4 != 0
USING SAMPLE reservoir (150000 ROWS) REPEATABLE (42)
""")

print("Sampling 150,000 hard negatives from candidate pool...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE train_cand_sample AS
SELECT c.source1_entity_id, c.candidate_entity_id AS matched_entity_id
FROM read_csv('{CANDIDATES}', delim='\\t', header=true) c
WHERE ABS(HASH(c.source1_entity_id)) % 4 != 0
USING SAMPLE reservoir (1000000 ROWS) REPEATABLE (43)
""")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE train_hard_neg AS
SELECT c.source1_entity_id, c.matched_entity_id, 0 AS label
FROM train_cand_sample c
LEFT JOIN read_csv('{GT_PAIRS}', delim='\\t', header=true) g
  ON c.source1_entity_id = g.source1_entity_id AND c.matched_entity_id = g.matched_entity_id
WHERE g.matched_entity_id IS NULL
USING SAMPLE reservoir (150000 ROWS) REPEATABLE (44)
""")

print("Sampling 50,000 random non-matching negatives to eliminate bias...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE all_s23 AS
SELECT entity_id, business_name, business_address, country FROM read_csv('{S2_FILE}', delim='\\t', header=true)
UNION ALL
SELECT entity_id, business_name, business_address, country FROM read_csv('{S3_FILE}', delim='\\t', header=true)
""")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE train_random_neg AS
WITH s1_sampled AS (
    SELECT entity_id, ROW_NUMBER() OVER () as rn
    FROM (SELECT entity_id FROM read_csv('{S1_FILE}', delim='\\t', header=true)
          WHERE ABS(HASH(entity_id)) % 4 != 0
          USING SAMPLE reservoir (50000 ROWS) REPEATABLE (45))
),
s23_sampled AS (
    SELECT entity_id, ROW_NUMBER() OVER () as rn
    FROM (SELECT entity_id FROM all_s23
          USING SAMPLE reservoir (50000 ROWS) REPEATABLE (46))
)
SELECT s1.entity_id AS source1_entity_id, s23.entity_id AS matched_entity_id, 0 AS label
FROM s1_sampled s1
JOIN s23_sampled s23 ON s1.rn = s23.rn
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE train_pairs_all AS
SELECT * FROM train_pos
UNION ALL
SELECT * FROM train_hard_neg
UNION ALL
SELECT * FROM train_random_neg
""")

print("Joining entity text directly in DuckDB...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE train_full AS
SELECT 
    p.source1_entity_id,
    p.matched_entity_id,
    p.label,
    COALESCE(s1.business_name, '') AS s1_name_raw,
    COALESCE(s1.business_address, '') AS s1_addr_raw,
    COALESCE(s1.country, '') AS s1_country,
    COALESCE(s23.business_name, '') AS m_name_raw,
    COALESCE(s23.business_address, '') AS m_addr_raw,
    COALESCE(s23.country, '') AS m_country
FROM train_pairs_all p
JOIN read_csv('{S1_FILE}', delim='\\t', header=true) s1 ON p.source1_entity_id = s1.entity_id
JOIN all_s23 s23 ON p.matched_entity_id = s23.entity_id
""")

print("Converting to Polars and precomputing normalizations...")
train_raw_pl = pl.from_arrow(con.execute("SELECT * FROM train_full").arrow())
print(f"Total training pairs: {len(train_raw_pl):,} (positives: {train_raw_pl['label'].sum():,})")

unique_n = set(train_raw_pl["s1_name_raw"].unique()).union(set(train_raw_pl["m_name_raw"].unique()))
unique_a = set(train_raw_pl["s1_addr_raw"].unique()).union(set(train_raw_pl["m_addr_raw"].unique()))

n_map = {x: normalize_business_name(x) for x in unique_n if x}
a_struct_map = {x: extract_structured_fields(x) for x in unique_a if x}

print("Extracting 52 features for training set...")
train_rows = []
for r in train_raw_pl.iter_rows(named=True):
    s1_a_struct = a_struct_map.get(r["s1_addr_raw"], {"address_normalized":"", "house_number":"", "postal_code":""})
    m_a_struct = a_struct_map.get(r["m_addr_raw"], {"address_normalized":"", "house_number":"", "postal_code":""})
    
    train_rows.append({
        "s1_name_norm": n_map.get(r["s1_name_raw"], ""),
        "matched_name_norm": n_map.get(r["m_name_raw"], ""),
        "s1_addr_norm": s1_a_struct["address_normalized"],
        "matched_addr_norm": m_a_struct["address_normalized"],
        "s1_house": s1_a_struct["house_number"],
        "matched_house": m_a_struct["house_number"],
        "s1_postal": s1_a_struct["postal_code"],
        "matched_postal": m_a_struct["postal_code"],
        "s1_country": r["s1_country"] or "",
        "matched_country": r["m_country"] or "",
        "matched_entity_id": r["matched_entity_id"],
        "label": r["label"]
    })

train_pairs_df = pl.DataFrame(train_rows)
train_feat_df = extract_features_df(train_pairs_df)

X_train = train_feat_df.select(V5_FEATURES_EXPANDED).fill_null(0).to_numpy().astype(np.float32)
y_train = train_pairs_df["label"].to_numpy().astype(np.int8)

print(f"X_train shape: {X_train.shape}, positive ratio: {np.mean(y_train):.2%}")

# Train XGBoost with histogram method
print("\nTraining XGBoost V8 with 52 features...")
t0 = time.time()
clf = xgb.XGBClassifier(
    n_estimators=750,
    max_depth=7,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.05,
    reg_alpha=0.1,
    reg_lambda=1.5,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    max_bin=256,
    random_state=42,
    n_jobs=2
)
clf.fit(X_train, y_train, verbose=150)
print(f"Training completed in {time.time()-t0:.1f}s")

model_v8_path = os.path.join(MODEL_DIR, "xgboost_v8_52features.json")
clf.save_model(model_v8_path)
print(f"Model saved to: {model_v8_path}")

# ========================================================
# EVALUATE ON VALIDATION BENCHMARK (V7 Candidates)
# ========================================================
print("\n" + "=" * 70)
print("EVALUATING MODEL V8 ON 77.98% RECALL VALIDATION CANDIDATES")
print("=" * 70)

s1_val, gt_val, _, s23_val = load_benchmark()
v7_cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v7_cands.parquet"))

v7_feat_cache = os.path.join(VAL_DIR, "val_v7_features52_X.npy")

if os.path.exists(v7_feat_cache):
    print("Loading cached 52-feature validation matrix...")
    X_val = np.load(v7_feat_cache)
else:
    print("Extracting 52 features for 405k validation candidates...")
    v7_pairs = v7_cands.join(
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
    
    val_un_names = set(v7_pairs["s1_name_raw"].unique().drop_nulls()).union(set(v7_pairs["m_name_raw"].unique().drop_nulls()))
    val_un_addrs = set(v7_pairs["s1_addr_raw"].unique().drop_nulls()).union(set(v7_pairs["m_addr_raw"].unique().drop_nulls()))
    
    val_n_map = {x: normalize_business_name(x) for x in val_un_names if x}
    val_a_map = {x: extract_structured_fields(x) for x in val_un_addrs if x}
    
    val_rows = []
    for r in v7_pairs.iter_rows(named=True):
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
    X_val = val_feat_df.select(V5_FEATURES_EXPANDED).fill_null(0).to_numpy().astype(np.float32)
    np.save(v7_feat_cache, X_val)

probs = clf.predict_proba(X_val)[:, 1]

print("\n--- Model V8 Threshold Calibration Curve ---")
best_score = -1
best_res = None
best_t = None

for t in [0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99]:
    pred_mask = probs >= t
    pred_pairs = v7_cands.select(["source1_entity_id", "matched_entity_id"]).filter(pred_mask)
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
print(f"PEAK MODEL V8 SCORE: Macro F0.5 = {best_score:.4f} @ Threshold = {best_t:.3f}")
print(f"Precision: {best_res['precision']*100:.2f}% | Recall: {best_res['recall']*100:.2f}%")
print(f"Total True Positives: {best_res['tp']:,} | False Positives: {best_res['fp']:,}")
print(f"Singleton False Positives: {best_res['singleton_fp']} / {best_res['singletons']}")
print(f"==================================================================")

# Append to log
import csv
with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["EXP-5", f"V7 Multi-Channel Cands (77.98% rec) + Model V8 (52 feats) @ t={best_t:.3f}", "77.98%", len(v7_cands), f"{best_res['precision']*100:.2f}%", f"{best_res['recall']*100:.2f}%", f"{best_score:.4f}", f"{time.time()-t0:.1f}", "52 features with character n-grams, acronyms, and gating"])
