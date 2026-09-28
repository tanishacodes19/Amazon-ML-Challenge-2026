"""
V13 RETRAIN ENSEMBLE: Retrain XGBoost + LightGBM with upgraded normalizer (V13)
- Regenerates training features from scratch with unidecode + zero-strip
- Trains stronger models (more trees, tuned hyperparams)
- Evaluates on V13 validation candidate set
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import os
import time
import duckdb
import pandas as pd
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb
from normalizer import normalize_business_name, extract_structured_fields
from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED
from eval_framework import load_benchmark, compute_macro_f05

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
MODEL_DIR = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

XGB_PATH = os.path.join(MODEL_DIR, "xgboost_v13_52features.json")
LGB_PATH = os.path.join(MODEL_DIR, "lightgbm_v13_52features.txt")
TRAIN_X_CACHE = os.path.join(MODEL_DIR, "train_v13_X.npy")
TRAIN_Y_CACHE = os.path.join(MODEL_DIR, "train_v13_y.npy")

print("=" * 70)
print("V13 RETRAIN: XGBOOST + LIGHTGBM ENSEMBLE WITH UPGRADED NORMALIZER")
print("=" * 70)

t0 = time.time()
GT_PAIRS = os.path.join(BASE, "ground_truth_pairs.tsv").replace("\\", "/")
CANDIDATES = os.path.join(BASE, "training_candidate_pairs.tsv").replace("\\", "/")
S1_FILE = os.path.join(BASE, "normalized_data", "train_source1_normalized.tsv").replace("\\", "/")
S2_FILE = os.path.join(BASE, "normalized_data", "train_source2_normalized.tsv").replace("\\", "/")
S3_FILE = os.path.join(BASE, "normalized_data", "train_source3_normalized.tsv").replace("\\", "/")

if os.path.exists(TRAIN_X_CACHE) and os.path.exists(TRAIN_Y_CACHE):
    print(f"Loading cached V13 training features from {TRAIN_X_CACHE}...")
    X_train = np.load(TRAIN_X_CACHE)
    y_train = np.load(TRAIN_Y_CACHE)
    print(f"Loaded X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
else:
    con = duckdb.connect()
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='3500MB'")
    
    # Sample MORE training data for stronger signal
    print("\n1. Sampling 200,000 true positives from training partition (% 4 != 0)...")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE train_pos AS
    SELECT g.source1_entity_id, g.matched_entity_id, 1 AS label
    FROM read_csv('{GT_PAIRS}', delim='\\t', header=true) g
    WHERE ABS(HASH(g.source1_entity_id)) % 4 != 0
    USING SAMPLE reservoir (200000 ROWS) REPEATABLE (42)
    """)
    pos_count = con.execute("SELECT COUNT(*) FROM train_pos").fetchone()[0]
    print(f"   -> Got {pos_count:,} positive pairs")
    
    print("2. Sampling 200,000 hard negatives from candidate pool (% 4 != 0)...")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE train_cand_sample AS
    SELECT c.source1_entity_id, c.candidate_entity_id AS matched_entity_id
    FROM read_csv('{CANDIDATES}', delim='\\t', header=true) c
    WHERE ABS(HASH(c.source1_entity_id)) % 4 != 0
    USING SAMPLE reservoir (1500000 ROWS) REPEATABLE (43)
    """)
    
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE train_hard_neg AS
    SELECT c.source1_entity_id, c.matched_entity_id, 0 AS label
    FROM train_cand_sample c
    LEFT JOIN read_csv('{GT_PAIRS}', delim='\\t', header=true) g
      ON c.source1_entity_id = g.source1_entity_id AND c.matched_entity_id = g.matched_entity_id
    WHERE g.matched_entity_id IS NULL
    USING SAMPLE reservoir (200000 ROWS) REPEATABLE (44)
    """)
    hard_neg_count = con.execute("SELECT COUNT(*) FROM train_hard_neg").fetchone()[0]
    print(f"   -> Got {hard_neg_count:,} hard negatives")
    
    print("3. Sampling 50,000 random non-matching negatives...")
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
    total_pairs = con.execute("SELECT COUNT(*) FROM train_pairs_all").fetchone()[0]
    print(f"   Total training pairs: {total_pairs:,}")
    
    print("4. Joining text directly in DuckDB...")
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
    
    print("5. Converting to Polars and precomputing normalizations with V13 normalizer...")
    train_raw_pl = pl.from_arrow(con.execute("SELECT * FROM train_full").to_arrow_table())
    print(f"   Total training pairs: {len(train_raw_pl):,} (positives: {train_raw_pl['label'].sum():,})")
    
    unique_n = set(train_raw_pl["s1_name_raw"].unique()).union(set(train_raw_pl["m_name_raw"].unique()))
    unique_a = set(train_raw_pl["s1_addr_raw"].unique()).union(set(train_raw_pl["m_addr_raw"].unique()))
    
    print(f"   Normalizing {len(unique_n):,} unique names & {len(unique_a):,} unique addresses with V13 normalizer...")
    n_map = {x: normalize_business_name(x) for x in unique_n if x}
    a_map = {x: extract_structured_fields(x) for x in unique_a if x}
    
    print("   Building structured training dataframe...")
    t_prep = time.time()
    train_rows = []
    for r in train_raw_pl.iter_rows(named=True):
        s1_a = a_map.get(r["s1_addr_raw"], {"address_normalized":"", "house_number":"", "postal_code":""})
        m_a = a_map.get(r["m_addr_raw"], {"address_normalized":"", "house_number":"", "postal_code":""})
        train_rows.append({
            "s1_name_norm": n_map.get(r["s1_name_raw"], ""),
            "matched_name_norm": n_map.get(r["m_name_raw"], ""),
            "s1_addr_norm": s1_a["address_normalized"],
            "matched_addr_norm": m_a["address_normalized"],
            "s1_house": s1_a["house_number"],
            "matched_house": m_a["house_number"],
            "s1_postal": s1_a["postal_code"],
            "matched_postal": m_a["postal_code"],
            "s1_country": r["s1_country"] or "",
            "matched_country": r["m_country"] or "",
            "matched_entity_id": r["matched_entity_id"],
        })
    
    print(f"   Preprocessed in {time.time()-t_prep:.1f}s. Extracting 52 features...")
    train_pl = pl.DataFrame(train_rows)
    feat_train_pl = extract_features_df(train_pl)
    
    X_train = feat_train_pl.select(V5_FEATURES_EXPANDED).to_numpy()
    y_train = train_raw_pl["label"].to_numpy()
    
    np.save(TRAIN_X_CACHE, X_train)
    np.save(TRAIN_Y_CACHE, y_train)
    print(f"   Saved V13 training feature cache to {TRAIN_X_CACHE}")

# ============================================================
# FIT MODEL A: Hist-XGBoost (stronger config)
# ============================================================
print("\n" + "=" * 70)
print("TRAINING MODEL A: HIST-XGBOOST (V13 - STRONGER)")
print("=" * 70)
t_xgb = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=1200,
    max_depth=8,
    learning_rate=0.025,
    tree_method="hist",
    subsample=0.80,
    colsample_bytree=0.80,
    min_child_weight=3,
    gamma=0.03,
    reg_alpha=0.05,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=2,
    scale_pos_weight=1.0
)
xgb_model.fit(X_train, y_train)
xgb_model.save_model(XGB_PATH)
print(f"Model A fitted in {time.time()-t_xgb:.1f}s and saved to {XGB_PATH}")

# ============================================================
# FIT MODEL B: LightGBM (stronger config)
# ============================================================
print("\n" + "=" * 70)
print("TRAINING MODEL B: LIGHTGBM (V13 - STRONGER)")
print("=" * 70)
t_lgb = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=1200,
    num_leaves=127,
    max_depth=8,
    learning_rate=0.025,
    subsample=0.80,
    colsample_bytree=0.80,
    min_child_samples=15,
    reg_alpha=0.05,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=2,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_model.booster_.save_model(LGB_PATH)
print(f"Model B fitted in {time.time()-t_lgb:.1f}s and saved to {LGB_PATH}")

# ============================================================
# EVALUATE V13 ENSEMBLE ON V13 CANDIDATES
# ============================================================
print("\n" + "=" * 70)
print("EVALUATING V13 ENSEMBLE ON V13 CANDIDATES")
print("=" * 70)

V13_FEAT_X = os.path.join(VAL_DIR, "val_v13_features52_X.npy")
V13_FEAT_DF = os.path.join(VAL_DIR, "val_v13_features52_df.parquet")

if not os.path.exists(V13_FEAT_X):
    print("ERROR: V13 features not found! Run build_v13_mega_pipeline.py first.")
    sys.exit(1)

X_val = np.load(V13_FEAT_X)
pairs_df = pl.read_parquet(V13_FEAT_DF)

s1_val, gt_val, _, _ = load_benchmark()

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
name_ratios = pairs_df["name_ratio"].to_numpy()
addr_ratios = pairs_df["address_ratio"].to_numpy()
house_matches = pairs_df["house_match"].to_numpy()

# Predictions
print("Generating predictions with V13 models...")
p_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_lgb = lgb_model.predict_proba(X_val)[:, 1]

# Try multiple blend ratios
blends = [
    ("50/50", 0.50, 0.50),
    ("55/45", 0.55, 0.45),
    ("60/40", 0.60, 0.40),
    ("65/35", 0.65, 0.35),
    ("70/30", 0.70, 0.30),
]

global_best_f = 0.0
global_best_config = ""

for blend_name, w_xgb, w_lgb in blends:
    p_ens = w_xgb * p_xgb + w_lgb * p_lgb
    print(f"\n--- Ensemble Blend ({blend_name}) on V13 Candidates ---")
    print(f"{'Threshold':>10} | {'Macro F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'TP':>7} | {'FP':>6} | {'Singl FP':>8}")
    print("-" * 80)
    best_f = 0.0
    best_t = 0.0
    for tau in [0.50, 0.60, 0.70, 0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]:
        gate = (p_ens >= tau) & ((name_ratios >= 0.20) | (addr_ratios >= 0.40) | (house_matches == 1))
        pred_df = pl.DataFrame({
            "source1_entity_id": s1_ids[gate],
            "matched_entity_id": m_ids[gate]
        })
        res = compute_macro_f05(pred_df, gt_val, s1_val)
        if res["macro_f05"] > best_f:
            best_f = res["macro_f05"]
            best_t = tau
        print(f"{tau:>10.3f} | {res['macro_f05']:>11.4f} | {res['precision']*100:>9.2f}% | {res['recall']*100:>7.2f}% | {res['tp']:>7,} | {res['fp']:>6,} | {res['singleton_fp']:>8,}")
    print(f"*** PEAK for {blend_name}: {best_f:.4f} @ tau={best_t:.3f} ***")
    
    if best_f > global_best_f:
        global_best_f = best_f
        global_best_config = f"{blend_name} @ tau={best_t:.3f}"

print(f"\n{'=' * 80}")
print(f"GLOBAL BEST: Macro F0.5 = {global_best_f:.4f} with {global_best_config}")
print(f"IMPROVEMENT FROM V12: {global_best_f - 0.9291:.4f}")
print(f"Total V13 Retrain Completed in {time.time()-t0:.1f}s")
print(f"{'=' * 80}")
