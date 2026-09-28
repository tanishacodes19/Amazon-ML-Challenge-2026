import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import time
import subprocess
import duckdb
import polars as pl
import numpy as np
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))

from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED

BASE_DIR = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_DIR = os.path.join(BASE_DIR, "model")
XGB_PATH = os.path.join(MODEL_DIR, "xgboost_v11_52features.json")
LGB_PATH = os.path.join(MODEL_DIR, "lightgbm_v11_52features.txt")

NORM_DIR = os.path.join(BASE_DIR, "normalized_data")
S1_NORM = os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")
S2_NORM = os.path.join(NORM_DIR, "test_source2_normalized.tsv").replace("\\", "/")
S3_NORM = os.path.join(NORM_DIR, "test_source3_normalized.tsv").replace("\\", "/")

CANDS_FILE = os.path.join(BASE_DIR, "test_scored_candidates.tsv").replace("\\", "/")
S1_RAW = r"D:\student_resource\student_resource\dataset\test\test_source1.tsv".replace("\\", "/")

MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv").replace("\\", "/")
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv").replace("\\", "/")

THRESHOLD = 0.900
CHUNK_SIZE = 100000

print("=" * 80)
print("AMAZON ML CHALLENGE 2026 — PRODUCTION STREAMING INFERENCE PIPELINE (V12)")
print("=" * 80)

t_start = time.time()

# -------------------------------------------------------------
# STEP 1: SETUP DUCKDB & PRE-PARSED NORMALIZED TABLES
# -------------------------------------------------------------
print("\n[Step 1/5] Initializing DuckDB and building pre-parsed in-memory lookup tables...")
con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE s1_tbl AS
SELECT 
    entity_id,
    COALESCE(name_normalized, '') AS name_norm,
    COALESCE(address_normalized, '') AS addr_norm,
    COALESCE(country, '') AS country,
    COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), '') AS house_number,
    COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS postal_code
FROM read_csv('{S1_NORM}', delim='\\t', header=true);
""")
print(f"  Loaded s1_tbl (1.73M records) in {time.time()-t0:.1f}s.")

t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE s23_tbl AS
SELECT 
    entity_id,
    COALESCE(name_normalized, '') AS name_norm,
    COALESCE(address_normalized, '') AS addr_norm,
    COALESCE(country, '') AS country,
    COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), '') AS house_number,
    COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS postal_code
FROM (
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('{S2_NORM}', delim='\\t', header=true)
    UNION ALL
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('{S3_NORM}', delim='\\t', header=true)
);
""")
print(f"  Loaded s23_tbl (9.97M records) in {time.time()-t0:.1f}s.")

# -------------------------------------------------------------
# STEP 2: GENERATE CANDIDATE_PAIRS.TSV DIRECTLY
# -------------------------------------------------------------
print("\n[Step 2/5] Checking candidate_pairs.tsv...")
if os.path.exists(CANDIDATE_PAIRS) and os.path.getsize(CANDIDATE_PAIRS) > 300000000:
    print(f"  candidate_pairs.tsv already exists ({os.path.getsize(CANDIDATE_PAIRS):,} bytes). Skipping generation.")
else:
    print("  Generating candidate_pairs.tsv via DuckDB streaming aggregation...")
    t0 = time.time()
    con.execute(f"""
    COPY (
        SELECT s1.entity_id AS source1_entity_id,
               COALESCE(STRING_AGG(c.matched_entity_id, ','), '') AS candidate_entity_ids
        FROM read_csv('{S1_RAW}', delim='\\t', header=true) s1
        LEFT JOIN read_csv('{CANDS_FILE}', delim='\\t', header=true) c
          ON s1.entity_id = c.source1_entity_id
        GROUP BY s1.entity_id
    ) TO '{CANDIDATE_PAIRS}' (HEADER, DELIMITER '\\t', QUOTE '');
    """)
    print(f"  candidate_pairs.tsv successfully generated in {time.time()-t0:.1f}s.")

# -------------------------------------------------------------
# STEP 3: LOAD TRAINED DUAL-MODEL ENSEMBLE
# -------------------------------------------------------------
print("\n[Step 3/5] Loading trained ensemble models...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_PATH)

lgb_model = lgb.Booster(model_file=LGB_PATH)
print("  Loaded Model A (Hist-XGBoost V11) and Model B (LightGBM V11).")

# -------------------------------------------------------------
# STEP 4: STREAMING CANDIDATE SCORING IN CHUNKS
# -------------------------------------------------------------
print("\n[Step 4/5] Scoring test candidates (probability >= 0.05) in streaming chunks...")

# Filter candidate pool to plausibles (prob >= 0.05)
con.execute(f"""
CREATE TEMP TABLE candidate_pool AS
SELECT source1_entity_id, matched_entity_id
FROM read_csv('{CANDS_FILE}', delim='\\t', header=true)
WHERE probability >= 0.05;
""")
n_cands = con.execute("SELECT COUNT(*) FROM candidate_pool").fetchone()[0]
print(f"  Candidate pool filtered to {n_cands:,} plausible candidate pairs (prob >= 0.05).")

con.execute("""
CREATE TEMP TABLE final_matches (
    source1_entity_id VARCHAR,
    matched_entity_id VARCHAR,
    prob DOUBLE
);
""")

con.execute("CREATE SEQUENCE chunk_seq START 1;")
n_chunks = (n_cands + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"  Processing {n_cands:,} pairs in {n_chunks} chunks of {CHUNK_SIZE:,}...")

idx_name_ratio = V5_FEATURES_EXPANDED.index("name_ratio")
idx_addr_ratio = V5_FEATURES_EXPANDED.index("address_ratio")
idx_house_match = V5_FEATURES_EXPANDED.index("house_match")

total_matches = 0
t_stream = time.time()

for chunk_idx in range(n_chunks):
    offset = chunk_idx * CHUNK_SIZE
    t_c = time.time()
    
    # Query joined chunk directly from DuckDB
    chunk_arrow = con.execute(f"""
    SELECT p.source1_entity_id, p.matched_entity_id,
           s1.name_norm AS s1_name_norm,
           s23.name_norm AS matched_name_norm,
           s1.addr_norm AS s1_addr_norm,
           s23.addr_norm AS matched_addr_norm,
           s1.house_number AS s1_house,
           s23.house_number AS matched_house,
           s1.postal_code AS s1_postal,
           s23.postal_code AS matched_postal,
           s1.country AS s1_country,
           s23.country AS matched_country
    FROM (
        SELECT source1_entity_id, matched_entity_id 
        FROM candidate_pool 
        LIMIT {CHUNK_SIZE} OFFSET {offset}
    ) p
    JOIN s1_tbl s1 ON p.source1_entity_id = s1.entity_id
    JOIN s23_tbl s23 ON p.matched_entity_id = s23.entity_id
    """).to_arrow_table()
    
    if chunk_arrow.num_rows == 0:
        break
        
    df_pl = pl.from_arrow(chunk_arrow)
    feat_pl = extract_features_df(df_pl)
    X_chunk = feat_pl.select(V5_FEATURES_EXPANDED).to_numpy()
    
    # Ensemble prediction
    p_a = xgb_model.predict_proba(X_chunk)[:, 1]
    p_b = lgb_model.predict(X_chunk)
    p_blend = 0.60 * p_a + 0.40 * p_b
    
    name_ratios = X_chunk[:, idx_name_ratio]
    addr_ratios = X_chunk[:, idx_addr_ratio]
    house_matches = X_chunk[:, idx_house_match]
    
    # Validated Decision Gate
    gate = (p_blend >= THRESHOLD) & ((name_ratios >= 0.25) | (addr_ratios >= 0.50) | (house_matches == 1))
    
    s_s1 = df_pl["source1_entity_id"].to_numpy()[gate]
    s_m = df_pl["matched_entity_id"].to_numpy()[gate]
    s_p = p_blend[gate]
    
    if len(s_s1) > 0:
        survived_df = pl.DataFrame({
            "source1_entity_id": s_s1,
            "matched_entity_id": s_m,
            "prob": s_p
        })
        con.register("survived_chunk", survived_df.to_arrow())
        con.execute("INSERT INTO final_matches SELECT * FROM survived_chunk;")
        con.unregister("survived_chunk")
        total_matches += len(s_s1)
        
    if (chunk_idx + 1) % 10 == 0 or (chunk_idx + 1) == n_chunks:
        elapsed = time.time() - t_stream
        rate = (chunk_idx + 1) * CHUNK_SIZE / elapsed
        print(f"    Chunk {chunk_idx+1:>3}/{n_chunks} ({min((chunk_idx+1)*CHUNK_SIZE, n_cands):,}/{n_cands:,} pairs) | Matches so far: {total_matches:,} | Speed: {rate:,.0f} pairs/s | Elapsed: {elapsed/60:.1f}m")

print(f"\n  Scoring complete! Total surviving matches: {total_matches:,} in {time.time()-t_stream:.1f}s.")

# -------------------------------------------------------------
# STEP 5: GENERATE MATCHING_RESULTS.TSV & VALIDATE
# -------------------------------------------------------------
print("\n[Step 5/5] Generating matching_results.tsv and running official validator...")
t0 = time.time()

# Apply Singleton Shield before export
con.execute("""
CREATE TEMP TABLE filtered_matches AS
WITH counts AS (
    SELECT source1_entity_id, COUNT(*) as cnt
    FROM final_matches
    GROUP BY source1_entity_id
)
SELECT m.source1_entity_id, m.matched_entity_id
FROM final_matches m
JOIN counts c ON m.source1_entity_id = c.source1_entity_id
WHERE NOT (c.cnt = 1 AND m.prob < 0.94);
""")

con.execute(f"""
COPY (
    SELECT s1.entity_id AS source1_entity_id,
           COALESCE(STRING_AGG(m.matched_entity_id, ','), '') AS matched_entity_ids
    FROM read_csv('{S1_RAW}', delim='\\t', header=true) s1
    LEFT JOIN filtered_matches m
      ON s1.entity_id = m.source1_entity_id
    GROUP BY s1.entity_id
) TO '{MATCHING_RESULTS}' (HEADER, DELIMITER '\\t', QUOTE '');
""")
print(f"  matching_results.tsv generated in {time.time()-t0:.1f}s.")

# Run official validator
print("\nRunning official submission validator:")
val_cmd = [
    sys.executable,
    r"D:\student_resource\student_resource\utils\validate_submission.py",
    "--matching", MATCHING_RESULTS,
    "--candidate", CANDIDATE_PAIRS,
    "--test-dir", r"D:\student_resource\student_resource\dataset\test"
]
ret = subprocess.run(val_cmd)

print("\n" + "=" * 80)
print(f"ALL STEPS COMPLETED IN {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} minutes)")
print("=" * 80)
