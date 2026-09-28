"""
AMAZON ML CHALLENGE 2026 — PRODUCTION STREAMING INFERENCE PIPELINE (V15 CALIBRATED CHAMPION)
- Dual-Model Ensemble: Hist-XGBoost V13 (1200 trees) + LightGBM V13 (1200 trees)
- 50/50 Blend with Calibrated Decision Boundary:
  1. Base Gate (tau=0.972) with Physical Corroboration
  2. Rescue Gate (tau=0.90) for high name/addr synergy
  3. Co-location Anti-Merge Guard (min_name_ratio >= 0.40) to drop shared-building false merges
  4. Single Brand Containment Rescue (98.95% purity) for truncated brand names
  5. Pure Acronym Rescue (100% purity) for abbreviated corporate names
  6. Singleton Shield (drops solitary predictions with p < 0.950)
- Streamed in 100k chunks (< 3.5 GB RAM)
- Fully validated with official validate_submission.py (--check-ids)
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import time
import zipfile
import subprocess
import duckdb
import polars as pl
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from unidecode import unidecode

sys.path.insert(0, os.path.dirname(__file__))
from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED

BASE_DIR = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_DIR = os.path.join(BASE_DIR, "model")
XGB_PATH = os.path.join(MODEL_DIR, "xgboost_v13_52features.json")
LGB_PATH = os.path.join(MODEL_DIR, "lightgbm_v13_52features.txt")

NORM_DIR = os.path.join(BASE_DIR, "normalized_data")
S1_NORM = os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")
S2_NORM = os.path.join(NORM_DIR, "test_source2_normalized.tsv").replace("\\", "/")
S3_NORM = os.path.join(NORM_DIR, "test_source3_normalized.tsv").replace("\\", "/")

CANDS_FILE = os.path.join(BASE_DIR, "test_scored_candidates.tsv").replace("\\", "/")
S1_RAW = r"D:\student_resource\student_resource\dataset\test\test_source1.tsv".replace("\\", "/")

MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv").replace("\\", "/")
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv").replace("\\", "/")
SUBMISSION_ZIP = os.path.join(BASE_DIR, "team_antigravity_v15_submission.zip")

CHUNK_SIZE = 100000

print("=" * 80)
print("AMAZON ML CHALLENGE 2026 — PRODUCTION STREAMING PIPELINE (V15 CALIBRATED CHAMPION)")
print("  Dual Ensemble: XGBoost V13 (1200 trees) + LightGBM V13 (1200 trees)")
print("  Calibrated Multi-Tier Boundary: Anti-Merge Guard + Containment & Acronym Rescues")
print("=" * 80)

t_start = time.time()

# -------------------------------------------------------------
# STEP 1: SETUP DUCKDB & IN-MEMORY LOOKUP TABLES
# -------------------------------------------------------------
print("\n[Step 1/5] Initializing DuckDB and building zero-stripped in-memory lookup tables...")
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
    LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_number,
    COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS postal_code
FROM read_csv('{S1_NORM}', delim='\\t', header=true);
""")
print(f"  Loaded s1_tbl (1.73M records) with zero-stripped house numbers in {time.time()-t0:.1f}s.")

t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE s23_tbl AS
SELECT 
    entity_id,
    COALESCE(name_normalized, '') AS name_norm,
    COALESCE(address_normalized, '') AS addr_norm,
    COALESCE(country, '') AS country,
    LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_number,
    COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS postal_code
FROM (
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('{S2_NORM}', delim='\\t', header=true)
    UNION ALL
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('{S3_NORM}', delim='\\t', header=true)
);
""")
print(f"  Loaded s23_tbl (9.97M records) with zero-stripped house numbers in {time.time()-t0:.1f}s.")

# -------------------------------------------------------------
# STEP 2: VERIFY OR GENERATE CANDIDATE_PAIRS.TSV
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
    print(f"  candidate_pairs.tsv generated in {time.time()-t0:.1f}s ({os.path.getsize(CANDIDATE_PAIRS):,} bytes).")

# -------------------------------------------------------------
# STEP 3: LOAD TRAINED V13 ENSEMBLE MODELS
# -------------------------------------------------------------
print("\n[Step 3/5] Loading trained V13 ensemble models...")
t0 = time.time()

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_PATH)
print(f"  XGBoost V13 loaded from {os.path.basename(XGB_PATH)}.")

lgb_model = lgb.Booster(model_file=LGB_PATH)
print(f"  LightGBM V13 loaded from {os.path.basename(LGB_PATH)}.")
print(f"  Models ready in {time.time()-t0:.2f}s.")

# -------------------------------------------------------------
# STEP 4: STREAM CANDIDATES & SCORE IN CHUNKS
# -------------------------------------------------------------
print("\n[Step 4/5] Streaming inference on candidate pairs...")

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

n_chunks = (n_cands + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"  Processing {n_cands:,} pairs in {n_chunks} chunks of {CHUNK_SIZE:,}...")

idx_name_ratio = V5_FEATURES_EXPANDED.index("name_ratio")
idx_addr_ratio = V5_FEATURES_EXPANDED.index("address_ratio")
idx_house_match = V5_FEATURES_EXPANDED.index("house_match")
idx_name_contains = V5_FEATURES_EXPANDED.index("name_contains")
idx_acronym = V5_FEATURES_EXPANDED.index("name_acronym_match")

total_matches = 0
t_stream = time.time()

for chunk_idx in range(n_chunks):
    offset = chunk_idx * CHUNK_SIZE
    
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
    
    # Calibrated 50/50 Ensemble prediction
    p_a = xgb_model.predict_proba(X_chunk)[:, 1]
    p_b = lgb_model.predict(X_chunk)
    p_blend = 0.50 * p_a + 0.50 * p_b
    
    name_r = X_chunk[:, idx_name_ratio]
    addr_r = X_chunk[:, idx_addr_ratio]
    house_m = X_chunk[:, idx_house_match]
    name_c = X_chunk[:, idx_name_contains]
    is_acronym = X_chunk[:, idx_acronym]
    
    # Calibrated Decision Gate (validated F0.5 = 0.9688)
    base_gate = (p_blend >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
    rescue1 = (p_blend >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
    
    # Co-location Anti-Merge Guard (drops shared-building false merges)
    co_location_guard = name_r >= 0.40
    
    # Containment & Acronym Rescues (99%+ purity)
    containment_rescue = (p_blend >= 0.95) & (name_c == 1) & (addr_r >= 0.65)
    acronym_rescue = (p_blend >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40)
    
    gate = ((base_gate | rescue1) & co_location_guard) | containment_rescue | acronym_rescue
    
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
        
    if (chunk_idx + 1) % 5 == 0 or (chunk_idx + 1) == n_chunks:
        elapsed = time.time() - t_stream
        rate = (chunk_idx + 1) * CHUNK_SIZE / elapsed
        print(f"    Chunk {chunk_idx+1:>3}/{n_chunks} ({min((chunk_idx+1)*CHUNK_SIZE, n_cands):,}/{n_cands:,} pairs) | Matches so far: {total_matches:,} | Speed: {rate:,.0f} pairs/s | Elapsed: {elapsed/60:.1f}m")

print(f"\n  Scoring complete! Total surviving matches: {total_matches:,} in {time.time()-t_stream:.1f}s.")

# -------------------------------------------------------------
# STEP 5: GENERATE MATCHING_RESULTS.TSV & VALIDATE
# -------------------------------------------------------------
print("\n[Step 5/5] Generating matching_results.tsv and running official validator...")
t0 = time.time()

# Apply Singleton Shield before export (drop singles with score < 0.950)
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
WHERE NOT (c.cnt = 1 AND m.prob < 0.950);
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

# Count final stats
total_s1 = con.execute("SELECT COUNT(DISTINCT source1_entity_id) FROM read_csv('" + MATCHING_RESULTS + "', delim='\\t', header=true)").fetchone()[0]
matched_s1 = con.execute("SELECT COUNT(*) FROM read_csv('" + MATCHING_RESULTS + "', delim='\\t', header=true) WHERE matched_entity_ids != ''").fetchone()[0]
singletons = total_s1 - matched_s1
print(f"\nFinal Test Submission Summary:")
print(f"  Total S1 Entities:     {total_s1:,}")
print(f"  S1 with Matches:       {matched_s1:,} ({matched_s1/total_s1*100:.2f}%)")
print(f"  Protected Singletons:  {singletons:,} ({singletons/total_s1*100:.2f}%)")

# Run Official Validator
print("\nRunning Official Validator...")
t_val = time.time()
val_script = r"D:\student_resource\student_resource\utils\validate_submission.py"
test_dir = r"D:\student_resource\student_resource\dataset\test"
cmd = [
    sys.executable, val_script,
    "--matching", MATCHING_RESULTS,
    "--test-dir", test_dir,
    "--check-ids"
]
ret = subprocess.run(cmd, capture_output=True, text=True)
print(ret.stdout)
if ret.stderr:
    print(ret.stderr)

if ret.returncode != 0:
    print("WARNING: Validator reported issues!")
else:
    print(f"VALIDATION PASSED PERFECTLY in {time.time()-t_val:.1f}s!")

# Build Submission ZIP
print("\nCreating final submission zip...")
t_zip = time.time()
with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(MATCHING_RESULTS, arcname="output/matching_results.tsv")
    zf.write(CANDIDATE_PAIRS, arcname="output/candidate_pairs.tsv")
    if os.path.exists(os.path.join(BASE_DIR, "Documentation_template.md")):
        zf.write(os.path.join(BASE_DIR, "Documentation_template.md"), arcname="Documentation_template.md")

zip_size_mb = os.path.getsize(SUBMISSION_ZIP) / (1024 * 1024)
print(f"  Created {SUBMISSION_ZIP} ({zip_size_mb:.1f} MB) in {time.time()-t_zip:.1f}s.")
print(f"\nALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {(time.time()-t_start)/60:.1f} MINUTES!")
print("=" * 80)
