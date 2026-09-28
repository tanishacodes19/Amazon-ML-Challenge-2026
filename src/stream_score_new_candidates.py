import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, gc
import duckdb
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb
from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORM_DIR = os.path.join(BASE, "normalized_data")
S1_NORM = os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")
S2_NORM = os.path.join(NORM_DIR, "test_source2_normalized.tsv").replace("\\", "/")
S3_NORM = os.path.join(NORM_DIR, "test_source3_normalized.tsv").replace("\\", "/")
CANDS_FILE = os.path.join(BASE, "test_scored_candidates.tsv").replace("\\", "/")
MODEL_DIR = os.path.join(BASE, "model")
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NEW_MATCHES_FILE = os.path.join(OUTPUT_DIR, "new_accepted_matches_v18.parquet")

os.makedirs(r"D:\duckdb_temp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")
con.execute("PRAGMA max_temp_directory_size='500GiB'")

print("=" * 80)
print("AMAZON ML CHALLENGE 2026 — STREAM SCORING 4.65M NEW CANDIDATE PAIRS")
print("=" * 80)

t0 = time.time()
print("\n[Step 1/4] Loading S1 and S23 reference tables...")
con.execute(f"""
CREATE TEMP TABLE s1_tbl AS 
SELECT entity_id AS source1_entity_id, 
       name_normalized AS clean_name, 
       address_normalized AS clean_addr, 
       country,
       LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_number,
       COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS postal_code
FROM read_csv('{S1_NORM}', delim='\t', header=true);

CREATE TEMP TABLE s23_tbl AS 
SELECT entity_id AS matched_entity_id, 
       name_normalized AS clean_name, 
       address_normalized AS clean_addr, 
       country,
       LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_number,
       COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS postal_code
FROM (
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('{S2_NORM}', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('{S3_NORM}', delim='\t', header=true)
);
""")
print(f"Loaded reference tables in {time.time()-t0:.1f}s.")

print("\n[Step 2/4] Generating 4.65M new candidate pairs...")
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE existing_cands AS
SELECT source1_entity_id, matched_entity_id
FROM read_csv('{CANDS_FILE}', delim='\t', header=true)
WHERE probability >= 0.05;

CREATE TEMP TABLE new_cands AS
WITH s1_k2 AS (
    SELECT source1_entity_id, country,
           CASE WHEN words[1] < words[2] THEN words[1] || '_' || words[2] ELSE words[2] || '_' || words[1] END AS k2
    FROM (SELECT source1_entity_id, country, LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 3 AND x NOT IN ('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')) AS words FROM s1_tbl)
    WHERE LEN(words) >= 2
),
s23_k2 AS (
    SELECT matched_entity_id, country,
           CASE WHEN words[1] < words[2] THEN words[1] || '_' || words[2] ELSE words[2] || '_' || words[1] END AS k2
    FROM (SELECT matched_entity_id, country, LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 3 AND x NOT IN ('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')) AS words FROM s23_tbl)
    WHERE LEN(words) >= 2
),
vk1 AS (SELECT country, k2 FROM s1_k2 GROUP BY country, k2 HAVING COUNT(*) <= 50),
p1 AS (
    SELECT s1.source1_entity_id, s23.matched_entity_id
    FROM s1_k2 s1 JOIN vk1 ON s1.country = vk1.country AND s1.k2 = vk1.k2
    JOIN s23_k2 s23 ON s1.country = s23.country AND s1.k2 = s23.k2
    QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15
),
s1_dl AS (
    SELECT source1_entity_id, country, house_number || '_' || word as k
    FROM (
        SELECT source1_entity_id, country, house_number,
               UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 6 AND x NOT IN ('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata') AND x NOT SIMILAR TO '[0-9].*')) as word
        FROM s1_tbl
    ) WHERE LENGTH(house_number) >= 1
),
s23_dl AS (
    SELECT matched_entity_id, country, house_number || '_' || word as k
    FROM (
        SELECT matched_entity_id, country, house_number,
               UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 6 AND x NOT IN ('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata') AND x NOT SIMILAR TO '[0-9].*')) as word
        FROM s23_tbl
    ) WHERE LENGTH(house_number) >= 1
),
vk2 AS (SELECT country, k FROM s1_dl GROUP BY country, k HAVING COUNT(*) <= 50),
p2 AS (
    SELECT s1.source1_entity_id, s23.matched_entity_id
    FROM s1_dl s1 JOIN vk2 ON s1.country = vk2.country AND s1.k = vk2.k
    JOIN s23_dl s23 ON s1.country = s23.country AND s1.k = s23.k
    QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15
)
SELECT DISTINCT source1_entity_id, matched_entity_id
FROM (SELECT * FROM p1 UNION ALL SELECT * FROM p2)
WHERE (source1_entity_id, matched_entity_id) NOT IN (SELECT source1_entity_id, matched_entity_id FROM existing_cands);
""")
total_new_cands = con.execute("SELECT COUNT(*) FROM new_cands").fetchone()[0]
print(f"Generated {total_new_cands:,} candidate pairs in {time.time()-t0:.1f}s.")

print("\n[Step 3/4] Loading trained XGBoost and LightGBM models...")
t0 = time.time()
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(MODEL_DIR, "xgboost_v13_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lightgbm_v13_52features.txt"))
print(f"Models loaded in {time.time()-t0:.1f}s.")

idx_name_ratio = V5_FEATURES_EXPANDED.index("name_ratio")
idx_addr_ratio = V5_FEATURES_EXPANDED.index("address_ratio")
idx_house_match = V5_FEATURES_EXPANDED.index("house_match")
idx_name_contains = V5_FEATURES_EXPANDED.index("name_contains")
idx_acronym = V5_FEATURES_EXPANDED.index("name_acronym_match")

con.execute("CREATE TEMP TABLE accepted_new_matches (source1_entity_id VARCHAR, matched_entity_id VARCHAR, prob FLOAT);")

CHUNK_SIZE = 150000
total_chunks = (total_new_cands + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"\n[Step 4/4] Streaming feature extraction & scoring across {total_chunks} chunks ({CHUNK_SIZE:,} per chunk)...")

total_accepted = 0
t_stream_start = time.time()

for chunk_idx in range(total_chunks):
    t_c0 = time.time()
    offset = chunk_idx * CHUNK_SIZE
    chunk_arrow = con.execute(f"""
    SELECT p.source1_entity_id,
           p.matched_entity_id,
           s1.clean_name AS s1_name_norm,
           s23.clean_name AS matched_name_norm,
           s1.clean_addr AS s1_addr_norm,
           s23.clean_addr AS matched_addr_norm,
           s1.house_number AS s1_house,
           s23.house_number AS matched_house,
           s1.postal_code AS s1_postal,
           s23.postal_code AS matched_postal,
           s1.country AS s1_country,
           s23.country AS matched_country
    FROM (
        SELECT source1_entity_id, matched_entity_id
        FROM new_cands
        LIMIT {CHUNK_SIZE} OFFSET {offset}
    ) p
    JOIN s1_tbl s1 ON p.source1_entity_id = s1.source1_entity_id
    JOIN s23_tbl s23 ON p.matched_entity_id = s23.matched_entity_id;
    """).to_arrow_table()
    
    if chunk_arrow.num_rows == 0:
        break
        
    df_pl = pl.from_arrow(chunk_arrow)
    feat_pl = extract_features_df(df_pl)
    X_chunk = feat_pl.select(V5_FEATURES_EXPANDED).to_numpy()
    
    # 50/50 Ensemble
    p_xgb = xgb_model.predict_proba(X_chunk)[:, 1]
    p_lgb = lgb_model.predict(X_chunk)
    p_blend = 0.50 * p_xgb + 0.50 * p_lgb
    
    name_r = X_chunk[:, idx_name_ratio]
    addr_r = X_chunk[:, idx_addr_ratio]
    house_m = X_chunk[:, idx_house_match]
    name_c = X_chunk[:, idx_name_contains]
    is_acronym = X_chunk[:, idx_acronym]
    
    # Calibrated Gate
    base_gate = (p_blend >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
    rescue1 = (p_blend >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
    co_location = (name_r >= 0.40)
    containment = (p_blend >= 0.95) & (name_c == 1) & (addr_r >= 0.65)
    acronym = (p_blend >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40)
    
    gate = ((base_gate | rescue1) & co_location) | containment | acronym
    
    n_acc = gate.sum()
    if n_acc > 0:
        s_s1 = df_pl["source1_entity_id"].to_numpy()[gate]
        s_m = df_pl["matched_entity_id"].to_numpy()[gate]
        s_p = p_blend[gate]
        
        acc_df = pl.DataFrame({
            "source1_entity_id": s_s1,
            "matched_entity_id": s_m,
            "prob": s_p
        })
        con.register("acc_chunk", acc_df.to_arrow())
        con.execute("INSERT INTO accepted_new_matches SELECT * FROM acc_chunk;")
        con.unregister("acc_chunk")
        total_accepted += n_acc
        
    elapsed = time.time() - t_stream_start
    c_time = time.time() - t_c0
    rate = (offset + len(df_pl)) / elapsed if elapsed > 0 else 0
    rem = (total_new_cands - (offset + len(df_pl))) / rate if rate > 0 else 0
    print(f"  Chunk {chunk_idx+1}/{total_chunks} ({len(df_pl):,} pairs) -> Accepted +{n_acc:,} in {c_time:.1f}s | Total Accepted: {total_accepted:,} | ETA: {rem/60:.1f}m")
    
    del df_pl, feat_pl, X_chunk, p_xgb, p_lgb, p_blend
    gc.collect()

print("\n" + "=" * 80)
print(f"Scoring Complete in {time.time()-t_stream_start:.1f}s!")
print(f"Total Brand-New Accepted Matches: {total_accepted:,}!")
print("=" * 80)

# Save to parquet
con.execute(f"COPY accepted_new_matches TO '{NEW_MATCHES_FILE.replace(chr(92), '/')}' (FORMAT PARQUET);")
print(f"Saved new accepted matches to {NEW_MATCHES_FILE} ({os.path.getsize(NEW_MATCHES_FILE):,} bytes).")
