import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, gc, zipfile
import duckdb
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb
from collections import defaultdict
from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORM_DIR = os.path.join(BASE, "normalized_data")
OUTPUT_DIR = os.path.join(BASE, "output")
MODEL_DIR = os.path.join(BASE, "model")
S1_NORM = os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")
S2_NORM = os.path.join(NORM_DIR, "test_source2_normalized.tsv").replace("\\", "/")
S3_NORM = os.path.join(NORM_DIR, "test_source3_normalized.tsv").replace("\\", "/")
S1_RAW = r"D:\student_resource\student_resource\dataset\test\test_source1.tsv".replace("\\", "/")

CURRENT_MATCHING_TSV = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATES_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
SUBMISSION_ZIP = os.path.join(BASE, "team_antigravity_final_v20_submission.zip")

os.makedirs(r"D:\duckdb_temp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")
con.execute("PRAGMA max_temp_directory_size='500GiB'")

print("=" * 80, flush=True)
print("AMAZON ML CHALLENGE 2026 — FINAL V20 MEGA RECALL & PRECISION PIPELINE", flush=True)
print("FINAL SUBMISSION ATTEMPT: Maximizing Recall while Enforcing 1-to-1 Injective Purity", flush=True)
print("=" * 80, flush=True)

t0 = time.time()
print("\n[Step 1/6] Loading test reference tables...", flush=True)
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
print(f"Loaded reference tables in {time.time()-t0:.1f}s.", flush=True)

# Identify currently matched vs unmatched entities
con.execute(f"""
CREATE TEMP TABLE current_matches AS
SELECT source1_entity_id, UNNEST(STR_SPLIT(matched_entity_ids, ',')) AS matched_entity_id
FROM read_csv('{CURRENT_MATCHING_TSV.replace(chr(92), '/')}', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';

CREATE TEMP TABLE current_matched_s1 AS
SELECT DISTINCT source1_entity_id FROM current_matches;

CREATE TEMP TABLE s1_unmatched AS
SELECT s.* 
FROM s1_tbl s
LEFT JOIN current_matched_s1 m ON s.source1_entity_id = m.source1_entity_id
WHERE m.source1_entity_id IS NULL;
""")
n_curr_matched = con.execute("SELECT COUNT(*) FROM current_matched_s1").fetchone()[0]
n_curr_pairs = con.execute("SELECT COUNT(*) FROM current_matches").fetchone()[0]
n_curr_unmatched = con.execute("SELECT COUNT(*) FROM s1_unmatched").fetchone()[0]
print(f"Current V19 baseline: {n_curr_matched:,} matched S1 entities ({n_curr_pairs:,} strictly injective match pairs).", flush=True)
print(f"Currently unmatched S1 entities: {n_curr_unmatched:,}.", flush=True)

ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"
STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"

print("\n[Step 2/6] Mining multi-channel candidate pairs across FULL test set & targeted singletons...", flush=True)
t1 = time.time()

# 1. Sorted 2 Distinctive Address Words across full test set
t_ch1 = time.time()
print("  Generating Channel 1: Sorted 2 Address Words (full test set)...", flush=True)
con.execute(f"""
CREATE TEMP TABLE cand_top2 AS
WITH s1_top2 AS (
    SELECT source1_entity_id, country,
           CASE WHEN sorted_words[1] < sorted_words[2] 
                THEN sorted_words[1] || '_' || sorted_words[2] 
                ELSE sorted_words[2] || '_' || sorted_words[1] END as k
    FROM (
        SELECT source1_entity_id, country,
               LIST_SLICE(LIST_REVERSE_SORT(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')), 1, 2) as sorted_words
        FROM s1_tbl
    ) WHERE LEN(sorted_words) >= 2
),
s23_top2 AS (
    SELECT matched_entity_id, country,
           CASE WHEN sorted_words[1] < sorted_words[2] 
                THEN sorted_words[1] || '_' || sorted_words[2] 
                ELSE sorted_words[2] || '_' || sorted_words[1] END as k
    FROM (
        SELECT matched_entity_id, country,
               LIST_SLICE(LIST_REVERSE_SORT(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')), 1, 2) as sorted_words
        FROM s23_tbl
    ) WHERE LEN(sorted_words) >= 2
),
vk_top2 AS (SELECT country, k FROM s1_top2 GROUP BY country, k HAVING COUNT(*) <= 50)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_top2 s1
JOIN vk_top2 ON s1.country = vk_top2.country AND s1.k = vk_top2.k
JOIN s23_top2 s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
cnt_top2 = con.execute("SELECT COUNT(*) FROM cand_top2").fetchone()[0]
print(f"    Ch1: {cnt_top2:,} pairs in {time.time()-t_ch1:.1f}s", flush=True)

# 2. House Number + Locality word (length >= 6) across full test set
t_ch2 = time.time()
print("  Generating Channel 2: House Number + Locality (len >= 6, full test set)...", flush=True)
con.execute(f"""
CREATE TEMP TABLE cand_dl AS
WITH s1_dl AS (
    SELECT source1_entity_id, country,
           house_number || '_' || word as k
    FROM (
        SELECT source1_entity_id, country, house_number,
               UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 6 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as word
        FROM s1_tbl
    ) WHERE LENGTH(house_number) >= 1
),
s23_dl AS (
    SELECT matched_entity_id, country,
           house_number || '_' || word as k
    FROM (
        SELECT matched_entity_id, country, house_number,
               UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 6 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as word
        FROM s23_tbl
    ) WHERE LENGTH(house_number) >= 1
),
vk_dl AS (SELECT country, k FROM s1_dl GROUP BY country, k HAVING COUNT(*) <= 50)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_dl s1
JOIN vk_dl ON s1.country = vk_dl.country AND s1.k = vk_dl.k
JOIN s23_dl s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
cnt_dl = con.execute("SELECT COUNT(*) FROM cand_dl").fetchone()[0]
print(f"    Ch2: {cnt_dl:,} pairs in {time.time()-t_ch2:.1f}s", flush=True)

# 3. Targeted Address Prefix 10 chars (on unmatched S1)
t_ch3 = time.time()
print("  Generating Channel 3: Address 10-char prefix (unmatched S1)...", flush=True)
con.execute("""
CREATE TEMP TABLE cand_a10 AS
WITH s1_a10 AS (
    SELECT source1_entity_id, country, SUBSTRING(clean_addr, 1, 10) as k
    FROM s1_unmatched WHERE LENGTH(clean_addr) >= 10
),
s23_a10 AS (
    SELECT matched_entity_id, country, SUBSTRING(clean_addr, 1, 10) as k
    FROM s23_tbl WHERE LENGTH(clean_addr) >= 10
),
vk_a10 AS (SELECT country, k FROM s1_a10 GROUP BY country, k HAVING COUNT(*) <= 50)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_a10 s1
JOIN vk_a10 ON s1.country = vk_a10.country AND s1.k = vk_a10.k
JOIN s23_a10 s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
cnt_a10 = con.execute("SELECT COUNT(*) FROM cand_a10").fetchone()[0]
print(f"    Ch3: {cnt_a10:,} pairs in {time.time()-t_ch3:.1f}s", flush=True)

# 4. Targeted Distinctive Brand Word len >= 6 (on unmatched S1)
t_ch4 = time.time()
print("  Generating Channel 4: Distinctive Brand Word (unmatched S1)...", flush=True)
con.execute(f"""
CREATE TEMP TABLE cand_w6 AS
WITH s1_w6 AS (
    SELECT source1_entity_id, country,
           UNNEST(LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 6 AND x NOT IN {STOP_WORDS})) as w
    FROM s1_unmatched
),
s23_w6 AS (
    SELECT matched_entity_id, country,
           UNNEST(LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 6 AND x NOT IN {STOP_WORDS})) as w
    FROM s23_tbl
),
vk_w6 AS (SELECT country, w FROM s1_w6 GROUP BY country, w HAVING COUNT(*) <= 30)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_w6 s1
JOIN vk_w6 ON s1.country = vk_w6.country AND s1.w = vk_w6.w
JOIN s23_w6 s23 ON s1.country = s23.country AND s1.w = s23.w
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
cnt_w6 = con.execute("SELECT COUNT(*) FROM cand_w6").fetchone()[0]
print(f"    Ch4: {cnt_w6:,} pairs in {time.time()-t_ch4:.1f}s", flush=True)

# Union and EXCEPT existing matches
print("  Unioning all channels and subtracting existing matches with DuckDB EXCEPT...", flush=True)
t_exc = time.time()
con.execute("""
CREATE TEMP TABLE all_new_cands AS
SELECT source1_entity_id, matched_entity_id FROM (
    SELECT * FROM cand_top2
    UNION ALL
    SELECT * FROM cand_dl
    UNION ALL
    SELECT * FROM cand_a10
    UNION ALL
    SELECT * FROM cand_w6
)
EXCEPT
SELECT source1_entity_id, matched_entity_id FROM current_matches;
""")
n_cands, cov_s1 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM all_new_cands").fetchone()
print(f"Generated {n_cands:,} BRAND-NEW candidate pairs covering {cov_s1:,} S1 entities in {time.time()-t1:.1f}s!", flush=True)

# Step 3: Load models
print("\n[Step 3/6] Loading trained ensemble models...", flush=True)
t2 = time.time()
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(MODEL_DIR, "xgboost_v13_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lightgbm_v13_52features.txt"))
print(f"Models loaded in {time.time()-t2:.1f}s.", flush=True)

idx_name_ratio = V5_FEATURES_EXPANDED.index("name_ratio")
idx_addr_ratio = V5_FEATURES_EXPANDED.index("address_ratio")
idx_house_match = V5_FEATURES_EXPANDED.index("house_match")
idx_name_contains = V5_FEATURES_EXPANDED.index("name_contains")
idx_acronym = V5_FEATURES_EXPANDED.index("name_acronym_match")

con.execute("CREATE TEMP TABLE accepted_new_matches (source1_entity_id VARCHAR, matched_entity_id VARCHAR, prob FLOAT);")

CHUNK_SIZE = 150000
total_chunks = (n_cands + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"\n[Step 4/6] Streaming scoring across {total_chunks} chunks ({CHUNK_SIZE:,} per chunk)...", flush=True)

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
        FROM all_new_cands
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
    
    # Calibrated Precision Gates
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
        
    c_time = time.time() - t_c0
    print(f"  Chunk {chunk_idx+1}/{total_chunks} ({len(df_pl):,} pairs) -> Accepted +{n_acc:,} in {c_time:.1f}s | Cumulative Accepted: {total_accepted:,}", flush=True)
    
    del df_pl, feat_pl, X_chunk, p_xgb, p_lgb, p_blend
    gc.collect()

print("\n" + "=" * 80, flush=True)
print(f"Scoring Complete in {time.time()-t_stream_start:.1f}s! Total Accepted New Matches: {total_accepted:,}!", flush=True)
print("=" * 80, flush=True)

# Step 5: Merge with existing matches and apply injective constraint
print("\n[Step 5/6] Injective 1-to-1 bipartite resolution & matching_results.tsv generation...", flush=True)
t_merge = time.time()

# 1. Reference row order
row_order = []
with open(S1_RAW, "r", encoding="utf-8") as f:
    header = next(f)
    for line in f:
        row_order.append(line.split("\t")[0].strip())

# 2. Existing matches get highest priority (0.999) to protect validated baseline
existing_pairs = con.execute("SELECT source1_entity_id, matched_entity_id, 0.999 as prob FROM current_matches").fetchall()

# 3. New matches get their ensemble probability
new_pairs = con.execute("SELECT source1_entity_id, matched_entity_id, prob FROM accepted_new_matches").fetchall()

combined_pairs = [(r[0], r[1], float(r[2])) for r in existing_pairs] + [(r[0], r[1], float(r[2])) for r in new_pairs]
combined_pairs.sort(key=lambda x: -x[2])

seen_m = set()
s1_matches = defaultdict(list)
accepted_final = 0

for sid, mid, prob in combined_pairs:
    if mid not in seen_m:
        seen_m.add(mid)
        s1_matches[sid].append(mid)
        accepted_final += 1

# Write matching_results.tsv
non_empty = 0
empty = 0
with open(CURRENT_MATCHING_TSV, "w", encoding="utf-8") as f_out:
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in row_order:
        matches = s1_matches.get(sid, [])
        if matches:
            f_out.write(f"{sid}\t{','.join(matches)}\n")
            non_empty += 1
        else:
            f_out.write(f"{sid}\t\n")
            empty += 1

print(f"Final matching_results.tsv written in {time.time()-t_merge:.1f}s:", flush=True)
print(f"  Matched S1 entities: {non_empty:,} (was {n_curr_matched:,}, +{non_empty - n_curr_matched:,} gained!)", flush=True)
print(f"  Empty singletons:    {empty:,} (was {n_curr_unmatched:,}, dropped by {n_curr_unmatched - empty:,})", flush=True)
print(f"  Total match pairs:   {accepted_final:,} (was {n_curr_pairs:,}, +{accepted_final - n_curr_pairs:,} surge!)", flush=True)

# Step 6: Synchronize candidate_pairs.tsv & validate
print("\n[Step 6/6] Synchronizing candidate_pairs.tsv with all accepted matches...", flush=True)
t_sync = time.time()
TEMP_CAND_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs_temp.tsv")
added_cands = 0

with open(CANDIDATES_TSV, "r", encoding="utf-8") as f_in, open(TEMP_CAND_FILE, "w", encoding="utf-8") as f_out:
    f_out.write("source1_entity_id\tcandidate_entity_ids\n")
    header = next(f_in)
    for line in f_in:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0]
        cands = [x for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
        cand_set = set(cands)
        true_matches = set(s1_matches.get(sid, []))
        
        # Add any true matches missing from candidate list
        missing_cands = true_matches - cand_set
        if missing_cands:
            cands.extend(list(missing_cands))
            added_cands += len(missing_cands)
            
        if cands:
            f_out.write(f"{sid}\t{','.join(cands)}\n")
        else:
            f_out.write(f"{sid}\t\n")

# Replace candidate_pairs.tsv
os.replace(TEMP_CAND_FILE, CANDIDATES_TSV)
print(f"Synchronized candidate_pairs.tsv in {time.time()-t_sync:.1f}s (+{added_cands:,} IDs added).", flush=True)

# Package final zip
print(f"\nPackaging final submission zip: {SUBMISSION_ZIP}...", flush=True)
t_zip = time.time()
with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(CURRENT_MATCHING_TSV, arcname="matching_results.tsv")
    zf.write(CANDIDATES_TSV, arcname="candidate_pairs.tsv")
zip_size_mb = os.path.getsize(SUBMISSION_ZIP) / (1024 * 1024)
print(f"Zip created: {SUBMISSION_ZIP} ({zip_size_mb:.1f} MB) in {time.time()-t_zip:.1f}s.", flush=True)

print("\n" + "=" * 80, flush=True)
print(f"V20 FINAL PIPELINE COMPLETED SUCCESSFULLY IN {time.time()-t0:.1f}s!", flush=True)
print("=" * 80, flush=True)
