import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, duckdb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORM_DIR = os.path.join(BASE, "normalized_data")
S1_NORM = os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")
S2_NORM = os.path.join(NORM_DIR, "test_source2_normalized.tsv").replace("\\", "/")
S3_NORM = os.path.join(NORM_DIR, "test_source3_normalized.tsv").replace("\\", "/")

print("1. Connecting DuckDB...")
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")

t0 = time.time()
print("2. Loading test tables...")
con.execute(f"""
CREATE TEMP TABLE s1 AS 
SELECT entity_id AS source1_entity_id, 
       name_normalized AS clean_name, 
       address_normalized AS clean_addr, 
       country 
FROM read_csv('{S1_NORM}', delim='\t', header=true);

CREATE TEMP TABLE s23 AS 
SELECT entity_id AS matched_entity_id, 
       name_normalized AS clean_name, 
       address_normalized AS clean_addr, 
       country 
FROM (
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('{S2_NORM}', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('{S3_NORM}', delim='\t', header=true)
);
""")
print(f"Loaded tables in {time.time()-t0:.1f}s.")

# Check Channel 1: Cleaned brand first word + Country
STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"
t1 = time.time()
print("\n3. Running Channel 1 (First Brand Token >= 4 chars)...")
con.execute(f"""
CREATE TEMP TABLE ch1_s1 AS
WITH base AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS}) AS words
    FROM s1
)
SELECT source1_entity_id, country, words[1] AS tok
FROM base WHERE LEN(words) >= 1;

CREATE TEMP TABLE ch1_s23 AS
WITH base AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS}) AS words
    FROM s23
)
SELECT matched_entity_id, country, words[1] AS tok
FROM base WHERE LEN(words) >= 1;

CREATE TEMP TABLE ch1_pairs AS
SELECT source1_entity_id, matched_entity_id FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM ch1_s1 s1 JOIN ch1_s23 s23 ON s1.country = s23.country AND s1.tok = s23.tok
    WHERE LENGTH(s1.tok) >= 4
) WHERE cnt <= 15;
""")
ch1_cnt = con.execute("SELECT COUNT(*) FROM ch1_pairs").fetchone()[0]
print(f"Ch1: {ch1_cnt:,} pairs in {time.time()-t1:.1f}s")
