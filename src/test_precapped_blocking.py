import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, duckdb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORM_DIR = os.path.join(BASE, "normalized_data")
S1_NORM = os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")
S2_NORM = os.path.join(NORM_DIR, "test_source2_normalized.tsv").replace("\\", "/")
S3_NORM = os.path.join(NORM_DIR, "test_source3_normalized.tsv").replace("\\", "/")

# Use D drive for duckdb temp storage (927 GB free!)
os.makedirs(r"D:\duckdb_temp", exist_ok=True)

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")
con.execute("PRAGMA max_temp_directory_size='500GiB'")

t0 = time.time()
print("1. Loading test tables...")
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

# Channel 3: Sorted 2-Token Key with PRE-JOIN Frequency Cap (<= 50)
STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"
t1 = time.time()
print("\n2. Testing Pre-Capped Channel 3 (Sorted 2-Token Key)...")
con.execute(f"""
CREATE TEMP TABLE s1_k2 AS
WITH base AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) AS words
    FROM s1
)
SELECT source1_entity_id, country,
       CASE WHEN words[1] < words[2] THEN words[1] || '_' || words[2] ELSE words[2] || '_' || words[1] END AS k2
FROM base WHERE LEN(words) >= 2;

CREATE TEMP TABLE s23_k2 AS
WITH base AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) AS words
    FROM s23
)
SELECT matched_entity_id, country,
       CASE WHEN words[1] < words[2] THEN words[1] || '_' || words[2] ELSE words[2] || '_' || words[1] END AS k2
FROM base WHERE LEN(words) >= 2;

-- Pre-cap keys that appear <= 50 times in S1 to eliminate combinatorial explosion
CREATE TEMP TABLE valid_keys AS
SELECT country, k2
FROM s1_k2
GROUP BY country, k2
HAVING COUNT(*) <= 50;

CREATE TEMP TABLE ch3_pairs AS
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_k2 s1
JOIN valid_keys vk ON s1.country = vk.country AND s1.k2 = vk.k2
JOIN s23_k2 s23 ON s1.country = s23.country AND s1.k2 = s23.k2
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")

ch3_cnt = con.execute("SELECT COUNT(*) FROM ch3_pairs").fetchone()[0]
print(f"Ch3 generated {ch3_cnt:,} pairs across the ENTIRE 1.73M test set in {time.time()-t1:.1f}s!")
