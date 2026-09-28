import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, duckdb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORM_DIR = os.path.join(BASE, "normalized_data")
S1_NORM = os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")
S2_NORM = os.path.join(NORM_DIR, "test_source2_normalized.tsv").replace("\\", "/")
S3_NORM = os.path.join(NORM_DIR, "test_source3_normalized.tsv").replace("\\", "/")
CANDS_FILE = os.path.join(BASE, "test_scored_candidates.tsv").replace("\\", "/")

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")
con.execute("PRAGMA max_temp_directory_size='500GiB'")

print("1. Loading existing candidate pool (prob >= 0.05)...")
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE existing_cands AS
SELECT source1_entity_id, matched_entity_id
FROM read_csv('{CANDS_FILE}', delim='\t', header=true)
WHERE probability >= 0.05;
""")
n_exist = con.execute("SELECT COUNT(*) FROM existing_cands").fetchone()[0]
print(f"Loaded {n_exist:,} existing candidates in {time.time()-t0:.1f}s")

# Load S1 and S23
print("\n2. Loading S1 & S23 tables...")
t0 = time.time()
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
print(f"Loaded in {time.time()-t0:.1f}s")

STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"
ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"

print("\n3. Generating union of new channels (excluding existing candidates)...")
t0 = time.time()
con.execute(f"""
CREATE TEMP TABLE new_cands AS
-- Ch1: Sorted 2 Brand Words
WITH s1_k2 AS (
    SELECT source1_entity_id, country,
           CASE WHEN words[1] < words[2] THEN words[1] || '_' || words[2] ELSE words[2] || '_' || words[1] END AS k2
    FROM (SELECT source1_entity_id, country, LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) AS words FROM s1)
    WHERE LEN(words) >= 2
),
s23_k2 AS (
    SELECT matched_entity_id, country,
           CASE WHEN words[1] < words[2] THEN words[1] || '_' || words[2] ELSE words[2] || '_' || words[1] END AS k2
    FROM (SELECT matched_entity_id, country, LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) AS words FROM s23)
    WHERE LEN(words) >= 2
),
vk1 AS (SELECT country, k2 FROM s1_k2 GROUP BY country, k2 HAVING COUNT(*) <= 50),
p1 AS (
    SELECT s1.source1_entity_id, s23.matched_entity_id
    FROM s1_k2 s1 JOIN vk1 ON s1.country = vk1.country AND s1.k2 = vk1.k2
    JOIN s23_k2 s23 ON s1.country = s23.country AND s1.k2 = s23.k2
    QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15
),

-- Ch2: Door Number + Locality
s1_dl AS (
    SELECT source1_entity_id, country, house_number || '_' || word as k
    FROM (
        SELECT source1_entity_id, country,
               LTRIM(COALESCE(REGEXP_EXTRACT(clean_addr, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') as house_number,
               UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 6 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as word
        FROM s1
    ) WHERE LENGTH(house_number) >= 1
),
s23_dl AS (
    SELECT matched_entity_id, country, house_number || '_' || word as k
    FROM (
        SELECT matched_entity_id, country,
               LTRIM(COALESCE(REGEXP_EXTRACT(clean_addr, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') as house_number,
               UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 6 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as word
        FROM s23
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

n_new = con.execute("SELECT COUNT(*) FROM new_cands").fetchone()[0]
print(f"Discovered {n_new:,} BRAND-NEW candidate pairs in {time.time()-t0:.1f}s!")
