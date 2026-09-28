import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, duckdb

os.makedirs(r"D:\duckdb_temp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")
con.execute("PRAGMA max_temp_directory_size='500GiB'")

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORM_DIR = os.path.join(BASE, "normalized_data")
OUTPUT_DIR = os.path.join(BASE, "output")
S1_NORM = os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")
S2_NORM = os.path.join(NORM_DIR, "test_source2_normalized.tsv").replace("\\", "/")
S3_NORM = os.path.join(NORM_DIR, "test_source3_normalized.tsv").replace("\\", "/")
CURRENT_MATCHING = os.path.join(OUTPUT_DIR, "matching_results.tsv").replace("\\", "/")

t0 = time.time()
print("1. Loading test reference tables into DuckDB...")
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

# Load existing matches to exclude them
con.execute(f"""
CREATE TEMP TABLE existing_matches AS
SELECT source1_entity_id, UNNEST(STR_SPLIT(matched_entity_ids, ',')) AS matched_entity_id
FROM read_csv('{CURRENT_MATCHING}', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';
""")
n_exist = con.execute("SELECT COUNT(*) FROM existing_matches").fetchone()[0]
print(f"Loaded {n_exist:,} existing accepted matches.")

ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"

print("\n2. Generating Full Test Channels (Address Words + House Street + Addr10)...")
t1 = time.time()
con.execute(f"""
-- 1. Sorted 2 Distinctive Address Words across full test set
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

-- 2. House Number + Street Word across full test set
CREATE TEMP TABLE cand_hs AS
WITH s1_hs AS (
    SELECT source1_entity_id, country, house_number,
           UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as street_tok
    FROM s1_tbl WHERE LENGTH(house_number) >= 1
),
s23_hs AS (
    SELECT matched_entity_id, country, house_number,
           UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as street_tok
    FROM s23_tbl WHERE LENGTH(house_number) >= 1
),
vk_hs AS (SELECT country, house_number || '_' || street_tok as k FROM s1_hs GROUP BY country, k HAVING COUNT(*) <= 50)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_hs s1
JOIN vk_hs ON s1.country = vk_hs.country AND s1.house_number || '_' || s1.street_tok = vk_hs.k
JOIN s23_hs s23 ON s1.country = s23.country AND s23.house_number || '_' || s23.street_tok = vk_hs.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;

-- Union new candidates (excluding existing matches)
CREATE TEMP TABLE full_new_cands AS
SELECT DISTINCT source1_entity_id, matched_entity_id
FROM (
    SELECT * FROM cand_top2
    UNION ALL
    SELECT * FROM cand_hs
) sub
WHERE (source1_entity_id, matched_entity_id) NOT IN (SELECT source1_entity_id, matched_entity_id FROM existing_matches);
""")

n_new = con.execute("SELECT COUNT(*) FROM full_new_cands").fetchone()[0]
print(f"Generated {n_new:,} BRAND-NEW candidate pairs across full test set in {time.time()-t1:.1f}s!")
