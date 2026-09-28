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
print("1. Loading test reference tables into DuckDB...", flush=True)
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

# Load existing matches to exclude them
con.execute(f"""
CREATE TEMP TABLE existing_matches AS
SELECT source1_entity_id, UNNEST(STR_SPLIT(matched_entity_ids, ',')) AS matched_entity_id
FROM read_csv('{CURRENT_MATCHING}', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';

CREATE TEMP TABLE current_matched_s1 AS
SELECT DISTINCT source1_entity_id FROM existing_matches;

CREATE TEMP TABLE s1_unmatched AS
SELECT s.* FROM s1_tbl s
LEFT JOIN current_matched_s1 m ON s.source1_entity_id = m.source1_entity_id
WHERE m.source1_entity_id IS NULL;
""")
n_exist = con.execute("SELECT COUNT(*) FROM existing_matches").fetchone()[0]
n_unm = con.execute("SELECT COUNT(*) FROM s1_unmatched").fetchone()[0]
print(f"Existing accepted matches: {n_exist:,}. Unmatched S1 entities: {n_unm:,}.", flush=True)

ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"
STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"

# 1. Sorted 2 Distinctive Address Words across full test set
t1 = time.time()
print("Generating Channel 1: Sorted 2 Address Words (full test set)...", flush=True)
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
print(f"  Ch1 generated {cnt_top2:,} pairs in {time.time()-t1:.1f}s", flush=True)

# 2. House Number + Locality word (length >= 6)
t2 = time.time()
print("Generating Channel 2: House Number + Locality (len >= 6)...", flush=True)
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
print(f"  Ch2 generated {cnt_dl:,} pairs in {time.time()-t2:.1f}s", flush=True)

# 3. Targeted Postal Code + Name Prefix 3 (on unmatched S1)
t3 = time.time()
print("Generating Channel 3: Postal Code + Name Prefix 3 (unmatched S1)...", flush=True)
con.execute("""
CREATE TEMP TABLE cand_p3 AS
WITH s1_p3 AS (
    SELECT source1_entity_id, country, postal_code || '_' || SUBSTRING(clean_name, 1, 3) as k
    FROM s1_unmatched
    WHERE LENGTH(postal_code) >= 5 AND LENGTH(clean_name) >= 3
),
s23_p3 AS (
    SELECT matched_entity_id, country, postal_code || '_' || SUBSTRING(clean_name, 1, 3) as k
    FROM s23_tbl
    WHERE LENGTH(postal_code) >= 5 AND LENGTH(clean_name) >= 3
),
vk_p3 AS (SELECT country, k FROM s1_p3 GROUP BY country, k HAVING COUNT(*) <= 50)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_p3 s1
JOIN vk_p3 ON s1.country = vk_p3.country AND s1.k = vk_p3.k
JOIN s23_p3 s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
cnt_p3 = con.execute("SELECT COUNT(*) FROM cand_p3").fetchone()[0]
print(f"  Ch3 generated {cnt_p3:,} pairs in {time.time()-t3:.1f}s", flush=True)

# 4. Targeted Address Prefix 10 chars (on unmatched S1)
t4 = time.time()
print("Generating Channel 4: Address 10-char prefix (unmatched S1)...", flush=True)
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
print(f"  Ch4 generated {cnt_a10:,} pairs in {time.time()-t4:.1f}s", flush=True)

# 5. Targeted Distinctive Brand Word len >= 6 (on unmatched S1)
t5 = time.time()
print("Generating Channel 5: Distinctive Brand Word (unmatched S1)...", flush=True)
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
print(f"  Ch5 generated {cnt_w6:,} pairs in {time.time()-t5:.1f}s", flush=True)

# Union and EXCEPT existing matches
t6 = time.time()
print("Unioning all channels and subtracting existing matches...", flush=True)
con.execute("""
CREATE TEMP TABLE all_new_cands AS
SELECT source1_entity_id, matched_entity_id FROM (
    SELECT * FROM cand_top2
    UNION ALL
    SELECT * FROM cand_dl
    UNION ALL
    SELECT * FROM cand_p3
    UNION ALL
    SELECT * FROM cand_a10
    UNION ALL
    SELECT * FROM cand_w6
)
EXCEPT
SELECT source1_entity_id, matched_entity_id FROM existing_matches;
""")
total_new, cov_s1 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM all_new_cands").fetchone()
print(f"\n=======================================================", flush=True)
print(f"TOTAL BRAND-NEW CANDIDATE PAIRS: {total_new:,}", flush=True)
print(f"Covering {cov_s1:,} S1 entities in {time.time()-t6:.1f}s!", flush=True)
print(f"Total time: {time.time()-t0:.1f}s", flush=True)
print(f"=======================================================", flush=True)
