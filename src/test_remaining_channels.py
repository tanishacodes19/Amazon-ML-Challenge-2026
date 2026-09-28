import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, duckdb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORM_DIR = os.path.join(BASE, "normalized_data")
S1_NORM = os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")
S2_NORM = os.path.join(NORM_DIR, "test_source2_normalized.tsv").replace("\\", "/")
S3_NORM = os.path.join(NORM_DIR, "test_source3_normalized.tsv").replace("\\", "/")
OUTPUT_DIR = os.path.join(BASE, "output")

os.makedirs(r"D:\duckdb_temp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")
con.execute("PRAGMA max_temp_directory_size='500GiB'")

t0 = time.time()
print("1. Loading test reference tables...")
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

# Check how many S1 currently have zero matches
CURRENT_MATCHING = os.path.join(OUTPUT_DIR, "matching_results.tsv")
con.execute(f"""
CREATE TEMP TABLE current_matched_s1 AS
SELECT source1_entity_id 
FROM read_csv('{CURRENT_MATCHING.replace(chr(92), '/')}', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';
""")
curr_matched = con.execute("SELECT COUNT(*) FROM current_matched_s1").fetchone()[0]
print(f"Currently matched S1 entities: {curr_matched:,} / 1,732,544 ({curr_matched/1732544*100:.2f}%)")
print(f"S1 entities STILL NEEDING MATCHES: {1732544 - curr_matched:,}")

# Let's focus especially on uncovering candidates for the 308,150 empty S1 entities!
con.execute("""
CREATE TEMP TABLE s1_unmatched AS
SELECT s.* 
FROM s1_tbl s
LEFT JOIN current_matched_s1 m ON s.source1_entity_id = m.source1_entity_id
WHERE m.source1_entity_id IS NULL;
""")
n_unm = con.execute("SELECT COUNT(*) FROM s1_unmatched").fetchone()[0]
print(f"Loaded {n_unm:,} unmatched S1 entities for targeted candidate mining!")

STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"
ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"

# TARGETED CHANNEL 1: House Number + Street Word (for unmatched S1)
print("\n[Targeted Ch1] House Number + Street Word on unmatched S1...")
t1 = time.time()
con.execute(f"""
CREATE TEMP TABLE s1_hs AS
WITH base AS (
    SELECT source1_entity_id, country, house_number,
           UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as street_tok
    FROM s1_unmatched
    WHERE LENGTH(house_number) >= 1
)
SELECT source1_entity_id, country, house_number || '_' || street_tok as k
FROM base;

CREATE TEMP TABLE s23_hs AS
WITH base AS (
    SELECT matched_entity_id, country, house_number,
           UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as street_tok
    FROM s23_tbl
    WHERE LENGTH(house_number) >= 1
)
SELECT matched_entity_id, country, house_number || '_' || street_tok as k
FROM base;

CREATE TEMP TABLE vk_hs AS
SELECT country, k FROM s1_hs GROUP BY country, k HAVING COUNT(*) <= 50;

CREATE TEMP TABLE cand_hs AS
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_hs s1
JOIN vk_hs ON s1.country = vk_hs.country AND s1.k = vk_hs.k
JOIN s23_hs s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
cnt_hs = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_hs").fetchall()[0]
print(f"  Targeted Ch1 generated {cnt_hs[0]:,} candidates covering {cnt_hs[1]:,} unmatched S1 entities in {time.time()-t1:.1f}s!")

# TARGETED CHANNEL 2: Sorted 2 Longest Distinctive Address Words
print("\n[Targeted Ch2] Sorted 2 Address Words on unmatched S1...")
t2 = time.time()
con.execute(f"""
CREATE TEMP TABLE s1_top2 AS
SELECT source1_entity_id, country,
       CASE WHEN sorted_words[1] < sorted_words[2] 
            THEN sorted_words[1] || '_' || sorted_words[2] 
            ELSE sorted_words[2] || '_' || sorted_words[1] END as k
FROM (
    SELECT source1_entity_id, country,
           LIST_SLICE(LIST_REVERSE_SORT(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')), 1, 2) as sorted_words
    FROM s1_unmatched
)
WHERE LEN(sorted_words) >= 2;

CREATE TEMP TABLE s23_top2 AS
SELECT matched_entity_id, country,
       CASE WHEN sorted_words[1] < sorted_words[2] 
            THEN sorted_words[1] || '_' || sorted_words[2] 
            ELSE sorted_words[2] || '_' || sorted_words[1] END as k
FROM (
    SELECT matched_entity_id, country,
           LIST_SLICE(LIST_REVERSE_SORT(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')), 1, 2) as sorted_words
    FROM s23_tbl
)
WHERE LEN(sorted_words) >= 2;

CREATE TEMP TABLE vk_top2 AS
SELECT country, k FROM s1_top2 GROUP BY country, k HAVING COUNT(*) <= 50;

CREATE TEMP TABLE cand_top2 AS
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_top2 s1
JOIN vk_top2 ON s1.country = vk_top2.country AND s1.k = vk_top2.k
JOIN s23_top2 s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
cnt_top2 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_top2").fetchall()[0]
print(f"  Targeted Ch2 generated {cnt_top2[0]:,} candidates covering {cnt_top2[1]:,} unmatched S1 entities in {time.time()-t2:.1f}s!")

# TARGETED CHANNEL 3: Single Distinctive Brand Word (length >= 6)
print("\n[Targeted Ch3] Distinctive Single Brand Word (len >= 6) on unmatched S1...")
t3 = time.time()
con.execute(f"""
CREATE TEMP TABLE s1_w6 AS
SELECT source1_entity_id, country,
       UNNEST(LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 6 AND x NOT IN {STOP_WORDS})) as w
FROM s1_unmatched;

CREATE TEMP TABLE s23_w6 AS
SELECT matched_entity_id, country,
       UNNEST(LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 6 AND x NOT IN {STOP_WORDS})) as w
FROM s23_tbl;

CREATE TEMP TABLE vk_w6 AS
SELECT country, w FROM s1_w6 GROUP BY country, w HAVING COUNT(*) <= 30;

CREATE TEMP TABLE cand_w6 AS
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_w6 s1
JOIN vk_w6 ON s1.country = vk_w6.country AND s1.w = vk_w6.w
JOIN s23_w6 s23 ON s1.country = s23.country AND s1.w = s23.w
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
cnt_w6 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_w6").fetchall()[0]
print(f"  Targeted Ch3 generated {cnt_w6[0]:,} candidates covering {cnt_w6[1]:,} unmatched S1 entities in {time.time()-t3:.1f}s!")

# Union of all targeted candidates
print("\nUnioning all targeted candidates for previously empty S1 entities...")
con.execute("""
CREATE TEMP TABLE all_targeted_cands AS
SELECT DISTINCT source1_entity_id, matched_entity_id
FROM (
    SELECT * FROM cand_hs
    UNION ALL
    SELECT * FROM cand_top2
    UNION ALL
    SELECT * FROM cand_w6
);
""")
total_t = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM all_targeted_cands").fetchall()[0]
print(f"\nTOTAL TARGETED CANDIDATE POOL: {total_t[0]:,} pairs covering {total_t[1]:,} / {n_unm:,} previously empty S1 entities ({total_t[1]/n_unm*100:.2f}%)!")
