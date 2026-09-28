import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, duckdb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORM_DIR = os.path.join(BASE, "normalized_data")
S1_NORM = os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")
S2_NORM = os.path.join(NORM_DIR, "test_source2_normalized.tsv").replace("\\", "/")
S3_NORM = os.path.join(NORM_DIR, "test_source3_normalized.tsv").replace("\\", "/")

os.makedirs(r"D:\duckdb_temp", exist_ok=True)

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")
con.execute("PRAGMA max_temp_directory_size='500GiB'")

t0 = time.time()
print("1. Loading test tables into DuckDB...")
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
print(f"Loaded test tables in {time.time()-t0:.1f}s.")

STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"
ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"

# CHANNEL 1: Sorted 2 Brand Words
t1 = time.time()
print("\n2. Channel 1: Sorted 2 Brand Words...")
con.execute(f"""
CREATE TEMP TABLE ch1_pairs AS
WITH s1_k2 AS (
    SELECT source1_entity_id, country,
           CASE WHEN words[1] < words[2] THEN words[1] || '_' || words[2] ELSE words[2] || '_' || words[1] END AS k2
    FROM (
        SELECT source1_entity_id, country,
               LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) AS words
        FROM s1
    ) WHERE LEN(words) >= 2
),
s23_k2 AS (
    SELECT matched_entity_id, country,
           CASE WHEN words[1] < words[2] THEN words[1] || '_' || words[2] ELSE words[2] || '_' || words[1] END AS k2
    FROM (
        SELECT matched_entity_id, country,
               LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) AS words
        FROM s23
    ) WHERE LEN(words) >= 2
),
vk AS (
    SELECT country, k2 FROM s1_k2 GROUP BY country, k2 HAVING COUNT(*) <= 50
)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_k2 s1
JOIN vk ON s1.country = vk.country AND s1.k2 = vk.k2
JOIN s23_k2 s23 ON s1.country = s23.country AND s1.k2 = s23.k2
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
ch1_cnt = con.execute("SELECT COUNT(*) FROM ch1_pairs").fetchone()[0]
print(f"  Ch1: {ch1_cnt:,} pairs in {time.time()-t1:.1f}s")

# CHANNEL 2: Door Number + Locality Token
t2 = time.time()
print("\n3. Channel 2: Door Number + Distinctive Locality...")
con.execute(f"""
CREATE TEMP TABLE ch2_pairs AS
WITH s1_dl AS (
    SELECT source1_entity_id, country,
           house_number || '_' || word as k
    FROM (
        SELECT source1_entity_id, country,
               LTRIM(COALESCE(REGEXP_EXTRACT(clean_addr, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') as house_number,
               UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 6 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as word
        FROM s1
    )
    WHERE LENGTH(house_number) >= 1
),
s23_dl AS (
    SELECT matched_entity_id, country,
           house_number || '_' || word as k
    FROM (
        SELECT matched_entity_id, country,
               LTRIM(COALESCE(REGEXP_EXTRACT(clean_addr, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') as house_number,
               UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 6 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as word
        FROM s23
    )
    WHERE LENGTH(house_number) >= 1
),
vk AS (
    SELECT country, k FROM s1_dl GROUP BY country, k HAVING COUNT(*) <= 50
)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_dl s1
JOIN vk ON s1.country = vk.country AND s1.k = vk.k
JOIN s23_dl s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
ch2_cnt = con.execute("SELECT COUNT(*) FROM ch2_pairs").fetchone()[0]
print(f"  Ch2: {ch2_cnt:,} pairs in {time.time()-t2:.1f}s")

# CHANNEL 3: Sorted 2 Longest Distinctive Address Words
t3 = time.time()
print("\n4. Channel 3: Sorted 2 Longest Address Words...")
con.execute(f"""
CREATE TEMP TABLE ch3_pairs AS
WITH s1_top2 AS (
    SELECT source1_entity_id, country,
           CASE WHEN sorted_words[1] < sorted_words[2] 
                THEN sorted_words[1] || '_' || sorted_words[2] 
                ELSE sorted_words[2] || '_' || sorted_words[1] END as k
    FROM (
        SELECT source1_entity_id, country,
               LIST_SLICE(LIST_REVERSE_SORT(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')), 1, 2) as sorted_words
        FROM s1
    )
    WHERE LEN(sorted_words) >= 2
),
s23_top2 AS (
    SELECT matched_entity_id, country,
           CASE WHEN sorted_words[1] < sorted_words[2] 
                THEN sorted_words[1] || '_' || sorted_words[2] 
                ELSE sorted_words[2] || '_' || sorted_words[1] END as k
    FROM (
        SELECT matched_entity_id, country,
               LIST_SLICE(LIST_REVERSE_SORT(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')), 1, 2) as sorted_words
        FROM s23
    )
    WHERE LEN(sorted_words) >= 2
),
vk AS (
    SELECT country, k FROM s1_top2 GROUP BY country, k HAVING COUNT(*) <= 50
)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_top2 s1
JOIN vk ON s1.country = vk.country AND s1.k = vk.k
JOIN s23_top2 s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
ch3_cnt = con.execute("SELECT COUNT(*) FROM ch3_pairs").fetchone()[0]
print(f"  Ch3: {ch3_cnt:,} pairs in {time.time()-t3:.1f}s")
