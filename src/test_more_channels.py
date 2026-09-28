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
S1_NORM = os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")
S2_NORM = os.path.join(NORM_DIR, "test_source2_normalized.tsv").replace("\\", "/")
S3_NORM = os.path.join(NORM_DIR, "test_source3_normalized.tsv").replace("\\", "/")
OUTPUT_DIR = os.path.join(BASE, "output")

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
print(f"Loaded in {time.time()-t0:.1f}s.")

CURRENT_MATCHING = os.path.join(OUTPUT_DIR, "matching_results.tsv")
con.execute(f"""
CREATE TEMP TABLE current_matched_s1 AS
SELECT source1_entity_id 
FROM read_csv('{CURRENT_MATCHING.replace(chr(92), '/')}', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';
""")

con.execute("""
CREATE TEMP TABLE s1_unmatched AS
SELECT s.* 
FROM s1_tbl s
LEFT JOIN current_matched_s1 m ON s.source1_entity_id = m.source1_entity_id
WHERE m.source1_entity_id IS NULL;
""")

# Additional Targeted Channels:
# Channel 4: Postal Code (5-6 digits) + Brand First 3 Letters
print("\n[Targeted Ch4] Postal Code + Brand First 3 Letters...")
t1 = time.time()
con.execute("""
CREATE TEMP TABLE s1_p3 AS
SELECT source1_entity_id, country, postal_code || '_' || SUBSTRING(clean_name, 1, 3) as k
FROM s1_unmatched
WHERE LENGTH(postal_code) >= 5 AND LENGTH(clean_name) >= 3;

CREATE TEMP TABLE s23_p3 AS
SELECT matched_entity_id, country, postal_code || '_' || SUBSTRING(clean_name, 1, 3) as k
FROM s23_tbl
WHERE LENGTH(postal_code) >= 5 AND LENGTH(clean_name) >= 3;

CREATE TEMP TABLE vk_p3 AS
SELECT country, k FROM s1_p3 GROUP BY country, k HAVING COUNT(*) <= 50;

CREATE TEMP TABLE cand_p3 AS
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_p3 s1
JOIN vk_p3 ON s1.country = vk_p3.country AND s1.k = vk_p3.k
JOIN s23_p3 s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
cnt_p3 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_p3").fetchall()[0]
print(f"  Targeted Ch4 generated {cnt_p3[0]:,} candidates covering {cnt_p3[1]:,} unmatched S1 entities in {time.time()-t1:.1f}s!")

# Channel 5: Address 10-char prefix (when clean_addr >= 10 chars)
print("\n[Targeted Ch5] Address 10-char prefix...")
t2 = time.time()
con.execute("""
CREATE TEMP TABLE s1_a10 AS
SELECT source1_entity_id, country, SUBSTRING(clean_addr, 1, 10) as k
FROM s1_unmatched
WHERE LENGTH(clean_addr) >= 10;

CREATE TEMP TABLE s23_a10 AS
SELECT matched_entity_id, country, SUBSTRING(clean_addr, 1, 10) as k
FROM s23_tbl
WHERE LENGTH(clean_addr) >= 10;

CREATE TEMP TABLE vk_a10 AS
SELECT country, k FROM s1_a10 GROUP BY country, k HAVING COUNT(*) <= 50;

CREATE TEMP TABLE cand_a10 AS
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_a10 s1
JOIN vk_a10 ON s1.country = vk_a10.country AND s1.k = vk_a10.k
JOIN s23_a10 s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
cnt_a10 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_a10").fetchall()[0]
print(f"  Targeted Ch5 generated {cnt_a10[0]:,} candidates covering {cnt_a10[1]:,} unmatched S1 entities in {time.time()-t2:.1f}s!")
