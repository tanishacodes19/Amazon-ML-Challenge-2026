import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, time

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")

print("Investigating how to match the 242,511 empty S1 entities...", flush=True)
t0 = time.time()

# 1. Load empty S1 entities
con.execute("""
CREATE TEMP TABLE empty_s1 AS
SELECT s.entity_id as s1_id, s.country, s.name_normalized as n1, s.address_normalized as a1,
       LTRIM(COALESCE(REGEXP_EXTRACT(s.address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_number,
       COALESCE(REGEXP_EXTRACT(s.address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS postal_code
FROM read_csv('normalized_data/test_source1_normalized.tsv', delim='\t', header=true) s
JOIN (
    SELECT source1_entity_id 
    FROM read_csv('output/matching_results.tsv', delim='\t', header=true) 
    WHERE matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = ''
) e ON s.entity_id = e.source1_entity_id;
""")
n_empty = con.execute("SELECT COUNT(*) FROM empty_s1").fetchone()[0]
print(f"Loaded {n_empty:,} empty S1 entities in {time.time()-t0:.1f}s.", flush=True)

# 2. Load all available S23 entities that are NOT yet matched to anyone in V20
con.execute("""
CREATE TEMP TABLE matched_s23 AS
SELECT UNNEST(STR_SPLIT(matched_entity_ids, ',')) as m_id
FROM read_csv('output/matching_results.tsv', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';

CREATE TEMP TABLE free_s23 AS
SELECT entity_id as m_id, country, name_normalized as n2, address_normalized as a2,
       LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_number,
       COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS postal_code
FROM (
    SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/test_source2_normalized.tsv', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/test_source3_normalized.tsv', delim='\t', header=true)
) s
WHERE entity_id NOT IN (SELECT m_id FROM matched_s23);
""")
n_free = con.execute("SELECT COUNT(*) FROM free_s23").fetchone()[0]
print(f"Loaded {n_free:,} available/free S2/S3 entities (not matched yet).", flush=True)

# Test Method 1: Exact Name (len >= 4) in same country
t1 = time.time()
n_m1 = con.execute("""
SELECT COUNT(DISTINCT e.s1_id)
FROM empty_s1 e
JOIN free_s23 f ON e.country = f.country AND e.n1 = f.n2
WHERE LENGTH(e.n1) >= 4;
""").fetchone()[0]
print(f"  Method 1 (Exact Name): {n_m1:,} empty S1 entities match an available S23 in {time.time()-t1:.1f}s", flush=True)

# Test Method 2: Exact Address (len >= 8) in same country
t2 = time.time()
n_m2 = con.execute("""
SELECT COUNT(DISTINCT e.s1_id)
FROM empty_s1 e
JOIN free_s23 f ON e.country = f.country AND e.a1 = f.a2
WHERE LENGTH(e.a1) >= 8;
""").fetchone()[0]
print(f"  Method 2 (Exact Address): {n_m2:,} empty S1 entities match an available S23 in {time.time()-t2:.1f}s", flush=True)

# Test Method 3: Same House Number + First Word of Name (len >= 4)
t3 = time.time()
STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"
n_m3 = con.execute(f"""
WITH e_w AS (
    SELECT s1_id, country, house_number,
           (LIST_FILTER(STR_SPLIT(n1, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS}))[1] as w1
    FROM empty_s1 WHERE LENGTH(house_number) >= 1
),
f_w AS (
    SELECT m_id, country, house_number,
           (LIST_FILTER(STR_SPLIT(n2, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS}))[1] as w1
    FROM free_s23 WHERE LENGTH(house_number) >= 1
)
SELECT COUNT(DISTINCT e.s1_id)
FROM e_w e
JOIN f_w f ON e.country = f.country AND e.house_number = f.house_number AND e.w1 = f.w1
WHERE e.w1 IS NOT NULL AND f.w1 IS NOT NULL;
""").fetchone()[0]
print(f"  Method 3 (House No + First Name Word): {n_m3:,} empty S1 entities match in {time.time()-t3:.1f}s", flush=True)

# Test Method 4: Name Prefix 6 chars + Address Prefix 6 chars
t4 = time.time()
n_m4 = con.execute("""
SELECT COUNT(DISTINCT e.s1_id)
FROM empty_s1 e
JOIN free_s23 f ON e.country = f.country 
               AND SUBSTRING(e.n1, 1, 6) = SUBSTRING(f.n2, 1, 6)
               AND SUBSTRING(e.a1, 1, 6) = SUBSTRING(f.a2, 1, 6)
WHERE LENGTH(e.n1) >= 6 AND LENGTH(e.a1) >= 6;
""").fetchone()[0]
print(f"  Method 4 (Name Prefix 6 + Addr Prefix 6): {n_m4:,} empty S1 entities match in {time.time()-t4:.1f}s", flush=True)

# Total unique empty S1 matched by these 4 high-precision methods:
t5 = time.time()
total_cov = con.execute(f"""
WITH m1 AS (
    SELECT DISTINCT e.s1_id FROM empty_s1 e JOIN free_s23 f ON e.country = f.country AND e.n1 = f.n2 WHERE LENGTH(e.n1) >= 4
),
m2 AS (
    SELECT DISTINCT e.s1_id FROM empty_s1 e JOIN free_s23 f ON e.country = f.country AND e.a1 = f.a2 WHERE LENGTH(e.a1) >= 8
),
m3 AS (
    SELECT DISTINCT e.s1_id
    FROM (SELECT s1_id, country, house_number, (LIST_FILTER(STR_SPLIT(n1, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS}))[1] as w1 FROM empty_s1 WHERE LENGTH(house_number) >= 1) e
    JOIN (SELECT m_id, country, house_number, (LIST_FILTER(STR_SPLIT(n2, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS}))[1] as w1 FROM free_s23 WHERE LENGTH(house_number) >= 1) f
      ON e.country = f.country AND e.house_number = f.house_number AND e.w1 = f.w1
    WHERE e.w1 IS NOT NULL AND f.w1 IS NOT NULL
),
m4 AS (
    SELECT DISTINCT e.s1_id FROM empty_s1 e JOIN free_s23 f ON e.country = f.country AND SUBSTRING(e.n1, 1, 6) = SUBSTRING(f.n2, 1, 6) AND SUBSTRING(e.a1, 1, 6) = SUBSTRING(f.a2, 1, 6) WHERE LENGTH(e.n1) >= 6 AND LENGTH(e.a1) >= 6
)
SELECT COUNT(DISTINCT s1_id) FROM (
    SELECT * FROM m1 UNION SELECT * FROM m2 UNION SELECT * FROM m3 UNION SELECT * FROM m4
);
""").fetchone()[0]
print(f"\n=======================================================", flush=True)
print(f"TOTAL RECOVERABLE EMPTY S1 ENTITIES: {total_cov:,} / {n_empty:,} ({total_cov/n_empty*100:.2f}%) in {time.time()-t5:.1f}s!", flush=True)
print(f"=======================================================", flush=True)
