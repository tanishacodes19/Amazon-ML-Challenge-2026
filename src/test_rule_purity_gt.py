import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, time

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")

print("Testing Rule Purity on Training Set Ground Truth...", flush=True)
t0 = time.time()

# 1. Load train reference tables
con.execute("""
CREATE TEMP TABLE s1 AS
SELECT entity_id as s1_id, country, name_normalized as n1, address_normalized as a1,
       LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_number
FROM read_csv('normalized_data/train_source1_normalized.tsv', delim='\t', header=true);

CREATE TEMP TABLE s23 AS
SELECT entity_id as m_id, country, name_normalized as n2, address_normalized as a2,
       LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_number
FROM (
    SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/train_source2_normalized.tsv', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/train_source3_normalized.tsv', delim='\t', header=true)
);

CREATE TEMP TABLE gt AS
SELECT source1_entity_id as s1_id, matched_entity_id as m_id
FROM read_csv('ground_truth_pairs.tsv', delim='\t', header=true);
""")
print(f"Loaded train tables in {time.time()-t0:.1f}s.", flush=True)

# Test Rule 1: Exact Name + Same Country (where name is unique per S1)
print("\n--- Rule 1: Exact Name (len >= 6) ---", flush=True)
t1 = time.time()
r1 = con.execute("""
WITH pairs AS (
    SELECT s1.s1_id, s23.m_id
    FROM s1 JOIN s23 ON s1.country = s23.country AND s1.n1 = s23.n2
    WHERE LENGTH(s1.n1) >= 6
    QUALIFY COUNT(*) OVER (PARTITION BY s1.n1, s1.country) <= 5
)
SELECT COUNT(*) as total_preds,
       SUM(CASE WHEN g.m_id IS NOT NULL THEN 1 ELSE 0 END) as true_positives,
       SUM(CASE WHEN g.m_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as precision
FROM pairs p
LEFT JOIN gt g ON p.s1_id = g.s1_id AND p.m_id = g.m_id;
""").fetchall()[0]
print(f"  Rule 1: {r1[0]:,} pairs | {r1[1]:,} TP | Precision = {r1[2]:.2f}% in {time.time()-t1:.1f}s", flush=True)

# Test Rule 2: Exact Address (len >= 12) + Same Country
print("\n--- Rule 2: Exact Address (len >= 12) ---", flush=True)
t2 = time.time()
r2 = con.execute("""
WITH pairs AS (
    SELECT s1.s1_id, s23.m_id
    FROM s1 JOIN s23 ON s1.country = s23.country AND s1.a1 = s23.a2
    WHERE LENGTH(s1.a1) >= 12
    QUALIFY COUNT(*) OVER (PARTITION BY s1.a1, s1.country) <= 5
)
SELECT COUNT(*) as total_preds,
       SUM(CASE WHEN g.m_id IS NOT NULL THEN 1 ELSE 0 END) as true_positives,
       SUM(CASE WHEN g.m_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as precision
FROM pairs p
LEFT JOIN gt g ON p.s1_id = g.s1_id AND p.m_id = g.m_id;
""").fetchall()[0]
print(f"  Rule 2: {r2[0]:,} pairs | {r2[1]:,} TP | Precision = {r2[2]:.2f}% in {time.time()-t2:.1f}s", flush=True)

# Test Rule 3: Name Prefix 6 + Addr Prefix 6 (len >= 6)
print("\n--- Rule 3: Name Prefix 6 + Addr Prefix 6 ---", flush=True)
t3 = time.time()
r3 = con.execute("""
WITH pairs AS (
    SELECT s1.s1_id, s23.m_id
    FROM s1 JOIN s23 ON s1.country = s23.country 
                    AND SUBSTRING(s1.n1, 1, 6) = SUBSTRING(s23.n2, 1, 6)
                    AND SUBSTRING(s1.a1, 1, 6) = SUBSTRING(s23.a2, 1, 6)
    WHERE LENGTH(s1.n1) >= 6 AND LENGTH(s1.a1) >= 6
    QUALIFY COUNT(*) OVER (PARTITION BY s1.s1_id) <= 5
)
SELECT COUNT(*) as total_preds,
       SUM(CASE WHEN g.m_id IS NOT NULL THEN 1 ELSE 0 END) as true_positives,
       SUM(CASE WHEN g.m_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as precision
FROM pairs p
LEFT JOIN gt g ON p.s1_id = g.s1_id AND p.m_id = g.m_id;
""").fetchall()[0]
print(f"  Rule 3: {r3[0]:,} pairs | {r3[1]:,} TP | Precision = {r3[2]:.2f}% in {time.time()-t3:.1f}s", flush=True)
