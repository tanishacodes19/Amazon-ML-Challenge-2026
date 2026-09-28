import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, time

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")

print("Analyzing S2/S3 coverage across the test set...", flush=True)
t0 = time.time()

con.execute("""
CREATE TEMP TABLE s1 AS
SELECT entity_id as s1_id, country, name_normalized as n1, address_normalized as a1
FROM read_csv('normalized_data/test_source1_normalized.tsv', delim='\t', header=true);

CREATE TEMP TABLE s23 AS
SELECT entity_id as m_id, country, name_normalized as n2, address_normalized as a2
FROM (
    SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/test_source2_normalized.tsv', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/test_source3_normalized.tsv', delim='\t', header=true)
);
""")
print(f"Loaded tables in {time.time()-t0:.1f}s.", flush=True)

# Check how many S23 entities share exact name with some S1 entity in same country
t1 = time.time()
n_exact_name = con.execute("""
SELECT COUNT(DISTINCT s23.m_id)
FROM s1 JOIN s23 ON s1.country = s23.country AND s1.n1 = s23.n2
WHERE LENGTH(s1.n1) >= 3;
""").fetchone()[0]
print(f"Exact Name Matches: {n_exact_name:,} distinct S2/S3 entities in {time.time()-t1:.1f}s", flush=True)

# Check how many S23 entities share exact address with some S1 entity in same country
t2 = time.time()
n_exact_addr = con.execute("""
SELECT COUNT(DISTINCT s23.m_id)
FROM s1 JOIN s23 ON s1.country = s23.country AND s1.a1 = s23.a2
WHERE LENGTH(s1.a1) >= 10;
""").fetchone()[0]
print(f"Exact Address Matches: {n_exact_addr:,} distinct S2/S3 entities in {time.time()-t2:.1f}s", flush=True)
