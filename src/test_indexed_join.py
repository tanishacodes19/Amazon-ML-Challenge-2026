import sys
import duckdb
import time

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

S1_NORM = 'C:/Users/Admin/Documents/Amazon-ML-Challenge-2026/normalized_data/test_source1_normalized.tsv'
S2_NORM = 'C:/Users/Admin/Documents/Amazon-ML-Challenge-2026/normalized_data/test_source2_normalized.tsv'
S3_NORM = 'C:/Users/Admin/Documents/Amazon-ML-Challenge-2026/normalized_data/test_source3_normalized.tsv'
CANDS = 'C:/Users/Admin/Documents/Amazon-ML-Challenge-2026/test_scored_candidates.tsv'

t_start = time.time()
print("1. Creating s1_tbl in memory...")
con.execute(f"CREATE TEMP TABLE s1_tbl AS SELECT entity_id, business_name, business_address, country FROM read_csv('{S1_NORM}', delim='\\t', header=true)")

print(f"Loaded s1_tbl in {time.time()-t_start:.1f}s. Creating s23_tbl...")
t_s23 = time.time()
con.execute(f"""
CREATE TEMP TABLE s23_tbl AS 
SELECT entity_id, business_name, business_address, country FROM read_csv('{S2_NORM}', delim='\\t', header=true)
UNION ALL
SELECT entity_id, business_name, business_address, country FROM read_csv('{S3_NORM}', delim='\\t', header=true)
""")
print(f"Loaded s23_tbl in {time.time()-t_s23:.1f}s.")

print("2. Joining 100,000 candidates against in-memory tables...")
t_join = time.time()
res = con.execute(f"""
SELECT p.source1_entity_id, p.matched_entity_id,
       COALESCE(s1.business_name,'') as s1_name_raw,
       COALESCE(s1.business_address,'') as s1_addr_raw,
       COALESCE(s1.country,'') as s1_country,
       COALESCE(s23.business_name,'') as m_name_raw,
       COALESCE(s23.business_address,'') as m_addr_raw,
       COALESCE(s23.country,'') as m_country
FROM (SELECT source1_entity_id, matched_entity_id FROM read_csv('{CANDS}', delim='\\t', header=true) LIMIT 100000) p
JOIN s1_tbl s1 ON p.source1_entity_id = s1.entity_id
JOIN s23_tbl s23 ON p.matched_entity_id = s23.entity_id
""")
df = res.df()
print(f"Retrieved {len(df):,} joined rows in {time.time()-t_join:.2f}s!")
