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

con.execute(f"CREATE VIEW s1_data AS SELECT entity_id, business_name, business_address, country FROM read_csv('{S1_NORM}', delim='\\t', header=true)")
con.execute(f"""
CREATE VIEW s23_data AS 
SELECT entity_id, business_name, business_address, country FROM read_csv('{S2_NORM}', delim='\\t', header=true)
UNION ALL
SELECT entity_id, business_name, business_address, country FROM read_csv('{S3_NORM}', delim='\\t', header=true)
""")

t0 = time.time()
print("Testing streaming query...")
res = con.execute(f"""
SELECT p.source1_entity_id, p.matched_entity_id,
       COALESCE(s1.business_name,'') as s1_name_raw,
       COALESCE(s1.business_address,'') as s1_addr_raw,
       COALESCE(s1.country,'') as s1_country,
       COALESCE(s23.business_name,'') as m_name_raw,
       COALESCE(s23.business_address,'') as m_addr_raw,
       COALESCE(s23.country,'') as m_country
FROM (SELECT source1_entity_id, matched_entity_id FROM read_csv('{CANDS}', delim='\\t', header=true) LIMIT 100000) p
JOIN s1_data s1 ON p.source1_entity_id = s1.entity_id
JOIN s23_data s23 ON p.matched_entity_id = s23.entity_id
""")

df = res.df()
print(f"Retrieved {len(df):,} joined rows in {time.time()-t0:.2f}s!")
