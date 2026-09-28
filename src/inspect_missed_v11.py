import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import polars as pl
import duckdb

con = duckdb.connect()
con.execute("""
CREATE TEMP TABLE gt AS SELECT * FROM read_parquet('validation_benchmark/val_gt.parquet');
CREATE TEMP TABLE cands AS SELECT source1_entity_id, matched_entity_id FROM read_parquet('validation_benchmark/val_v11_cands.parquet');
CREATE TEMP TABLE s1 AS SELECT * FROM read_parquet('validation_benchmark/val_s1.parquet');
CREATE TEMP TABLE s23 AS SELECT * FROM read_parquet('validation_benchmark/val_s23.parquet');

CREATE TEMP TABLE missed AS
SELECT g.source1_entity_id, g.matched_entity_id,
       s1.business_name as s1_name, s1.business_address as s1_addr, s1.country as s1_country,
       s23.business_name as m_name, s23.business_address as m_addr, s23.country as m_country
FROM gt g
LEFT JOIN cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
JOIN s1 ON g.source1_entity_id = s1.source1_entity_id
JOIN s23 ON g.matched_entity_id = s23.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")

cnt = con.execute("SELECT COUNT(*) FROM missed").fetchone()[0]
print(f"Total missed pairs in V11: {cnt:,}")

sample = con.execute("SELECT s1_name, m_name, s1_addr, m_addr, s1_country FROM missed LIMIT 15").df()
for _, r in sample.iterrows():
    print(f"S1: [{r['s1_name']}] | [{r['s1_addr']}]")
    print(f"M:  [{r['m_name']}] | [{r['m_addr']}]")
    print(f"Country: {r['s1_country']}")
    print("-" * 60)
