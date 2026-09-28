import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import duckdb, os
import polars as pl
from eval_framework import load_benchmark

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

s1, gt, _, s23 = load_benchmark()
v8_path = os.path.join(VAL_DIR, "val_v8_cands.parquet").replace("\\", "/")

con = duckdb.connect()
con.register("s1", s1.to_pandas())
con.register("gt", gt.to_pandas())
con.register("s23", s23.to_pandas())

con.execute(f"""
CREATE OR REPLACE TEMP TABLE v8_recovered AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
JOIN read_parquet('{v8_path}') c 
  ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE missed AS
SELECT g.source1_entity_id, g.matched_entity_id,
       s1.business_name AS s1_name, s1.business_address AS s1_addr, s1.country AS s1_country,
       s23.business_name AS m_name, s23.business_address AS m_addr, s23.country AS m_country
FROM gt g
JOIN s1 ON g.source1_entity_id = s1.source1_entity_id
JOIN s23 ON g.matched_entity_id = s23.matched_entity_id
LEFT JOIN v8_recovered r 
  ON g.source1_entity_id = r.source1_entity_id AND g.matched_entity_id = r.matched_entity_id
WHERE r.matched_entity_id IS NULL
""")

total_missed = con.execute("SELECT COUNT(*) FROM missed").fetchone()[0]
print(f"Total missed true pairs: {total_missed:,}")

print("\n--- SAMPLE OF 20 MISSED TRUE MATCHES ---")
samples = con.execute("SELECT s1_name, m_name, s1_addr, m_addr, s1_country, m_country FROM missed USING SAMPLE reservoir (20 ROWS) REPEATABLE (42)").fetchall()
for i, (n1, n2, a1, a2, c1, c2) in enumerate(samples, 1):
    print(f"[{i}] Country: {c1} vs {c2}")
    print(f"    S1 Name:    {n1}")
    print(f"    Match Name: {n2}")
    print(f"    S1 Addr:    {a1}")
    print(f"    Match Addr: {a2}")
    print("-" * 60)
