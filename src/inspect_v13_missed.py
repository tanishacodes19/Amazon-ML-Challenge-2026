import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
sys.path.insert(0, 'src')
import os
import duckdb
import polars as pl
from eval_framework import load_benchmark
from normalizer import normalize_business_name, normalize_address, extract_structured_fields

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

s1, gt, _, s23 = load_benchmark()
v13_cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v13_cands.parquet"))

con = duckdb.connect()
con.register("s1", s1.to_pandas())
con.register("gt", gt.to_pandas())
con.register("s23", s23.to_pandas())
con.register("cands", v13_cands.to_pandas())

# Identify the 5,221 missed pairs
con.execute("""
CREATE OR REPLACE TEMP TABLE missed AS
SELECT g.source1_entity_id, g.matched_entity_id,
       s1.business_name AS s1_name, s1.business_address AS s1_addr, s1.country AS s1_country,
       s23.business_name AS m_name, s23.business_address AS m_addr, s23.country AS m_country
FROM gt g
LEFT JOIN cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
JOIN s1 ON g.source1_entity_id = s1.source1_entity_id
JOIN s23 ON g.matched_entity_id = s23.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")

missed_cnt = con.execute("SELECT COUNT(*) FROM missed").fetchone()[0]
print(f"Total Missed Ground Truth in V13: {missed_cnt:,} / {len(gt):,} ({missed_cnt/len(gt)*100:.2f}%)")

# Country breakdown
countries = con.execute("SELECT s1_country, COUNT(*) as cnt FROM missed GROUP BY s1_country").fetchall()
print("\nMissed by Country:")
for c, cnt in countries:
    print(f"  {c}: {cnt:,} ({cnt/missed_cnt*100:.1f}%)")

# Inspect sample missed pairs
sample = con.execute("SELECT s1_name, m_name, s1_addr, m_addr, s1_country FROM missed LIMIT 20").fetchall()
print("\nSample 20 Missed Pairs in V13:")
for i, (n1, n2, a1, a2, country) in enumerate(sample, 1):
    print(f"\n--- Missed #{i} ({country}) ---")
    print(f"S1 Name:  {n1}")
    print(f"S23 Name: {n2}")
    print(f"S1 Addr:  {a1}")
    print(f"S23 Addr: {a2}")
