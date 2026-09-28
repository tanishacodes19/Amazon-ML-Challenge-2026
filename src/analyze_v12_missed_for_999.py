import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
sys.path.insert(0, 'src')
import duckdb
import polars as pl
import re

con = duckdb.connect()
con.execute("""
CREATE TEMP TABLE gt AS SELECT * FROM read_parquet('validation_benchmark/val_gt.parquet');
CREATE TEMP TABLE cands AS SELECT source1_entity_id, matched_entity_id FROM read_parquet('validation_benchmark/val_v12_cands.parquet');
CREATE TEMP TABLE s1 AS SELECT * FROM read_parquet('validation_benchmark/val_s1.parquet');
CREATE TEMP TABLE s23 AS SELECT * FROM read_parquet('validation_benchmark/val_s23.parquet');

CREATE TEMP TABLE missed_v12 AS
SELECT g.source1_entity_id, g.matched_entity_id,
       s1.business_name as s1_name, s1.business_address as s1_addr, s1.country as s1_country,
       s23.business_name as m_name, s23.business_address as m_addr, s23.country as m_country
FROM gt g
LEFT JOIN cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
JOIN s1 ON g.source1_entity_id = s1.source1_entity_id
JOIN s23 ON g.matched_entity_id = s23.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")

cnt = con.execute("SELECT COUNT(*) FROM missed_v12").fetchone()[0]
total_gt = con.execute("SELECT COUNT(*) FROM gt").fetchone()[0]
print(f"Total True Pairs in Validation: {total_gt:,}")
print(f"Remaining Missed in V12: {cnt:,} ({cnt/total_gt*100:.2f}% of all matches)")

# Breakdown by country
print("\nMissed pairs by country:")
print(con.execute("SELECT s1_country, COUNT(*) as cnt, COUNT(*)*100.0/" + str(cnt) + " as pct FROM missed_v12 GROUP BY s1_country ORDER BY cnt DESC").df())

# Check Indic / non-ASCII script in missed
df = con.execute("SELECT s1_name, m_name, s1_addr, m_addr, s1_country FROM missed_v12").df()

indic_count = 0
for _, r in df.iterrows():
    # Check if Devanagari or Bengali or Tamil or other non-ascii
    has_indic = bool(re.search(r'[\u0900-\u0D7F]', str(r['s1_name']) + str(r['m_name'])))
    if has_indic:
        indic_count += 1

print(f"\nMissed pairs involving Indic non-Latin scripts: {indic_count:,} / {cnt:,} ({indic_count/cnt*100:.1f}%)")

# Check sample of missed
print("\nSample 10 missed pairs:")
sample = df.head(10)
for idx, r in sample.iterrows():
    print(f"[{idx+1}] S1: {r['s1_name']} | {r['s1_addr']}")
    print(f"    M:  {r['m_name']} | {r['m_addr']}")
    print(f"    Country: {r['s1_country']}")
    print("-" * 70)
