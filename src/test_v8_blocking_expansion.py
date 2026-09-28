import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import duckdb
import polars as pl
from eval_framework import load_benchmark

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

print("=" * 70)
print("TESTING V8 CANDIDATE EXPANSION CHANNELS ON TOP OF V7")
print("=" * 70)

s1, gt, _, s23 = load_benchmark()
v7_path = os.path.join(VAL_DIR, "val_v7_cands.parquet").replace("\\", "/")

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='2500MB'")

con.register("s1", s1.to_pandas())
con.register("gt", gt.to_pandas())
con.register("s23", s23.to_pandas())

# Identify ground truth pairs missed by V7
con.execute(f"""
CREATE OR REPLACE TEMP TABLE v7_recovered AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
JOIN read_parquet('{v7_path}') c 
  ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""")

v7_rec_cnt = con.execute("SELECT COUNT(*) FROM v7_recovered").fetchone()[0]
total_gt = len(gt)
print(f"Total Ground Truth:      {total_gt:,}")
print(f"V7 Recovered Matches:    {v7_rec_cnt:,} ({v7_rec_cnt/total_gt*100:.2f}%)")
print(f"Remaining Missed in V7:  {total_gt - v7_rec_cnt:,}")

# Prepare cleaned fields
print("\nPreparing clean fields...")
con.execute("""
CREATE OR REPLACE TEMP TABLE s1_prep AS
SELECT 
    source1_entity_id,
    country,
    LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))) AS name_squashed,
    LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', '', 'g'))) AS addr_squashed,
    REGEXP_EXTRACT(business_address, '(\\b\\d{4,6}\\b)', 1) AS postal,
    REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1) AS house,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', '', 'g'))), 1, 10) AS addr_clean10,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', '', 'g'))), 1, 14) AS addr_clean14
FROM s1
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE s23_prep AS
SELECT 
    matched_entity_id,
    country,
    LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))) AS name_squashed,
    LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', '', 'g'))) AS addr_squashed,
    REGEXP_EXTRACT(business_address, '(\\b\\d{4,6}\\b)', 1) AS postal,
    REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1) AS house,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', '', 'g'))), 1, 10) AS addr_clean10,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', '', 'g'))), 1, 14) AS addr_clean14
FROM s23
""")

# Test Candidates
tests = [
    # 1. Address Clean Prefix 10 chars (country matched, capped at 15)
    ("addr_clean10_country", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_prep s1
        JOIN s23_prep s23 ON s1.country = s23.country AND s1.addr_clean10 = s23.addr_clean10
        WHERE LENGTH(s1.addr_clean10) >= 10
    """, 15),

    # 2. Address Clean Prefix 14 chars (any country, capped at 15)
    ("addr_clean14_global", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_prep s1
        JOIN s23_prep s23 ON s1.addr_clean14 = s23.addr_clean14
        WHERE LENGTH(s1.addr_clean14) >= 14
    """, 15),

    # 3. Squashed Name Prefix 5 + Postal match
    ("name_p5_postal", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_prep s1
        JOIN s23_prep s23 ON s1.postal = s23.postal AND SUBSTR(s1.name_squashed, 1, 5) = SUBSTR(s23.name_squashed, 1, 5)
        WHERE s1.postal != '' AND LENGTH(s1.name_squashed) >= 5
    """, 15),

    # 4. Squashed Name Prefix 5 + House number match
    ("name_p5_house", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_prep s1
        JOIN s23_prep s23 ON s1.house = s23.house AND SUBSTR(s1.name_squashed, 1, 5) = SUBSTR(s23.name_squashed, 1, 5)
        WHERE s1.house != '' AND LENGTH(s1.name_squashed) >= 5
    """, 15),
]

con.execute("""
CREATE OR REPLACE TEMP TABLE v7_missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN v7_recovered r 
  ON g.source1_entity_id = r.source1_entity_id AND g.matched_entity_id = r.matched_entity_id
WHERE r.matched_entity_id IS NULL
""")

for name, sql, cap in tests:
    con.execute(f"CREATE OR REPLACE TEMP TABLE t_raw AS {sql}")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE t_capped AS
    SELECT source1_entity_id, matched_entity_id
    FROM (
        SELECT source1_entity_id, matched_entity_id,
               COUNT(*) OVER (PARTITION BY source1_entity_id) AS cnt
        FROM t_raw
    ) WHERE cnt <= {cap}
    """)
    cnt = con.execute("SELECT COUNT(*) FROM t_capped").fetchone()[0]
    rec = con.execute("""
    SELECT COUNT(*) FROM v7_missed m
    JOIN t_capped c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id
    """).fetchone()[0]
    eff = (rec / cnt * 100) if cnt > 0 else 0.0
    print(f"Channel: {name:<22} | Candidates: {cnt:>7,} | Recovered Missed: {rec:>5,} | Efficiency: {eff:.2f}%")
