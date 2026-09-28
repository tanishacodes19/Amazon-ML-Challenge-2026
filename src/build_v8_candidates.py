import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import duckdb
import polars as pl
from eval_framework import load_benchmark

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

s1, gt, _, s23 = load_benchmark()
v7_path = os.path.join(VAL_DIR, "val_v7_cands.parquet").replace("\\", "/")

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3GB'")

con.register("s1", s1.to_pandas())
con.register("gt", gt.to_pandas())
con.register("s23", s23.to_pandas())

con.execute(f"""
CREATE OR REPLACE TEMP TABLE v7_cands AS
SELECT source1_entity_id, matched_entity_id FROM read_parquet('{v7_path}')
""")

# 1. addr_clean10
con.execute("""
CREATE OR REPLACE TEMP TABLE addr_ch AS
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM (
    SELECT source1_entity_id, country,
           SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', '', 'g'))), 1, 10) AS a10
    FROM s1
) s1
JOIN (
    SELECT matched_entity_id, country,
           SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', '', 'g'))), 1, 10) AS a10
    FROM s23
) s23 ON s1.country = s23.country AND s1.a10 = s23.a10
WHERE LENGTH(s1.a10) >= 10
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE addr_ch_capped AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT source1_entity_id, matched_entity_id,
           COUNT(*) OVER (PARTITION BY source1_entity_id) AS cnt
    FROM addr_ch
) WHERE cnt <= 15
""")

# 2. 4-grams
con.execute("""
CREATE OR REPLACE TEMP TABLE s1_ngrams AS
WITH base AS (
    SELECT source1_entity_id, country,
           LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))) AS name_squashed
    FROM s1
)
SELECT source1_entity_id, country, SUBSTR(name_squashed, 1, 4) AS g1, SUBSTR(name_squashed, 2, 4) AS g2, SUBSTR(name_squashed, 3, 4) AS g3
FROM base WHERE LENGTH(name_squashed) >= 5;

CREATE OR REPLACE TEMP TABLE s23_ngrams AS
WITH base AS (
    SELECT matched_entity_id, country,
           LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))) AS name_squashed
    FROM s23
)
SELECT matched_entity_id, country, SUBSTR(name_squashed, 1, 4) AS g1, SUBSTR(name_squashed, 2, 4) AS g2, SUBSTR(name_squashed, 3, 4) AS g3
FROM base WHERE LENGTH(name_squashed) >= 5;

CREATE OR REPLACE TEMP TABLE s1_inv AS
SELECT source1_entity_id, country, g1 AS gram FROM s1_ngrams WHERE LENGTH(g1) = 4
UNION ALL
SELECT source1_entity_id, country, g2 AS gram FROM s1_ngrams WHERE LENGTH(g2) = 4
UNION ALL
SELECT source1_entity_id, country, g3 AS gram FROM s1_ngrams WHERE LENGTH(g3) = 4;

CREATE OR REPLACE TEMP TABLE s23_inv AS
SELECT matched_entity_id, country, g1 AS gram FROM s23_ngrams WHERE LENGTH(g1) = 4
UNION ALL
SELECT matched_entity_id, country, g2 AS gram FROM s23_ngrams WHERE LENGTH(g2) = 4
UNION ALL
SELECT matched_entity_id, country, g3 AS gram FROM s23_ngrams WHERE LENGTH(g3) = 4;

CREATE OR REPLACE TEMP TABLE shared_4grams AS
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_inv s1
JOIN s23_inv s23 ON s1.country = s23.country AND s1.gram = s23.gram
GROUP BY s1.source1_entity_id, s23.matched_entity_id
HAVING COUNT(*) >= 2;

CREATE OR REPLACE TEMP TABLE ngrams_capped AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT source1_entity_id, matched_entity_id,
           COUNT(*) OVER (PARTITION BY source1_entity_id) AS cnt
    FROM shared_4grams
) WHERE cnt <= 12;
""")

# 3. Combine into V8 candidate set
con.execute("""
CREATE OR REPLACE TEMP TABLE v8_candidates AS
SELECT source1_entity_id, matched_entity_id FROM v7_cands
UNION
SELECT source1_entity_id, matched_entity_id FROM addr_ch_capped
UNION
SELECT source1_entity_id, matched_entity_id FROM ngrams_capped
""")

v8_total = con.execute("SELECT COUNT(*) FROM v8_candidates").fetchone()[0]
total_gt = len(gt)
v8_rec = con.execute("""
SELECT COUNT(*) FROM gt g
JOIN v8_candidates c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

v7_rec = con.execute("""
SELECT COUNT(*) FROM gt g
JOIN v7_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

print("=" * 70)
print(f"V7 Candidates:  {len(pl.read_parquet(v7_path)):,} | Matches: {v7_rec:,} ({v7_rec/total_gt*100:.2f}%)")
print(f"V8 Candidates:  {v8_total:,} | Matches: {v8_rec:,} ({v8_rec/total_gt*100:.2f}%)")
print(f"Net Gained:     +{v8_rec - v7_rec:,} true matches (+{(v8_rec - v7_rec)/total_gt*100:.2f}% recall)")
print(f"Avg per S1:     {v8_total / len(s1):.2f}")
print("=" * 70)

# Save to parquet
out_p = os.path.join(VAL_DIR, "val_v8_cands.parquet")
con.execute("SELECT source1_entity_id, matched_entity_id FROM v8_candidates").df().to_parquet(out_p, index=False)
print(f"Saved: {out_p}")
