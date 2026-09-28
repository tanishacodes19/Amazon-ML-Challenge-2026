import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
sys.path.insert(0, 'src')
import os
import duckdb
import polars as pl
from eval_framework import load_benchmark

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

s1, gt, _, s23 = load_benchmark()
v13_cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v13_cands.parquet"))

s1_clean_path = os.path.join(VAL_DIR, "val_s1_v13_clean.parquet")
s23_clean_path = os.path.join(VAL_DIR, "val_s23_v13_clean.parquet")
s1_df = pl.read_parquet(s1_clean_path).to_pandas()
s23_df = pl.read_parquet(s23_clean_path).to_pandas()

con = duckdb.connect()
con.register("s1", s1_df)
con.register("gt", gt.to_pandas())
con.register("s23", s23_df)
con.register("v13_cands", v13_cands.to_pandas())

total_gt = len(gt)

# Missed pairs
con.execute("""
CREATE OR REPLACE TEMP TABLE v13_missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN v13_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")

# Channel 9: Squashed Brand
LEGAL_WORDS = "('private', 'limited', 'incorporated', 'corporation', 'company', 'llc', 'llp', 'services')"
con.execute(f"""
CREATE OR REPLACE TEMP TABLE ch9_squash AS
WITH s1_sq AS (
    SELECT source1_entity_id, country,
           ARRAY_TO_STRING(LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> x NOT IN {LEGAL_WORDS} AND LENGTH(x) >= 3), '') as k
    FROM s1
),
s23_sq AS (
    SELECT matched_entity_id, country,
           ARRAY_TO_STRING(LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> x NOT IN {LEGAL_WORDS} AND LENGTH(x) >= 3), '') as k
    FROM s23
)
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_sq s1 JOIN s23_sq s23 ON s1.country = s23.country AND s1.k = s23.k
    WHERE LENGTH(s1.k) >= 6
) WHERE cnt <= 15;
""")

# Channel 11: Distinctive Address Bigram
ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"
con.execute(f"""
CREATE OR REPLACE TEMP TABLE ch11_addr_bi AS
WITH s1_b AS (
    SELECT source1_entity_id, country,
           words[1] || '_' || words[2] as bi
    FROM (SELECT source1_entity_id, country, LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*') as words FROM s1)
    WHERE LEN(words) >= 2 AND LENGTH(words[1] || '_' || words[2]) >= 10
),
s23_b AS (
    SELECT matched_entity_id, country,
           words[1] || '_' || words[2] as bi
    FROM (SELECT matched_entity_id, country, LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*') as words FROM s23)
    WHERE LEN(words) >= 2 AND LENGTH(words[1] || '_' || words[2]) >= 10
)
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_b s1 JOIN s23_b s23 ON s1.country = s23.country AND s1.bi = s23.bi
) WHERE cnt <= 12;
""")

# Channel 12: Leet-speak
con.execute("""
CREATE OR REPLACE TEMP TABLE ch12_leet AS
WITH s1_t AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(REPLACE(REPLACE(REPLACE(clean_name, '0', 'o'), '5', 's'), '1', 'l'), ' '), x -> LENGTH(x) >= 4) as words
    FROM s1
),
s23_t AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(REPLACE(REPLACE(REPLACE(clean_name, '0', 'o'), '5', 's'), '1', 'l'), ' '), x -> LENGTH(x) >= 4) as words
    FROM s23
)
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_t s1 JOIN s23_t s23 ON s1.country = s23.country AND s1.words[1] = s23.words[1]
    WHERE LEN(s1.words) >= 1 AND LEN(s23.words) >= 1
) WHERE cnt <= 15;
""")

# Channel 13: 2-char Initial + Numeric Anchor
con.execute("""
CREATE OR REPLACE TEMP TABLE ch13_init_loc AS
WITH s1_i AS (
    SELECT source1_entity_id, country,
           SUBSTR(clean_name, 1, 2) AS init2,
           REGEXP_EXTRACT(clean_addr, '(\\b\\d{3,6}\\b)', 1) AS num_anchor
    FROM s1 WHERE LENGTH(clean_name) >= 2 AND LENGTH(REGEXP_EXTRACT(clean_addr, '(\\b\\d{3,6}\\b)', 1)) >= 3
),
s23_i AS (
    SELECT matched_entity_id, country,
           SUBSTR(clean_name, 1, 2) AS init2,
           REGEXP_EXTRACT(clean_addr, '(\\b\\d{3,6}\\b)', 1) AS num_anchor
    FROM s23 WHERE LENGTH(clean_name) >= 2 AND LENGTH(REGEXP_EXTRACT(clean_addr, '(\\b\\d{3,6}\\b)', 1)) >= 3
)
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_i s1 JOIN s23_i s23 
      ON s1.country = s23.country AND s1.init2 = s23.init2 AND s1.num_anchor = s23.num_anchor
) WHERE cnt <= 15;
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE v14_all_cands AS
SELECT source1_entity_id, matched_entity_id FROM v13_cands
UNION SELECT * FROM ch9_squash
UNION SELECT * FROM ch11_addr_bi
UNION SELECT * FROM ch12_leet
UNION SELECT * FROM ch13_init_loc;
""")

v14_cnt = con.execute("SELECT COUNT(*) FROM v14_all_cands").fetchone()[0]
v14_rec = con.execute("""
SELECT COUNT(*) FROM gt g
JOIN v14_all_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

print("=" * 70)
print(f"V14 TOTAL CANDIDATES:     {v14_cnt:,} ({v14_cnt/25000:.2f} cands/S1)")
print(f"V14 RECOVERED GT:         {v14_rec:,} / {total_gt:,} ({v14_rec/total_gt*100:.2f}%)")
print(f"NET GT RECOVERED OVER V13: +{v14_rec - 81054:,} pairs")
print(f"REMAINING MISSED:         {total_gt - v14_rec:,}")
print("=" * 70)

# Save V14 candidates
out_path = os.path.join(VAL_DIR, "val_v14_cands.parquet")
con.execute("SELECT source1_entity_id, matched_entity_id FROM v14_all_cands").df().to_parquet(out_path, index=False)
print(f"Saved V14 Candidates to: {out_path}")
