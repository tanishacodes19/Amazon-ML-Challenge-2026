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
v14_cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v14_cands.parquet"))

s1_clean_path = os.path.join(VAL_DIR, "val_s1_v13_clean.parquet")
s23_clean_path = os.path.join(VAL_DIR, "val_s23_v13_clean.parquet")
s1_df = pl.read_parquet(s1_clean_path).to_pandas()
s23_df = pl.read_parquet(s23_clean_path).to_pandas()

con = duckdb.connect()
con.register("s1", s1_df)
con.register("gt", gt.to_pandas())
con.register("s23", s23_df)
con.register("v14_cands", v14_cands.to_pandas())

# Missed pairs
con.execute("""
CREATE OR REPLACE TEMP TABLE v14_missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN v14_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")

print("Testing Channel 15: Distinctive Street Name Tri-gram (3 consecutive words >= 4 chars)...")
ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_tri AS
WITH parsed AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*') as words
    FROM s1
)
SELECT source1_entity_id, country,
       words[1] || '_' || words[2] || '_' || words[3] as tri
FROM parsed WHERE LEN(words) >= 3 AND LENGTH(words[1] || '_' || words[2] || '_' || words[3]) >= 14;

CREATE OR REPLACE TEMP TABLE s23_tri AS
WITH parsed AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*') as words
    FROM s23
)
SELECT matched_entity_id, country,
       words[1] || '_' || words[2] || '_' || words[3] as tri
FROM parsed WHERE LEN(words) >= 3 AND LENGTH(words[1] || '_' || words[2] || '_' || words[3]) >= 14;

CREATE OR REPLACE TEMP TABLE ch15_tri AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_tri s1
    JOIN s23_tri s23 ON s1.country = s23.country AND s1.tri = s23.tri
) WHERE cnt <= 12;
""")

ch15_cnt = con.execute("SELECT COUNT(*) FROM ch15_tri").fetchone()[0]
ch15_rec = con.execute("SELECT COUNT(*) FROM v14_missed m JOIN ch15_tri c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"Ch15: {ch15_cnt:,} pairs | Recovered missed GT: {ch15_rec:,}")

print("\nTesting Channel 16: Door / Unit / Plot Alpha-numeric Normalization (e.g. H0091 -> H91)...")
con.execute("""
CREATE OR REPLACE TEMP TABLE s1_alpha_plot AS
SELECT source1_entity_id, country,
       REGEXP_EXTRACT(clean_addr, '(?i)(?:plot|door|flat|unit|h\\.?no\\.?|q\\.?no\\.?)\\s*(?:no\\.?)?\\s*([a-z]?[0-9]+)', 1) as plot_raw
FROM s1
WHERE REGEXP_EXTRACT(clean_addr, '(?i)(?:plot|door|flat|unit|h\\.?no\\.?|q\\.?no\\.?)\\s*(?:no\\.?)?\\s*([a-z]?[0-9]+)', 1) IS NOT NULL;

CREATE OR REPLACE TEMP TABLE s23_alpha_plot AS
SELECT matched_entity_id, country,
       REGEXP_EXTRACT(clean_addr, '(?i)(?:plot|door|flat|unit|h\\.?no\\.?|q\\.?no\\.?)\\s*(?:no\\.?)?\\s*([a-z]?[0-9]+)', 1) as plot_raw
FROM s23
WHERE REGEXP_EXTRACT(clean_addr, '(?i)(?:plot|door|flat|unit|h\\.?no\\.?|q\\.?no\\.?)\\s*(?:no\\.?)?\\s*([a-z]?[0-9]+)', 1) IS NOT NULL;

CREATE OR REPLACE TEMP TABLE ch16_plot AS
WITH s1_p AS (
    SELECT source1_entity_id, country,
           REGEXP_REPLACE(plot_raw, '0*([1-9][0-9]*)', '\\1') as plot_norm
    FROM s1_alpha_plot WHERE LENGTH(plot_raw) >= 2
),
s23_p AS (
    SELECT matched_entity_id, country,
           REGEXP_REPLACE(plot_raw, '0*([1-9][0-9]*)', '\\1') as plot_norm
    FROM s23_alpha_plot WHERE LENGTH(plot_raw) >= 2
),
s1_w AS (
    SELECT source1_entity_id,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT SIMILAR TO '[0-9].*') as words
    FROM s1
),
s23_w AS (
    SELECT matched_entity_id,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT SIMILAR TO '[0-9].*') as words
    FROM s23
)
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT p1.source1_entity_id, p2.matched_entity_id,
           COUNT(*) OVER (PARTITION BY p1.source1_entity_id) as cnt
    FROM s1_p p1
    JOIN s23_p p2 ON p1.country = p2.country AND p1.plot_norm = p2.plot_norm
    JOIN s1_w w1 ON p1.source1_entity_id = w1.source1_entity_id
    JOIN s23_w w2 ON p2.matched_entity_id = w2.matched_entity_id
    WHERE LEN(w1.words) >= 1 AND LEN(w2.words) >= 1 AND w1.words[1] = w2.words[1]
) WHERE cnt <= 12;
""")

ch16_cnt = con.execute("SELECT COUNT(*) FROM ch16_plot").fetchone()[0]
ch16_rec = con.execute("SELECT COUNT(*) FROM v14_missed m JOIN ch16_plot c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"Ch16: {ch16_cnt:,} pairs | Recovered missed GT: {ch16_rec:,}")
