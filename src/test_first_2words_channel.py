import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import os, duckdb, polars as pl

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

con = duckdb.connect()
s1 = pl.read_parquet(os.path.join(VAL_DIR, "val_s1_v13_clean.parquet")).to_pandas()
s23 = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet")).to_pandas()
gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet")).to_pandas()
v16 = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet")).to_pandas()

con.register("s1_src", s1)
con.register("s23_src", s23)
con.register("gt_src", gt)
con.register("v16_src", v16)

con.execute("""
CREATE TEMP TABLE missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt_src g
LEFT JOIN v16_src c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")
missed_count = con.execute("SELECT COUNT(*) FROM missed").fetchone()[0]
print(f"Total missed GT by V16: {missed_count:,}")

STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"

# Channel 1: First 2 Non-stop Words
con.execute(f"""
CREATE TEMP TABLE s1_2w AS
WITH b AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(clean_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 2 AND x NOT IN {STOP_WORDS}) as w
    FROM s1_src
)
SELECT source1_entity_id, country, w[1] || '_' || w[2] as k2
FROM b WHERE LEN(w) >= 2;

CREATE TEMP TABLE s23_2w AS
WITH b AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(clean_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 2 AND x NOT IN {STOP_WORDS}) as w
    FROM s23_src
)
SELECT matched_entity_id, country, w[1] || '_' || w[2] as k2
FROM b WHERE LEN(w) >= 2;

CREATE TEMP TABLE ch_2w AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_2w s1
    JOIN s23_2w s23 ON s1.country = s23.country AND s1.k2 = s23.k2
) sub WHERE cnt <= 20;
""")
rec_2w = con.execute("SELECT COUNT(*) FROM missed m JOIN ch_2w c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
total_2w = con.execute("SELECT COUNT(*) FROM ch_2w").fetchone()[0]
print(f"[Channel: First 2 Non-Stop Words] Recovered {rec_2w:,} missed GT pairs! Total candidate pairs: {total_2w:,}")

# Channel 2: Sorted 2 Longest Words in Clean Name (Order Invariant Name Match)
con.execute(f"""
CREATE TEMP TABLE s1_sort2w AS
WITH b AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(clean_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) as w
    FROM s1_src
),
top2 AS (
    SELECT source1_entity_id, country,
           CASE 
               WHEN LEN(w) >= 2 THEN 
                   CASE WHEN w[1] < w[2] THEN w[1] || '_' || w[2] ELSE w[2] || '_' || w[1] END
               ELSE ''
           END as k_sort2
    FROM b
)
SELECT * FROM top2 WHERE k_sort2 != '';

CREATE TEMP TABLE s23_sort2w AS
WITH b AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(clean_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) as w
    FROM s23_src
),
top2 AS (
    SELECT matched_entity_id, country,
           CASE 
               WHEN LEN(w) >= 2 THEN 
                   CASE WHEN w[1] < w[2] THEN w[1] || '_' || w[2] ELSE w[2] || '_' || w[1] END
               ELSE ''
           END as k_sort2
    FROM b
)
SELECT * FROM top2 WHERE k_sort2 != '';

CREATE TEMP TABLE ch_sort2w AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_sort2w s1
    JOIN s23_sort2w s23 ON s1.country = s23.country AND s1.k_sort2 = s23.k_sort2
) sub WHERE cnt <= 20;
""")
rec_sort2w = con.execute("SELECT COUNT(*) FROM missed m JOIN ch_sort2w c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
total_sort2w = con.execute("SELECT COUNT(*) FROM ch_sort2w").fetchone()[0]
print(f"[Channel: Sorted 2 Words] Recovered {rec_sort2w:,} missed GT pairs! Total candidate pairs: {total_sort2w:,}")

# Combined
con.execute("""
CREATE TEMP TABLE combined_new AS
SELECT * FROM ch_2w
UNION
SELECT * FROM ch_sort2w;
""")
rec_comb = con.execute("SELECT COUNT(*) FROM missed m JOIN combined_new c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
total_comb = con.execute("SELECT COUNT(*) FROM combined_new").fetchone()[0]
print(f"\n[COMBINED NEW NAME CHANNELS] Recovered {rec_comb:,} missed GT pairs! Total candidate pairs: {total_comb:,}")
print(f"Total Blocking Recall would increase from 83,484 to {83484 + rec_comb:,} / 86,275 ({(83484 + rec_comb)/86275:.2%})!")
