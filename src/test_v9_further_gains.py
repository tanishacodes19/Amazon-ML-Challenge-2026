import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import duckdb, os
import polars as pl
from eval_framework import load_benchmark

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

s1, gt, _, s23 = load_benchmark()
v9_path = os.path.join(VAL_DIR, "val_v9_cands.parquet").replace("\\", "/")

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

con.register("s1", s1.to_pandas())
con.register("gt", gt.to_pandas())
con.register("s23", s23.to_pandas())

total_gt = len(gt)

con.execute(f"""
CREATE OR REPLACE TEMP TABLE v9_cands AS
SELECT source1_entity_id, matched_entity_id FROM read_parquet('{v9_path}');
""")
v9_rec = con.execute("""
SELECT COUNT(*) FROM gt g
JOIN v9_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

con.execute("""
CREATE OR REPLACE TEMP TABLE v9_missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN (SELECT source1_entity_id, matched_entity_id FROM v9_cands) c
  ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")

missed_cnt = con.execute("SELECT COUNT(*) FROM v9_missed").fetchone()[0]
print(f"Current V9 Benchmark: {v9_rec:,} / {total_gt:,} ({v9_rec/total_gt*100:.2f}%) | Remaining Missed: {missed_cnt:,}")

# Test A: Address Landmark Tokens length >= 6 (freq between 2 and 25)
print("\n--- Testing Channel A: Expanded Landmark Tokens (len >= 6, freq <= 25) ---")
ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s23_rare_tokens_a AS
SELECT token
FROM (
    SELECT UNNEST(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) AS token
    FROM s23
)
WHERE LENGTH(token) >= 6 AND token NOT IN {ADDR_STOP}
GROUP BY token
HAVING COUNT(*) BETWEEN 2 AND 25;

CREATE OR REPLACE TEMP TABLE s1_rare_addr_a AS
SELECT s1.source1_entity_id, s1.country, t.token
FROM (
    SELECT source1_entity_id, country,
           UNNEST(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) AS token
    FROM s1
) s1
JOIN s23_rare_tokens_a t ON s1.token = t.token;

CREATE OR REPLACE TEMP TABLE s23_rare_addr_a AS
SELECT s23.matched_entity_id, s23.country, t.token
FROM (
    SELECT matched_entity_id, country,
           UNNEST(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) AS token
    FROM s23
) s23
JOIN s23_rare_tokens_a t ON s23.token = t.token;

CREATE OR REPLACE TEMP TABLE ch_a AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_rare_addr_a s1
    JOIN s23_rare_addr_a s23 ON s1.country = s23.country AND s1.token = s23.token
) WHERE cnt <= 12;
""")
ch_a_cnt = con.execute("SELECT COUNT(*) FROM ch_a").fetchone()[0]
ch_a_rec = con.execute("SELECT COUNT(*) FROM v9_missed m JOIN ch_a c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  Channel A: {ch_a_cnt:,} candidates -> Recovered {ch_a_rec:,} missed matches ({ch_a_rec/max(1,ch_a_cnt)*100:.2f}% eff)")

# Test B: First 4 characters of sorted tokens (Handles minor typos in transposed names!)
print("\n--- Testing Channel B: Sorted First 4-prefix Tokens ---")
STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"
con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_sort_p4 AS
WITH base AS (
    SELECT source1_entity_id, country,
           LIST_SORT(LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS})) AS words
    FROM s1
)
SELECT source1_entity_id, country,
       CASE WHEN LEN(words) >= 2 THEN SUBSTR(words[1], 1, 4) || '_' || SUBSTR(words[2], 1, 4) ELSE '' END AS p4_key
FROM base WHERE LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE s23_sort_p4 AS
WITH base AS (
    SELECT matched_entity_id, country,
           LIST_SORT(LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS})) AS words
    FROM s23
)
SELECT matched_entity_id, country,
       CASE WHEN LEN(words) >= 2 THEN SUBSTR(words[1], 1, 4) || '_' || SUBSTR(words[2], 1, 4) ELSE '' END AS p4_key
FROM base WHERE LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE ch_b AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_sort_p4 s1
    JOIN s23_sort_p4 s23 ON s1.country = s23.country AND s1.p4_key = s23.p4_key
    WHERE s1.p4_key != ''
) WHERE cnt <= 12;
""")
ch_b_cnt = con.execute("SELECT COUNT(*) FROM ch_b").fetchone()[0]
ch_b_rec = con.execute("SELECT COUNT(*) FROM v9_missed m JOIN ch_b c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  Channel B: {ch_b_cnt:,} candidates -> Recovered {ch_b_rec:,} missed matches ({ch_b_rec/max(1,ch_b_cnt)*100:.2f}% eff)")
