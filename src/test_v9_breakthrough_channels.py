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
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3GB'")

con.register("s1", s1.to_pandas())
con.register("gt", gt.to_pandas())
con.register("s23", s23.to_pandas())

con.execute(f"""
CREATE OR REPLACE TEMP TABLE v8_cands AS
SELECT source1_entity_id, matched_entity_id FROM read_parquet('{v8_path}')
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE v8_recovered AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
JOIN v8_cands c 
  ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""")

v8_cnt = con.execute("SELECT COUNT(*) FROM v8_cands").fetchone()[0]
v8_rec = con.execute("SELECT COUNT(*) FROM v8_recovered").fetchone()[0]
total_gt = len(gt)
print("=" * 70)
print(f"BASELINE V8 CANDIDATES: {v8_cnt:,} | Recovered: {v8_rec:,} / {total_gt:,} ({v8_rec/total_gt*100:.2f}%)")
print(f"Remaining Missed in V8:  {total_gt - v8_rec:,}")
print("=" * 70)

con.execute("""
CREATE OR REPLACE TEMP TABLE v8_missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN v8_recovered r 
  ON g.source1_entity_id = r.source1_entity_id AND g.matched_entity_id = r.matched_entity_id
WHERE r.matched_entity_id IS NULL
""")

# Test 1: DBA / Trade Name Stripping
print("\n--- Testing Channel 1: DBA / Trade Name Stripping ---")
# Regex to extract text after d/b/a, t/a, or remove M/s, etc.
DBA_REGEX = r"(?i)\b(?:d/?b/?a|t/?a|trading as|doing business as|m/s\.?|c/o)\s+(.*)"
con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_dba AS
SELECT source1_entity_id, country,
       LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))) AS name_clean,
       LOWER(TRIM(REGEXP_EXTRACT(business_name, '{DBA_REGEX}', 1))) AS dba_name
FROM s1;

CREATE OR REPLACE TEMP TABLE s23_dba AS
SELECT matched_entity_id, country,
       LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))) AS name_clean,
       LOWER(TRIM(REGEXP_EXTRACT(business_name, '{DBA_REGEX}', 1))) AS dba_name
FROM s23;

CREATE OR REPLACE TEMP TABLE dba_matches AS
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_dba s1
JOIN s23_dba s23 
  ON s1.country = s23.country 
 AND (
     (LENGTH(s1.dba_name) >= 5 AND s1.dba_name = s23.name_clean)
  OR (LENGTH(s23.dba_name) >= 5 AND s23.dba_name = s1.name_clean)
 );
""")
dba_cands = con.execute("SELECT COUNT(*) FROM dba_matches").fetchone()[0]
dba_rec = con.execute("SELECT COUNT(*) FROM v8_missed m JOIN dba_matches d ON m.source1_entity_id = d.source1_entity_id AND m.matched_entity_id = d.matched_entity_id").fetchone()[0]
print(f"DBA Channel: {dba_cands:,} candidates -> Recovered {dba_rec:,} missed matches")

# Test 2: Distinctive Address Token Inverted Index (Handles Cross-Lingual & Vernacular Indian records!)
print("\n--- Testing Channel 2: Distinctive Address Token Inverted Index ---")
# Unnest address tokens of length >= 6, filter common stopwords
ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"
con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_addr_toks AS
SELECT source1_entity_id, country, token
FROM (
    SELECT source1_entity_id, country,
           UNNEST(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) AS token
    FROM s1
)
WHERE LENGTH(token) >= 7 AND token NOT IN {ADDR_STOP};

CREATE OR REPLACE TEMP TABLE s23_addr_toks AS
SELECT matched_entity_id, country, token
FROM (
    SELECT matched_entity_id, country,
           UNNEST(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) AS token
    FROM s23
)
WHERE LENGTH(token) >= 7 AND token NOT IN {ADDR_STOP};

-- Require 2 shared distinctive address tokens!
CREATE OR REPLACE TEMP TABLE addr_dual_toks AS
SELECT s1.source1_entity_id, s23.matched_entity_id, COUNT(*) AS shared_toks
FROM s1_addr_toks s1
JOIN s23_addr_toks s23 
  ON s1.country = s23.country AND s1.token = s23.token
GROUP BY s1.source1_entity_id, s23.matched_entity_id
HAVING COUNT(*) >= 2;

CREATE OR REPLACE TEMP TABLE addr_dual_capped AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT source1_entity_id, matched_entity_id,
           COUNT(*) OVER (PARTITION BY source1_entity_id) AS cnt
    FROM addr_dual_toks
) WHERE cnt <= 15;
""")
dual_cands = con.execute("SELECT COUNT(*) FROM addr_dual_capped").fetchone()[0]
dual_rec = con.execute("SELECT COUNT(*) FROM v8_missed m JOIN addr_dual_capped d ON m.source1_entity_id = d.source1_entity_id AND m.matched_entity_id = d.matched_entity_id").fetchone()[0]
print(f"Dual Address Distinctive Tokens (>=2 shared, len >= 7, cap <= 15): {dual_cands:,} candidates -> Recovered {dual_rec:,} missed matches ({dual_rec/dual_cands*100:.2f}% eff)")

# Test 3: Sorted First Two Brand Tokens (Handles word order transposition!)
print("\n--- Testing Channel 3: Sorted First Two Brand Tokens ---")
con.execute("""
CREATE OR REPLACE TEMP TABLE s1_sorted_tokens AS
WITH toks AS (
    SELECT source1_entity_id, country,
           LIST_SORT(LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 3 AND x NOT IN ('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center'))) AS sorted_list
    FROM s1
)
SELECT source1_entity_id, country,
       CASE WHEN LEN(sorted_list) >= 2 THEN sorted_list[1] || '_' || sorted_list[2] ELSE '' END AS sort_key
FROM toks
WHERE LEN(sorted_list) >= 2;

CREATE OR REPLACE TEMP TABLE s23_sorted_tokens AS
WITH toks AS (
    SELECT matched_entity_id, country,
           LIST_SORT(LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 3 AND x NOT IN ('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center'))) AS sorted_list
    FROM s23
)
SELECT matched_entity_id, country,
       CASE WHEN LEN(sorted_list) >= 2 THEN sorted_list[1] || '_' || sorted_list[2] ELSE '' END AS sort_key
FROM toks
WHERE LEN(sorted_list) >= 2;

CREATE OR REPLACE TEMP TABLE sorted_tok_cands AS
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) AS cnt
    FROM s1_sorted_tokens s1
    JOIN s23_sorted_tokens s23 ON s1.country = s23.country AND s1.sort_key = s23.sort_key
    WHERE s1.sort_key != ''
) WHERE cnt <= 15;
""")
sort_cands = con.execute("SELECT COUNT(*) FROM sorted_tok_cands").fetchone()[0]
sort_rec = con.execute("SELECT COUNT(*) FROM v8_missed m JOIN sorted_tok_cands d ON m.source1_entity_id = d.source1_entity_id AND m.matched_entity_id = d.matched_entity_id").fetchone()[0]
print(f"Sorted Token Pair Channel (cap <= 15): {sort_cands:,} candidates -> Recovered {sort_rec:,} missed matches ({sort_rec/sort_cands*100:.2f}% eff)")
