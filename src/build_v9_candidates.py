import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import time
import duckdb
import polars as pl
from eval_framework import load_benchmark

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

print("=" * 70)
print("PHASE 1: BUILDING V9 MULTI-CHANNEL BLOCKING ENGINE (TARGETING 90-95%+ RECALL)")
print("=" * 70)

t0 = time.time()
s1, gt, _, s23 = load_benchmark()
v8_path = os.path.join(VAL_DIR, "val_v8_cands.parquet").replace("\\", "/")

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

con.register("s1", s1.to_pandas())
con.register("gt", gt.to_pandas())
con.register("s23", s23.to_pandas())

total_gt = len(gt)

# Base V8 candidates
con.execute(f"""
CREATE OR REPLACE TEMP TABLE v8_cands AS
SELECT source1_entity_id, matched_entity_id FROM read_parquet('{v8_path}');
""")
v8_cnt = con.execute("SELECT COUNT(*) FROM v8_cands").fetchone()[0]
v8_rec = con.execute("""
SELECT COUNT(*) FROM gt g
JOIN v8_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

print(f"Starting V8 Candidates: {v8_cnt:,} | Recovered: {v8_rec:,} / {total_gt:,} ({v8_rec/total_gt*100:.2f}%)")
print(f"Missed Ground Truth to Target: {total_gt - v8_rec:,}")

# Missed GT tracker
con.execute("""
CREATE OR REPLACE TEMP TABLE v8_missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN (SELECT source1_entity_id, matched_entity_id FROM v8_cands) c
  ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")

# ============================================================
# CHANNEL 1: DBA & Trade-Name Extraction
# ============================================================
print("\n[Channel 1] Extracting DBA & Trade Name anchors...")
# Extracts text after d/b/a, t/a, c/o, or m/s
DBA_REGEX = r"(?i)\b(?:d/?b/?a|t/?a|trading as|doing business as|m/s\.?|c/o)\s+(.*)"
LEGAL_CLEAN = r"(?i)\b(llc|llp|inc|incorporated|corp|corporation|pvt|private|ltd|limited|co|company)\b"

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_dba AS
SELECT source1_entity_id, country,
       LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))) AS name_clean,
       TRIM(REGEXP_REPLACE(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), '{LEGAL_CLEAN}', '', 'g')) AS name_stripped,
       LOWER(TRIM(REGEXP_EXTRACT(business_name, '{DBA_REGEX}', 1))) AS dba_raw,
       TRIM(REGEXP_REPLACE(LOWER(TRIM(REGEXP_REPLACE(COALESCE(REGEXP_EXTRACT(business_name, '{DBA_REGEX}', 1), ''), '[^a-zA-Z0-9]+', ' ', 'g'))), '{LEGAL_CLEAN}', '', 'g')) AS dba_stripped
FROM s1;

CREATE OR REPLACE TEMP TABLE s23_dba AS
SELECT matched_entity_id, country,
       LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))) AS name_clean,
       TRIM(REGEXP_REPLACE(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), '{LEGAL_CLEAN}', '', 'g')) AS name_stripped,
       LOWER(TRIM(REGEXP_EXTRACT(business_name, '{DBA_REGEX}', 1))) AS dba_raw,
       TRIM(REGEXP_REPLACE(LOWER(TRIM(REGEXP_REPLACE(COALESCE(REGEXP_EXTRACT(business_name, '{DBA_REGEX}', 1), ''), '[^a-zA-Z0-9]+', ' ', 'g'))), '{LEGAL_CLEAN}', '', 'g')) AS dba_stripped
FROM s23;

CREATE OR REPLACE TEMP TABLE ch1_dba AS
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_dba s1
JOIN s23_dba s23 
  ON s1.country = s23.country
 AND (
     (LENGTH(s1.dba_stripped) >= 4 AND (s1.dba_stripped = s23.name_stripped OR s1.dba_stripped = s23.dba_stripped))
  OR (LENGTH(s23.dba_stripped) >= 4 AND (s23.dba_stripped = s1.name_stripped))
 );

CREATE OR REPLACE TEMP TABLE ch1_capped AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT source1_entity_id, matched_entity_id,
           COUNT(*) OVER (PARTITION BY source1_entity_id) as cnt
    FROM ch1_dba
) WHERE cnt <= 12;
""")
ch1_cnt = con.execute("SELECT COUNT(*) FROM ch1_capped").fetchone()[0]
ch1_rec = con.execute("SELECT COUNT(*) FROM v8_missed m JOIN ch1_capped c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  -> Generated {ch1_cnt:,} pairs | Recovered {ch1_rec:,} missed matches ({ch1_rec/max(1,ch1_cnt)*100:.2f}% eff)")

# ============================================================
# CHANNEL 2: Order-Invariant Sorted Brand Token Bags
# ============================================================
print("\n[Channel 2] Generating order-invariant sorted brand token pairs...")
STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_sort_toks AS
WITH base AS (
    SELECT source1_entity_id, country,
           LIST_SORT(LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS})) AS words
    FROM s1
)
SELECT source1_entity_id, country,
       CASE WHEN LEN(words) >= 2 THEN words[1] || '_' || words[2] ELSE '' END AS bag_key
FROM base WHERE LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE s23_sort_toks AS
WITH base AS (
    SELECT matched_entity_id, country,
           LIST_SORT(LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS})) AS words
    FROM s23
)
SELECT matched_entity_id, country,
       CASE WHEN LEN(words) >= 2 THEN words[1] || '_' || words[2] ELSE '' END AS bag_key
FROM base WHERE LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE ch2_sort_bag AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_sort_toks s1
    JOIN s23_sort_toks s23 ON s1.country = s23.country AND s1.bag_key = s23.bag_key
    WHERE s1.bag_key != ''
) WHERE cnt <= 12;
""")
ch2_cnt = con.execute("SELECT COUNT(*) FROM ch2_sort_bag").fetchone()[0]
ch2_rec = con.execute("SELECT COUNT(*) FROM v8_missed m JOIN ch2_sort_bag c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  -> Generated {ch2_cnt:,} pairs | Recovered {ch2_rec:,} missed matches ({ch2_rec/max(1,ch2_cnt)*100:.2f}% eff)")

# ============================================================
# CHANNEL 3: Rare Landmark & Address Substring Co-Location
# ============================================================
print("\n[Channel 3] Indexing rare physical landmark & address tokens (Cross-lingual bridge)...")
# Select rare address tokens (length >= 7) that appear between 2 and 15 times
ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s23_rare_tokens AS
SELECT token
FROM (
    SELECT UNNEST(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) AS token
    FROM s23
)
WHERE LENGTH(token) >= 7 AND token NOT IN {ADDR_STOP}
GROUP BY token
HAVING COUNT(*) BETWEEN 2 AND 15;

CREATE OR REPLACE TEMP TABLE s1_rare_addr AS
SELECT s1.source1_entity_id, s1.country, t.token
FROM (
    SELECT source1_entity_id, country,
           UNNEST(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) AS token
    FROM s1
) s1
JOIN s23_rare_tokens t ON s1.token = t.token;

CREATE OR REPLACE TEMP TABLE s23_rare_addr AS
SELECT s23.matched_entity_id, s23.country, t.token
FROM (
    SELECT matched_entity_id, country,
           UNNEST(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) AS token
    FROM s23
) s23
JOIN s23_rare_tokens t ON s23.token = t.token;

CREATE OR REPLACE TEMP TABLE ch3_landmark AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_rare_addr s1
    JOIN s23_rare_addr s23 ON s1.country = s23.country AND s1.token = s23.token
) WHERE cnt <= 12;
""")
ch3_cnt = con.execute("SELECT COUNT(*) FROM ch3_landmark").fetchone()[0]
ch3_rec = con.execute("SELECT COUNT(*) FROM v8_missed m JOIN ch3_landmark c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  -> Generated {ch3_cnt:,} pairs | Recovered {ch3_rec:,} missed matches ({ch3_rec/max(1,ch3_cnt)*100:.2f}% eff)")

# ============================================================
# CHANNEL 4: Normalized House Number + Postal PIN Code
# ============================================================
print("\n[Channel 4] High-precision Physical Unit co-location (House + PIN)...")
con.execute("""
CREATE OR REPLACE TEMP TABLE s1_unit AS
SELECT source1_entity_id, country,
       REGEXP_EXTRACT(business_address, '(\\b\\d{5,6}\\b)', 1) AS postal,
       REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1) AS house
FROM s1
WHERE LENGTH(REGEXP_EXTRACT(business_address, '(\\b\\d{5,6}\\b)', 1)) >= 5
  AND LENGTH(REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1)) >= 1;

CREATE OR REPLACE TEMP TABLE s23_unit AS
SELECT matched_entity_id, country,
       REGEXP_EXTRACT(business_address, '(\\b\\d{5,6}\\b)', 1) AS postal,
       REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1) AS house
FROM s23
WHERE LENGTH(REGEXP_EXTRACT(business_address, '(\\b\\d{5,6}\\b)', 1)) >= 5
  AND LENGTH(REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1)) >= 1;

CREATE OR REPLACE TEMP TABLE ch4_unit AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_unit s1
    JOIN s23_unit s23 ON s1.postal = s23.postal AND s1.house = s23.house
) WHERE cnt <= 12;
""")
ch4_cnt = con.execute("SELECT COUNT(*) FROM ch4_unit").fetchone()[0]
ch4_rec = con.execute("SELECT COUNT(*) FROM v8_missed m JOIN ch4_unit c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  -> Generated {ch4_cnt:,} pairs | Recovered {ch4_rec:,} missed matches ({ch4_rec/max(1,ch4_cnt)*100:.2f}% eff)")

# ============================================================
# COMBINE ALL CHANNELS INTO V9 CANDIDATE SET
# ============================================================
print("\n" + "=" * 70)
print("BUILDING UNIFIED V9 CANDIDATE SET")
print("=" * 70)
con.execute("""
CREATE OR REPLACE TEMP TABLE v9_candidates AS
SELECT source1_entity_id, matched_entity_id FROM v8_cands
UNION
SELECT source1_entity_id, matched_entity_id FROM ch1_capped
UNION
SELECT source1_entity_id, matched_entity_id FROM ch2_sort_bag
UNION
SELECT source1_entity_id, matched_entity_id FROM ch3_landmark
UNION
SELECT source1_entity_id, matched_entity_id FROM ch4_unit;
""")

v9_total = con.execute("SELECT COUNT(*) FROM v9_candidates").fetchone()[0]
v9_rec = con.execute("""
SELECT COUNT(*) FROM gt g
JOIN v9_candidates c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

v9_recall = (v9_rec / total_gt) * 100
gain_v9 = v9_rec - v8_rec

print(f"Baseline V8 Recovered:   {v8_rec:,} / {total_gt:,} ({v8_rec/total_gt*100:.2f}%)")
print(f"Unified V9 Candidates:   {v9_total:,} (+{v9_total - v8_cnt:,} candidates)")
print(f"Unified V9 Recovered:    {v9_rec:,} / {total_gt:,} ({v9_recall:.2f}%)")
print(f"Net True Matches Gained: +{gain_v9:,} (+{gain_v9/total_gt*100:.2f}% recall gain)")
print(f"Average Candidates/S1:   {v9_total / len(s1):.2f}")
print("=" * 70)

# Save V9 candidates to parquet
out_path = os.path.join(VAL_DIR, "val_v9_cands.parquet")
con.execute("SELECT source1_entity_id, matched_entity_id FROM v9_candidates").df().to_parquet(out_path, index=False)
print(f"Saved V9 Candidate Set to: {out_path} in {time.time()-t0:.1f}s")
