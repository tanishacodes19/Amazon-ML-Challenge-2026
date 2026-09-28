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
print("PHASE 1: BUILDING V11 CANDIDATE SET (TARGETING 92-95%+ BLOCKING RECALL)")
print("=" * 70)

t0 = time.time()
s1, gt, _, s23 = load_benchmark()
v10_path = os.path.join(VAL_DIR, "val_v10_cands.parquet").replace("\\", "/")

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

con.register("s1", s1.to_pandas())
con.register("gt", gt.to_pandas())
con.register("s23", s23.to_pandas())

total_gt = len(gt)

# Base V10 candidates
con.execute(f"""
CREATE OR REPLACE TEMP TABLE v10_cands AS
SELECT source1_entity_id, matched_entity_id FROM read_parquet('{v10_path}');
""")
v10_cnt = con.execute("SELECT COUNT(*) FROM v10_cands").fetchone()[0]
v10_rec = con.execute("""
SELECT COUNT(*) FROM gt g
JOIN v10_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

print(f"Starting V10 Candidates: {v10_cnt:,} | Recovered: {v10_rec:,} / {total_gt:,} ({v10_rec/total_gt*100:.2f}%)")
print(f"Remaining Missed in V10: {total_gt - v10_rec:,}")

con.execute("""
CREATE OR REPLACE TEMP TABLE v10_missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN (SELECT source1_entity_id, matched_entity_id FROM v10_cands) c
  ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")

STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"

# ============================================================
# CHANNEL 1: Exact First Two Brand Tokens
# ============================================================
print("\n[Channel 1] Exact First Two Brand Tokens (e.g. select_mobility, womens_health)...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_two_tokens AS
WITH base AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) AS words
    FROM s1
)
SELECT source1_entity_id, country,
       CASE WHEN LEN(words) >= 2 THEN words[1] || '_' || words[2] ELSE '' END AS two_tok_key
FROM base WHERE LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE s23_two_tokens AS
WITH base AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) AS words
    FROM s23
)
SELECT matched_entity_id, country,
       CASE WHEN LEN(words) >= 2 THEN words[1] || '_' || words[2] ELSE '' END AS two_tok_key
FROM base WHERE LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE ch1_two_tokens AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_two_tokens s1
    JOIN s23_two_tokens s23 ON s1.country = s23.country AND s1.two_tok_key = s23.two_tok_key
    WHERE s1.two_tok_key != '' AND LENGTH(s1.two_tok_key) >= 7
) WHERE cnt <= 15;
""")
ch1_cnt = con.execute("SELECT COUNT(*) FROM ch1_two_tokens").fetchone()[0]
ch1_rec = con.execute("SELECT COUNT(*) FROM v10_missed m JOIN ch1_two_tokens c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  -> Generated {ch1_cnt:,} pairs | Recovered {ch1_rec:,} missed matches ({ch1_rec/max(1,ch1_cnt)*100:.2f}% eff)")

# ============================================================
# CHANNEL 2: House Number + Street 8-Char Prefix
# ============================================================
print("\n[Channel 2] House Number + Street 8-Char Prefix (e.g. 1516 kenilwor, 6331 blue chu)...")
con.execute("""
CREATE OR REPLACE TEMP TABLE s1_house_street AS
WITH parsed AS (
    SELECT source1_entity_id, country,
           REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1) AS house,
           LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z]+', ' ', 'g'))) AS clean_addr
    FROM s1
)
SELECT source1_entity_id, country, house,
       SUBSTR(clean_addr, 1, 8) AS street_p8
FROM parsed
WHERE LENGTH(house) >= 1 AND LENGTH(clean_addr) >= 8;

CREATE OR REPLACE TEMP TABLE s23_house_street AS
WITH parsed AS (
    SELECT matched_entity_id, country,
           REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1) AS house,
           LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z]+', ' ', 'g'))) AS clean_addr
    FROM s23
)
SELECT matched_entity_id, country, house,
       SUBSTR(clean_addr, 1, 8) AS street_p8
FROM parsed
WHERE LENGTH(house) >= 1 AND LENGTH(clean_addr) >= 8;

CREATE OR REPLACE TEMP TABLE ch2_house_street AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_house_street s1
    JOIN s23_house_street s23 
      ON s1.country = s23.country 
     AND s1.house = s23.house 
     AND s1.street_p8 = s23.street_p8
) WHERE cnt <= 12;
""")
ch2_cnt = con.execute("SELECT COUNT(*) FROM ch2_house_street").fetchone()[0]
ch2_rec = con.execute("SELECT COUNT(*) FROM v10_missed m JOIN ch2_house_street c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  -> Generated {ch2_cnt:,} pairs | Recovered {ch2_rec:,} missed matches ({ch2_rec/max(1,ch2_cnt)*100:.2f}% eff)")

# ============================================================
# CHANNEL 3: Shared Distinctive Address Bigrams
# ============================================================
print("\n[Channel 3] Shared Distinctive Address Token Pairs (Colony/Building anchors)...")
ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s23_rare_tokens_v11 AS
SELECT token
FROM (
    SELECT UNNEST(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) AS token
    FROM s23
)
WHERE LENGTH(token) >= 5 AND token NOT IN {ADDR_STOP}
GROUP BY token
HAVING COUNT(*) BETWEEN 2 AND 10;

CREATE OR REPLACE TEMP TABLE s1_rare_v11 AS
SELECT s1.source1_entity_id, s1.country, t.token
FROM (
    SELECT source1_entity_id, country,
           UNNEST(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) AS token
    FROM s1
) s1
JOIN s23_rare_tokens_v11 t ON s1.token = t.token;

CREATE OR REPLACE TEMP TABLE s23_rare_v11 AS
SELECT s23.matched_entity_id, s23.country, t.token
FROM (
    SELECT matched_entity_id, country,
           UNNEST(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) AS token
    FROM s23
) s23
JOIN s23_rare_tokens_v11 t ON s23.token = t.token;

CREATE OR REPLACE TEMP TABLE ch3_rare_addr AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_rare_v11 s1
    JOIN s23_rare_v11 s23 ON s1.country = s23.country AND s1.token = s23.token
) WHERE cnt <= 12;
""")
ch3_cnt = con.execute("SELECT COUNT(*) FROM ch3_rare_addr").fetchone()[0]
ch3_rec = con.execute("SELECT COUNT(*) FROM v10_missed m JOIN ch3_rare_addr c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  -> Generated {ch3_cnt:,} pairs | Recovered {ch3_rec:,} missed matches ({ch3_rec/max(1,ch3_cnt)*100:.2f}% eff)")

# ============================================================
# UNIFY INTO V11 CANDIDATE SET
# ============================================================
print("\n" + "=" * 70)
print("BUILDING UNIFIED V11 CANDIDATE SET")
print("=" * 70)
con.execute("""
CREATE OR REPLACE TEMP TABLE v11_candidates AS
SELECT source1_entity_id, matched_entity_id FROM v10_cands
UNION
SELECT source1_entity_id, matched_entity_id FROM ch1_two_tokens
UNION
SELECT source1_entity_id, matched_entity_id FROM ch2_house_street
UNION
SELECT source1_entity_id, matched_entity_id FROM ch3_rare_addr;
""")

v11_total = con.execute("SELECT COUNT(*) FROM v11_candidates").fetchone()[0]
v11_rec = con.execute("""
SELECT COUNT(*) FROM gt g
JOIN v11_candidates c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

v11_recall = (v11_rec / total_gt) * 100
gain_v11 = v11_rec - v10_rec

print(f"Baseline V10 Recovered:  {v10_rec:,} / {total_gt:,} ({v10_rec/total_gt*100:.2f}%)")
print(f"Unified V11 Candidates:  {v11_total:,} (+{v11_total - v10_cnt:,} candidates)")
print(f"Unified V11 Recovered:   {v11_rec:,} / {total_gt:,} ({v11_recall:.2f}%)")
print(f"Net True Matches Gained: +{gain_v11:,} (+{gain_v11/total_gt*100:.2f}% recall gain)")
print(f"Average Candidates/S1:   {v11_total / len(s1):.2f}")
print("=" * 70)

out_path = os.path.join(VAL_DIR, "val_v11_cands.parquet")
con.execute("SELECT source1_entity_id, matched_entity_id FROM v11_candidates").df().to_parquet(out_path, index=False)
print(f"Saved V11 Candidate Set to: {out_path} in {time.time()-t0:.1f}s")
