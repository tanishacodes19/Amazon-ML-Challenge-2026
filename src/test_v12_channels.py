import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
sys.path.insert(0, 'src')
import os
import time
import duckdb
import polars as pl
from eval_framework import load_benchmark
from normalizer import normalize_business_name

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

print("=" * 70)
print("TESTING V12 CHANNELS: TARGETING 93-96%+ BLOCKING RECALL")
print("=" * 70)

t0 = time.time()
s1, gt, _, s23 = load_benchmark()
v11_path = os.path.join(VAL_DIR, "val_v11_cands.parquet").replace("\\", "/")

# Pre-normalize business names using the upgraded normalizer
print("Applying upgraded normalizer to validation entities...")
s1_df = s1.to_pandas()
s23_df = s23.to_pandas()

s1_df["clean_name"] = s1_df["business_name"].apply(normalize_business_name)
s23_df["clean_name"] = s23_df["business_name"].apply(normalize_business_name)

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

con.register("s1", s1_df)
con.register("gt", gt.to_pandas())
con.register("s23", s23_df)

total_gt = len(gt)

# Base V11 candidates
con.execute(f"""
CREATE OR REPLACE TEMP TABLE v11_cands AS
SELECT source1_entity_id, matched_entity_id FROM read_parquet('{v11_path}');
""")
v11_cnt = con.execute("SELECT COUNT(*) FROM v11_cands").fetchone()[0]
v11_rec = con.execute("""
SELECT COUNT(*) FROM gt g
JOIN v11_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

print(f"Starting V11 Candidates: {v11_cnt:,} | Recovered: {v11_rec:,} / {total_gt:,} ({v11_rec/total_gt*100:.2f}%)")
print(f"Remaining Missed in V11: {total_gt - v11_rec:,}")

con.execute("""
CREATE OR REPLACE TEMP TABLE v11_missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN (SELECT source1_entity_id, matched_entity_id FROM v11_cands) c
  ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")

STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"

# CHANNEL A: Cleaned Brand First Token
print("\n[Channel A] Cleaned Brand First Token (capturing stripped domains, honorifics, etc.)...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_clean_first AS
WITH base AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(clean_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS}) AS words
    FROM s1
)
SELECT source1_entity_id, country,
       CASE WHEN LEN(words) >= 1 THEN words[1] ELSE '' END AS first_tok
FROM base WHERE LEN(words) >= 1;

CREATE OR REPLACE TEMP TABLE s23_clean_first AS
WITH base AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(clean_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS}) AS words
    FROM s23
)
SELECT matched_entity_id, country,
       CASE WHEN LEN(words) >= 1 THEN words[1] ELSE '' END AS first_tok
FROM base WHERE LEN(words) >= 1;

CREATE OR REPLACE TEMP TABLE ch_clean_first AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_clean_first s1
    JOIN s23_clean_first s23 ON s1.country = s23.country AND s1.first_tok = s23.first_tok
    WHERE s1.first_tok != '' AND LENGTH(s1.first_tok) >= 4
) WHERE cnt <= 15;
""")
ch_a_cnt = con.execute("SELECT COUNT(*) FROM ch_clean_first").fetchone()[0]
ch_a_rec = con.execute("SELECT COUNT(*) FROM v11_missed m JOIN ch_clean_first c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  -> Generated {ch_a_cnt:,} pairs | Recovered {ch_a_rec:,} missed matches ({ch_a_rec/max(1,ch_a_cnt)*100:.2f}% eff)")

# CHANNEL B: Unit/Plot + Address Numeric Pattern
print("\n[Channel B] Unit / Plot + Address First Distinctive Word...")
con.execute("""
CREATE OR REPLACE TEMP TABLE s1_unit_word AS
WITH parsed AS (
    SELECT source1_entity_id, country,
           LOWER(REGEXP_EXTRACT(business_address, '(?i)(?:unit|plot|flat|no\\.?|room|shop)\\s*(?:no\\.?)?\\s*([a-z0-9\\-\\/]+)', 1)) as unit,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 5) AS words
    FROM s1
)
SELECT source1_entity_id, country, unit,
       CASE WHEN LEN(words) >= 1 THEN words[1] ELSE '' END as first_word
FROM parsed
WHERE unit IS NOT NULL AND unit != '' AND LENGTH(unit) >= 2;

CREATE OR REPLACE TEMP TABLE s23_unit_word AS
WITH parsed AS (
    SELECT matched_entity_id, country,
           LOWER(REGEXP_EXTRACT(business_address, '(?i)(?:unit|plot|flat|no\\.?|room|shop)\\s*(?:no\\.?)?\\s*([a-z0-9\\-\\/]+)', 1)) as unit,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 5) AS words
    FROM s23
)
SELECT matched_entity_id, country, unit,
       CASE WHEN LEN(words) >= 1 THEN words[1] ELSE '' END as first_word
FROM parsed
WHERE unit IS NOT NULL AND unit != '' AND LENGTH(unit) >= 2;

CREATE OR REPLACE TEMP TABLE ch_unit_word AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_unit_word s1
    JOIN s23_unit_word s23 
      ON s1.country = s23.country 
     AND s1.unit = s23.unit 
     AND s1.first_word = s23.first_word
    WHERE s1.first_word != ''
) WHERE cnt <= 12;
""")
ch_b_cnt = con.execute("SELECT COUNT(*) FROM ch_unit_word").fetchone()[0]
ch_b_rec = con.execute("SELECT COUNT(*) FROM v11_missed m JOIN ch_unit_word c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  -> Generated {ch_b_cnt:,} pairs | Recovered {ch_b_rec:,} missed matches ({ch_b_rec/max(1,ch_b_cnt)*100:.2f}% eff)")

# Combine into V12 Candidate Pool
print("\n[Combining] Merging V11 + Channel A + Channel B...")
con.execute("""
CREATE OR REPLACE TEMP TABLE v12_combined AS
SELECT source1_entity_id, matched_entity_id FROM v11_cands
UNION
SELECT source1_entity_id, matched_entity_id FROM ch_clean_first
UNION
SELECT source1_entity_id, matched_entity_id FROM ch_unit_word;
""")

v12_total = con.execute("SELECT COUNT(*) FROM v12_combined").fetchone()[0]
v12_rec = con.execute("""
SELECT COUNT(*) FROM gt g
JOIN v12_combined c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

print(f"\n" + "=" * 70)
print(f"V12 CANDIDATE SET RESULTS:")
print(f"Total Candidate Pairs: {v12_total:,} ({v12_total/25000:.2f} cands/S1)")
print(f"Recovered Ground Truth: {v12_rec:,} / {total_gt:,} ({v12_rec/total_gt*100:.2f}%)")
print(f"Gain over V11: +{v12_rec - v11_rec:,} true matches!")
print(f"Total time: {time.time()-t0:.1f}s")
print("=" * 70)
