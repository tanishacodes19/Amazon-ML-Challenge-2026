import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
sys.path.insert(0, 'src')
import os
import duckdb
import polars as pl
from eval_framework import load_benchmark
from normalizer import normalize_business_name, normalize_address

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

# Remaining 5,221 missed
con.execute("""
CREATE OR REPLACE TEMP TABLE v13_missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN v13_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")
missed_cnt = con.execute("SELECT COUNT(*) FROM v13_missed").fetchone()[0]
print(f"Targeting {missed_cnt:,} remaining missed ground truth pairs...\n")

# TEST CHANNEL 9: Squashed Brand Name (removes all spaces)
print("[Channel 9] Squashed Brand Name (Domain / Concatenation Bridge)...")
LEGAL_WORDS = "('private', 'limited', 'incorporated', 'corporation', 'company', 'llc', 'llp', 'services')"
con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_squash AS
WITH base AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> x NOT IN {LEGAL_WORDS} AND LENGTH(x) >= 3) as words
    FROM s1
)
SELECT source1_entity_id, country,
       ARRAY_TO_STRING(words, '') as squash_key
FROM base WHERE LEN(words) >= 1 AND LENGTH(ARRAY_TO_STRING(words, '')) >= 6;

CREATE OR REPLACE TEMP TABLE s23_squash AS
WITH base AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> x NOT IN {LEGAL_WORDS} AND LENGTH(x) >= 3) as words
    FROM s23
)
SELECT matched_entity_id, country,
       ARRAY_TO_STRING(words, '') as squash_key
FROM base WHERE LEN(words) >= 1 AND LENGTH(ARRAY_TO_STRING(words, '')) >= 6;

CREATE OR REPLACE TEMP TABLE ch9_squash AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_squash s1
    JOIN s23_squash s23 ON s1.country = s23.country AND s1.squash_key = s23.squash_key
) WHERE cnt <= 15;
""")
ch9_cnt = con.execute("SELECT COUNT(*) FROM ch9_squash").fetchone()[0]
ch9_rec = con.execute("SELECT COUNT(*) FROM v13_missed m JOIN ch9_squash c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  Ch9: {ch9_cnt:,} pairs | Recovered missed GT: {ch9_rec:,}")

# TEST CHANNEL 10: "Formerly:" / "AKA" Extraction
print("\n[Channel 10] 'Formerly:' / 'AKA' / 'DBA' Extraction...")
con.execute("""
CREATE OR REPLACE TEMP TABLE s23_formerly AS
WITH parsed AS (
    SELECT matched_entity_id, country,
           LOWER(TRIM(REGEXP_EXTRACT(business_name, '(?i)(?:formerly|aka|d/?b/?a|t/?a|c/?o)\\s*:?\\s*(.*)', 1))) as sub_name
    FROM s23
)
SELECT matched_entity_id, country,
       LIST_FILTER(STR_SPLIT(LOWER(REGEXP_REPLACE(sub_name, '[^a-z0-9]+', ' ', 'g')), ' '), x -> LENGTH(x) >= 4) as words
FROM parsed WHERE sub_name IS NOT NULL AND LENGTH(sub_name) >= 4;

CREATE OR REPLACE TEMP TABLE ch10_formerly AS
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 4) as words
    FROM s1
) s1
JOIN s23_formerly s23 ON s1.country = s23.country
WHERE LEN(s1.words) >= 1 AND LEN(s23.words) >= 1
  AND s1.words[1] = s23.words[1];
""")
ch10_cnt = con.execute("SELECT COUNT(*) FROM ch10_formerly").fetchone()[0]
ch10_rec = con.execute("SELECT COUNT(*) FROM v13_missed m JOIN ch10_formerly c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  Ch10: {ch10_cnt:,} pairs | Recovered missed GT: {ch10_rec:,}")

# TEST CHANNEL 11: Distinctive Address Bigram (Cross-Lingual Physical Match)
print("\n[Channel 11] Distinctive Address Bigram (Bridge across scripts)...")
ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"
con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_addr_bi AS
WITH parsed AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*') as words
    FROM s1
)
SELECT source1_entity_id, country,
       words[1] || '_' || words[2] as bi
FROM parsed WHERE LEN(words) >= 2 AND LENGTH(words[1] || '_' || words[2]) >= 10;

CREATE OR REPLACE TEMP TABLE s23_addr_bi AS
WITH parsed AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*') as words
    FROM s23
)
SELECT matched_entity_id, country,
       words[1] || '_' || words[2] as bi
FROM parsed WHERE LEN(words) >= 2 AND LENGTH(words[1] || '_' || words[2]) >= 10;

CREATE OR REPLACE TEMP TABLE ch11_addr_bi AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_addr_bi s1
    JOIN s23_addr_bi s23 ON s1.country = s23.country AND s1.bi = s23.bi
) WHERE cnt <= 12;
""")
ch11_cnt = con.execute("SELECT COUNT(*) FROM ch11_addr_bi").fetchone()[0]
ch11_rec = con.execute("SELECT COUNT(*) FROM v13_missed m JOIN ch11_addr_bi c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  Ch11: {ch11_cnt:,} pairs | Recovered missed GT: {ch11_rec:,}")

# TEST CHANNEL 12: Leet-speak Normalization (0->o, 5->s, 1->l)
print("\n[Channel 12] Leet-speak Brand Token Normalization (0->o, 5->s)...")
con.execute("""
CREATE OR REPLACE TEMP TABLE s1_leet AS
SELECT source1_entity_id, country,
       REPLACE(REPLACE(REPLACE(clean_name, '0', 'o'), '5', 's'), '1', 'l') as leet_name
FROM s1;

CREATE OR REPLACE TEMP TABLE s23_leet AS
SELECT matched_entity_id, country,
       REPLACE(REPLACE(REPLACE(clean_name, '0', 'o'), '5', 's'), '1', 'l') as leet_name
FROM s23;

CREATE OR REPLACE TEMP TABLE ch12_leet AS
WITH s1_toks AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(leet_name, ' '), x -> LENGTH(x) >= 4) as words
    FROM s1_leet
),
s23_toks AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(leet_name, ' '), x -> LENGTH(x) >= 4) as words
    FROM s23_leet
)
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_toks s1
    JOIN s23_toks s23 ON s1.country = s23.country AND s1.words[1] = s23.words[1]
    WHERE LEN(s1.words) >= 1 AND LEN(s23.words) >= 1
) WHERE cnt <= 15;
""")
ch12_cnt = con.execute("SELECT COUNT(*) FROM ch12_leet").fetchone()[0]
ch12_rec = con.execute("SELECT COUNT(*) FROM v13_missed m JOIN ch12_leet c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  Ch12: {ch12_cnt:,} pairs | Recovered missed GT: {ch12_rec:,}")

# COMBINED RECOVERY
con.execute("""
CREATE OR REPLACE TEMP TABLE all_new_cands AS
SELECT * FROM ch9_squash
UNION SELECT * FROM ch10_formerly
UNION SELECT * FROM ch11_addr_bi
UNION SELECT * FROM ch12_leet;
""")
new_total = con.execute("SELECT COUNT(*) FROM all_new_cands").fetchone()[0]
new_rec = con.execute("SELECT COUNT(*) FROM v13_missed m JOIN all_new_cands c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]

total_recovered = con.execute("""
SELECT COUNT(*) FROM gt g
WHERE EXISTS (SELECT 1 FROM v13_cands c WHERE g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id)
   OR EXISTS (SELECT 1 FROM all_new_cands c WHERE g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id)
""").fetchone()[0]

print(f"\n{'=' * 70}")
print(f"SUMMARY OF NEW CHANNELS (9-12):")
print(f"  New Candidates Added:   {new_total:,}")
print(f"  Net Missed Recovered:   +{new_rec:,} / {missed_cnt:,} ({new_rec/missed_cnt*100:.2f}%)")
print(f"  PROJECTED TOTAL RECALL: {total_recovered:,} / {total_gt:,} ({total_recovered/total_gt*100:.2f}%)")
print(f"{'=' * 70}")
