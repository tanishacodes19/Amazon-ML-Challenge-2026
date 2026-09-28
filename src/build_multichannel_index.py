import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import duckdb
import polars as pl
import pandas as pd
from eval_framework import load_benchmark
from normalizer import normalize_business_name, normalize_address, extract_structured_fields

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

print("=" * 70)
print("PHASE 1 & 2: MULTI-CHANNEL INVERTED INDEX RETRIEVAL")
print("=" * 70)

s1, gt, baseline_cands, s23 = load_benchmark()
v6_path = os.path.join(VAL_DIR, "val_v6_cands.parquet").replace("\\", "/")

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3GB'")

con.register("s1", s1.to_pandas())
con.register("gt", gt.to_pandas())
con.register("s23", s23.to_pandas())

# Identify remaining missed ground truth
con.execute(f"""
CREATE OR REPLACE TEMP TABLE v6_recovered AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
JOIN read_parquet('{v6_path}') c 
  ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE missed_gt AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN v6_recovered r 
  ON g.source1_entity_id = r.source1_entity_id AND g.matched_entity_id = r.matched_entity_id
WHERE r.matched_entity_id IS NULL
""")

total_gt = len(gt)
v6_rec_count = con.execute("SELECT COUNT(*) FROM v6_recovered").fetchone()[0]
total_missed = con.execute("SELECT COUNT(*) FROM missed_gt").fetchone()[0]

print(f"Total Ground Truth:      {total_gt:,}")
print(f"Current V6 Recovered:    {v6_rec_count:,} ({v6_rec_count/total_gt*100:.2f}%)")
print(f"Remaining Missed:        {total_missed:,} ({total_missed/total_gt*100:.2f}%)")

# Precompute stripped legal names, house numbers, postal codes, and city/state tokens
print("\nExtracting multi-channel indexing keys...")

# Legal suffixes to strip
LEGAL_REGEX = r"\b(llc|llp|inc|incorporated|corp|corporation|pvt|private|ltd|limited|co|company)\b"

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_channels AS
SELECT 
    source1_entity_id,
    country,
    LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))) AS name_clean,
    TRIM(REGEXP_REPLACE(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), '{LEGAL_REGEX}', '', 'g')) AS name_stripped,
    SPLIT_PART(TRIM(REGEXP_REPLACE(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), '{LEGAL_REGEX}', '', 'g')), ' ', 1) AS stripped_first_tok,
    SPLIT_PART(TRIM(REGEXP_REPLACE(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), '{LEGAL_REGEX}', '', 'g')), ' ', 2) AS stripped_second_tok,
    REGEXP_EXTRACT(business_address, '(\\b\\d{{4,6}}\\b)', 1) AS postal,
    REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1) AS house,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))), 1, 3) AS name_p3,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))), 1, 4) AS name_p4,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', '', 'g'))), 1, 6) AS addr_p6
FROM s1
""")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s23_channels AS
SELECT 
    matched_entity_id,
    country,
    LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))) AS name_clean,
    TRIM(REGEXP_REPLACE(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), '{LEGAL_REGEX}', '', 'g')) AS name_stripped,
    SPLIT_PART(TRIM(REGEXP_REPLACE(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), '{LEGAL_REGEX}', '', 'g')), ' ', 1) AS stripped_first_tok,
    SPLIT_PART(TRIM(REGEXP_REPLACE(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), '{LEGAL_REGEX}', '', 'g')), ' ', 2) AS stripped_second_tok,
    REGEXP_EXTRACT(business_address, '(\\b\\d{{4,6}}\\b)', 1) AS postal,
    REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1) AS house,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))), 1, 3) AS name_p3,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))), 1, 4) AS name_p4,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', '', 'g'))), 1, 6) AS addr_p6
FROM s23
""")

# Test New Channels against remaining missed true pairs
channels = [
    # Channel A1: Exact Name After Stripping Legal Suffixes (captures "LLC Creative Works" vs "Creative Works LLC")
    ("stripped_name_exact", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_channels s1
        JOIN s23_channels s23 ON s1.name_stripped = s23.name_stripped
        WHERE LENGTH(s1.name_stripped) >= 4
    """, 15),

    # Channel A2: First Token After Stripping Legal Suffixes
    ("stripped_first_tok_f15", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_channels s1
        JOIN s23_channels s23 ON s1.country = s23.country AND s1.stripped_first_tok = s23.stripped_first_tok
        WHERE LENGTH(s1.stripped_first_tok) >= 4
    """, 15),

    # Channel A3: Second Token After Stripping Legal Suffixes
    ("stripped_second_tok_f15", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_channels s1
        JOIN s23_channels s23 ON s1.country = s23.country AND s1.stripped_second_tok = s23.stripped_second_tok
        WHERE LENGTH(s1.stripped_second_tok) >= 4
    """, 15),

    # Channel D: Postal Code + Stripped First Word Token
    ("postal_stripped_first_f20", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_channels s1
        JOIN s23_channels s23 ON s1.postal = s23.postal AND s1.stripped_first_tok = s23.stripped_first_tok
        WHERE s1.postal != '' AND LENGTH(s1.stripped_first_tok) >= 3
    """, 20),

    # Channel E: House Number + Stripped First Word Token
    ("house_stripped_first_f15", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_channels s1
        JOIN s23_channels s23 ON s1.house = s23.house AND s1.stripped_first_tok = s23.stripped_first_tok
        WHERE s1.house != '' AND LENGTH(s1.stripped_first_tok) >= 3
    """, 15),

    # Channel E2: House Number + Postal Code (Exact Same Physical Unit / Building)
    ("house_postal_f15", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_channels s1
        JOIN s23_channels s23 ON s1.house = s23.house AND s1.postal = s23.postal
        WHERE s1.house != '' AND s1.postal != ''
    """, 15),
]

print("\n--- Evaluating New Multi-Channel Retrieval Passes on Missed Matches ---")
channel_dfs = []
for name, sql, max_f in channels:
    con.execute(f"CREATE OR REPLACE TEMP TABLE ch_raw AS {sql}")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE ch_capped AS
    SELECT source1_entity_id, matched_entity_id
    FROM (
        SELECT source1_entity_id, matched_entity_id,
               COUNT(*) OVER (PARTITION BY source1_entity_id) AS cnt
        FROM ch_raw
    )
    WHERE cnt <= {max_f}
    """)
    
    total_c = con.execute("SELECT COUNT(*) FROM ch_capped").fetchone()[0]
    rec = con.execute("""
    SELECT COUNT(*) FROM missed_gt m
    JOIN ch_capped c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id
    """).fetchone()[0]
    
    eff = (rec / total_c * 100) if total_c > 0 else 0.0
    print(f"Channel: {name:<26} | Candidates: {total_c:>7,} | Recovered Missed: {rec:>5,} | Efficiency: {eff:.2f}%")

# Create High-Recall Multi-Channel Union (V7)
print("\n--- Building V7 Multi-Channel Union Candidate Set ---")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE v7_candidates AS
SELECT source1_entity_id, matched_entity_id FROM read_parquet('{v6_path}')

UNION

-- Stripped exact name
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_channels s1
    JOIN s23_channels s23 ON s1.name_stripped = s23.name_stripped
    WHERE LENGTH(s1.name_stripped) >= 4
) WHERE cnt <= 15

UNION

-- Stripped first token
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_channels s1
    JOIN s23_channels s23 ON s1.country = s23.country AND s1.stripped_first_tok = s23.stripped_first_tok
    WHERE LENGTH(s1.stripped_first_tok) >= 4
) WHERE cnt <= 10

UNION

-- Stripped second token
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_channels s1
    JOIN s23_channels s23 ON s1.country = s23.country AND s1.stripped_second_tok = s23.stripped_second_tok
    WHERE LENGTH(s1.stripped_second_tok) >= 4
) WHERE cnt <= 10

UNION

-- Postal + stripped first token
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_channels s1
    JOIN s23_channels s23 ON s1.postal = s23.postal AND s1.stripped_first_tok = s23.stripped_first_tok
    WHERE s1.postal != '' AND LENGTH(s1.stripped_first_tok) >= 3
) WHERE cnt <= 15

UNION

-- House + stripped first token
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_channels s1
    JOIN s23_channels s23 ON s1.house = s23.house AND s1.stripped_first_tok = s23.stripped_first_tok
    WHERE s1.house != '' AND LENGTH(s1.stripped_first_tok) >= 3
) WHERE cnt <= 15

UNION

-- House + postal (same building)
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_channels s1
    JOIN s23_channels s23 ON s1.house = s23.house AND s1.postal = s23.postal
    WHERE s1.house != '' AND s1.postal != ''
) WHERE cnt <= 10
""")

v7_total = con.execute("SELECT COUNT(*) FROM v7_candidates").fetchone()[0]
v7_rec = con.execute("""
SELECT COUNT(*) FROM gt g
JOIN v7_candidates c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

v7_recall = (v7_rec / total_gt) * 100
gain_v7 = v7_rec - v6_rec_count

print(f"\n=======================================================")
print(f"V6 Candidates:           {len(pl.read_parquet(v6_path)):,} (72.94% recall)")
print(f"V7 Multi-Channel Cands:  {v7_total:,} (+{v7_total - len(pl.read_parquet(v6_path)):,})")
print(f"V7 True Matches:         {v7_rec:,} / {total_gt:,} ({v7_recall:.2f}%)")
print(f"Net True Matches Gained: +{gain_v7:,} (+{v7_recall - 72.94:.2f}% recall)")
print(f"Avg candidates per S1:   {v7_total / len(s1):.2f}")
print(f"=======================================================")

# Save V7 candidate set
v7_df = con.execute("SELECT source1_entity_id, matched_entity_id FROM v7_candidates").df()
v7_path = os.path.join(VAL_DIR, "val_v7_cands.parquet")
v7_df.to_parquet(v7_path, index=False)
print(f"\nSaved V7 candidate set to: {v7_path}")
