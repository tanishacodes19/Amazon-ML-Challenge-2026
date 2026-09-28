import os
import duckdb
import polars as pl
import pandas as pd
from eval_framework import load_benchmark
from normalizer import normalize_business_name, normalize_address, extract_structured_fields

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

print("=" * 70)
print("PHASE 3: BENCHMARKING NEW CANDIDATE BLOCKS ON VALIDATION")
print("=" * 70)

s1, gt, cands, s23 = load_benchmark()

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3GB'")

con.register("s1", s1.to_pandas())
con.register("gt", gt.to_pandas())
con.register("cands", cands.to_pandas())
con.register("s23", s23.to_pandas())

# Identify missed ground truth pairs
con.execute("""
CREATE OR REPLACE TEMP TABLE baseline_recovered AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
JOIN cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE missed_gt AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN baseline_recovered r 
  ON g.source1_entity_id = r.source1_entity_id AND g.matched_entity_id = r.matched_entity_id
WHERE r.matched_entity_id IS NULL
""")

total_missed = con.execute("SELECT COUNT(*) FROM missed_gt").fetchone()[0]
print(f"Total ground truth:         {len(gt):,}")
print(f"Baseline recovered:         {len(cands):,} candidates -> {len(gt) - total_missed:,} true ({((len(gt)-total_missed)/len(gt))*100:.2f}%)")
print(f"Total missed true pairs:    {total_missed:,} ({total_missed/len(gt)*100:.2f}%)")

# Precompute normalized columns and structured fields in DuckDB
print("\nPreparing enriched entity tables with normalized columns...")

con.execute("""
CREATE OR REPLACE TEMP TABLE s1_enriched AS
SELECT 
    source1_entity_id,
    country,
    LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))) AS name_clean,
    LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))) AS addr_clean,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))), 1, 3) AS name_p3,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))), 1, 4) AS name_p4,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', '', 'g'))), 1, 6) AS addr_p6,
    SPLIT_PART(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ', 1) AS first_tok,
    REGEXP_EXTRACT(business_address, '(\\b\\d{4,6}\\b)', 1) AS postal,
    REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1) AS house
FROM s1
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE s23_enriched AS
SELECT 
    matched_entity_id,
    country,
    LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))) AS name_clean,
    LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))) AS addr_clean,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))), 1, 3) AS name_p3,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))), 1, 4) AS name_p4,
    SUBSTR(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', '', 'g'))), 1, 6) AS addr_p6,
    SPLIT_PART(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ', 1) AS first_tok,
    REGEXP_EXTRACT(business_address, '(\\b\\d{4,6}\\b)', 1) AS postal,
    REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1) AS house
FROM s23
""")

# Test candidate blocking strategies
blocks = [
    # 1. Country + Name Prefix 4 (captures typos after char 4)
    ("country_name_p4_f15", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_enriched s1
        JOIN s23_enriched s23 
          ON s1.country = s23.country AND s1.name_p4 = s23.name_p4
        WHERE LENGTH(s1.name_p4) >= 4
    """, 15),
    
    # 2. Country + First Word Token (captures same first word within country)
    ("country_first_tok_f15", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_enriched s1
        JOIN s23_enriched s23 
          ON s1.country = s23.country AND s1.first_tok = s23.first_tok
        WHERE LENGTH(s1.first_tok) >= 4
    """, 15),
    
    # 3. Postal / PIN code + Name Prefix 3 (same locality + similar start)
    ("postal_name_p3_f20", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_enriched s1
        JOIN s23_enriched s23 
          ON s1.postal = s23.postal AND s1.name_p3 = s23.name_p3
        WHERE s1.postal != '' AND LENGTH(s1.name_p3) >= 3
    """, 20),
    
    # 4. House Number + Name Prefix 3 (same premise/building + name start)
    ("house_name_p3_f15", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_enriched s1
        JOIN s23_enriched s23 
          ON s1.house = s23.house AND s1.name_p3 = s23.name_p3
        WHERE s1.house != '' AND LENGTH(s1.name_p3) >= 3
    """, 15),
    
    # 5. House Number + Address Prefix 6 (same building + same road start)
    ("house_addr_p6_f15", """
        SELECT s1.source1_entity_id, s23.matched_entity_id
        FROM s1_enriched s1
        JOIN s23_enriched s23 
          ON s1.house = s23.house AND s1.addr_p6 = s23.addr_p6
        WHERE s1.house != '' AND LENGTH(s1.addr_p6) >= 5
    """, 15),
]

print("\n--- Evaluating Individual Blocks on Missed Ground Truth ---")
results = []
for name, query, max_f in blocks:
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE block_raw AS
    {query}
    """)
    
    # Apply frequency cap per S1
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE block_capped AS
    SELECT source1_entity_id, matched_entity_id
    FROM (
        SELECT source1_entity_id, matched_entity_id,
               COUNT(*) OVER (PARTITION BY source1_entity_id) AS s1_cnt
        FROM block_raw
    )
    WHERE s1_cnt <= {max_f}
    """)
    
    total_c = con.execute("SELECT COUNT(*) FROM block_capped").fetchone()[0]
    
    # How many missed true matches recovered?
    rec_missed = con.execute("""
    SELECT COUNT(*) FROM missed_gt m
    JOIN block_capped b ON m.source1_entity_id = b.source1_entity_id AND m.matched_entity_id = b.matched_entity_id
    """).fetchone()[0]
    
    eff = (rec_missed / total_c * 100) if total_c > 0 else 0.0
    print(f"Block: {name:<22} | Candidates: {total_c:>7,} | Recovered Missed: {rec_missed:>5,} | Efficiency: {eff:.2f}%")
    results.append({
        "block": name,
        "candidates": total_c,
        "recovered_missed": rec_missed,
        "efficiency": eff
    })

# Now evaluate UNION of the most efficient blocks with baseline candidates!
print("\n--- Unioning High-Efficiency Blocks with Baseline Candidates ---")
con.execute("""
CREATE OR REPLACE TEMP TABLE v6_cands AS
SELECT source1_entity_id, matched_entity_id FROM cands

UNION

-- Add country + name_p4 (freq <= 10)
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_enriched s1
    JOIN s23_enriched s23 ON s1.country = s23.country AND s1.name_p4 = s23.name_p4
    WHERE LENGTH(s1.name_p4) >= 4
) WHERE cnt <= 10

UNION

-- Add country + first_token (freq <= 10)
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_enriched s1
    JOIN s23_enriched s23 ON s1.country = s23.country AND s1.first_tok = s23.first_tok
    WHERE LENGTH(s1.first_tok) >= 4
) WHERE cnt <= 10

UNION

-- Add house + addr_p6 (freq <= 10)
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_enriched s1
    JOIN s23_enriched s23 ON s1.house = s23.house AND s1.addr_p6 = s23.addr_p6
    WHERE s1.house != '' AND LENGTH(s1.addr_p6) >= 5
) WHERE cnt <= 10

UNION

-- Add postal + name_p3 (freq <= 10)
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_enriched s1
    JOIN s23_enriched s23 ON s1.postal = s23.postal AND s1.name_p3 = s23.name_p3
    WHERE s1.postal != '' AND LENGTH(s1.name_p3) >= 3
) WHERE cnt <= 10
""")

new_cand_count = con.execute("SELECT COUNT(*) FROM v6_cands").fetchone()[0]
new_recovered = con.execute("""
SELECT COUNT(*) FROM gt g
JOIN v6_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

new_recall = (new_recovered / len(gt)) * 100
gain_recall = new_recall - 57.85
added_cands = new_cand_count - len(cands)
added_true = new_recovered - (len(gt) - total_missed)

print(f"\n=======================================================")
print(f"Baseline Candidates:     {len(cands):,}")
print(f"New Union Candidates:    {new_cand_count:,} (+{added_cands:,})")
print(f"Baseline True Matches:   {len(gt)-total_missed:,} (57.85%)")
print(f"New True Matches:        {new_recovered:,} ({new_recall:.2f}%)")
print(f"Net True Matches Gained: +{added_true:,} (+{gain_recall:.2f}% recall)")
print(f"New Efficiency:          {(added_true/max(1, added_cands))*100:.2f}% of added candidates are TRUE matches!")
print(f"Avg candidates per S1:   {new_cand_count/len(s1):.2f}")
print(f"=======================================================")

# Save the improved candidate set
v6_df = con.execute("SELECT source1_entity_id, matched_entity_id FROM v6_cands").df()
v6_df.to_parquet(os.path.join(VAL_DIR, "val_v6_cands.parquet"), index=False)
print(f"\nSaved improved candidate set to {os.path.join(VAL_DIR, 'val_v6_cands.parquet')}")
