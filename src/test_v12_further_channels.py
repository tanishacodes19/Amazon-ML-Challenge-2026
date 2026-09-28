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

s1, gt, _, s23 = load_benchmark()
v11_path = os.path.join(VAL_DIR, "val_v11_cands.parquet").replace("\\", "/")

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

con.execute(f"""
CREATE OR REPLACE TEMP TABLE base_cands AS
SELECT source1_entity_id, matched_entity_id FROM read_parquet('{v11_path}');
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN (SELECT source1_entity_id, matched_entity_id FROM base_cands) c
  ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")

print(f"Remaining Missed in V11: {con.execute('SELECT COUNT(*) FROM missed').fetchone()[0]:,}")

# CHANNEL C: House Number + First 3 chars of Brand
print("\n[Channel C] House Number + First 3 chars of Brand...")
con.execute("""
CREATE OR REPLACE TEMP TABLE s1_house_brand3 AS
WITH parsed AS (
    SELECT source1_entity_id, country,
           REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1) AS house,
           SUBSTR(clean_name, 1, 3) as brand_p3
    FROM s1
)
SELECT source1_entity_id, country, house, brand_p3
FROM parsed
WHERE house IS NOT NULL AND house != '' AND brand_p3 != '' AND LENGTH(brand_p3) >= 3;

CREATE OR REPLACE TEMP TABLE s23_house_brand3 AS
WITH parsed AS (
    SELECT matched_entity_id, country,
           REGEXP_EXTRACT(business_address, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?)\\s*(\\d+)', 1) AS house,
           SUBSTR(clean_name, 1, 3) as brand_p3
    FROM s23
)
SELECT matched_entity_id, country, house, brand_p3
FROM parsed
WHERE house IS NOT NULL AND house != '' AND brand_p3 != '' AND LENGTH(brand_p3) >= 3;

CREATE OR REPLACE TEMP TABLE ch_house_brand3 AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_house_brand3 s1
    JOIN s23_house_brand3 s23 
      ON s1.country = s23.country 
     AND s1.house = s23.house 
     AND s1.brand_p3 = s23.brand_p3
) WHERE cnt <= 12;
""")
ch_c_cnt = con.execute("SELECT COUNT(*) FROM ch_house_brand3").fetchone()[0]
ch_c_rec = con.execute("SELECT COUNT(*) FROM missed m JOIN ch_house_brand3 c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  -> Generated {ch_c_cnt:,} pairs | Recovered {ch_c_rec:,} missed matches ({ch_c_rec/max(1,ch_c_cnt)*100:.2f}% eff)")

# CHANNEL D: Cleaned Street Name (first 2 words of address) + Brand First Char
print("\n[Channel D] Cleaned Street Name (first 2 words of address) + Brand First Char...")
con.execute("""
CREATE OR REPLACE TEMP TABLE s1_street2_brand1 AS
WITH parsed AS (
    SELECT source1_entity_id, country,
           SUBSTR(clean_name, 1, 1) as brand_c1,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 3) AS words
    FROM s1
)
SELECT source1_entity_id, country, brand_c1,
       CASE WHEN LEN(words) >= 2 THEN words[1] || '_' || words[2] ELSE '' END as street2
FROM parsed
WHERE brand_c1 != '' AND LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE s23_street2_brand1 AS
WITH parsed AS (
    SELECT matched_entity_id, country,
           SUBSTR(clean_name, 1, 1) as brand_c1,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_address, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 3) AS words
    FROM s23
)
SELECT matched_entity_id, country, brand_c1,
       CASE WHEN LEN(words) >= 2 THEN words[1] || '_' || words[2] ELSE '' END as street2
FROM parsed
WHERE brand_c1 != '' AND LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE ch_street2_brand1 AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_street2_brand1 s1
    JOIN s23_street2_brand1 s23 
      ON s1.country = s23.country 
     AND s1.street2 = s23.street2 
     AND s1.brand_c1 = s23.brand_c1
    WHERE s1.street2 != '' AND LENGTH(s1.street2) >= 8
) WHERE cnt <= 12;
""")
ch_d_cnt = con.execute("SELECT COUNT(*) FROM ch_street2_brand1").fetchone()[0]
ch_d_rec = con.execute("SELECT COUNT(*) FROM missed m JOIN ch_street2_brand1 c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"  -> Generated {ch_d_cnt:,} pairs | Recovered {ch_d_rec:,} missed matches ({ch_d_rec/max(1,ch_d_cnt)*100:.2f}% eff)")
