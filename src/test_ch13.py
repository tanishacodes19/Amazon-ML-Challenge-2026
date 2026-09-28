import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
sys.path.insert(0, 'src')
import os
import duckdb
import polars as pl
from eval_framework import load_benchmark

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

# Missed pairs
con.execute("""
CREATE OR REPLACE TEMP TABLE v13_missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN v13_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")

print("Testing Channel 13: Brand Initial (2 chars) + Postal Code / Door Number...")
con.execute("""
CREATE OR REPLACE TEMP TABLE s1_init_loc AS
SELECT source1_entity_id, country,
       SUBSTR(clean_name, 1, 2) AS init2,
       REGEXP_EXTRACT(clean_addr, '(\\b\\d{3,6}\\b)', 1) AS num_anchor
FROM s1
WHERE LENGTH(clean_name) >= 2 AND LENGTH(REGEXP_EXTRACT(clean_addr, '(\\b\\d{3,6}\\b)', 1)) >= 3;

CREATE OR REPLACE TEMP TABLE s23_init_loc AS
SELECT matched_entity_id, country,
       SUBSTR(clean_name, 1, 2) AS init2,
       REGEXP_EXTRACT(clean_addr, '(\\b\\d{3,6}\\b)', 1) AS num_anchor
FROM s23
WHERE LENGTH(clean_name) >= 2 AND LENGTH(REGEXP_EXTRACT(clean_addr, '(\\b\\d{3,6}\\b)', 1)) >= 3;

CREATE OR REPLACE TEMP TABLE ch13_init_loc AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM s1_init_loc s1
    JOIN s23_init_loc s23 
      ON s1.country = s23.country 
     AND s1.init2 = s23.init2 
     AND s1.num_anchor = s23.num_anchor
) WHERE cnt <= 15;
""")

ch13_cnt = con.execute("SELECT COUNT(*) FROM ch13_init_loc").fetchone()[0]
ch13_rec = con.execute("SELECT COUNT(*) FROM v13_missed m JOIN ch13_init_loc c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"Ch13: {ch13_cnt:,} pairs | Recovered missed GT: {ch13_rec:,}")
