import sys
sys.stdout.reconfigure(encoding='utf-8')
import duckdb, os

con = duckdb.connect()
con.execute('PRAGMA threads=2')
con.execute('PRAGMA memory_limit="3GB"')

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
v7_path = os.path.join(VAL_DIR, "val_v7_cands.parquet").replace("\\", "/")
gt_path = os.path.join(VAL_DIR, "val_gt.parquet").replace("\\", "/")
s1_path = os.path.join(VAL_DIR, "val_s1.parquet").replace("\\", "/")
s23_path = os.path.join(VAL_DIR, "val_s23.parquet").replace("\\", "/")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE v7_missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM read_parquet('{gt_path}') g
LEFT JOIN read_parquet('{v7_path}') c 
  ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL
""")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_tokens AS
SELECT 
    source1_entity_id,
    country,
    UNNEST(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) AS token
FROM read_parquet('{s1_path}')
""")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s23_tokens AS
SELECT 
    matched_entity_id,
    country,
    UNNEST(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' ')) AS token
FROM read_parquet('{s23_path}')
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE token_match AS
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM (
    SELECT DISTINCT source1_entity_id, country, token 
    FROM s1_tokens 
    WHERE LENGTH(token) >= 5 
      AND token NOT IN ('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises')
) s1
JOIN (
    SELECT DISTINCT matched_entity_id, country, token 
    FROM s23_tokens 
    WHERE LENGTH(token) >= 5 
      AND token NOT IN ('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises')
) s23 ON s1.country = s23.country AND s1.token = s23.token
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE token_match_capped AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT source1_entity_id, matched_entity_id,
           COUNT(*) OVER (PARTITION BY source1_entity_id) as cnt
    FROM token_match
) WHERE cnt <= 15
""")

cands_cnt = con.execute("SELECT COUNT(*) FROM token_match_capped").fetchone()[0]
rec = con.execute("""
SELECT COUNT(*) FROM v7_missed m
JOIN token_match_capped c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

print(f"Distinctive Name Token Match (len >= 5, freq <= 15): {cands_cnt:,} candidates -> recovered {rec:,} missed matches ({rec/cands_cnt*100:.2f}% eff)")
