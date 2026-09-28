import sys
sys.stdout.reconfigure(encoding='utf-8')
import duckdb, os

con = duckdb.connect()
con.execute('PRAGMA threads=2')
con.execute('PRAGMA memory_limit="2500MB"')

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
v7_path = os.path.join(VAL_DIR, "val_v7_cands.parquet").replace("\\", "/")
gt_path = os.path.join(VAL_DIR, "val_gt.parquet").replace("\\", "/")
s1_path = os.path.join(VAL_DIR, "val_s1.parquet").replace("\\", "/")
s23_path = os.path.join(VAL_DIR, "val_s23.parquet").replace("\\", "/")

print("=" * 70)
print("CHANNEL B: MEMORY-SAFE CHARACTER 4-GRAM INVERTED INDEX")
print("=" * 70)

con.execute(f"""
CREATE OR REPLACE TEMP TABLE v7_missed AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM read_parquet('{gt_path}') g
LEFT JOIN read_parquet('{v7_path}') c 
  ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL
""")

n_missed = con.execute("SELECT COUNT(*) FROM v7_missed").fetchone()[0]
print(f"Remaining Missed in V7: {n_missed:,}")

# Extract character 4-grams from clean business names (positions 1, 2, 3, 4, 5)
# e.g. for "creative", 4-grams are: "crea", "reat", "eati", "ativ", "tive"
print("Generating character 4-gram positional keys...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_ngrams AS
WITH base AS (
    SELECT source1_entity_id, country,
           LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))) AS name_squashed
    FROM read_parquet('{s1_path}')
)
SELECT source1_entity_id, country, SUBSTR(name_squashed, 1, 4) AS g1, SUBSTR(name_squashed, 2, 4) AS g2, SUBSTR(name_squashed, 3, 4) AS g3, SUBSTR(name_squashed, 4, 4) AS g4
FROM base
WHERE LENGTH(name_squashed) >= 5
""")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s23_ngrams AS
WITH base AS (
    SELECT matched_entity_id, country,
           LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', '', 'g'))) AS name_squashed
    FROM read_parquet('{s23_path}')
)
SELECT matched_entity_id, country, SUBSTR(name_squashed, 1, 4) AS g1, SUBSTR(name_squashed, 2, 4) AS g2, SUBSTR(name_squashed, 3, 4) AS g3, SUBSTR(name_squashed, 4, 4) AS g4
FROM base
WHERE LENGTH(name_squashed) >= 5
""")

# Unpivot into an inverted index: (id, country, gram, pos)
con.execute("""
CREATE OR REPLACE TEMP TABLE s1_inv AS
SELECT source1_entity_id, country, g1 AS gram FROM s1_ngrams WHERE LENGTH(g1) = 4
UNION ALL
SELECT source1_entity_id, country, g2 AS gram FROM s1_ngrams WHERE LENGTH(g2) = 4
UNION ALL
SELECT source1_entity_id, country, g3 AS gram FROM s1_ngrams WHERE LENGTH(g3) = 4
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE s23_inv AS
SELECT matched_entity_id, country, g1 AS gram FROM s23_ngrams WHERE LENGTH(g1) = 4
UNION ALL
SELECT matched_entity_id, country, g2 AS gram FROM s23_ngrams WHERE LENGTH(g2) = 4
UNION ALL
SELECT matched_entity_id, country, g3 AS gram FROM s23_ngrams WHERE LENGTH(g3) = 4
""")

print("Testing 4-gram inverted index retrieval...")

# Require 2 shared 4-grams to filter noisy single-gram matches!
con.execute("""
CREATE OR REPLACE TEMP TABLE shared_4grams AS
SELECT s1.source1_entity_id, s23.matched_entity_id, COUNT(*) AS shared_count
FROM s1_inv s1
JOIN s23_inv s23 ON s1.country = s23.country AND s1.gram = s23.gram
GROUP BY s1.source1_entity_id, s23.matched_entity_id
HAVING COUNT(*) >= 2
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE ngram_cands_capped AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT source1_entity_id, matched_entity_id,
           COUNT(*) OVER (PARTITION BY source1_entity_id) AS cnt
    FROM shared_4grams
) WHERE cnt <= 15
""")

total_c = con.execute("SELECT COUNT(*) FROM ngram_cands_capped").fetchone()[0]
rec = con.execute("""
SELECT COUNT(*) FROM v7_missed m
JOIN ngram_cands_capped c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

print(f"Shared 4-grams (>=2 shared, freq <= 15): {total_c:,} candidates -> Recovered Missed: {rec:,} ({(rec/max(1, total_c))*100:.2f}% eff)")
