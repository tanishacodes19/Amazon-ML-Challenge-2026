import duckdb
import os

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

NORMALIZED = os.path.join(BASE, "normalized_data")
CANDIDATES = os.path.join(BASE, "training_candidate_pairs.tsv")
GROUND_TRUTH = os.path.join(BASE, "ground_truth_pairs.tsv")
db = duckdb.connect()

db.execute("PRAGMA memory_limit='1500MB'")
db.execute("PRAGMA threads=2")
db.execute(
    f"PRAGMA temp_directory='{os.path.join(BASE, 'duckdb_temp')}'"
)

print("=" * 70)
print("EVALUATING NEW BLOCK RECALL")
print("=" * 70)

# ---------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------

print("\n[1/5] Loading ground truth...")

db.execute(f"""
CREATE OR REPLACE TABLE truth AS
SELECT
    source1_entity_id AS s1_id,
    matched_entity_id AS candidate_id
FROM read_csv_auto(
    '{GROUND_TRUTH}',
    delim='\\t',
    header=true
)
""")

truth_count = db.execute(
    "SELECT COUNT(*) FROM truth"
).fetchone()[0]

print(f"True pairs: {truth_count:,}")

# ---------------------------------------------------------
# Find currently missed pairs
# ---------------------------------------------------------

print("\n[2/5] Finding currently missed pairs...")

db.execute(f"""
CREATE OR REPLACE TABLE existing AS
SELECT
    source1_entity_id AS s1_id,
    candidate_entity_id AS candidate_id
FROM read_csv_auto(
    '{CANDIDATES}',
    delim='\\t',
    header=true
)
""")

db.execute("""
CREATE OR REPLACE TABLE missed AS
SELECT t.s1_id, t.candidate_id
FROM truth t
LEFT JOIN existing e
  ON t.s1_id = e.s1_id
 AND t.candidate_id = e.candidate_id
WHERE e.s1_id IS NULL
""")

missed_count = db.execute(
    "SELECT COUNT(*) FROM missed"
).fetchone()[0]

print(f"Currently missed: {missed_count:,}")

# ---------------------------------------------------------
# Load normalized sources
# ---------------------------------------------------------

print("\n[3/5] Loading normalized data...")

db.execute(f"""
CREATE OR REPLACE TABLE s1 AS
SELECT
    entity_id AS s1_id,
    name_normalized AS name,
    address_normalized AS address,
    country
FROM read_csv_auto(
    '{NORMALIZED}/train_source1_normalized.tsv',
    delim='\\t',
    header=true
)
""")

db.execute(f"""
CREATE OR REPLACE TABLE s23 AS

SELECT
    entity_id AS candidate_id,
    name_normalized AS name,
    address_normalized AS address,
    country
FROM read_csv_auto(
    '{NORMALIZED}/train_source2_normalized.tsv',
    delim='\\t',
    header=true
)

UNION ALL

SELECT
    entity_id AS candidate_id,
    name_normalized AS name,
    address_normalized AS address,
    country
FROM read_csv_auto(
    '{NORMALIZED}/train_source3_normalized.tsv',
    delim='\\t',
    header=true
)
""")

# ---------------------------------------------------------
# Build keys
# ---------------------------------------------------------

print("\n[4/5] Building keys...")

db.execute("""
ALTER TABLE s1 ADD COLUMN IF NOT EXISTS name_prefix_4 VARCHAR;
ALTER TABLE s23 ADD COLUMN IF NOT EXISTS name_prefix_4 VARCHAR;

ALTER TABLE s1 ADD COLUMN IF NOT EXISTS first_token VARCHAR;
ALTER TABLE s23 ADD COLUMN IF NOT EXISTS first_token VARCHAR;

ALTER TABLE s1 ADD COLUMN IF NOT EXISTS house_number VARCHAR;
ALTER TABLE s23 ADD COLUMN IF NOT EXISTS house_number VARCHAR;
""")

db.execute("""
UPDATE s1
SET
    name_prefix_4 = LEFT(name, 4),
    first_token = split_part(name, ' ', 1),
    house_number = regexp_extract(address, '^([0-9]+)', 1)
""")

db.execute("""
UPDATE s23
SET
    name_prefix_4 = LEFT(name, 4),
    first_token = split_part(name, ' ', 1),
    house_number = regexp_extract(address, '^([0-9]+)', 1)
""")

# ---------------------------------------------------------
# Evaluate missed-pair recovery
# ---------------------------------------------------------

print("\n[5/5] Evaluating recovery...")

# House + name prefix 4
house_prefix_recovered = db.execute("""
SELECT COUNT(*)
FROM missed m
JOIN s1 a
  ON m.s1_id = a.s1_id
JOIN s23 b
  ON m.candidate_id = b.candidate_id
WHERE a.country = b.country
  AND a.house_number <> ''
  AND a.house_number = b.house_number
  AND a.name_prefix_4 = b.name_prefix_4
  AND a.name_prefix_4 <> ''
""").fetchone()[0]

# House + first token
house_first_recovered = db.execute("""
SELECT COUNT(*)
FROM missed m
JOIN s1 a
  ON m.s1_id = a.s1_id
JOIN s23 b
  ON m.candidate_id = b.candidate_id
WHERE a.country = b.country
  AND a.house_number <> ''
  AND a.house_number = b.house_number
  AND a.first_token = b.first_token
  AND a.first_token <> ''
""").fetchone()[0]

# Combined
combined_recovered = db.execute("""
SELECT COUNT(*)
FROM (
    SELECT m.s1_id, m.candidate_id
    FROM missed m
    JOIN s1 a ON m.s1_id = a.s1_id
    JOIN s23 b ON m.candidate_id = b.candidate_id
    WHERE a.country = b.country
      AND a.house_number <> ''
      AND a.house_number = b.house_number
      AND a.name_prefix_4 = b.name_prefix_4
      AND a.name_prefix_4 <> ''

    UNION

    SELECT m.s1_id, m.candidate_id
    FROM missed m
    JOIN s1 a ON m.s1_id = a.s1_id
    JOIN s23 b ON m.candidate_id = b.candidate_id
    WHERE a.country = b.country
      AND a.house_number <> ''
      AND a.house_number = b.house_number
      AND a.first_token = b.first_token
      AND a.first_token <> ''
) x
""").fetchone()[0]

print("\n" + "=" * 70)
print("NEW BLOCK RECALL")
print("=" * 70)

print(
    f"House + name prefix 4 : "
    f"{house_prefix_recovered:,} "
    f"({house_prefix_recovered / missed_count * 100:.2f}% of missed)"
)

print(
    f"House + first token   : "
    f"{house_first_recovered:,} "
    f"({house_first_recovered / missed_count * 100:.2f}% of missed)"
)

print(
    f"COMBINED              : "
    f"{combined_recovered:,} "
    f"({combined_recovered / missed_count * 100:.2f}% of missed)"
)

print("\nCurrent pair recall:")
print(f"{(truth_count - missed_count) / truth_count * 100:.2f}%")

new_recall = (
    truth_count - missed_count + combined_recovered
) / truth_count * 100

print("Projected pair recall after blocks:")
print(f"{new_recall:.2f}%")

print("\nDone.")