import duckdb
import os

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORMALIZED = os.path.join(BASE, "normalized_data")
CANDIDATES = os.path.join(BASE, "training_candidate_pairs.tsv")
db = duckdb.connect()

db.execute("PRAGMA memory_limit='1500MB'")
db.execute("PRAGMA threads=2")
db.execute(
    f"PRAGMA temp_directory='{os.path.join(BASE, 'duckdb_temp')}'"
)

print("=" * 70)
print("EVALUATING NEW BLOCK OVERLAP")
print("=" * 70)

# ---------------------------------------------------------
# Load existing candidates
# ---------------------------------------------------------

print("\n[1/4] Loading existing candidates...")

db.execute(f"""
CREATE OR REPLACE TABLE existing_candidates AS
SELECT
    source1_entity_id AS s1_id,
    candidate_entity_id AS candidate_id
FROM read_csv_auto(
    '{CANDIDATES}',
    delim='\\t',
    header=true
)
""")

print(
    "Existing candidates:",
    f"{db.execute('SELECT COUNT(*) FROM existing_candidates').fetchone()[0]:,}"
)

# ---------------------------------------------------------
# Load normalized data
# ---------------------------------------------------------

print("\n[2/4] Loading normalized data...")

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

print("Normalized data loaded.")

# ---------------------------------------------------------
# Build keys
# ---------------------------------------------------------

print("\n[3/4] Building keys...")

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
# Create candidate tables for the two useful blocks
# ---------------------------------------------------------

print("\n[4/4] Measuring actual new candidates...")

db.execute("""
CREATE OR REPLACE TABLE house_prefix AS
SELECT
    a.s1_id,
    b.candidate_id
FROM s1 a
JOIN s23 b
  ON a.country = b.country
 AND a.house_number <> ''
 AND a.house_number = b.house_number
 AND a.name_prefix_4 = b.name_prefix_4
 AND a.name_prefix_4 <> ''
""")

db.execute("""
CREATE OR REPLACE TABLE house_first AS
SELECT
    a.s1_id,
    b.candidate_id
FROM s1 a
JOIN s23 b
  ON a.country = b.country
 AND a.house_number <> ''
 AND a.house_number = b.house_number
 AND a.first_token = b.first_token
 AND a.first_token <> ''
""")

# ---------------------------------------------------------
# Measure overlap
# ---------------------------------------------------------

def report(name, table):

    total = db.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    overlap = db.execute(f"""
        SELECT COUNT(*)
        FROM {table} n
        INNER JOIN existing_candidates e
          ON n.s1_id = e.s1_id
         AND n.candidate_id = e.candidate_id
    """).fetchone()[0]

    new_count = total - overlap

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)
    print(f"Raw block candidates       : {total:,}")
    print(f"Already existing           : {overlap:,}")
    print(f"Actually NEW candidates   : {new_count:,}")

    return total, overlap, new_count


hp = report(
    "HOUSE NUMBER + NAME PREFIX 4",
    "house_prefix"
)

hf = report(
    "HOUSE NUMBER + FIRST TOKEN",
    "house_first"
)

# ---------------------------------------------------------
# Combined new candidates
# ---------------------------------------------------------

combined_new = db.execute("""
SELECT COUNT(*)
FROM (
    SELECT s1_id, candidate_id
    FROM house_prefix

    UNION

    SELECT s1_id, candidate_id
    FROM house_first
) n
LEFT JOIN existing_candidates e
  ON n.s1_id = e.s1_id
 AND n.candidate_id = e.candidate_id
WHERE e.s1_id IS NULL
""").fetchone()[0]

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(f"Existing candidates        : 31,327,226")
print(f"New candidates combined    : {combined_new:,}")
print(
    f"New total candidate set    : {31_327_226 + combined_new:,}"
)

print("\nDone.")