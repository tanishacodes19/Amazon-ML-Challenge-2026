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
print("COMBINED BLOCKING EVALUATION")
print("=" * 70)

# =========================================================
# 1. EXISTING CANDIDATES
# =========================================================

print("\n[1/7] Loading existing candidates...")

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

existing_count = db.execute(
    "SELECT COUNT(*) FROM existing"
).fetchone()[0]

print(f"Existing candidates: {existing_count:,}")

# =========================================================
# 2. GROUND TRUTH + MISSED PAIRS
# =========================================================

print("\n[2/7] Loading ground truth...")

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

db.execute("""
CREATE OR REPLACE TABLE missed AS
SELECT
    t.s1_id,
    t.candidate_id
FROM truth t
LEFT JOIN existing e
  ON t.s1_id = e.s1_id
 AND t.candidate_id = e.candidate_id
WHERE e.s1_id IS NULL
""")

missed_count = db.execute(
    "SELECT COUNT(*) FROM missed"
).fetchone()[0]

print(f"True pairs:   {truth_count:,}")
print(f"Missed pairs: {missed_count:,}")

# =========================================================
# 3. LOAD NORMALIZED DATA
# =========================================================

print("\n[3/7] Loading normalized data...")

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

# =========================================================
# 4. BUILD BLOCKING KEYS
# =========================================================

print("\n[4/7] Building blocking keys...")

db.execute("""
ALTER TABLE s1 ADD COLUMN IF NOT EXISTS
    name_prefix_4 VARCHAR;

ALTER TABLE s23 ADD COLUMN IF NOT EXISTS
    name_prefix_4 VARCHAR;

ALTER TABLE s1 ADD COLUMN IF NOT EXISTS
    first_token VARCHAR;

ALTER TABLE s23 ADD COLUMN IF NOT EXISTS
    first_token VARCHAR;

ALTER TABLE s1 ADD COLUMN IF NOT EXISTS
    house_number VARCHAR;

ALTER TABLE s23 ADD COLUMN IF NOT EXISTS
    house_number VARCHAR;
""")

db.execute("""
UPDATE s1
SET
    name_prefix_4 = LEFT(name, 4),
    first_token = split_part(name, ' ', 1),
    house_number = regexp_extract(
        address,
        '^([0-9]+)',
        1
    )
""")

db.execute("""
UPDATE s23
SET
    name_prefix_4 = LEFT(name, 4),
    first_token = split_part(name, ' ', 1),
    house_number = regexp_extract(
        address,
        '^([0-9]+)',
        1
    )
""")

# =========================================================
# 5. PREFIX FREQUENCIES
# =========================================================

print("\n[5/7] Calculating prefix frequencies...")

db.execute("""
CREATE OR REPLACE TABLE prefix4_freq AS
SELECT
    country,
    name_prefix_4,
    COUNT(*) AS freq
FROM s23
WHERE name_prefix_4 <> ''
GROUP BY country, name_prefix_4
""")

# =========================================================
# 6. BUILD THREE NEW BLOCKS
# =========================================================

print("\n[6/7] Building new blocks...")

# ---------------------------------------------------------
# BLOCK B
# House number + name prefix 4
# ---------------------------------------------------------

print("Building Block B...")

db.execute("""
CREATE OR REPLACE TABLE block_b AS
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

b_count = db.execute(
    "SELECT COUNT(*) FROM block_b"
).fetchone()[0]

print(f"Block B: {b_count:,}")

# ---------------------------------------------------------
# BLOCK C
# House number + first token
# ---------------------------------------------------------

print("Building Block C...")

db.execute("""
CREATE OR REPLACE TABLE block_c AS
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

c_count = db.execute(
    "SELECT COUNT(*) FROM block_c"
).fetchone()[0]

print(f"Block C: {c_count:,}")

# ---------------------------------------------------------
# BLOCK D
# Country + prefix 4, frequency <= 50
# ---------------------------------------------------------

print("Building Block D...")

db.execute("""
CREATE OR REPLACE TABLE block_d AS
SELECT
    a.s1_id,
    b.candidate_id
FROM s1 a
JOIN s23 b
  ON a.country = b.country
 AND a.name_prefix_4 = b.name_prefix_4
 AND a.name_prefix_4 <> ''
JOIN prefix4_freq f
  ON b.country = f.country
 AND b.name_prefix_4 = f.name_prefix_4
WHERE f.freq <= 50
""")

d_count = db.execute(
    "SELECT COUNT(*) FROM block_d"
).fetchone()[0]

print(f"Block D: {d_count:,}")

# =========================================================
# 7. COMBINE + MEASURE
# =========================================================

print("\n[7/7] Measuring combined result...")

# ---------------------------------------------------------
# Combined new candidates
# ---------------------------------------------------------

db.execute("""
CREATE OR REPLACE TABLE new_blocks AS

SELECT s1_id, candidate_id
FROM block_b

UNION

SELECT s1_id, candidate_id
FROM block_c

UNION

SELECT s1_id, candidate_id
FROM block_d
""")

combined_raw = db.execute(
    "SELECT COUNT(*) FROM new_blocks"
).fetchone()[0]

# Remove candidates already present
db.execute("""
CREATE OR REPLACE TABLE actually_new AS
SELECT
    n.s1_id,
    n.candidate_id
FROM new_blocks n
LEFT JOIN existing e
  ON n.s1_id = e.s1_id
 AND n.candidate_id = e.candidate_id
WHERE e.s1_id IS NULL
""")

new_count = db.execute(
    "SELECT COUNT(*) FROM actually_new"
).fetchone()[0]

# ---------------------------------------------------------
# Recover missed true pairs
# ---------------------------------------------------------

recovered = db.execute("""
SELECT COUNT(*)
FROM missed m
JOIN actually_new n
  ON m.s1_id = n.s1_id
 AND m.candidate_id = n.candidate_id
""").fetchone()[0]

# ---------------------------------------------------------
# Per-block recovery
# ---------------------------------------------------------

b_recovered = db.execute("""
SELECT COUNT(*)
FROM missed m
JOIN block_b b
  ON m.s1_id = b.s1_id
 AND m.candidate_id = b.candidate_id
""").fetchone()[0]

c_recovered = db.execute("""
SELECT COUNT(*)
FROM missed m
JOIN block_c c
  ON m.s1_id = c.s1_id
 AND m.candidate_id = c.candidate_id
""").fetchone()[0]

d_recovered = db.execute("""
SELECT COUNT(*)
FROM missed m
JOIN block_d d
  ON m.s1_id = d.s1_id
 AND m.candidate_id = d.candidate_id
""").fetchone()[0]

# ---------------------------------------------------------
# Projected recall
# ---------------------------------------------------------

new_total_recovered = (
    truth_count - missed_count + recovered
)

projected_recall = (
    new_total_recovered / truth_count * 100
)

# =========================================================
# FINAL REPORT
# =========================================================

print("\n" + "=" * 70)
print("COMBINED BLOCKING RESULT")
print("=" * 70)

print(f"Current candidates        : {existing_count:,}")
print(f"Raw new-block candidates  : {combined_raw:,}")
print(f"Actually NEW candidates   : {new_count:,}")

print("\nBlock recovery individually:")
print(
    f"House + name prefix 4    : "
    f"{b_recovered:,}"
)

print(
    f"House + first token      : "
    f"{c_recovered:,}"
)

print(
    f"Prefix 4 frequency <=50  : "
    f"{d_recovered:,}"
)

print("\nCombined recovery:")
print(
    f"New missed pairs recovered: "
    f"{recovered:,}"
)

print(
    f"Recovery of missed pairs  : "
    f"{recovered / missed_count * 100:.2f}%"
)

print("\nPair recall:")
print(
    f"Current                  : "
    f"{(truth_count - missed_count) / truth_count * 100:.2f}%"
)

print(
    f"Projected                : "
    f"{projected_recall:.2f}%"
)

print("\nProjected candidate count:")
print(
    f"{existing_count + new_count:,}"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)