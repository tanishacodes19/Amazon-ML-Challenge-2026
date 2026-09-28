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
print("TOKEN + EXISTING BLOCK COMBINATION")
print("=" * 70)

# =========================================================
# 1. EXISTING CANDIDATES
# =========================================================

print("\n[1/8] Loading existing candidates...")

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
# 2. GROUND TRUTH
# =========================================================

print("\n[2/8] Loading ground truth...")

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

print("\n[3/8] Loading normalized data...")

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
# 4. BUILD KEYS
# =========================================================

print("\n[4/8] Building keys...")

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
# 5. PREFIX / TOKEN FREQUENCIES
# =========================================================

print("\n[5/8] Calculating frequencies...")

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

db.execute("""
CREATE OR REPLACE TABLE first_token_freq AS
SELECT
    country,
    first_token,
    COUNT(*) AS freq
FROM s23
WHERE first_token <> ''
GROUP BY country, first_token
""")

# =========================================================
# 6. BUILD NEW BLOCKS
# =========================================================

print("\n[6/8] Building blocks...")

# ---------------------------------------------------------
# Block B: House + name prefix 4
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
# Block C: House + first token
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
# Block D: Prefix 4 frequency <= 50
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

# ---------------------------------------------------------
# Block E: First token frequency <= 50
# ---------------------------------------------------------

print("Building Block E...")

db.execute("""
CREATE OR REPLACE TABLE block_e AS
SELECT
    a.s1_id,
    b.candidate_id
FROM s1 a
JOIN s23 b
  ON a.country = b.country
 AND a.first_token = b.first_token
 AND a.first_token <> ''
JOIN first_token_freq f
  ON b.country = f.country
 AND b.first_token = f.first_token
WHERE f.freq <= 50
""")

e_count = db.execute(
    "SELECT COUNT(*) FROM block_e"
).fetchone()[0]

print(f"Block E: {e_count:,}")

# =========================================================
# 7. COMBINE
# =========================================================

print("\n[7/8] Combining blocks...")

db.execute("""
CREATE OR REPLACE TABLE new_blocks_without_e AS

SELECT s1_id, candidate_id
FROM block_b

UNION

SELECT s1_id, candidate_id
FROM block_c

UNION

SELECT s1_id, candidate_id
FROM block_d
""")

db.execute("""
CREATE OR REPLACE TABLE new_blocks_with_e AS

SELECT s1_id, candidate_id
FROM block_b

UNION

SELECT s1_id, candidate_id
FROM block_c

UNION

SELECT s1_id, candidate_id
FROM block_d

UNION

SELECT s1_id, candidate_id
FROM block_e
""")

# Remove existing candidates

db.execute("""
CREATE OR REPLACE TABLE actually_new_without_e AS
SELECT n.*
FROM new_blocks_without_e n
LEFT JOIN existing e
  ON n.s1_id = e.s1_id
 AND n.candidate_id = e.candidate_id
WHERE e.s1_id IS NULL
""")

db.execute("""
CREATE OR REPLACE TABLE actually_new_with_e AS
SELECT n.*
FROM new_blocks_with_e n
LEFT JOIN existing e
  ON n.s1_id = e.s1_id
 AND n.candidate_id = e.candidate_id
WHERE e.s1_id IS NULL
""")

# =========================================================
# 8. MEASURE
# =========================================================

print("\n[8/8] Measuring recovery...")

without_e_count = db.execute("""
SELECT COUNT(*)
FROM actually_new_without_e
""").fetchone()[0]

with_e_count = db.execute("""
SELECT COUNT(*)
FROM actually_new_with_e
""").fetchone()[0]

recovered_without_e = db.execute("""
SELECT COUNT(*)
FROM missed m
JOIN actually_new_without_e n
  ON m.s1_id = n.s1_id
 AND m.candidate_id = n.candidate_id
""").fetchone()[0]

recovered_with_e = db.execute("""
SELECT COUNT(*)
FROM missed m
JOIN actually_new_with_e n
  ON m.s1_id = n.s1_id
 AND m.candidate_id = n.candidate_id
""").fetchone()[0]

additional_recovery = recovered_with_e - recovered_without_e
additional_candidates = with_e_count - without_e_count

recall_without_e = (
    truth_count - missed_count + recovered_without_e
) / truth_count * 100

recall_with_e = (
    truth_count - missed_count + recovered_with_e
) / truth_count * 100

# =========================================================
# FINAL REPORT
# =========================================================

print("\n" + "=" * 70)
print("TOKEN COMBINATION RESULT")
print("=" * 70)

print("\nWITHOUT FIRST-TOKEN <=50:")
print(
    f"New candidates          : "
    f"{without_e_count:,}"
)

print(
    f"Missed pairs recovered  : "
    f"{recovered_without_e:,}"
)

print(
    f"Projected pair recall   : "
    f"{recall_without_e:.2f}%"
)

print("\nWITH FIRST-TOKEN <=50:")
print(
    f"New candidates          : "
    f"{with_e_count:,}"
)

print(
    f"Missed pairs recovered  : "
    f"{recovered_with_e:,}"
)

print(
    f"Projected pair recall   : "
    f"{recall_with_e:.2f}%"
)

print("\nADDITIONAL EFFECT OF FIRST-TOKEN <=50:")
print(
    f"Additional candidates   : "
    f"{additional_candidates:,}"
)

print(
    f"Additional pairs found  : "
    f"{additional_recovery:,}"
)

if additional_candidates > 0:
    print(
        f"Pairs / 1M candidates  : "
        f"{additional_recovery / additional_candidates * 1_000_000:.1f}"
    )

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)