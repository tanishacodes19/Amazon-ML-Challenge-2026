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
print("ANALYZING REMAINING MISSED PAIRS")
print("=" * 70)

# =========================================================
# 1. CURRENT CANDIDATES
# =========================================================

print("\n[1/7] Loading current candidates...")

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

# =========================================================
# 2. GROUND TRUTH
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

print("\n[4/7] Building keys...")

db.execute("""
ALTER TABLE s1 ADD COLUMN IF NOT EXISTS prefix_3 VARCHAR;
ALTER TABLE s23 ADD COLUMN IF NOT EXISTS prefix_3 VARCHAR;

ALTER TABLE s1 ADD COLUMN IF NOT EXISTS prefix_4 VARCHAR;
ALTER TABLE s23 ADD COLUMN IF NOT EXISTS prefix_4 VARCHAR;

ALTER TABLE s1 ADD COLUMN IF NOT EXISTS first_token VARCHAR;
ALTER TABLE s23 ADD COLUMN IF NOT EXISTS first_token VARCHAR;

ALTER TABLE s1 ADD COLUMN IF NOT EXISTS second_token VARCHAR;
ALTER TABLE s23 ADD COLUMN IF NOT EXISTS second_token VARCHAR;

ALTER TABLE s1 ADD COLUMN IF NOT EXISTS last_token VARCHAR;
ALTER TABLE s23 ADD COLUMN IF NOT EXISTS last_token VARCHAR;

ALTER TABLE s1 ADD COLUMN IF NOT EXISTS house_number VARCHAR;
ALTER TABLE s23 ADD COLUMN IF NOT EXISTS house_number VARCHAR;
""")

db.execute("""
UPDATE s1
SET
    prefix_3 = LEFT(name, 3),
    prefix_4 = LEFT(name, 4),
    first_token = split_part(name, ' ', 1),
    second_token = split_part(name, ' ', 2),
    last_token = reverse(split_part(reverse(name), ' ', 1)),
    house_number = regexp_extract(address, '^([0-9]+)', 1)
""")

db.execute("""
UPDATE s23
SET
    prefix_3 = LEFT(name, 3),
    prefix_4 = LEFT(name, 4),
    first_token = split_part(name, ' ', 1),
    second_token = split_part(name, ' ', 2),
    last_token = reverse(split_part(reverse(name), ' ', 1)),
    house_number = regexp_extract(address, '^([0-9]+)', 1)
""")

# =========================================================
# 5. BUILD THE FOUR NEW BLOCKS
# =========================================================

print("\n[5/7] Building tested blocks...")

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
 AND a.prefix_4 = b.prefix_4
 AND a.prefix_4 <> ''
""")

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

db.execute("""
CREATE OR REPLACE TABLE prefix4_freq AS
SELECT
    country,
    prefix_4,
    COUNT(*) AS freq
FROM s23
WHERE prefix_4 <> ''
GROUP BY country, prefix_4
""")

db.execute("""
CREATE OR REPLACE TABLE block_d AS
SELECT
    a.s1_id,
    b.candidate_id
FROM s1 a
JOIN s23 b
  ON a.country = b.country
 AND a.prefix_4 = b.prefix_4
 AND a.prefix_4 <> ''
JOIN prefix4_freq f
  ON b.country = f.country
 AND b.prefix_4 = f.prefix_4
WHERE f.freq <= 50
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

# =========================================================
# 6. FIND REMAINING MISSED PAIRS
# =========================================================

print("\n[6/7] Finding remaining missed pairs...")

db.execute("""
CREATE OR REPLACE TABLE tested_blocks AS

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

db.execute("""
CREATE OR REPLACE TABLE covered AS

SELECT
    s1_id,
    candidate_id
FROM existing

UNION

SELECT
    s1_id,
    candidate_id
FROM tested_blocks
""")

db.execute("""
CREATE OR REPLACE TABLE remaining_missed AS
SELECT
    t.s1_id,
    t.candidate_id
FROM truth t
LEFT JOIN covered c
  ON t.s1_id = c.s1_id
 AND t.candidate_id = c.candidate_id
WHERE c.s1_id IS NULL
""")

remaining = db.execute(
    "SELECT COUNT(*) FROM remaining_missed"
).fetchone()[0]

print(f"\nRemaining missed pairs: {remaining:,}")

# =========================================================
# 7. ANALYZE REMAINING PAIRS
# =========================================================

print("\n[7/7] Analyzing remaining pairs...")

db.execute("""
CREATE OR REPLACE TABLE remaining_pairs AS
SELECT
    m.s1_id,
    m.candidate_id,

    a.name AS s1_name,
    b.name AS s23_name,

    a.address AS s1_address,
    b.address AS s23_address,

    a.country,

    a.prefix_3 AS s1_prefix_3,
    b.prefix_3 AS s23_prefix_3,

    a.prefix_4 AS s1_prefix_4,
    b.prefix_4 AS s23_prefix_4,

    a.first_token AS s1_first,
    b.first_token AS s23_first,

    a.second_token AS s1_second,
    b.second_token AS s23_second,

    a.last_token AS s1_last,
    b.last_token AS s23_last,

    a.house_number AS s1_house,
    b.house_number AS s23_house

FROM remaining_missed m

JOIN s1 a
  ON m.s1_id = a.s1_id

JOIN s23 b
  ON m.candidate_id = b.candidate_id
""")

# ---------------------------------------------------------
# Simple exact / partial signals
# ---------------------------------------------------------

def signal_count(title, condition):

    count = db.execute(f"""
        SELECT COUNT(*)
        FROM remaining_pairs
        WHERE {condition}
    """).fetchone()[0]

    pct = count / remaining * 100 if remaining else 0

    print(
        f"{title:<38} "
        f"{count:>12,} "
        f"({pct:>6.2f}%)"
    )

    return count


print("\n" + "=" * 70)
print("REMAINING MISSED-PAIR SIGNALS")
print("=" * 70)

signal_count(
    "Same country",
    "country IS NOT NULL"
)

signal_count(
    "Same prefix 3",
    "s1_prefix_3 = s23_prefix_3 AND s1_prefix_3 <> ''"
)

signal_count(
    "Same prefix 4",
    "s1_prefix_4 = s23_prefix_4 AND s1_prefix_4 <> ''"
)

signal_count(
    "Same first token",
    "s1_first = s23_first AND s1_first <> ''"
)

signal_count(
    "Same second token",
    "s1_second = s23_second AND s1_second <> ''"
)

signal_count(
    "Same last token",
    "s1_last = s23_last AND s1_last <> ''"
)

signal_count(
    "Same house number",
    "s1_house = s23_house AND s1_house <> ''"
)

signal_count(
    "Exact address",
    "s1_address = s23_address AND s1_address <> ''"
)

signal_count(
    "Name contains other first token",
    """
    s1_name <> ''
    AND s23_first <> ''
    AND s1_name LIKE '%' || s23_first || '%'
    """
)

signal_count(
    "Other name contains first token",
    """
    s23_name <> ''
    AND s1_first <> ''
    AND s23_name LIKE '%' || s1_first || '%'
    """
)

signal_count(
    "Name prefix3 but prefix4 differs",
    """
    s1_prefix_3 = s23_prefix_3
    AND s1_prefix_3 <> ''
    AND s1_prefix_4 <> s23_prefix_4
    """
)

signal_count(
    "Same first token but different prefix4",
    """
    s1_first = s23_first
    AND s1_first <> ''
    AND s1_prefix_4 <> s23_prefix_4
    """
)

# ---------------------------------------------------------
# Name length difference
# ---------------------------------------------------------

signal_count(
    "Name length difference <= 2",
    """
    abs(length(s1_name) - length(s23_name)) <= 2
    """
)

signal_count(
    "Name length difference <= 4",
    """
    abs(length(s1_name) - length(s23_name)) <= 4
    """
)

# ---------------------------------------------------------
# Address prefix/suffix
# ---------------------------------------------------------

signal_count(
    "Address prefix 5",
    """
    left(s1_address, 5) =
    left(s23_address, 5)
    AND left(s1_address, 5) <> ''
    """
)

signal_count(
    "Address prefix 8",
    """
    left(s1_address, 8) =
    left(s23_address, 8)
    AND left(s1_address, 8) <> ''
    """
)

signal_count(
    "Address suffix 8",
    """
    right(s1_address, 8) =
    right(s23_address, 8)
    AND right(s1_address, 8) <> ''
    """
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)