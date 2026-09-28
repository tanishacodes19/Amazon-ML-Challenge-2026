import duckdb
import os

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

DATA = os.path.join(BASE, "training_data")
NORMALIZED = os.path.join(BASE, "normalized_data")

db = duckdb.connect()

db.execute("PRAGMA memory_limit='1500MB'")
db.execute("PRAGMA threads=2")
db.execute(f"PRAGMA temp_directory='{os.path.join(BASE, 'duckdb_temp')}'")

print("=" * 70)
print("BENCHMARKING NEW BLOCKS")
print("=" * 70)

print("\nLoading normalized data...")

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

print("Data loaded.")

# ---------------------------------------------------------
# Build reusable keys
# ---------------------------------------------------------

print("\nBuilding blocking keys...")

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
# Benchmark helper
# ---------------------------------------------------------

def benchmark(title, query):

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)

    result = db.execute(query).fetchone()

    count = result[0]

    print(f"Candidate pairs: {count:,}")

    return count


results = {}

# ---------------------------------------------------------
# 1. Country + name prefix 4
# ---------------------------------------------------------

results["country_name_prefix_4"] = benchmark(
    "1. COUNTRY + NAME PREFIX 4",
    """
    SELECT COUNT(*)
    FROM s1 a
    JOIN s23 b
      ON a.country = b.country
     AND a.name_prefix_4 = b.name_prefix_4
     AND a.name_prefix_4 <> ''
    """
)

# ---------------------------------------------------------
# 2. Country + first token
# ---------------------------------------------------------

results["country_first_token"] = benchmark(
    "2. COUNTRY + FIRST TOKEN",
    """
    SELECT COUNT(*)
    FROM s1 a
    JOIN s23 b
      ON a.country = b.country
     AND a.first_token = b.first_token
     AND a.first_token <> ''
    """
)

# ---------------------------------------------------------
# 3. House number + name prefix 4
# ---------------------------------------------------------

results["house_name_prefix_4"] = benchmark(
    "3. HOUSE NUMBER + NAME PREFIX 4",
    """
    SELECT COUNT(*)
    FROM s1 a
    JOIN s23 b
      ON a.country = b.country
     AND a.house_number <> ''
     AND a.house_number = b.house_number
     AND a.name_prefix_4 = b.name_prefix_4
     AND a.name_prefix_4 <> ''
    """
)

# ---------------------------------------------------------
# 4. House number + first token
# ---------------------------------------------------------

results["house_first_token"] = benchmark(
    "4. HOUSE NUMBER + FIRST TOKEN",
    """
    SELECT COUNT(*)
    FROM s1 a
    JOIN s23 b
      ON a.country = b.country
     AND a.house_number <> ''
     AND a.house_number = b.house_number
     AND a.first_token = b.first_token
     AND a.first_token <> ''
    """
)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for name, count in results.items():
    print(f"{name:35s} {count:>15,}")

print("\nExisting candidate pairs:")
print("31,327,226")

print("\nDone.")