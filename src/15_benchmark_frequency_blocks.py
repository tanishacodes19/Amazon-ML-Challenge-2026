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
print("FREQUENCY-AWARE BLOCKING BENCHMARK")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load current candidates
# ---------------------------------------------------------

print("\n[1/6] Loading current candidates...")

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

# ---------------------------------------------------------
# 2. Load ground truth and find missed pairs
# ---------------------------------------------------------

print("\n[2/6] Loading ground truth...")

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

truth_count = db.execute(
    "SELECT COUNT(*) FROM truth"
).fetchone()[0]

print(f"True pairs:       {truth_count:,}")
print(f"Missed pairs:     {missed_count:,}")

# ---------------------------------------------------------
# 3. Load normalized data
# ---------------------------------------------------------

print("\n[3/6] Loading normalized data...")

db.execute(f"""
CREATE OR REPLACE TABLE s1 AS
SELECT
    entity_id AS s1_id,
    name_normalized AS name,
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
    country
FROM read_csv_auto(
    '{NORMALIZED}/train_source3_normalized.tsv',
    delim='\\t',
    header=true
)
""")

# ---------------------------------------------------------
# 4. Build prefixes
# ---------------------------------------------------------

print("\n[4/6] Building prefix keys...")

db.execute("""
ALTER TABLE s1 ADD COLUMN IF NOT EXISTS prefix_3 VARCHAR;
ALTER TABLE s1 ADD COLUMN IF NOT EXISTS prefix_4 VARCHAR;

ALTER TABLE s23 ADD COLUMN IF NOT EXISTS prefix_3 VARCHAR;
ALTER TABLE s23 ADD COLUMN IF NOT EXISTS prefix_4 VARCHAR;
""")

db.execute("""
UPDATE s1
SET
    prefix_3 = LEFT(name, 3),
    prefix_4 = LEFT(name, 4)
""")

db.execute("""
UPDATE s23
SET
    prefix_3 = LEFT(name, 3),
    prefix_4 = LEFT(name, 4)
""")

# ---------------------------------------------------------
# 5. Calculate prefix frequencies
# ---------------------------------------------------------

print("\n[5/6] Calculating prefix frequencies...")

db.execute("""
CREATE OR REPLACE TABLE prefix3_freq AS
SELECT
    country,
    prefix_3,
    COUNT(*) AS freq
FROM s23
WHERE prefix_3 <> ''
GROUP BY country, prefix_3
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

print("Frequencies calculated.")

# ---------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------

def benchmark_prefix(prefix_length, limit):

    prefix_col = f"prefix_{prefix_length}"

    freq_table = f"prefix{prefix_length}_freq"

    print("\n" + "-" * 70)
    print(
        f"PREFIX {prefix_length} | "
        f"FREQUENCY <= {limit}"
    )
    print("-" * 70)

    # Number of raw candidates
    raw = db.execute(f"""
        SELECT COUNT(*)
        FROM s1 a
        JOIN s23 b
          ON a.country = b.country
         AND a.{prefix_col} = b.{prefix_col}
         AND a.{prefix_col} <> ''
        JOIN {freq_table} f
          ON b.country = f.country
         AND b.{prefix_col} = f.{prefix_col}
        WHERE f.freq <= {limit}
    """).fetchone()[0]

    # Number of currently missed true pairs recovered
    recovered = db.execute(f"""
        SELECT COUNT(*)
        FROM missed m
        JOIN s1 a
          ON m.s1_id = a.s1_id
        JOIN s23 b
          ON m.candidate_id = b.candidate_id
        JOIN {freq_table} f
          ON b.country = f.country
         AND b.{prefix_col} = f.{prefix_col}
        WHERE a.country = b.country
          AND a.{prefix_col} = b.{prefix_col}
          AND a.{prefix_col} <> ''
          AND f.freq <= {limit}
    """).fetchone()[0]

    # How many are actually new?
    new_candidates = db.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                a.s1_id,
                b.candidate_id
            FROM s1 a
            JOIN s23 b
              ON a.country = b.country
             AND a.{prefix_col} = b.{prefix_col}
             AND a.{prefix_col} <> ''
            JOIN {freq_table} f
              ON b.country = f.country
             AND b.{prefix_col} = f.{prefix_col}
            WHERE f.freq <= {limit}
        ) n
        LEFT JOIN existing e
          ON n.s1_id = e.s1_id
         AND n.candidate_id = e.candidate_id
        WHERE e.s1_id IS NULL
    """).fetchone()[0]

    print(f"Raw candidates       : {raw:,}")
    print(f"New candidates       : {new_candidates:,}")
    print(
        f"Missed recovered     : "
        f"{recovered:,} "
        f"({recovered / missed_count * 100:.2f}%)"
    )

    return raw, new_candidates, recovered


# ---------------------------------------------------------
# 6. Run benchmarks
# ---------------------------------------------------------

print("\n[6/6] Running benchmarks...")

results = []

for limit in [10, 25, 50, 100, 250, 500]:
    results.append(
        (
            3,
            limit,
            *benchmark_prefix(3, limit)
        )
    )

for limit in [10, 25, 50, 100, 250, 500]:
    results.append(
        (
            4,
            limit,
            *benchmark_prefix(4, limit)
        )
    )

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"{'Prefix':<10}"
    f"{'Limit':<10}"
    f"{'Raw':>18}"
    f"{'New':>18}"
    f"{'Recovered':>15}"
)

print("-" * 70)

for prefix, limit, raw, new, recovered in results:

    print(
        f"{prefix:<10}"
        f"{limit:<10}"
        f"{raw:>18,}"
        f"{new:>18,}"
        f"{recovered:>15,}"
    )

print("\nCurrent pair recall: 57.54%")

print("\nDone.")