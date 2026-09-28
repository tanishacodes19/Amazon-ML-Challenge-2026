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
print("FREQUENCY-AWARE TOKEN BLOCKING BENCHMARK")
print("=" * 70)

# =========================================================
# 1. EXISTING CANDIDATES
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

# =========================================================
# 4. BUILD TOKEN KEYS
# =========================================================

print("\n[4/7] Building token keys...")

db.execute("""
ALTER TABLE s1 ADD COLUMN IF NOT EXISTS first_token VARCHAR;
ALTER TABLE s1 ADD COLUMN IF NOT EXISTS second_token VARCHAR;
ALTER TABLE s1 ADD COLUMN IF NOT EXISTS last_token VARCHAR;

ALTER TABLE s23 ADD COLUMN IF NOT EXISTS first_token VARCHAR;
ALTER TABLE s23 ADD COLUMN IF NOT EXISTS second_token VARCHAR;
ALTER TABLE s23 ADD COLUMN IF NOT EXISTS last_token VARCHAR;
""")

db.execute("""
UPDATE s1
SET
    first_token = split_part(name, ' ', 1),
    second_token = split_part(name, ' ', 2),
    last_token = reverse(split_part(reverse(name), ' ', 1))
""")

db.execute("""
UPDATE s23
SET
    first_token = split_part(name, ' ', 1),
    second_token = split_part(name, ' ', 2),
    last_token = reverse(split_part(reverse(name), ' ', 1))
""")

# =========================================================
# 5. TOKEN FREQUENCIES
# =========================================================

print("\n[5/7] Calculating token frequencies...")

for token in ["first_token", "second_token", "last_token"]:

    db.execute(f"""
    CREATE OR REPLACE TABLE {token}_freq AS
    SELECT
        country,
        {token},
        COUNT(*) AS freq
    FROM s23
    WHERE {token} <> ''
    GROUP BY country, {token}
    """)

print("Frequencies calculated.")

# =========================================================
# BENCHMARK FUNCTION
# =========================================================

def benchmark(token, limit):

    freq_table = f"{token}_freq"

    print("\n" + "-" * 70)
    print(f"{token.upper()} | FREQUENCY <= {limit}")
    print("-" * 70)

    # Raw candidates
    raw = db.execute(f"""
        SELECT COUNT(*)
        FROM s1 a
        JOIN s23 b
          ON a.country = b.country
         AND a.{token} = b.{token}
         AND a.{token} <> ''
        JOIN {freq_table} f
          ON b.country = f.country
         AND b.{token} = f.{token}
        WHERE f.freq <= {limit}
    """).fetchone()[0]

    # Actually new candidates
    new_candidates = db.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                a.s1_id,
                b.candidate_id
            FROM s1 a
            JOIN s23 b
              ON a.country = b.country
             AND a.{token} = b.{token}
             AND a.{token} <> ''
            JOIN {freq_table} f
              ON b.country = f.country
             AND b.{token} = f.{token}
            WHERE f.freq <= {limit}
        ) n
        LEFT JOIN existing e
          ON n.s1_id = e.s1_id
         AND n.candidate_id = e.candidate_id
        WHERE e.s1_id IS NULL
    """).fetchone()[0]

    # Missed true pairs recovered
    recovered = db.execute(f"""
        SELECT COUNT(*)
        FROM missed m
        JOIN s1 a
          ON m.s1_id = a.s1_id
        JOIN s23 b
          ON m.candidate_id = b.candidate_id
        JOIN {freq_table} f
          ON b.country = f.country
         AND b.{token} = f.{token}
        WHERE a.country = b.country
          AND a.{token} = b.{token}
          AND a.{token} <> ''
          AND f.freq <= {limit}
    """).fetchone()[0]

    print(f"Raw candidates   : {raw:,}")
    print(f"New candidates   : {new_candidates:,}")
    print(
        f"Missed recovered : {recovered:,} "
        f"({recovered / missed_count * 100:.2f}%)"
    )

    return raw, new_candidates, recovered


# =========================================================
# 6. RUN BENCHMARKS
# =========================================================

print("\n[6/7] Running benchmarks...")

results = []

limits = [10, 25, 50, 100, 250, 500]

for token in ["first_token", "second_token", "last_token"]:

    for limit in limits:

        raw, new, recovered = benchmark(token, limit)

        results.append(
            (
                token,
                limit,
                raw,
                new,
                recovered
            )
        )

# =========================================================
# 7. SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"{'Token':<16}"
    f"{'Limit':<8}"
    f"{'Raw':>18}"
    f"{'New':>18}"
    f"{'Recovered':>15}"
)

print("-" * 70)

for token, limit, raw, new, recovered in results:

    print(
        f"{token:<16}"
        f"{limit:<8}"
        f"{raw:>18,}"
        f"{new:>18,}"
        f"{recovered:>15,}"
    )

print("\nCurrent pair recall: 57.54%")
print("\nDone.")