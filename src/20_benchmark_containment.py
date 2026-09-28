import duckdb
import os

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORM = os.path.join(BASE, "normalized_data")

DB = duckdb.connect()

DB.execute("PRAGMA threads=2")
DB.execute("PRAGMA memory_limit='1500MB'")
DB.execute(f"PRAGMA temp_directory='{BASE}\\duckdb_tmp'")

print("=" * 70)
print("LIGHTWEIGHT CONTAINMENT ANALYSIS")
print("=" * 70)


# =========================================================
# 1. Load ground truth
# =========================================================

print("\n[1/5] Loading ground truth...")

DB.execute("""
CREATE TEMP TABLE truth AS
SELECT
    CAST(source1_entity_id AS VARCHAR) AS s1_id,
    CAST(matched_entity_id AS VARCHAR) AS s23_id
FROM read_csv_auto(
    'ground_truth_pairs.tsv',
    delim='\t',
    header=true
)
""")

truth_count = DB.execute(
    "SELECT COUNT(*) FROM truth"
).fetchone()[0]

print(f"True pairs: {truth_count:,}")


# =========================================================
# 2. Load original candidates
# =========================================================

print("\n[2/5] Loading existing candidates...")

DB.execute("""
CREATE TEMP TABLE existing AS
SELECT
    CAST(source1_entity_id AS VARCHAR) AS s1_id,
    CAST(candidate_entity_id AS VARCHAR) AS s23_id
FROM read_csv_auto(
    'training_candidate_pairs.tsv',
    delim='\t',
    header=true
)
""")

existing_count = DB.execute(
    "SELECT COUNT(*) FROM existing"
).fetchone()[0]

print(f"Existing candidates: {existing_count:,}")


# =========================================================
# 3. Find originally missed pairs
# =========================================================

print("\n[3/5] Finding missed pairs...")

DB.execute("""
CREATE TEMP TABLE missed AS
SELECT
    t.s1_id,
    t.s23_id
FROM truth t
LEFT JOIN existing e
    ON t.s1_id = e.s1_id
   AND t.s23_id = e.s23_id
WHERE e.s1_id IS NULL
""")

missed_count = DB.execute(
    "SELECT COUNT(*) FROM missed"
).fetchone()[0]

print(f"Originally missed pairs: {missed_count:,}")


# =========================================================
# 4. Load only the records needed for missed pairs
# =========================================================

print("\n[4/5] Loading records for missed pairs...")

DB.execute(f"""
CREATE TEMP TABLE s1 AS
SELECT
    CAST(entity_id AS VARCHAR) AS s1_id,
    business_name AS name,
    country
FROM read_csv_auto(
    '{NORM}\\train_source1_normalized.tsv',
    delim='\t',
    header=true
)
""")

DB.execute(f"""
CREATE TEMP TABLE s23 AS
SELECT
    CAST(entity_id AS VARCHAR) AS s23_id,
    business_name AS name,
    country
FROM read_csv_auto(
    '{NORM}\\train_source2_normalized.tsv',
    delim='\t',
    header=true
)
UNION ALL
SELECT
    CAST(entity_id AS VARCHAR) AS s23_id,
    business_name AS name,
    country
FROM read_csv_auto(
    '{NORM}\\train_source3_normalized.tsv',
    delim='\t',
    header=true
)
""")


# Only retrieve the actual missed pair records.
DB.execute("""
CREATE TEMP TABLE missed_data AS
SELECT
    m.s1_id,
    m.s23_id,

    lower(trim(s1.name)) AS s1_name,
    lower(trim(s23.name)) AS s23_name,

    s1.country AS country

FROM missed m

JOIN s1
  ON m.s1_id = s1.s1_id

JOIN s23
  ON m.s23_id = s23.s23_id
""")


print(
    "Missed pair records loaded."
)


# =========================================================
# 5. Analyze containment
# =========================================================

print("\n[5/5] Analyzing containment signals...")
print()


# ---------------------------------------------------------
# Basic token containment
# ---------------------------------------------------------

DB.execute("""
CREATE TEMP TABLE analysis AS
SELECT
    *,

    split_part(s1_name, ' ', 1) AS s1_first,
    split_part(s23_name, ' ', 1) AS s23_first,

    -- S1 first token appears as a complete token in S23
    (
        ' ' || s23_name || ' '
        LIKE '% ' || split_part(s1_name, ' ', 1) || ' %'
    ) AS s1_first_in_s23,

    -- S23 first token appears as a complete token in S1
    (
        ' ' || s1_name || ' '
        LIKE '% ' || split_part(s23_name, ' ', 1) || ' %'
    ) AS s23_first_in_s1

FROM missed_data
""")


total = DB.execute(
    "SELECT COUNT(*) FROM analysis"
).fetchone()[0]


# ---------------------------------------------------------
# Calculate token frequencies
# ---------------------------------------------------------

print("Calculating token frequencies...")

DB.execute("""
CREATE TEMP TABLE s1_first_freq AS
SELECT
    split_part(lower(trim(business_name)), ' ', 1) AS token,
    country,
    COUNT(*) AS freq
FROM read_csv_auto(
    'normalized_data/train_source1_normalized.tsv',
    delim='\t',
    header=true
)
WHERE business_name IS NOT NULL
  AND trim(business_name) <> ''
GROUP BY token, country
""")


DB.execute("""
CREATE TEMP TABLE s23_first_freq AS
SELECT
    split_part(lower(trim(business_name)), ' ', 1) AS token,
    country,
    COUNT(*) AS freq
FROM (
    SELECT
        business_name,
        country
    FROM read_csv_auto(
        'normalized_data/train_source2_normalized.tsv',
        delim='\t',
        header=true
    )

    UNION ALL

    SELECT
        business_name,
        country
    FROM read_csv_auto(
        'normalized_data/train_source3_normalized.tsv',
        delim='\t',
        header=true
    )
)
WHERE business_name IS NOT NULL
  AND trim(business_name) <> ''
GROUP BY token, country
""")


DB.execute("""
CREATE TEMP TABLE final_analysis AS
SELECT
    a.*,

    COALESCE(f1.freq, 999999999) AS s1_first_freq,
    COALESCE(f23.freq, 999999999) AS s23_first_freq

FROM analysis a

LEFT JOIN s1_first_freq f1
  ON a.s1_first = f1.token
 AND a.country = f1.country

LEFT JOIN s23_first_freq f23
  ON a.s23_first = f23.token
 AND a.country = f23.country
""")


# =========================================================
# Results
# =========================================================

print()
print("=" * 70)
print("CONTAINMENT RESULTS")
print("=" * 70)

print(f"\nMissed pairs analyzed: {total:,}")


def run_test(label, condition):

    result = DB.execute(
        f"""
        SELECT COUNT(*)
        FROM final_analysis
        WHERE {condition}
        """
    ).fetchone()[0]

    percentage = result / total * 100

    print(
        f"{label:<55} "
        f"{result:>10,} "
        f"({percentage:6.2f}%)"
    )

    return result


print("\nBasic containment:")
print("-" * 70)

run_test(
    "S1 first token appears in S23 name",
    "s1_first_in_s23"
)

run_test(
    "S23 first token appears in S1 name",
    "s23_first_in_s1"
)


print("\nFrequency-filtered containment:")
print("-" * 70)

limits = [10, 25, 50, 100, 250]

for limit in limits:

    run_test(
        f"S1 first -> S23 name | S1 frequency <= {limit}",
        f"s1_first_in_s23 AND s1_first_freq <= {limit}"
    )

for limit in limits:

    run_test(
        f"S23 first -> S1 name | S23 frequency <= {limit}",
        f"s23_first_in_s1 AND s23_first_freq <= {limit}"
    )


# =========================================================
# Combination
# =========================================================

print("\nCombined containment:")
print("-" * 70)

for limit in limits:

    run_test(
        f"EITHER direction | frequency <= {limit}",
        f"""
        (
            s1_first_in_s23
            AND s1_first_freq <= {limit}
        )
        OR
        (
            s23_first_in_s1
            AND s23_first_freq <= {limit}
        )
        """
    )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)