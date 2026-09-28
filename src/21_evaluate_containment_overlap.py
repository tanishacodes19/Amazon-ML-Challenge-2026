import duckdb
import os

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORM = os.path.join(BASE, "normalized_data")

DB = duckdb.connect()

DB.execute("PRAGMA threads=2")
DB.execute("PRAGMA memory_limit='1500MB'")
DB.execute(f"PRAGMA temp_directory='{BASE}\\duckdb_tmp'")

print("=" * 70)
print("CONTAINMENT OVERLAP WITH CURRENT BEST BLOCKING")
print("=" * 70)


# =========================================================
# 1. Original candidates
# =========================================================

print("\n[1/7] Loading original candidates...")

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

print(f"Original candidates: {existing_count:,}")


# =========================================================
# 2. Ground truth
# =========================================================

print("\n[2/7] Loading ground truth...")

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
# 3. Build current best block set
#
# B = house + name prefix 4
# C = house + first token
# D = country + name prefix 4, frequency <= 50
# E = first token frequency <= 50
# =========================================================

print("\n[3/7] Building current best block set...")

DB.execute(f"""
CREATE TEMP TABLE s1 AS
SELECT
    CAST(entity_id AS VARCHAR) AS s1_id,
    lower(trim(business_name)) AS name,
    country,
    regexp_extract(
        lower(trim(business_address)),
        '^[0-9]+',
        0
    ) AS house
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
    lower(trim(business_name)) AS name,
    country,
    regexp_extract(
        lower(trim(business_address)),
        '^[0-9]+',
        0
    ) AS house
FROM read_csv_auto(
    '{NORM}\\train_source2_normalized.tsv',
    delim='\t',
    header=true
)
UNION ALL
SELECT
    CAST(entity_id AS VARCHAR) AS s23_id,
    lower(trim(business_name)) AS name,
    country,
    regexp_extract(
        lower(trim(business_address)),
        '^[0-9]+',
        0
    ) AS house
FROM read_csv_auto(
    '{NORM}\\train_source3_normalized.tsv',
    delim='\t',
    header=true
)
""")


# ---------------------------------------------------------
# Basic derived fields
# ---------------------------------------------------------

DB.execute("""
CREATE TEMP TABLE s1_base AS
SELECT
    *,
    left(name, 4) AS prefix4,
    split_part(name, ' ', 1) AS first_token
FROM s1
""")

DB.execute("""
CREATE TEMP TABLE s23_base AS
SELECT
    *,
    left(name, 4) AS prefix4,
    split_part(name, ' ', 1) AS first_token
FROM s23
""")


# ---------------------------------------------------------
# Token frequencies
# ---------------------------------------------------------

print("Calculating token frequencies...")

DB.execute("""
CREATE TEMP TABLE first_freq AS
SELECT
    country,
    first_token,
    COUNT(*) AS freq
FROM s1_base
WHERE first_token <> ''
GROUP BY country, first_token
""")


# ---------------------------------------------------------
# Build blocks separately
# ---------------------------------------------------------

print("Building B: house + name prefix 4...")

DB.execute("""
CREATE TEMP TABLE block_b AS
SELECT DISTINCT
    a.s1_id,
    b.s23_id
FROM s1_base a
JOIN s23_base b
  ON a.country = b.country
 AND a.house <> ''
 AND a.house = b.house
 AND a.prefix4 <> ''
 AND a.prefix4 = b.prefix4
""")


print("Building C: house + first token...")

DB.execute("""
CREATE TEMP TABLE block_c AS
SELECT DISTINCT
    a.s1_id,
    b.s23_id
FROM s1_base a
JOIN s23_base b
  ON a.country = b.country
 AND a.house <> ''
 AND a.house = b.house
 AND a.first_token <> ''
 AND a.first_token = b.first_token
""")


print("Building D: country + prefix4 frequency <= 50...")

DB.execute("""
CREATE TEMP TABLE block_d AS
SELECT DISTINCT
    a.s1_id,
    b.s23_id
FROM s1_base a
JOIN first_freq f
  ON a.country = f.country
 AND a.first_token = f.first_token
JOIN s23_base b
  ON a.country = b.country
 AND a.prefix4 <> ''
 AND a.prefix4 = b.prefix4
WHERE f.freq <= 50
""")


print("Building E: first token frequency <= 50...")

DB.execute("""
CREATE TEMP TABLE block_e AS
SELECT DISTINCT
    a.s1_id,
    b.s23_id
FROM s1_base a
JOIN first_freq f
  ON a.country = f.country
 AND a.first_token = f.first_token
JOIN s23_base b
  ON a.country = b.country
 AND a.first_token = b.first_token
WHERE f.freq <= 50
""")


# =========================================================
# Combine B+C+D+E with original candidates
# =========================================================

print("\nCombining current best blocks...")

DB.execute("""
CREATE TEMP TABLE current_best AS

SELECT s1_id, s23_id
FROM existing

UNION

SELECT s1_id, s23_id
FROM block_b

UNION

SELECT s1_id, s23_id
FROM block_c

UNION

SELECT s1_id, s23_id
FROM block_d

UNION

SELECT s1_id, s23_id
FROM block_e
""")

best_count = DB.execute(
    "SELECT COUNT(*) FROM current_best"
).fetchone()[0]

print(f"Current best candidate set: {best_count:,}")


# =========================================================
# 4. Remaining truth after current best
# =========================================================

print("\n[4/7] Calculating remaining missed pairs...")

DB.execute("""
CREATE TEMP TABLE remaining_missed AS
SELECT
    t.s1_id,
    t.s23_id
FROM truth t
LEFT JOIN current_best c
  ON t.s1_id = c.s1_id
 AND t.s23_id = c.s23_id
WHERE c.s1_id IS NULL
""")

remaining_count = DB.execute(
    "SELECT COUNT(*) FROM remaining_missed"
).fetchone()[0]

print(f"Remaining missed pairs: {remaining_count:,}")


# =========================================================
# 5. Build containment candidates
# =========================================================

print("\n[5/7] Building containment candidates...")

# We only benchmark containment against S1/S23 records
# whose token can actually be useful.

DB.execute("""
CREATE TEMP TABLE s23_tokens AS
SELECT DISTINCT
    s23.s23_id,
    s23.country,
    token
FROM s23_base s23,
LATERAL unnest(
    string_split(s23.name, ' ')
) AS t(token)
WHERE token <> ''
""")


DB.execute("""
CREATE TEMP TABLE s1_tokens AS
SELECT DISTINCT
    s1.s1_id,
    s1.country,
    token
FROM s1_base s1,
LATERAL unnest(
    string_split(s1.name, ' ')
) AS t(token)
WHERE token <> ''
""")


# =========================================================
# Helper for measuring each containment rule
# =========================================================

def evaluate(label, query):

    print("\n" + "-" * 70)
    print(label)
    print("-" * 70)

    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE containment AS
    {query}
    """)

    raw = DB.execute(
        "SELECT COUNT(*) FROM containment"
    ).fetchone()[0]

    DB.execute("""
    CREATE OR REPLACE TEMP TABLE additional AS
    SELECT
        c.s1_id,
        c.s23_id
    FROM containment c
    LEFT JOIN current_best b
      ON c.s1_id = b.s1_id
     AND c.s23_id = b.s23_id
    WHERE b.s1_id IS NULL
    """)

    additional = DB.execute(
        "SELECT COUNT(*) FROM additional"
    ).fetchone()[0]

    recovered = DB.execute("""
    SELECT COUNT(*)
    FROM additional a
    INNER JOIN remaining_missed r
      ON a.s1_id = r.s1_id
     AND a.s23_id = r.s23_id
    """).fetchone()[0]

    new_recall = recovered / truth_count * 100

    efficiency = (
        recovered / additional * 1_000_000
        if additional > 0 else 0
    )

    projected_candidates = best_count + additional
    projected_recall = (
        (truth_count - remaining_count + recovered)
        / truth_count
        * 100
    )

    print(f"Raw containment candidates : {raw:,}")
    print(f"Additional candidates      : {additional:,}")
    print(f"Additional true pairs      : {recovered:,}")
    print(f"Pairs / 1M candidates      : {efficiency:,.1f}")
    print(f"Projected candidates       : {projected_candidates:,}")
    print(f"Projected pair recall      : {projected_recall:.4f}%")

    return (
        raw,
        additional,
        recovered,
        efficiency,
        projected_candidates,
        projected_recall
    )


# =========================================================
# 6. Test containment rules
# =========================================================

print("\n[6/7] Testing containment rules...")


results = {}


# ---------------------------------------------------------
# A: S1 first token -> S23 name
# ---------------------------------------------------------

for limit in [50, 100]:

    results[f"S1->S23 <= {limit}"] = evaluate(
        f"S1 FIRST TOKEN -> S23 NAME | S1 frequency <= {limit}",

        f"""
        SELECT DISTINCT
            f.s1_id,
            tok.s23_id

        FROM s1_base f

        JOIN first_freq ff
          ON f.country = ff.country
         AND f.first_token = ff.first_token

        JOIN s23_tokens tok
          ON f.country = tok.country
         AND f.first_token = tok.token

        WHERE ff.freq <= {limit}
        """
    )


# ---------------------------------------------------------
# B: Either direction
# ---------------------------------------------------------

for limit in [50, 100]:

    results[f"EITHER <= {limit}"] = evaluate(
        f"EITHER DIRECTION | frequency <= {limit}",

        f"""
        SELECT DISTINCT
            f.s1_id,
            tok.s23_id

        FROM s1_base f

        JOIN first_freq ff
          ON f.country = ff.country
         AND f.first_token = ff.first_token

        JOIN s23_tokens tok
          ON f.country = tok.country
         AND f.first_token = tok.token

        WHERE ff.freq <= {limit}

        UNION

        SELECT DISTINCT
            tok.s1_id,
            f.s23_id

        FROM s23_base f

        JOIN s1_tokens tok
          ON f.country = tok.country
         AND f.first_token = tok.token

        WHERE f.first_token <> ''
        """
    )


# =========================================================
# 7. Summary
# =========================================================

print("\n" + "=" * 70)
print("FINAL CONTAINMENT OVERLAP SUMMARY")
print("=" * 70)

print(f"\nCurrent best candidates : {best_count:,}")
print(f"Current missed pairs    : {remaining_count:,}")
print()

print(
    f"{'Rule':<25}"
    f"{'Additional':>15}"
    f"{'Recovered':>15}"
    f"{'Proj Recall':>15}"
)

print("-" * 70)

for name, values in results.items():

    raw, additional, recovered, efficiency, projected, recall = values

    print(
        f"{name:<25}"
        f"{additional:>15,}"
        f"{recovered:>15,}"
        f"{recall:>14.4f}%"
    )


print("\nDone.")