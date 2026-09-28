import duckdb

DB = duckdb.connect(
    "missed_pairs_analysis.duckdb",
    read_only=True
)

DB.execute("PRAGMA threads=2")
DB.execute("PRAGMA memory_limit='1500MB'")

print("=" * 70)
print("CONTAINMENT CANDIDATE-VOLUME BENCHMARK")
print("=" * 70)


# =========================================================
# 1. Load S1 / S23
# =========================================================

print("\n[1/6] Loading source data...")

DB.execute("""
CREATE TEMP TABLE s1 AS
SELECT
    source1_entity_id AS s1_id,
    lower(trim(s1_name)) AS name,
    s1_country AS country
FROM s1
""")

DB.execute("""
CREATE TEMP TABLE s23 AS
SELECT
    matched_entity_id AS s23_id,
    lower(trim(matched_name)) AS name,
    matched_country AS country
FROM s23
""")


# =========================================================
# 2. First-token frequency
# =========================================================

print("[2/6] Calculating frequencies...")

DB.execute("""
CREATE TEMP TABLE s1_first AS
SELECT
    s1_id,
    country,
    split_part(name, ' ', 1) AS first_token
FROM s1
WHERE name <> ''
""")

DB.execute("""
CREATE TEMP TABLE s23_first AS
SELECT
    s23_id,
    country,
    split_part(name, ' ', 1) AS first_token
FROM s23
WHERE name <> ''
""")


DB.execute("""
CREATE TEMP TABLE s1_freq AS
SELECT
    country,
    first_token,
    COUNT(*) AS freq
FROM s1_first
WHERE first_token <> ''
GROUP BY country, first_token
""")


DB.execute("""
CREATE TEMP TABLE s23_freq AS
SELECT
    country,
    first_token,
    COUNT(*) AS freq
FROM s23_first
WHERE first_token <> ''
GROUP BY country, first_token
""")


print("Frequency tables ready.")


# =========================================================
# 3. Candidate-volume functions
# =========================================================

def s1_to_s23(limit):

    print("\n" + "-" * 70)
    print(
        f"S1 FIRST TOKEN -> S23 NAME | "
        f"S1 frequency <= {limit}"
    )
    print("-" * 70)

    # Count candidate pairs without storing them.
    #
    # Each S23 name is tokenized and joined to S1 first tokens.
    # DISTINCT is counted through grouped IDs.

    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE volume_a AS

    SELECT
        f.country,
        f.first_token,
        f.freq,

        COUNT(DISTINCT s1.s1_id)
            AS s1_count,

        COUNT(DISTINCT tok.s23_id)
            AS s23_count

    FROM s1_freq f

    JOIN s1_first s1
      ON s1.country = f.country
     AND s1.first_token = f.first_token

    JOIN (
        SELECT DISTINCT
            s23_id,
            country,
            token

        FROM s23,
        LATERAL unnest(
            string_split(name, ' ')
        ) AS t(token)

        WHERE token <> ''
    ) tok

      ON tok.country = f.country
     AND tok.token = f.first_token

    WHERE f.freq <= {limit}

    GROUP BY
        f.country,
        f.first_token,
        f.freq
    """)

    count = DB.execute("""
        SELECT
            COALESCE(
                SUM(s1_count * s23_count),
                0
            )
        FROM volume_a
    """).fetchone()[0]

    print(f"Estimated candidate pairs: {count:,}")

    return count


def s23_to_s1(limit):

    print("\n" + "-" * 70)
    print(
        f"S23 FIRST TOKEN -> S1 NAME | "
        f"S23 frequency <= {limit}"
    )
    print("-" * 70)

    DB.execute(f"""
    CREATE OR REPLACE TEMP TABLE volume_b AS

    SELECT
        f.country,
        f.first_token,
        f.freq,

        COUNT(DISTINCT s23.s23_id)
            AS s23_count,

        COUNT(DISTINCT tok.s1_id)
            AS s1_count

    FROM s23_freq f

    JOIN s23_first s23
      ON s23.country = f.country
     AND s23.first_token = f.first_token

    JOIN (
        SELECT DISTINCT
            s1_id,
            country,
            token

        FROM s1,
        LATERAL unnest(
            string_split(name, ' ')
        ) AS t(token)

        WHERE token <> ''
    ) tok

      ON tok.country = f.country
     AND tok.token = f.first_token

    WHERE f.freq <= {limit}

    GROUP BY
        f.country,
        f.first_token,
        f.freq
    """)

    count = DB.execute("""
        SELECT
            COALESCE(
                SUM(s23_count * s1_count),
                0
            )
        FROM volume_b
    """).fetchone()[0]

    print(f"Estimated candidate pairs: {count:,}")

    return count


# =========================================================
# 4. Run volume tests
# =========================================================

print("\n[3/6] Testing candidate volume...")


results = {}


for limit in [25, 50, 100, 250]:

    results[
        f"S1 -> S23 <= {limit}"
    ] = s1_to_s23(limit)


for limit in [25, 50, 100, 250]:

    results[
        f"S23 -> S1 <= {limit}"
    ] = s23_to_s1(limit)


# =========================================================
# 5. Summary
# =========================================================

print("\n" + "=" * 70)
print("CANDIDATE VOLUME SUMMARY")
print("=" * 70)

print()

for name, count in results.items():

    print(
        f"{name:<25} "
        f"{count:>18,} candidates"
    )


# =========================================================
# 6. Compare with true-pair recovery
# =========================================================

print("\n" + "=" * 70)
print("EFFICIENCY REFERENCE")
print("=" * 70)

print("""
The earlier remaining-pair analysis found:

S1 -> S23:
  <= 25 : 157,135 true pairs
  <= 50 : 240,804 true pairs
  <=100 : 380,286 true pairs
  <=250 : 481,757 true pairs

S23 -> S1:
  <= 25 : 10,923 true pairs
  <= 50 : 28,861 true pairs
  <=100 : 54,458 true pairs
  <=250 : 133,837 true pairs

Either direction:
  <= 25 : 160,679 true pairs
  <= 50 : 245,552 true pairs
  <=100 : 387,041 true pairs
  <=250 : 490,473 true pairs
""")

print("\nDone.")

DB.close()