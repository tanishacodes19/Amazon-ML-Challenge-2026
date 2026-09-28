import duckdb
import time

DB = "missed_pairs_analysis.duckdb"

con = duckdb.connect(DB)

print("=" * 70)
print("HOUSE + CONTAINMENT CANDIDATE VOLUME")
print("=" * 70)

# ---------------------------------------------------------
# Build normalized S1 / S23 temporary tables
# ---------------------------------------------------------

print("\nPreparing S1...")

con.execute("""
CREATE OR REPLACE TEMP TABLE s1_norm AS
SELECT
    source1_entity_id,

    lower(trim(s1_country)) AS country,

    lower(trim(s1_name)) AS name,

    regexp_extract(
        lower(trim(s1_address)),
        '([0-9]+)',
        1
    ) AS house,

    split_part(
        regexp_replace(
            lower(trim(s1_name)),
            '[^a-z0-9]+',
            ' ',
            'g'
        ),
        ' ',
        1
    ) AS first_token

FROM s1
WHERE trim(s1_name) <> ''
  AND trim(s1_country) <> '';
""")

print("Preparing S23...")

con.execute("""
CREATE OR REPLACE TEMP TABLE s23_norm AS
SELECT
    matched_entity_id,

    lower(trim(matched_country)) AS country,

    lower(trim(matched_name)) AS name,

    regexp_extract(
        lower(trim(matched_address)),
        '([0-9]+)',
        1
    ) AS house

FROM s23
WHERE trim(matched_name) <> ''
  AND trim(matched_country) <> '';
""")

# ---------------------------------------------------------
# Create S23 token index.
# We only need names containing a token.
# ---------------------------------------------------------

print("Creating S23 token index...")

con.execute("""
CREATE OR REPLACE TEMP TABLE s23_tokens AS
SELECT DISTINCT
    s.matched_entity_id,
    s.country,
    s.house,
    token
FROM s23_norm s,
LATERAL unnest(
    string_split(
        regexp_replace(
            s.name,
            '[^a-z0-9]+',
            ' ',
            'g'
        ),
        ' '
    )
) AS t(token)
WHERE token <> '';
""")

print("S23 token index ready.")

# ---------------------------------------------------------
# BLOCK 1
# S1 first token -> S23 name
# same country + same house
# ---------------------------------------------------------

print("\nBenchmarking BLOCK 1...")
start = time.time()

count1 = con.execute("""
SELECT COUNT(*)
FROM s1_norm s1
JOIN s23_tokens s23
    ON s1.country = s23.country
   AND s1.house = s23.house
   AND s1.first_token = s23.token
WHERE s1.house <> ''
  AND s1.first_token <> ''
""").fetchone()[0]

print(f"BLOCK 1 candidates : {count1:,}")
print(f"Time: {time.time() - start:.1f}s")

# ---------------------------------------------------------
# BLOCK 2
# S23 first token -> S1 name
# same country + same house
# ---------------------------------------------------------

print("\nBenchmarking BLOCK 2...")
start = time.time()

con.execute("""
CREATE OR REPLACE TEMP TABLE s23_first AS
SELECT
    matched_entity_id,
    country,
    house,

    split_part(
        regexp_replace(
            name,
            '[^a-z0-9]+',
            ' ',
            'g'
        ),
        ' ',
        1
    ) AS first_token

FROM s23_norm;
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE s1_tokens AS
SELECT DISTINCT
    s.source1_entity_id,
    s.country,
    s.house,
    token
FROM s1_norm s,
LATERAL unnest(
    string_split(
        regexp_replace(
            s.name,
            '[^a-z0-9]+',
            ' ',
            'g'
        ),
        ' '
    )
) AS t(token)
WHERE token <> '';
""")

count2 = con.execute("""
SELECT COUNT(*)
FROM s23_first s23
JOIN s1_tokens s1
    ON s1.country = s23.country
   AND s1.house = s23.house
   AND s1.token = s23.first_token
WHERE s23.house <> ''
  AND s23.first_token <> ''
""").fetchone()[0]

print(f"BLOCK 2 candidates : {count2:,}")
print(f"Time: {time.time() - start:.1f}s")

# ---------------------------------------------------------
# Combined approximate volume
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Block 1 : {count1:,}")
print(f"Block 2 : {count2:,}")

print(f"Combined raw upper bound : {count1 + count2:,}")

con.close()