import duckdb
import time

DB = "missed_pairs_analysis.duckdb"

con = duckdb.connect(DB)

# Keep memory usage safe for your laptop
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='1500MB'")

print("=" * 70)
print("10K SAMPLE - CONTAINMENT + LENGTH + TOKEN COUNT VOLUME")
print("=" * 70)

# ---------------------------------------------------------
# S1
# ---------------------------------------------------------

print("\nPreparing S1...")

con.execute("""
CREATE OR REPLACE TEMP TABLE s1_norm AS
SELECT
    source1_entity_id,
    lower(trim(s1_country)) AS country,
    lower(trim(s1_name)) AS name,

    split_part(
        regexp_replace(
            lower(trim(s1_name)),
            '[^a-z0-9]+',
            ' ',
            'g'
        ),
        ' ',
        1
    ) AS first_token,

    length(
        regexp_replace(
            lower(trim(s1_name)),
            '[^a-z0-9]+',
            '',
            'g'
        )
    ) AS name_len,

    array_length(
        string_split(
            regexp_replace(
                lower(trim(s1_name)),
                '[^a-z0-9]+',
                ' ',
                'g'
            ),
            ' '
        )
    ) AS token_count

FROM s1

WHERE trim(s1_country) <> ''
  AND trim(s1_name) <> '';
""")

# ---------------------------------------------------------
# 10K SAMPLE
# ---------------------------------------------------------

print("Creating 10K sample...")

con.execute("""
CREATE OR REPLACE TEMP TABLE s1_sample AS
SELECT *
FROM s1_norm
LIMIT 10000;
""")

sample_count = con.execute("""
SELECT COUNT(*)
FROM s1_sample
""").fetchone()[0]

print(f"Sample S1 entities : {sample_count:,}")

# ---------------------------------------------------------
# S23
# ---------------------------------------------------------

print("\nPreparing S23...")

con.execute("""
CREATE OR REPLACE TEMP TABLE s23_norm AS
SELECT
    matched_entity_id,
    lower(trim(matched_country)) AS country,
    lower(trim(matched_name)) AS name,

    length(
        regexp_replace(
            lower(trim(matched_name)),
            '[^a-z0-9]+',
            '',
            'g'
        )
    ) AS name_len,

    array_length(
        string_split(
            regexp_replace(
                lower(trim(matched_name)),
                '[^a-z0-9]+',
                ' ',
                'g'
            ),
            ' '
        )
    ) AS token_count

FROM s23

WHERE trim(matched_country) <> ''
  AND trim(matched_name) <> '';
""")

# ---------------------------------------------------------
# S23 TOKEN INDEX
# ---------------------------------------------------------

print("Creating S23 token index...")

start = time.time()

con.execute("""
CREATE OR REPLACE TEMP TABLE s23_tokens AS
SELECT DISTINCT
    s.matched_entity_id,
    s.country,
    s.name_len,
    s.token_count,
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

print(f"Token index ready in {time.time() - start:.1f}s")

# ---------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------

print("\nRunning 10K candidate-volume benchmark...")

start = time.time()

count = con.execute("""
SELECT COUNT(*)

FROM s1_sample s1

JOIN s23_tokens s23

    ON s1.country = s23.country

   AND s1.first_token = s23.token

   AND abs(s1.name_len - s23.name_len) <= 4

   AND abs(s1.token_count - s23.token_count) <= 2

WHERE s1.first_token <> '';
""").fetchone()[0]

elapsed = time.time() - start

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"Sample S1              : {sample_count:,}")
print(f"Raw candidates         : {count:,}")

if sample_count > 0:
    candidates_per_s1 = count / sample_count
else:
    candidates_per_s1 = 0

print(f"Candidates / S1        : {candidates_per_s1:,.2f}")

# Full S1 count from the actual database
full_s1_count = con.execute("""
SELECT COUNT(*)
FROM s1_norm
""").fetchone()[0]

estimated_full = (
    count * full_s1_count / sample_count
    if sample_count > 0
    else 0
)

print(f"Full S1 entities       : {full_s1_count:,}")
print(f"Estimated full volume : {estimated_full:,.0f}")

# Existing candidate set from previous experiments
existing = 31_327_226

print(f"Existing candidate set: {existing:,}")

if existing > 0:
    ratio = estimated_full / existing
else:
    ratio = 0

print(f"Estimated ratio        : {ratio:.2f}x")

print(f"Query time             : {elapsed:.1f}s")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

con.close()