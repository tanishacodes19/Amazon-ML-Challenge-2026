import os
import duckdb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

OLD = os.path.join(BASE, "training_candidate_pairs.tsv")
V3_DB = os.path.join(BASE, "v3_block_evaluation.duckdb")

DB = os.path.join(BASE, "candidate_distribution.duckdb")

con = duckdb.connect(DB)

con.execute("PRAGMA memory_limit='4GB'")
con.execute("PRAGMA threads=2")

print("=" * 60)
print("BASELINE CANDIDATE DISTRIBUTION")
print("=" * 60)

con.execute("""
CREATE OR REPLACE TABLE old_candidates AS
SELECT
    source1_entity_id,
    matched_entity_id
FROM read_csv(
    ?,
    delim='\t',
    header=true,
    columns={
        'source1_entity_id':'VARCHAR',
        'matched_entity_id':'VARCHAR'
    }
)
""", [OLD])

con.execute("""
CREATE OR REPLACE TABLE old_counts AS
SELECT
    source1_entity_id,
    COUNT(*) AS candidate_count
FROM old_candidates
GROUP BY source1_entity_id
""")

print("\nBaseline candidate statistics:")

stats = con.execute("""
SELECT
    MIN(candidate_count) AS minimum,
    AVG(candidate_count) AS mean,
    MEDIAN(candidate_count) AS median,
    QUANTILE_CONT(candidate_count, 0.75) AS p75,
    QUANTILE_CONT(candidate_count, 0.90) AS p90,
    QUANTILE_CONT(candidate_count, 0.95) AS p95,
    QUANTILE_CONT(candidate_count, 0.99) AS p99,
    MAX(candidate_count) AS maximum
FROM old_counts
""").fetchone()

names = [
    "minimum",
    "mean",
    "median",
    "p75",
    "p90",
    "p95",
    "p99",
    "maximum"
]

for name, value in zip(names, stats):
    print(f"{name:>10}: {value}")


# ---------------------------------------------------------
# Candidate buckets
# ---------------------------------------------------------

print("\nBaseline candidate-count distribution:")

rows = con.execute("""
SELECT
    CASE
        WHEN candidate_count = 0 THEN '0'
        WHEN candidate_count <= 5 THEN '1-5'
        WHEN candidate_count <= 10 THEN '6-10'
        WHEN candidate_count <= 20 THEN '11-20'
        WHEN candidate_count <= 50 THEN '21-50'
        WHEN candidate_count <= 100 THEN '51-100'
        WHEN candidate_count <= 250 THEN '101-250'
        WHEN candidate_count <= 500 THEN '251-500'
        WHEN candidate_count <= 1000 THEN '501-1000'
        WHEN candidate_count <= 5000 THEN '1001-5000'
        ELSE '5000+'
    END AS bucket,
    COUNT(*) AS s1_entities,
    SUM(candidate_count) AS total_candidates
FROM old_counts
GROUP BY bucket
ORDER BY
    MIN(candidate_count)
""").fetchall()

for row in rows:
    print(row)


# =========================================================
# V3
# =========================================================

print("\n" + "=" * 60)
print("V3 CANDIDATE DISTRIBUTION")
print("=" * 60)

# V3 blocks already exist in v3_block_evaluation.duckdb.
# Attach that database.

con.execute(f"""
ATTACH '{V3_DB}' AS v3
""")

con.execute("""
CREATE OR REPLACE TABLE v3_new_counts AS
SELECT
    source1_entity_id,
    COUNT(*) AS new_candidate_count
FROM v3.new_candidates
GROUP BY source1_entity_id
""")

con.execute("""
CREATE OR REPLACE TABLE combined_counts AS
SELECT
    COALESCE(o.source1_entity_id, n.source1_entity_id)
        AS source1_entity_id,

    COALESCE(o.candidate_count, 0)
        AS old_count,

    COALESCE(n.new_candidate_count, 0)
        AS new_count,

    COALESCE(o.candidate_count, 0)
      + COALESCE(n.new_candidate_count, 0)
        AS combined_count

FROM old_counts o

FULL OUTER JOIN v3_new_counts n
ON o.source1_entity_id = n.source1_entity_id
""")

stats = con.execute("""
SELECT
    MIN(combined_count),
    AVG(combined_count),
    MEDIAN(combined_count),
    QUANTILE_CONT(combined_count, 0.75),
    QUANTILE_CONT(combined_count, 0.90),
    QUANTILE_CONT(combined_count, 0.95),
    QUANTILE_CONT(combined_count, 0.99),
    MAX(combined_count)
FROM combined_counts
""").fetchone()

print("\nV3 combined candidate statistics:")

for name, value in zip(names, stats):
    print(f"{name:>10}: {value}")


print("\nV3 candidate-count distribution:")

rows = con.execute("""
SELECT
    CASE
        WHEN combined_count = 0 THEN '0'
        WHEN combined_count <= 5 THEN '1-5'
        WHEN combined_count <= 10 THEN '6-10'
        WHEN combined_count <= 20 THEN '11-20'
        WHEN combined_count <= 50 THEN '21-50'
        WHEN combined_count <= 100 THEN '51-100'
        WHEN combined_count <= 250 THEN '101-250'
        WHEN combined_count <= 500 THEN '251-500'
        WHEN combined_count <= 1000 THEN '501-1000'
        WHEN combined_count <= 5000 THEN '1001-5000'
        ELSE '5000+'
    END AS bucket,
    COUNT(*) AS s1_entities,
    SUM(combined_count) AS total_candidates
FROM combined_counts
GROUP BY bucket
ORDER BY
    MIN(combined_count)
""").fetchall()

for row in rows:
    print(row)


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

con.close()