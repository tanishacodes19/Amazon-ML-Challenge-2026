from pathlib import Path
import duckdb

BASE_DIR = Path(".")

GROUND_TRUTH = BASE_DIR / "ground_truth_pairs.tsv"
CANDIDATES = BASE_DIR / "training_candidate_pairs.tsv"

DB_FILE = BASE_DIR / "blocking_recall.duckdb"

print("=" * 70)
print("BLOCKING RECALL EVALUATION")
print("=" * 70)

ground_truth_path = str(
    GROUND_TRUTH.resolve()
).replace("\\", "/")

candidates_path = str(
    CANDIDATES.resolve()
).replace("\\", "/")

con = duckdb.connect(str(DB_FILE))

# Safe for your 8 GB machine
con.execute("SET memory_limit='1500MB'")
con.execute("SET threads=1")
con.execute("SET preserve_insertion_order=false")
con.execute("SET temp_directory='blocking_recall_temp'")

print("\nDuckDB:")
print("  Memory : 1500 MB")
print("  Threads: 1")
print("  Disk spill: enabled")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\n[1/4] Loading ground truth...")

con.execute(f"""
CREATE OR REPLACE TABLE ground_truth AS
SELECT DISTINCT
    source1_entity_id,
    matched_entity_id
FROM read_csv(
    '{ground_truth_path}',
    delim='\\t',
    header=true,
    columns={{
        'source1_entity_id': 'VARCHAR',
        'matched_entity_id': 'VARCHAR'
    }}
)
""")

gt_count = con.execute("""
SELECT COUNT(*)
FROM ground_truth
""").fetchone()[0]

print(f"Ground-truth positive pairs: {gt_count:,}")


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("\n[2/4] Loading candidate pairs...")

con.execute(f"""
CREATE OR REPLACE TABLE candidates AS
SELECT DISTINCT
    source1_entity_id,
    candidate_entity_id AS matched_entity_id
FROM read_csv(
    '{candidates_path}',
    delim='\\t',
    header=true,
    columns={{
        'source1_entity_id': 'VARCHAR',
        'candidate_entity_id': 'VARCHAR'
    }}
)
""")

candidate_count = con.execute("""
SELECT COUNT(*)
FROM candidates
""").fetchone()[0]

print(f"Unique candidate pairs: {candidate_count:,}")


# ============================================================
# RECALL
# ============================================================

print("\n[3/4] Measuring candidate recall...")

result = con.execute("""
SELECT
    COUNT(*) AS total_true_pairs,

    SUM(
        CASE
            WHEN c.source1_entity_id IS NOT NULL
            THEN 1
            ELSE 0
        END
    ) AS recovered_true_pairs

FROM ground_truth g

LEFT JOIN candidates c
    ON g.source1_entity_id = c.source1_entity_id
    AND g.matched_entity_id = c.matched_entity_id
""").fetchone()

total_true = result[0]
recovered = result[1]
missed = total_true - recovered

recall = (
    recovered / total_true
    if total_true > 0
    else 0
)


# ============================================================
# S1-LEVEL COVERAGE
# ============================================================

print("\n[4/4] Measuring S1-level coverage...")

s1_result = con.execute("""
WITH truth AS (
    SELECT
        source1_entity_id,
        COUNT(*) AS true_matches
    FROM ground_truth
    GROUP BY source1_entity_id
),

recovered AS (
    SELECT
        g.source1_entity_id,
        COUNT(*) AS recovered_matches
    FROM ground_truth g
    INNER JOIN candidates c
        ON g.source1_entity_id = c.source1_entity_id
        AND g.matched_entity_id = c.matched_entity_id
    GROUP BY g.source1_entity_id
)

SELECT
    COUNT(*) AS total_matching_s1,

    SUM(
        CASE
            WHEN COALESCE(r.recovered_matches, 0) > 0
            THEN 1
            ELSE 0
        END
    ) AS s1_with_at_least_one_match,

    SUM(
        CASE
            WHEN COALESCE(r.recovered_matches, 0) = t.true_matches
            THEN 1
            ELSE 0
        END
    ) AS s1_with_all_matches_recovered

FROM truth t

LEFT JOIN recovered r
    ON t.source1_entity_id = r.source1_entity_id
""").fetchone()

total_matching_s1 = s1_result[0]
s1_some = s1_result[1]
s1_all = s1_result[2]

s1_some_pct = (
    s1_some / total_matching_s1
    if total_matching_s1 > 0
    else 0
)

s1_all_pct = (
    s1_all / total_matching_s1
    if total_matching_s1 > 0
    else 0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BLOCKING RECALL RESULT")
print("=" * 70)

print(
    f"True positive pairs       : {total_true:,}"
)

print(
    f"Recovered by candidates   : {recovered:,}"
)

print(
    f"Missed true pairs         : {missed:,}"
)

print(
    f"\nPAIR RECALL                : {recall:.6%}"
)

print("\nS1-level coverage:")

print(
    f"S1 with >=1 recovered match"
    f" : {s1_some:,} / {total_matching_s1:,}"
    f" ({s1_some_pct:.6%})"
)

print(
    f"S1 with ALL matches recovered"
    f" : {s1_all:,} / {total_matching_s1:,}"
    f" ({s1_all_pct:.6%})"
)

print("\nCandidate expansion:")

print(
    f"Candidates / true pairs   : "
    f"{candidate_count / total_true:.2f}x"
)

print("\n" + "=" * 70)
print("BLOCKING EVALUATION COMPLETE")
print("=" * 70)

con.close()