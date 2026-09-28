import duckdb
import os
import time

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

BLOCK = os.path.join(
    BASE,
    "training_data",
    "v5_candidate_blocks",
    "containment_length4.tsv"
)

GROUND_TRUTH = os.path.join(
    BASE,
    "ground_truth_pairs.tsv"
)

print("=" * 70)
print("EVALUATE TIGHT CONTAINMENT BLOCK")
print("=" * 70)

con = duckdb.connect()

con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='1500MB'")

print("\nLoading candidate block...")

start = time.time()

con.execute(f"""
CREATE OR REPLACE TEMP TABLE candidates AS

SELECT DISTINCT
    column0 AS source1_entity_id,
    column1 AS candidate_entity_id

FROM read_csv(
    '{BLOCK.replace("\\", "/")}',
    delim='\\t',
    header=true,
    columns={{
        'column0':'VARCHAR',
        'column1':'VARCHAR'
    }}
)
""")

candidate_count = con.execute("""
SELECT COUNT(*)
FROM candidates
""").fetchone()[0]

print(
    f"Candidates: {candidate_count:,}"
)

print(
    f"Load time: {time.time() - start:.1f}s"
)

print("\nLoading ground truth...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE truth AS

SELECT DISTINCT
    column0 AS source1_entity_id,
    column1 AS matched_entity_id

FROM read_csv(
    '{GROUND_TRUTH.replace("\\", "/")}',
    delim='\\t',
    header=true,
    columns={{
        'column0':'VARCHAR',
        'column1':'VARCHAR'
    }}
)
""")

truth_count = con.execute("""
SELECT COUNT(*)
FROM truth
""").fetchone()[0]

print(
    f"Ground-truth pairs: {truth_count:,}"
)

print("\nChecking recovered true pairs...")

start = time.time()

recovered = con.execute("""
SELECT COUNT(*)

FROM candidates c

INNER JOIN truth t

    ON c.source1_entity_id =
       t.source1_entity_id

   AND c.candidate_entity_id =
       t.matched_entity_id
""").fetchone()[0]

elapsed = time.time() - start

recall = recovered / truth_count

efficiency = recovered / candidate_count

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Candidate pairs       : {candidate_count:,}"
)

print(
    f"True pairs recovered   : {recovered:,}"
)

print(
    f"Pair recall            : {recall:.6%}"
)

print(
    f"True / candidate       : {efficiency:.8%}"
)

print(
    f"True pairs per 1M     : "
    f"{efficiency * 1_000_000:,.1f}"
)

print(
    f"Join time              : {elapsed:.1f}s"
)

print("=" * 70)

con.close()