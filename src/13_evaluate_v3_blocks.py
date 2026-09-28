import os
import duckdb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

OLD_CANDIDATES = os.path.join(
    BASE, "training_candidate_pairs.tsv"
)

GROUND_TRUTH = os.path.join(
    BASE, "ground_truth_pairs.tsv"
)

BLOCK_DIR = os.path.join(
    BASE, "training_data", "candidate_v3_blocks"
)

DB_FILE = os.path.join(
    BASE, "v3_block_evaluation.duckdb"
)

con = duckdb.connect(DB_FILE)

con.execute("PRAGMA memory_limit='4GB'")
con.execute("PRAGMA threads=2")

print("Loading existing candidates...")

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
""", [OLD_CANDIDATES])

print(
    "Old candidates:",
    con.execute("SELECT COUNT(*) FROM old_candidates").fetchone()[0]
)


print("\nLoading ground truth...")

con.execute("""
CREATE OR REPLACE TABLE ground_truth AS
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
""", [GROUND_TRUTH])

true_pairs = con.execute(
    "SELECT COUNT(*) FROM ground_truth"
).fetchone()[0]

print("True pairs:", true_pairs)


# ---------------------------------------------------------
# Load V3 blocks
# ---------------------------------------------------------

blocks = [
    "country_prefix4.tsv",
    "country_first_token.tsv",
    "house_prefix4.tsv",
    "house_first_token.tsv",
]

for i, filename in enumerate(blocks, 1):

    path = os.path.join(BLOCK_DIR, filename)

    table = f"block_{i}"

    print(f"\nLoading {filename}...")

    con.execute(f"""
    CREATE OR REPLACE TABLE {table} AS
    SELECT DISTINCT
        source1_entity_id,
        matched_entity_id
    FROM read_csv(
        ?,
        delim='\t',
        header=true,
        columns={{
            'source1_entity_id':'VARCHAR',
            'matched_entity_id':'VARCHAR'
        }}
    )
    """, [path])

    count = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print("Rows:", count)


# ---------------------------------------------------------
# Combine V3 blocks
# ---------------------------------------------------------

print("\nCombining V3 blocks...")

con.execute("""
CREATE OR REPLACE TABLE v3_blocks AS

SELECT source1_entity_id, matched_entity_id
FROM block_1

UNION

SELECT source1_entity_id, matched_entity_id
FROM block_2

UNION

SELECT source1_entity_id, matched_entity_id
FROM block_3

UNION

SELECT source1_entity_id, matched_entity_id
FROM block_4
""")

v3_count = con.execute(
    "SELECT COUNT(*) FROM v3_blocks"
).fetchone()[0]

print("Unique V3 block candidates:", v3_count)


# ---------------------------------------------------------
# Find genuinely NEW candidates
# ---------------------------------------------------------

print("\nFinding NEW candidates...")

con.execute("""
CREATE OR REPLACE TABLE new_candidates AS

SELECT
    v.source1_entity_id,
    v.matched_entity_id

FROM v3_blocks v

ANTI JOIN old_candidates o

ON v.source1_entity_id = o.source1_entity_id
AND v.matched_entity_id = o.matched_entity_id
""")

new_count = con.execute(
    "SELECT COUNT(*) FROM new_candidates"
).fetchone()[0]

print("Actually NEW candidates:", new_count)


# ---------------------------------------------------------
# Existing recall
# ---------------------------------------------------------

old_recovered = con.execute("""
SELECT COUNT(*)
FROM ground_truth g
SEMI JOIN old_candidates c
ON g.source1_entity_id = c.source1_entity_id
AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

old_recall = old_recovered / true_pairs

print("\n==============================")
print("BASELINE")
print("==============================")
print("Recovered:", old_recovered)
print("Recall:", f"{old_recall:.4%}")


# ---------------------------------------------------------
# New true matches
# ---------------------------------------------------------

new_recovered = con.execute("""
SELECT COUNT(*)
FROM ground_truth g
SEMI JOIN new_candidates n
ON g.source1_entity_id = n.source1_entity_id
AND g.matched_entity_id = n.matched_entity_id
""").fetchone()[0]

print("\n==============================")
print("V3 ADDITION")
print("==============================")
print("New candidates:", new_count)
print("New true pairs recovered:", new_recovered)


# ---------------------------------------------------------
# Combined recall
# ---------------------------------------------------------

combined_recovered = old_recovered + new_recovered

combined_recall = combined_recovered / true_pairs

print("\n==============================")
print("COMBINED")
print("==============================")
print("Total candidates:",
      con.execute("""
      SELECT COUNT(*)
      FROM (
          SELECT source1_entity_id, matched_entity_id
          FROM old_candidates

          UNION

          SELECT source1_entity_id, matched_entity_id
          FROM new_candidates
      )
      """).fetchone()[0]
)

print("Recovered true pairs:", combined_recovered)

print(
    "Pair recall:",
    f"{combined_recall:.4%}"
)

print(
    "Recall gain:",
    f"{(combined_recall - old_recall):.4%}"
)

print(
    "New candidates / new true pairs:",
    f"{new_count / max(new_recovered,1):.2f}"
)

print("\nDONE.")

con.close()