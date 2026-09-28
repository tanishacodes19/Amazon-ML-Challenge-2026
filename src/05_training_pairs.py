from pathlib import Path
import duckdb

BASE_DIR = Path(".")
GROUND_TRUTH = BASE_DIR / "ground_truth_pairs.tsv"
CANDIDATES = BASE_DIR / "training_candidate_pairs.tsv"

OUTPUT_DIR = BASE_DIR / "training_data"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT = OUTPUT_DIR / "training_pairs.tsv"
DB_FILE = BASE_DIR / "training_pairs.duckdb"

MAX_POSITIVES_PER_S1 = 3
MAX_NEGATIVES_PER_S1 = 5

print("=" * 70)
print("BUILDING TRAINING PAIRS - DISK SAFE")
print("=" * 70)

con = duckdb.connect(str(DB_FILE))

# Important for your 8 GB RAM machine
con.execute("SET memory_limit='1500MB'")
con.execute("SET threads=1")
con.execute("SET preserve_insertion_order=false")
con.execute("SET temp_directory='training_duckdb_temp'")

ground_truth_path = str(GROUND_TRUTH.resolve()).replace("\\", "/")
candidates_path = str(CANDIDATES.resolve()).replace("\\", "/")
output_path = str(OUTPUT.resolve()).replace("\\", "/")

print("\nDuckDB settings:")
print("  Memory limit : 1500 MB")
print("  Threads      : 1")
print("  Disk spilling : enabled")

# ------------------------------------------------------------
# STEP 1: Create DuckDB tables on disk
# ------------------------------------------------------------

print("\n[1/5] Loading ground truth into DuckDB...")

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

print("Ground truth loaded.")

# ------------------------------------------------------------
# STEP 2: Process candidates using DuckDB table
# ------------------------------------------------------------

print("\n[2/5] Loading candidate pairs into DuckDB...")

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

print("Candidate pairs loaded.")

# ------------------------------------------------------------
# STEP 3: Label candidates
# ------------------------------------------------------------

print("\n[3/5] Labelling candidate pairs...")

con.execute("""
CREATE OR REPLACE TABLE labelled AS
SELECT
    c.source1_entity_id,
    c.matched_entity_id,
    CASE
        WHEN g.matched_entity_id IS NOT NULL THEN 1
        ELSE 0
    END AS label
FROM candidates c
LEFT JOIN ground_truth g
    ON c.source1_entity_id = g.source1_entity_id
    AND c.matched_entity_id = g.matched_entity_id
""")

print("Labelling complete.")

# ------------------------------------------------------------
# STEP 4: Sample using a two-stage approach
# ------------------------------------------------------------

print("\n[4/5] Sampling positives and negatives...")

# Positives are already relatively rare.
# We keep up to 3 true matches per S1.
con.execute(f"""
CREATE OR REPLACE TABLE sampled_positives AS
SELECT
    source1_entity_id,
    matched_entity_id,
    label
FROM (
    SELECT
        source1_entity_id,
        matched_entity_id,
        label,
        ROW_NUMBER() OVER (
            PARTITION BY source1_entity_id
            ORDER BY matched_entity_id
        ) AS rn
    FROM labelled
    WHERE label = 1
)
WHERE rn <= {MAX_POSITIVES_PER_S1}
""")

print("Positive sampling complete.")

# Instead of a massive window operation over ALL negatives,
# use a deterministic hash to select a small subset first.
con.execute("""
CREATE OR REPLACE TABLE negative_pool AS
SELECT
    source1_entity_id,
    matched_entity_id,
    label
FROM labelled
WHERE label = 0
  AND MOD(
        ABS(HASH(matched_entity_id)),
        10
      ) = 0
""")

print("Negative pool created.")

con.execute(f"""
CREATE OR REPLACE TABLE sampled_negatives AS
SELECT
    source1_entity_id,
    matched_entity_id,
    label
FROM (
    SELECT
        source1_entity_id,
        matched_entity_id,
        label,
        ROW_NUMBER() OVER (
            PARTITION BY source1_entity_id
            ORDER BY matched_entity_id
        ) AS rn
    FROM negative_pool
)
WHERE rn <= {MAX_NEGATIVES_PER_S1}
""")

print("Negative sampling complete.")

# ------------------------------------------------------------
# STEP 5: Write final training pairs
# ------------------------------------------------------------

print("\n[5/5] Writing training_pairs.tsv...")

con.execute(f"""
COPY (
    SELECT
        source1_entity_id,
        matched_entity_id,
        label
    FROM sampled_positives

    UNION ALL

    SELECT
        source1_entity_id,
        matched_entity_id,
        label
    FROM sampled_negatives
)
TO '{output_path}'
(
    DELIMITER '\\t',
    HEADER true
)
""")

# ------------------------------------------------------------
# CHECK
# ------------------------------------------------------------

result = con.execute(f"""
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS positives,
    SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS negatives,
    COUNT(DISTINCT source1_entity_id) AS s1_entities
FROM read_csv(
    '{output_path}',
    delim='\\t',
    header=true
)
""").fetchone()

total, positives, negatives, s1_entities = result

print("\n" + "=" * 70)
print("TRAINING PAIRS COMPLETE")
print("=" * 70)

print(f"Total training pairs : {total:,}")
print(f"Positive pairs       : {positives:,}")
print(f"Negative pairs       : {negatives:,}")
print(f"S1 entities          : {s1_entities:,}")
print(f"Saved to             : {OUTPUT}")

con.close()