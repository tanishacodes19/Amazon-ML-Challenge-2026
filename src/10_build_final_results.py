import os
import duckdb
import polars as pl


# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

S1_FILE = os.path.join(
    BASE,
    "normalized_data",
    "test_source1_normalized.tsv"
)

SCORED_FILE = os.path.join(
    BASE,
    "test_scored_candidates.tsv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "output"
)

MATCHING_FILE = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv"
)

THRESHOLD_FILE = os.path.join(
    BASE,
    "model",
    "best_threshold_v2.txt"
)


# ============================================================
# LOAD THRESHOLD
# ============================================================

print("=" * 70)
print("BUILDING FINAL MATCHING RESULTS")
print("=" * 70)

with open(THRESHOLD_FILE, "r") as f:
    threshold = float(f.read().strip())

print(f"Threshold: {threshold}")


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not os.path.exists(SCORED_FILE):
    raise FileNotFoundError(
        f"\nScored candidate file not found:\n{SCORED_FILE}\n\n"
        "Run 09_predict_test_v2.py first."
    )

if not os.path.exists(S1_FILE):
    raise FileNotFoundError(
        f"\nS1 test file not found:\n{S1_FILE}"
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DUCKDB
# ============================================================

con = duckdb.connect()

scored_path = SCORED_FILE.replace("\\", "/")
s1_path = S1_FILE.replace("\\", "/")


# ============================================================
# FINAL QUERY
# ============================================================

query = f"""
WITH scored AS (

    SELECT
        source1_entity_id,
        matched_entity_id,
        CAST(probability AS DOUBLE) AS probability

    FROM read_csv_auto(
        '{scored_path}',
        delim='\\t',
        header=true
    )
),

positive AS (

    SELECT
        source1_entity_id,
        matched_entity_id,
        probability

    FROM scored

    WHERE probability >= {threshold}
),

aggregated AS (

    SELECT
        source1_entity_id,

        string_agg(
            matched_entity_id,
            ','
            ORDER BY probability DESC, matched_entity_id
        ) AS matched_entity_ids

    FROM positive

    GROUP BY source1_entity_id
)

SELECT

    s.entity_id AS source1_entity_id,

    COALESCE(
        a.matched_entity_ids,
        ''
    ) AS matched_entity_ids

FROM read_csv_auto(
    '{s1_path}',
    delim='\\t',
    header=true
) s

LEFT JOIN aggregated a

    ON s.entity_id = a.source1_entity_id

ORDER BY s.entity_id
"""


# ============================================================
# EXECUTE
# ============================================================

print("\nReading scored candidates and building matches...")

final_df = con.execute(query).pl()


# ============================================================
# SAFETY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING OUTPUT")
print("=" * 70)

expected_rows = 1_732_544
actual_rows = len(final_df)

print(f"Expected S1 rows : {expected_rows:,}")
print(f"Actual output rows: {actual_rows:,}")

if actual_rows != expected_rows:
    raise RuntimeError(
        f"WRONG ROW COUNT! Expected {expected_rows:,}, "
        f"got {actual_rows:,}"
    )


# Check columns

expected_columns = [
    "source1_entity_id",
    "matched_entity_ids"
]

if final_df.columns != expected_columns:
    raise RuntimeError(
        f"WRONG COLUMNS!\n"
        f"Expected: {expected_columns}\n"
        f"Got: {final_df.columns}"
    )


# Check duplicate S1 IDs

duplicate_count = (
    final_df
    .group_by("source1_entity_id")
    .len()
    .filter(pl.col("len") > 1)
    .height
)

print(f"Duplicate S1 IDs: {duplicate_count:,}")

if duplicate_count != 0:
    raise RuntimeError(
        "Duplicate source1_entity_id values found!"
    )


# Check empty matches

non_empty = final_df.filter(
    pl.col("matched_entity_ids") != ""
)

empty = final_df.filter(
    pl.col("matched_entity_ids") == ""
)

print(
    f"S1 with predicted matches : {len(non_empty):,}"
)

print(
    f"S1 with no predicted matches: {len(empty):,}"
)


# ============================================================
# REMOVE DUPLICATE MATCH IDS INSIDE EACH ROW
# ============================================================

print("\nChecking duplicate matched IDs...")

def remove_duplicate_ids(value):
    if value is None or value == "":
        return ""

    ids = value.split(",")

    seen = set()
    result = []

    for entity_id in ids:
        if entity_id and entity_id not in seen:
            seen.add(entity_id)
            result.append(entity_id)

    return ",".join(result)


final_df = final_df.with_columns(
    pl.col("matched_entity_ids")
    .map_elements(
        remove_duplicate_ids,
        return_dtype=pl.String
    )
    .alias("matched_entity_ids")
)


# ============================================================
# WRITE FINAL FILE
# ============================================================

print("\nWriting final matching_results.tsv...")

final_df.write_csv(
    MATCHING_FILE,
    separator="\t",
    include_header=True
)


# ============================================================
# FINAL CHECK
# ============================================================

if not os.path.exists(MATCHING_FILE):
    raise RuntimeError(
        "matching_results.tsv was not created!"
    )

file_size = os.path.getsize(MATCHING_FILE)

print("\n" + "=" * 70)
print("FINAL MATCHING RESULTS CREATED")
print("=" * 70)

print(f"Output file:")
print(MATCHING_FILE)

print(f"\nRows: {len(final_df):,}")

print(
    f"S1 with predicted matches: "
    f"{len(final_df.filter(pl.col('matched_entity_ids') != '')):,}"
)

print(
    f"S1 with no predicted matches: "
    f"{len(final_df.filter(pl.col('matched_entity_ids') == '')):,}"
)

print(
    f"\nFile size: {file_size / (1024 * 1024):.2f} MB"
)

print("\nDONE.")

con.close()