# src/21_build_v5_candidates.py

import duckdb
import os
import csv

# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

BASELINE = os.path.join(
    BASE,
    "training_candidate_pairs.tsv"
)

GROUND_TRUTH = os.path.join(
    BASE,
    "ground_truth_pairs.tsv"
)

BLOCK_DIR = os.path.join(
    BASE,
    "training_data",
    "v5_candidate_blocks"
)

OUTPUT_FILE = os.path.join(
    BASE,
    "training_candidate_pairs_v5.tsv"
)

SUMMARY_FILE = os.path.join(
    BASE,
    "v5_incremental_union_summary.csv"
)

TMP_DIR = os.path.join(
    BASE,
    "duckdb_tmp"
)

os.makedirs(TMP_DIR, exist_ok=True)


# ============================================================
# BLOCK FILES
# ============================================================

BLOCK_FILES = [
    os.path.join(
        BLOCK_DIR,
        "country_name_prefix3_f10.tsv"
    ),

    os.path.join(
        BLOCK_DIR,
        "country_first_token_f10.tsv"
    ),

    os.path.join(
        BLOCK_DIR,
        "country_name_prefix4_f10.tsv"
    ),

    os.path.join(
        BLOCK_DIR,
        "house_address_prefix5_f10.tsv"
    )
]


# ============================================================
# DUCKDB
# ============================================================

con = duckdb.connect()

con.execute(
    "PRAGMA threads=2"
)

con.execute(
    "PRAGMA memory_limit='3000MB'"
)

con.execute(
    "PRAGMA preserve_insertion_order=false"
)

con.execute(
    f"SET temp_directory='{TMP_DIR.replace(chr(92), '/')}'"
)

print("=" * 78)
print("V5 LIGHTWEIGHT INCREMENTAL UNION")
print("=" * 78)


# ============================================================
# LOAD BASELINE
# ============================================================

print("\nLoading baseline candidates...")

baseline_path = BASELINE.replace("\\", "/")

con.execute(f"""
    CREATE OR REPLACE TABLE baseline AS

    SELECT DISTINCT

        column0 AS source1_entity_id,
        column1 AS candidate_entity_id

    FROM read_csv(
        '{baseline_path}',
        delim='\\t',
        header=true,
        quote='',
        columns={{
            'column0':'VARCHAR',
            'column1':'VARCHAR'
        }}
    )
""")

baseline_count = con.execute("""
    SELECT COUNT(*)
    FROM baseline
""").fetchone()[0]

print(
    f"Baseline candidates: "
    f"{baseline_count:,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt_path = GROUND_TRUTH.replace("\\", "/")

con.execute(f"""
    CREATE OR REPLACE TABLE ground_truth AS

    SELECT DISTINCT

        column0 AS source1_entity_id,
        column1 AS matched_entity_id

    FROM read_csv(
        '{gt_path}',
        delim='\\t',
        header=true,
        quote='',
        columns={{
            'column0':'VARCHAR',
            'column1':'VARCHAR'
        }}
    )
""")

gt_count = con.execute("""
    SELECT COUNT(*)
    FROM ground_truth
""").fetchone()[0]

print(
    f"Ground truth: "
    f"{gt_count:,}"
)


# ============================================================
# CHECK BLOCK FILES
# ============================================================

print("\nChecking saved block files...")

for path in BLOCK_FILES:

    if not os.path.exists(path):

        print(
            f"\nERROR: Missing block file:"
        )

        print(path)

        con.close()
        raise SystemExit(1)

    size_mb = (
        os.path.getsize(path)
        / (1024 * 1024)
    )

    print(
        f"  OK: "
        f"{os.path.basename(path)} "
        f"({size_mb:.1f} MB)"
    )


# ============================================================
# LOAD EACH BLOCK FILE
# ============================================================

print("\nLoading block files...")

for i, path in enumerate(BLOCK_FILES):

    table_name = f"block_{i}"

    path_sql = path.replace("\\", "/")

    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS

        SELECT DISTINCT

            column0 AS source1_entity_id,
            column1 AS candidate_entity_id

        FROM read_csv(
            '{path_sql}',
            delim='\\t',
            header=true,
            quote='',
            columns={{
                'column0':'VARCHAR',
                'column1':'VARCHAR'
            }}
        )
    """)

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table_name}
        """
    ).fetchone()[0]

    print(
        f"  {os.path.basename(path)}:"
        f" {count:,}"
    )


# ============================================================
# UNION THE FOUR SMALL FILES
# ============================================================

print("\nCombining NEW candidate blocks...")

con.execute("""
    CREATE OR REPLACE TABLE new_block_union AS

    SELECT
        source1_entity_id,
        candidate_entity_id
    FROM block_0

    UNION

    SELECT
        source1_entity_id,
        candidate_entity_id
    FROM block_1

    UNION

    SELECT
        source1_entity_id,
        candidate_entity_id
    FROM block_2

    UNION

    SELECT
        source1_entity_id,
        candidate_entity_id
    FROM block_3
""")

raw_union_count = con.execute("""
    SELECT COUNT(*)
    FROM new_block_union
""").fetchone()[0]

print(
    f"Unique candidates across "
    f"all four blocks: "
    f"{raw_union_count:,}"
)


# ============================================================
# REMOVE ANY BASELINE DUPLICATES
# ============================================================

print("\nRemoving candidates already in baseline...")

con.execute("""
    CREATE OR REPLACE TABLE genuinely_new AS

    SELECT

        n.source1_entity_id,
        n.candidate_entity_id

    FROM new_block_union AS n

    WHERE NOT EXISTS (

        SELECT 1

        FROM baseline AS b

        WHERE
            b.source1_entity_id =
                n.source1_entity_id

            AND

            b.candidate_entity_id =
                n.candidate_entity_id
    )
""")

new_count = con.execute("""
    SELECT COUNT(*)
    FROM genuinely_new
""").fetchone()[0]

print(
    f"Genuinely NEW candidates: "
    f"{new_count:,}"
)


# ============================================================
# COUNT NEW TRUE PAIRS
# ============================================================

print("\nChecking newly recovered true pairs...")

new_true_pairs = con.execute("""
    SELECT COUNT(*)

    FROM genuinely_new AS n

    JOIN ground_truth AS g

        ON n.source1_entity_id =
           g.source1_entity_id

        AND

           n.candidate_entity_id =
           g.matched_entity_id
""").fetchone()[0]

print(
    f"NEW true pairs recovered: "
    f"{new_true_pairs:,}"
)


# ============================================================
# EFFICIENCY
# ============================================================

if new_count > 0:

    efficiency = (
        new_true_pairs /
        new_count
    )

else:

    efficiency = 0.0


print(
    f"NEW candidate efficiency: "
    f"{efficiency:.8f}"
)


# ============================================================
# BUILD V5 FILE
# ============================================================

print("\nBuilding final V5 candidate file...")

output_path = OUTPUT_FILE.replace("\\", "/")

con.execute(f"""
    COPY (

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM baseline

        UNION

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM genuinely_new

    )

    TO '{output_path}'

    (
        FORMAT CSV,
        DELIMITER '\\t',
        HEADER
    )
""")

# ============================================================
# VERIFY V5 COUNT
# ============================================================

v5_count = con.execute("""
    SELECT
        COUNT(*)

    FROM (

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM baseline

        UNION

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM genuinely_new

    )
""").fetchone()[0]

print(
    f"\nV5 total candidates: "
    f"{v5_count:,}"
)


# ============================================================
# EXPECTED COUNT CHECK
# ============================================================

expected = baseline_count + new_count

print(
    f"Expected count:      "
    f"{expected:,}"
)

if v5_count == expected:

    print(
        "COUNT CHECK: PASS"
    )

else:

    print(
        "COUNT CHECK: WARNING"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "baseline_candidates": baseline_count,
    "raw_union_candidates": raw_union_count,
    "new_candidates": new_count,
    "new_true_pairs": new_true_pairs,
    "new_efficiency": efficiency,
    "v5_total_candidates": v5_count
}

with open(
    SUMMARY_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(summary.keys())
    )

    writer.writeheader()
    writer.writerow(summary)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 78)
print("V5 COMPLETE")
print("=" * 78)

print(
    f"Baseline : "
    f"{baseline_count:,}"
)

print(
    f"New      : "
    f"{new_count:,}"
)

print(
    f"True     : "
    f"{new_true_pairs:,}"
)

print(
    f"V5 total : "
    f"{v5_count:,}"
)

print(
    f"Efficiency: "
    f"{efficiency:.8f}"
)

print("\nV5 file:")
print(OUTPUT_FILE)

print("\nSummary:")
print(SUMMARY_FILE)

print("=" * 78)

con.close()