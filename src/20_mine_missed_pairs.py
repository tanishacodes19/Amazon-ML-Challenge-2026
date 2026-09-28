# src/20_mine_missed_pairs.py

import duckdb
import os
import csv
import time

# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

GT_FILE = os.path.join(
    BASE,
    "ground_truth_pairs.tsv"
)

CANDIDATE_FILE = os.path.join(
    BASE,
    "training_candidate_pairs.tsv"
)

S1_FILE = os.path.join(
    BASE,
    "normalized_data",
    "train_source1_normalized.tsv"
)

S2_FILE = os.path.join(
    BASE,
    "normalized_data",
    "train_source2_normalized.tsv"
)

S3_FILE = os.path.join(
    BASE,
    "normalized_data",
    "train_source3_normalized.tsv"
)

OUT_FILE = os.path.join(
    BASE,
    "missed_block_analysis.csv"
)

TMP_DIR = os.path.join(
    BASE,
    "duckdb_tmp"
)

os.makedirs(TMP_DIR, exist_ok=True)


# ============================================================
# DUCKDB SETUP
# ============================================================

con = duckdb.connect()

con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3000MB'")

tmp_dir_sql = TMP_DIR.replace("\\", "/")

con.execute(
    f"SET temp_directory='{tmp_dir_sql}'"
)

print("=" * 75)
print("FAST MISSED-PAIR BLOCK MINING")
print("=" * 75)

print("\nDuckDB:")
print("  Threads      : 2")
print("  Memory       : 3000 MB")
print("  Temp folder  :", TMP_DIR)


# ============================================================
# 1. LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt_path = GT_FILE.replace("\\", "/")

con.execute(f"""
    CREATE OR REPLACE TABLE ground_truth AS
    SELECT
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

print(f"Ground truth: {gt_count:,}")


# ============================================================
# 2. LOAD CURRENT CANDIDATES
# ============================================================

print("\nLoading current candidates...")

candidate_path = CANDIDATE_FILE.replace("\\", "/")

con.execute(f"""
    CREATE OR REPLACE TABLE current_candidates AS

    SELECT
        column0 AS source1_entity_id,
        column1 AS candidate_entity_id

    FROM read_csv(
        '{candidate_path}',
        delim='\\t',
        header=true,
        quote='',
        columns={{
            'column0':'VARCHAR',
            'column1':'VARCHAR'
        }}
    )
""")

candidate_count = con.execute("""
    SELECT COUNT(*)
    FROM current_candidates
""").fetchone()[0]

print(f"Current candidates: {candidate_count:,}")


# ============================================================
# 3. FIND MISSED TRUE PAIRS
# ============================================================

print("\nFinding missed true pairs...")

con.execute("""
    CREATE OR REPLACE TABLE missed_pairs AS

    SELECT
        gt.source1_entity_id,
        gt.matched_entity_id

    FROM ground_truth AS gt

    LEFT JOIN current_candidates AS cc

        ON gt.source1_entity_id = cc.source1_entity_id
        AND gt.matched_entity_id = cc.candidate_entity_id

    WHERE cc.candidate_entity_id IS NULL
""")

missed_count = con.execute("""
    SELECT COUNT(*)
    FROM missed_pairs
""").fetchone()[0]

print(f"Missed true pairs: {missed_count:,}")

if missed_count == 0:

    print("\nNo missed true pairs found.")
    print("Nothing to mine.")

    con.close()
    raise SystemExit


# ============================================================
# 4. LOAD SOURCE 1
# ============================================================

print("\nLoading S1...")

s1_path = S1_FILE.replace("\\", "/")

con.execute(f"""
    CREATE OR REPLACE TABLE s1 AS

    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        name_normalized,
        address_normalized

    FROM read_csv(
        '{s1_path}',
        delim='\\t',
        header=true,
        quote='',
        columns={{
            'entity_id':'VARCHAR',
            'business_name':'VARCHAR',
            'business_address':'VARCHAR',
            'country':'VARCHAR',
            'name_normalized':'VARCHAR',
            'address_normalized':'VARCHAR'
        }}
    )
""")

s1_count = con.execute("""
    SELECT COUNT(*)
    FROM s1
""").fetchone()[0]

print(f"S1: {s1_count:,}")


# ============================================================
# 5. LOAD SOURCE 2 + SOURCE 3
# ============================================================

print("\nLoading S2 + S3...")

s2_path = S2_FILE.replace("\\", "/")
s3_path = S3_FILE.replace("\\", "/")

con.execute(f"""
    CREATE OR REPLACE TABLE s23 AS

    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        name_normalized,
        address_normalized

    FROM read_csv(
        '{s2_path}',
        delim='\\t',
        header=true,
        quote='',
        columns={{
            'entity_id':'VARCHAR',
            'business_name':'VARCHAR',
            'business_address':'VARCHAR',
            'country':'VARCHAR',
            'name_normalized':'VARCHAR',
            'address_normalized':'VARCHAR'
        }}
    )

    UNION ALL

    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        name_normalized,
        address_normalized

    FROM read_csv(
        '{s3_path}',
        delim='\\t',
        header=true,
        quote='',
        columns={{
            'entity_id':'VARCHAR',
            'business_name':'VARCHAR',
            'business_address':'VARCHAR',
            'country':'VARCHAR',
            'name_normalized':'VARCHAR',
            'address_normalized':'VARCHAR'
        }}
    )
""")

s23_count = con.execute("""
    SELECT COUNT(*)
    FROM s23
""").fetchone()[0]

print(f"S2 + S3: {s23_count:,}")


# ============================================================
# 6. CREATE S1 BLOCKING KEYS
# ============================================================

print("\nCreating blocking keys...")

con.execute("""
    CREATE OR REPLACE TABLE s1_keys AS

    SELECT
        entity_id,

        lower(trim(country)) AS country,

        -- NAME PREFIXES
        CASE
            WHEN length(name_normalized) >= 3
            THEN substr(name_normalized, 1, 3)
            ELSE ''
        END AS name_prefix3,

        CASE
            WHEN length(name_normalized) >= 4
            THEN substr(name_normalized, 1, 4)
            ELSE ''
        END AS name_prefix4,

        CASE
            WHEN length(name_normalized) >= 5
            THEN substr(name_normalized, 1, 5)
            ELSE ''
        END AS name_prefix5,

        -- NAME TOKENS
        CASE
            WHEN strpos(name_normalized, ' ') > 0
            THEN split_part(name_normalized, ' ', 1)
            ELSE name_normalized
        END AS first_token,

        CASE
            WHEN array_length(
                string_split(name_normalized, ' ')
            ) >= 2
            THEN split_part(name_normalized, ' ', 2)
            ELSE ''
        END AS second_token,

        CASE
            WHEN length(trim(name_normalized)) > 0
            THEN regexp_extract(
                name_normalized,
                '([^ ]+)$',
                1
            )
            ELSE ''
        END AS last_token,

        -- ADDRESS
        CASE
            WHEN length(address_normalized) >= 5
            THEN substr(address_normalized, 1, 5)
            ELSE ''
        END AS address_prefix5,

        CASE
            WHEN length(address_normalized) >= 8
            THEN substr(address_normalized, 1, 8)
            ELSE ''
        END AS address_prefix8,

        CASE
            WHEN length(address_normalized) >= 8
            THEN right(address_normalized, 8)
            ELSE ''
        END AS address_suffix8,

        -- HOUSE NUMBER
        CASE
            WHEN regexp_matches(
                trim(address_normalized),
                '^[0-9]+'
            )
            THEN regexp_extract(
                trim(address_normalized),
                '^([0-9]+)',
                1
            )
            ELSE ''
        END AS house_number

    FROM s1
""")


# ============================================================
# 7. CREATE S2/S3 BLOCKING KEYS
# ============================================================

con.execute("""
    CREATE OR REPLACE TABLE s23_keys AS

    SELECT
        entity_id,

        lower(trim(country)) AS country,

        -- NAME PREFIXES
        CASE
            WHEN length(name_normalized) >= 3
            THEN substr(name_normalized, 1, 3)
            ELSE ''
        END AS name_prefix3,

        CASE
            WHEN length(name_normalized) >= 4
            THEN substr(name_normalized, 1, 4)
            ELSE ''
        END AS name_prefix4,

        CASE
            WHEN length(name_normalized) >= 5
            THEN substr(name_normalized, 1, 5)
            ELSE ''
        END AS name_prefix5,

        -- NAME TOKENS
        CASE
            WHEN strpos(name_normalized, ' ') > 0
            THEN split_part(name_normalized, ' ', 1)
            ELSE name_normalized
        END AS first_token,

        CASE
            WHEN array_length(
                string_split(name_normalized, ' ')
            ) >= 2
            THEN split_part(name_normalized, ' ', 2)
            ELSE ''
        END AS second_token,

        CASE
            WHEN length(trim(name_normalized)) > 0
            THEN regexp_extract(
                name_normalized,
                '([^ ]+)$',
                1
            )
            ELSE ''
        END AS last_token,

        -- ADDRESS
        CASE
            WHEN length(address_normalized) >= 5
            THEN substr(address_normalized, 1, 5)
            ELSE ''
        END AS address_prefix5,

        CASE
            WHEN length(address_normalized) >= 8
            THEN substr(address_normalized, 1, 8)
            ELSE ''
        END AS address_prefix8,

        CASE
            WHEN length(address_normalized) >= 8
            THEN right(address_normalized, 8)
            ELSE ''
        END AS address_suffix8,

        -- HOUSE NUMBER
        CASE
            WHEN regexp_matches(
                trim(address_normalized),
                '^[0-9]+'
            )
            THEN regexp_extract(
                trim(address_normalized),
                '^([0-9]+)',
                1
            )
            ELSE ''
        END AS house_number

    FROM s23
""")


# ============================================================
# 8. PREPARE MISSED PAIR KEYS
# ============================================================

print("\nPreparing missed-pair keys...")

con.execute("""
    CREATE OR REPLACE TABLE missed_keys AS

    SELECT
        m.source1_entity_id,
        m.matched_entity_id,

        s.country,
        s.name_prefix3,
        s.name_prefix4,
        s.name_prefix5,

        s.first_token,
        s.second_token,
        s.last_token,

        s.address_prefix5,
        s.address_prefix8,
        s.address_suffix8,

        s.house_number

    FROM missed_pairs AS m

    JOIN s1_keys AS s
        ON m.source1_entity_id = s.entity_id
""")


# ============================================================
# 9. BLOCK TEST FUNCTION
# ============================================================

results = []


def test_block(
    block_name,
    s1_key,
    s23_key,
    max_frequency
):

    print(
        f"\nTesting {block_name} "
        f"(frequency <= {max_frequency}) ..."
    )

    start = time.time()

    # --------------------------------------------------------
    # Remove previous frequency table
    # --------------------------------------------------------

    con.execute("""
        DROP TABLE IF EXISTS temp_block_freq
    """)

    # --------------------------------------------------------
    # Build frequency table
    #
    # IMPORTANT:
    # s23_key contains explicit "s." aliases.
    # --------------------------------------------------------

    con.execute(f"""
        CREATE TEMP TABLE temp_block_freq AS

        SELECT
            {s23_key} AS block_key,
            COUNT(*) AS freq

        FROM s23_keys AS s

        WHERE
            {s23_key} IS NOT NULL
            AND {s23_key} <> ''

        GROUP BY
            {s23_key}

        HAVING
            COUNT(*) <= {max_frequency}
    """)

    # --------------------------------------------------------
    # Candidate count
    # --------------------------------------------------------

    candidate_count = con.execute(f"""
        SELECT
            COALESCE(
                SUM(f.freq),
                0
            )

        FROM (

            SELECT DISTINCT

                m.source1_entity_id,

                {s1_key} AS block_key

            FROM missed_keys AS m

            WHERE
                {s1_key} IS NOT NULL
                AND {s1_key} <> ''

        ) AS x

        JOIN temp_block_freq AS f

            ON x.block_key = f.block_key
    """).fetchone()[0]

    # --------------------------------------------------------
    # Recover missed true pairs
    #
    # IMPORTANT:
    # Both expressions are explicitly qualified:
    #
    #       s.xxx
    #       m.xxx
    #
    # --------------------------------------------------------

    recovered = con.execute(f"""
        SELECT
            COUNT(*)

        FROM missed_keys AS m

        JOIN s23_keys AS s

            ON s.entity_id = m.matched_entity_id

        WHERE

            {s23_key} = {s1_key}

            AND {s23_key} <> ''
    """).fetchone()[0]

    candidate_count = int(
        candidate_count or 0
    )

    recovered = int(
        recovered or 0
    )

    # --------------------------------------------------------
    # Efficiency
    # --------------------------------------------------------

    if candidate_count > 0:

        efficiency = (
            recovered /
            candidate_count
        )

    else:

        efficiency = 0.0

    elapsed = time.time() - start

    print(
        f"  Candidates : "
        f"{candidate_count:,}"
    )

    print(
        f"  Recovered  : "
        f"{recovered:,}"
    )

    print(
        f"  Efficiency : "
        f"{efficiency:.8f}"
    )

    print(
        f"  Time       : "
        f"{elapsed:.1f}s"
    )

    results.append({

        "block":
            block_name,

        "max_frequency":
            max_frequency,

        "candidates":
            candidate_count,

        "recovered_missed_pairs":
            recovered,

        "efficiency":
            efficiency,

        "time_seconds":
            elapsed
    })


# ============================================================
# 10. TEST BLOCKS
# ============================================================

frequency_limits = [
    10,
    25,
    50,
    100
]


# ------------------------------------------------------------
# COUNTRY + NAME PREFIX
# ------------------------------------------------------------

for freq in frequency_limits:

    test_block(
        "country_name_prefix3",

        "m.country || '|' || m.name_prefix3",

        "s.country || '|' || s.name_prefix3",

        freq
    )

    test_block(
        "country_name_prefix4",

        "m.country || '|' || m.name_prefix4",

        "s.country || '|' || s.name_prefix4",

        freq
    )


# ------------------------------------------------------------
# COUNTRY + FIRST TOKEN
# ------------------------------------------------------------

for freq in frequency_limits:

    test_block(
        "country_first_token",

        "m.country || '|' || m.first_token",

        "s.country || '|' || s.first_token",

        freq
    )


# ------------------------------------------------------------
# HOUSE + NAME PREFIX
# ------------------------------------------------------------

for freq in frequency_limits:

    test_block(
        "house_name_prefix3",

        "m.house_number || '|' || m.name_prefix3",

        "s.house_number || '|' || s.name_prefix3",

        freq
    )

    test_block(
        "house_name_prefix4",

        "m.house_number || '|' || m.name_prefix4",

        "s.house_number || '|' || s.name_prefix4",

        freq
    )


# ------------------------------------------------------------
# HOUSE + FIRST TOKEN
# ------------------------------------------------------------

for freq in frequency_limits:

    test_block(
        "house_first_token",

        "m.house_number || '|' || m.first_token",

        "s.house_number || '|' || s.first_token",

        freq
    )


# ------------------------------------------------------------
# FIRST + LAST TOKEN
# ------------------------------------------------------------

for freq in frequency_limits:

    test_block(
        "first_last_token",

        "m.first_token || '|' || m.last_token",

        "s.first_token || '|' || s.last_token",

        freq
    )


# ------------------------------------------------------------
# HOUSE + ADDRESS PREFIX
# ------------------------------------------------------------

for freq in frequency_limits:

    test_block(
        "house_address_prefix5",

        "m.house_number || '|' || m.address_prefix5",

        "s.house_number || '|' || s.address_prefix5",

        freq
    )


# ------------------------------------------------------------
# COUNTRY + ADDRESS PREFIX
# ------------------------------------------------------------

for freq in frequency_limits:

    test_block(
        "country_address_prefix8",

        "m.country || '|' || m.address_prefix8",

        "s.country || '|' || s.address_prefix8",

        freq
    )


# ============================================================
# 11. SAVE RESULTS
# ============================================================

print("\n" + "=" * 75)
print("SAVING RESULTS")
print("=" * 75)

with open(
    OUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "block",
            "max_frequency",
            "candidates",
            "recovered_missed_pairs",
            "efficiency",
            "time_seconds"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# 12. SORT RESULTS BY EFFICIENCY
# ============================================================

print("\n" + "=" * 75)
print("BLOCK RESULTS — SORTED BY EFFICIENCY")
print("=" * 75)

results_sorted = sorted(
    results,
    key=lambda x: x["efficiency"],
    reverse=True
)

for r in results_sorted:

    print(
        f"{r['block']:28s} "
        f"freq<={r['max_frequency']:3d} | "
        f"cand={r['candidates']:12,d} | "
        f"recovered="
        f"{r['recovered_missed_pairs']:9,d} | "
        f"eff="
        f"{r['efficiency']:.8f}"
    )


# ============================================================
# 13. DONE
# ============================================================

print("\nResults saved to:")
print(OUT_FILE)

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)

con.close()