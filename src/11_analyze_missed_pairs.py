from pathlib import Path
import duckdb

BASE_DIR = Path(".")

GROUND_TRUTH = BASE_DIR / "ground_truth_pairs.tsv"
CANDIDATES = BASE_DIR / "training_candidate_pairs.tsv"

S1_FILE = BASE_DIR / "normalized_data" / "train_source1_normalized.tsv"
S2_FILE = BASE_DIR / "normalized_data" / "train_source2_normalized.tsv"
S3_FILE = BASE_DIR / "normalized_data" / "train_source3_normalized.tsv"

DB_FILE = BASE_DIR / "missed_pairs_analysis.duckdb"

print("=" * 70)
print("ANALYZING MISSED TRUE MATCHES")
print("=" * 70)

con = duckdb.connect(str(DB_FILE))

con.execute("SET memory_limit='1500MB'")
con.execute("SET threads=1")
con.execute("SET preserve_insertion_order=false")
con.execute("SET temp_directory='missed_pairs_temp'")

gt = str(GROUND_TRUTH.resolve()).replace("\\", "/")
cand = str(CANDIDATES.resolve()).replace("\\", "/")
s1 = str(S1_FILE.resolve()).replace("\\", "/")
s2 = str(S2_FILE.resolve()).replace("\\", "/")
s3 = str(S3_FILE.resolve()).replace("\\", "/")


# ============================================================
# 1. GROUND TRUTH
# ============================================================

print("\n[1/5] Loading ground truth...")

con.execute(f"""
CREATE OR REPLACE TABLE ground_truth AS
SELECT DISTINCT
    source1_entity_id,
    matched_entity_id
FROM read_csv(
    '{gt}',
    delim='\\t',
    header=true,
    columns={{
        'source1_entity_id': 'VARCHAR',
        'matched_entity_id': 'VARCHAR'
    }}
)
""")


# ============================================================
# 2. CURRENT CANDIDATES
# ============================================================

print("[2/5] Loading current candidates...")

con.execute(f"""
CREATE OR REPLACE TABLE candidates AS
SELECT DISTINCT
    source1_entity_id,
    candidate_entity_id AS matched_entity_id
FROM read_csv(
    '{cand}',
    delim='\\t',
    header=true,
    columns={{
        'source1_entity_id': 'VARCHAR',
        'candidate_entity_id': 'VARCHAR'
    }}
)
""")


# ============================================================
# 3. FIND MISSED TRUE PAIRS
# ============================================================

print("[3/5] Finding missed true pairs...")

con.execute("""
CREATE OR REPLACE TABLE missed AS
SELECT
    g.source1_entity_id,
    g.matched_entity_id
FROM ground_truth g
LEFT JOIN candidates c
    ON g.source1_entity_id = c.source1_entity_id
    AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL
""")

missed_count = con.execute("""
SELECT COUNT(*) FROM missed
""").fetchone()[0]

print(f"Missed true pairs: {missed_count:,}")


# ============================================================
# 4. LOAD SOURCE DATA
# ============================================================

print("[4/5] Loading normalized source records...")

con.execute(f"""
CREATE OR REPLACE TABLE s1 AS
SELECT
    entity_id AS source1_entity_id,
    business_name AS s1_name,
    business_address AS s1_address,
    country AS s1_country
FROM read_csv(
    '{s1}',
    delim='\\t',
    header=true,
    auto_detect=true
)
""")

con.execute(f"""
CREATE OR REPLACE TABLE s23 AS
SELECT
    entity_id AS matched_entity_id,
    business_name AS matched_name,
    business_address AS matched_address,
    country AS matched_country
FROM read_csv(
    '{s2}',
    delim='\\t',
    header=true,
    auto_detect=true
)

UNION ALL

SELECT
    entity_id AS matched_entity_id,
    business_name AS matched_name,
    business_address AS matched_address,
    country AS matched_country
FROM read_csv(
    '{s3}',
    delim='\\t',
    header=true,
    auto_detect=true
)
""")


# ============================================================
# 5. BUILD MISSED PAIR FEATURES
# ============================================================

print("[5/5] Building block-key diagnostics...")

con.execute("""
CREATE OR REPLACE TABLE missed_features AS

SELECT
    m.source1_entity_id,
    m.matched_entity_id,

    s1.s1_name,
    s23.matched_name,

    s1.s1_address,
    s23.matched_address,

    s1.s1_country,
    s23.matched_country,

    -- --------------------------------------------------------
    -- NAME PREFIXES
    -- --------------------------------------------------------

    left(s1.s1_name, 3)
        = left(s23.matched_name, 3)
        AS name_prefix_3,

    left(s1.s1_name, 4)
        = left(s23.matched_name, 4)
        AS name_prefix_4,

    left(s1.s1_name, 5)
        = left(s23.matched_name, 5)
        AS name_prefix_5,

    left(s1.s1_name, 6)
        = left(s23.matched_name, 6)
        AS name_prefix_6,

    -- --------------------------------------------------------
    -- NAME SUFFIXES
    -- --------------------------------------------------------

    right(s1.s1_name, 4)
        = right(s23.matched_name, 4)
        AS name_suffix_4,

    right(s1.s1_name, 6)
        = right(s23.matched_name, 6)
        AS name_suffix_6,

    right(s1.s1_name, 8)
        = right(s23.matched_name, 8)
        AS name_suffix_8,

    -- --------------------------------------------------------
    -- FIRST TOKEN
    -- --------------------------------------------------------

    split_part(s1.s1_name, ' ', 1)
        =
    split_part(s23.matched_name, ' ', 1)
        AS first_name_token,

    -- --------------------------------------------------------
    -- SECOND TOKEN
    -- --------------------------------------------------------

    split_part(s1.s1_name, ' ', 2)
        =
    split_part(s23.matched_name, ' ', 2)
        AS second_name_token,

    -- --------------------------------------------------------
    -- LAST TOKEN
    -- --------------------------------------------------------

    regexp_extract(
        s1.s1_name,
        '([^ ]+)$',
        1
    )
    =
    regexp_extract(
        s23.matched_name,
        '([^ ]+)$',
        1
    )
        AS last_name_token,

    -- --------------------------------------------------------
    -- ADDRESS PREFIX
    -- --------------------------------------------------------

    left(s1.s1_address, 5)
        = left(s23.matched_address, 5)
        AS address_prefix_5,

    left(s1.s1_address, 8)
        = left(s23.matched_address, 8)
        AS address_prefix_8,

    left(s1.s1_address, 12)
        = left(s23.matched_address, 12)
        AS address_prefix_12,

    -- --------------------------------------------------------
    -- ADDRESS SUFFIX
    -- --------------------------------------------------------

    right(s1.s1_address, 8)
        = right(s23.matched_address, 8)
        AS address_suffix_8,

    right(s1.s1_address, 12)
        = right(s23.matched_address, 12)
        AS address_suffix_12,

    -- --------------------------------------------------------
    -- COUNTRY
    -- --------------------------------------------------------

    lower(s1.s1_country)
        =
    lower(s23.matched_country)
        AS country_match,

    -- --------------------------------------------------------
    -- EXACT NAME / ADDRESS
    -- --------------------------------------------------------

    s1.s1_name = s23.matched_name
        AS exact_name,

    s1.s1_address = s23.matched_address
        AS exact_address,

    -- --------------------------------------------------------
    -- HOUSE NUMBER
    -- --------------------------------------------------------

    regexp_extract(
        s1.s1_address,
        '^([0-9]+)',
        1
    )
    =
    regexp_extract(
        s23.matched_address,
        '^([0-9]+)',
        1
    )
        AS same_house_number

FROM missed m

INNER JOIN s1
    ON m.source1_entity_id = s1.source1_entity_id

INNER JOIN s23
    ON m.matched_entity_id = s23.matched_entity_id
""")


# ============================================================
# PRINT COVERAGE OF EACH SIGNAL
# ============================================================

print("\n" + "=" * 70)
print("MISSED-PAIR BLOCK ANALYSIS")
print("=" * 70)

signals = [
    "name_prefix_3",
    "name_prefix_4",
    "name_prefix_5",
    "name_prefix_6",
    "name_suffix_4",
    "name_suffix_6",
    "name_suffix_8",
    "first_name_token",
    "second_name_token",
    "last_name_token",
    "address_prefix_5",
    "address_prefix_8",
    "address_prefix_12",
    "address_suffix_8",
    "address_suffix_12",
    "country_match",
    "exact_name",
    "exact_address",
    "same_house_number",
]

for signal in signals:

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM missed_features
        WHERE {signal}
        """
    ).fetchone()[0]

    percentage = (
        count / missed_count * 100
        if missed_count
        else 0
    )

    print(
        f"{signal:<25} "
        f"{count:>12,} "
        f"({percentage:>7.2f}%)"
    )


# ============================================================
# COMBINATIONS
# ============================================================

print("\n" + "-" * 70)
print("USEFUL COMBINATIONS")
print("-" * 70)

combinations = {
    "country + name prefix 4":
        "country_match AND name_prefix_4",

    "country + name prefix 5":
        "country_match AND name_prefix_5",

    "country + name prefix 6":
        "country_match AND name_prefix_6",

    "country + first token":
        "country_match AND first_name_token",

    "country + last token":
        "country_match AND last_name_token",

    "country + address prefix 8":
        "country_match AND address_prefix_8",

    "house number + name prefix 4":
        "same_house_number AND name_prefix_4",

    "house number + first token":
        "same_house_number AND first_name_token",

    "name prefix 4 + address prefix 8":
        "name_prefix_4 AND address_prefix_8",
}

for name, condition in combinations.items():

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM missed_features
        WHERE {condition}
        """
    ).fetchone()[0]

    percentage = (
        count / missed_count * 100
        if missed_count
        else 0
    )

    print(
        f"{name:<35} "
        f"{count:>12,} "
        f"({percentage:>7.2f}%)"
    )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

con.close()