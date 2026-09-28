import duckdb
import time

DB = "missed_pairs_analysis.duckdb"

con = duckdb.connect(DB)

print("=" * 70)
print("TIGHT CONTAINMENT + ADDRESS BENCHMARK")
print("=" * 70)

# ---------------------------------------------------------
# 1. Start from original missed true pairs
# ---------------------------------------------------------

print("\nLoading missed features...")

con.execute("""
DROP TABLE IF EXISTS remaining_test;
""")

# Reconstruct the remaining population using the strongest
# previously tested cheap blocks:
#
# B = same house + name prefix4
# C = same house + first token
# D = name prefix4 + rare prefix frequency <= 50
# E = first token + rare first-token frequency <= 50
#
# First create frequency tables.

print("Building S1 name-prefix frequencies...")

con.execute("""
DROP TABLE IF EXISTS s1_prefix_freq;
CREATE TEMP TABLE s1_prefix_freq AS
SELECT
    s1_country,
    LEFT(
        regexp_replace(
            lower(trim(s1_name)),
            '[^a-z0-9]+',
            ' ',
            'g'
        ),
        4
    ) AS prefix4,
    COUNT(*) AS freq
FROM s1
GROUP BY 1, 2;
""")

print("Building S1 first-token frequencies...")

con.execute("""
DROP TABLE IF EXISTS s1_first_freq;
CREATE TEMP TABLE s1_first_freq AS
SELECT
    s1_country,
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
    COUNT(*) AS freq
FROM s1
GROUP BY 1, 2;
""")

# ---------------------------------------------------------
# 2. Add normalized first token / prefix to missed pairs
# ---------------------------------------------------------

print("Creating remaining population...")

con.execute("""
CREATE TEMP TABLE remaining_test AS
SELECT
    mf.*,

    split_part(
        regexp_replace(
            lower(trim(mf.s1_name)),
            '[^a-z0-9]+',
            ' ',
            'g'
        ),
        ' ',
        1
    ) AS s1_first,

    split_part(
        regexp_replace(
            lower(trim(mf.matched_name)),
            '[^a-z0-9]+',
            ' ',
            'g'
        ),
        ' ',
        1
    ) AS s23_first,

    LEFT(
        regexp_replace(
            lower(trim(mf.s1_name)),
            '[^a-z0-9]+',
            ' ',
            'g'
        ),
        4
    ) AS s1_prefix4,

    regexp_extract(
        lower(trim(mf.s1_address)),
        '([0-9]+)',
        1
    ) AS s1_house,

    regexp_extract(
        lower(trim(mf.matched_address)),
        '([0-9]+)',
        1
    ) AS s23_house

FROM missed_features mf
JOIN s1_prefix_freq pf
    ON pf.s1_country = mf.s1_country
   AND pf.prefix4 =
       LEFT(
           regexp_replace(
               lower(trim(mf.s1_name)),
               '[^a-z0-9]+',
               ' ',
               'g'
           ),
           4
       )
JOIN s1_first_freq ff
    ON ff.s1_country = mf.s1_country
   AND ff.first_token =
       split_part(
           regexp_replace(
               lower(trim(mf.s1_name)),
               '[^a-z0-9]+',
               ' ',
               'g'
           ),
           ' ',
           1
       )

WHERE NOT (
    -- B: house + prefix4
    (
        mf.same_house_number
        AND mf.name_prefix_4
    )

    OR

    -- C: house + first token
    (
        mf.same_house_number
        AND mf.first_name_token
    )

    OR

    -- D: prefix4 frequency <= 50
    (
        mf.name_prefix_4
        AND pf.freq <= 50
    )

    OR

    -- E: first-token frequency <= 50
    (
        mf.first_name_token
        AND ff.freq <= 50
    )
);
""")

remaining = con.execute(
    "SELECT COUNT(*) FROM remaining_test"
).fetchone()[0]

print(f"\nRemaining true pairs: {remaining:,}")

# ---------------------------------------------------------
# 3. Calculate containment
# ---------------------------------------------------------

print("\nCalculating containment signals...")

con.execute("""
ALTER TABLE remaining_test
ADD COLUMN s1_first_in_s23 BOOLEAN;
""")

con.execute("""
UPDATE remaining_test
SET s1_first_in_s23 =
    list_contains(
        string_split(
            regexp_replace(
                lower(trim(matched_name)),
                '[^a-z0-9]+',
                ' ',
                'g'
            ),
            ' '
        ),
        s1_first
    );
""")

con.execute("""
ALTER TABLE remaining_test
ADD COLUMN s23_first_in_s1 BOOLEAN;
""")

con.execute("""
UPDATE remaining_test
SET s23_first_in_s1 =
    list_contains(
        string_split(
            regexp_replace(
                lower(trim(s1_name)),
                '[^a-z0-9]+',
                ' ',
                'g'
            ),
            ' '
        ),
        s23_first
    );
""")

# ---------------------------------------------------------
# 4. Benchmark containment + address signals
# ---------------------------------------------------------

tests = [
    (
        "S1 first in S23 name + address prefix5",
        "s1_first_in_s23 AND address_prefix_5"
    ),
    (
        "S1 first in S23 name + address prefix8",
        "s1_first_in_s23 AND address_prefix_8"
    ),
    (
        "S1 first in S23 name + address suffix8",
        "s1_first_in_s23 AND address_suffix_8"
    ),
    (
        "S1 first in S23 name + same house",
        "s1_first_in_s23 AND same_house_number"
    ),
    (
        "S1 first in S23 name + address prefix5 + same house",
        "s1_first_in_s23 AND address_prefix_5 AND same_house_number"
    ),
    (
        "S1 first in S23 name + address prefix8 + same house",
        "s1_first_in_s23 AND address_prefix_8 AND same_house_number"
    ),
    (
        "S1 first in S23 name + address suffix8 + same house",
        "s1_first_in_s23 AND address_suffix_8 AND same_house_number"
    ),
    (
        "S23 first in S1 name + address prefix8",
        "s23_first_in_s1 AND address_prefix_8"
    ),
    (
        "EITHER containment + address prefix8",
        "(s1_first_in_s23 OR s23_first_in_s1) AND address_prefix_8"
    ),
    (
        "EITHER containment + address suffix8",
        "(s1_first_in_s23 OR s23_first_in_s1) AND address_suffix_8"
    ),
    (
        "EITHER containment + same house",
        "(s1_first_in_s23 OR s23_first_in_s1) AND same_house_number"
    ),
]

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

for name, condition in tests:

    count = con.execute(f"""
        SELECT COUNT(*)
        FROM remaining_test
        WHERE {condition}
    """).fetchone()[0]

    recall_remaining = (
        count / remaining * 100
        if remaining else 0
    )

    total_recall = (
        (7_638_365 - remaining + count)
        / 7_638_365
        * 100
    )

    print(f"\n{name}")
    print(f"  True pairs recovered : {count:,}")
    print(f"  Remaining recall     : {recall_remaining:.2f}%")
    print(f"  Overall pair recall  : {total_recall:.2f}%")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

con.close()