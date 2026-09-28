import duckdb

DB = "missed_pairs_analysis.duckdb"

con = duckdb.connect(DB)

print("=" * 70)
print("CONTAINMENT + LENGTH + TOKEN COUNT BENCHMARK")
print("=" * 70)

# ---------------------------------------------------------
# Build remaining population
# ---------------------------------------------------------

print("\nBuilding frequency tables...")

con.execute("""
CREATE OR REPLACE TEMP TABLE s1_prefix_freq AS
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

con.execute("""
CREATE OR REPLACE TEMP TABLE s1_first_freq AS
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

print("Building remaining population...")

con.execute("""
CREATE OR REPLACE TEMP TABLE remaining_test AS
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

    length(
        regexp_replace(
            lower(trim(mf.s1_name)),
            '[^a-z0-9]+',
            '',
            'g'
        )
    ) AS s1_name_len,

    length(
        regexp_replace(
            lower(trim(mf.matched_name)),
            '[^a-z0-9]+',
            '',
            'g'
        )
    ) AS s23_name_len,

    array_length(
        string_split(
            regexp_replace(
                lower(trim(mf.s1_name)),
                '[^a-z0-9]+',
                ' ',
                'g'
            ),
            ' '
        )
    ) AS s1_token_count,

    array_length(
        string_split(
            regexp_replace(
                lower(trim(mf.matched_name)),
                '[^a-z0-9]+',
                ' ',
                'g'
            ),
            ' '
        )
    ) AS s23_token_count

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
    (
        mf.same_house_number
        AND mf.name_prefix_4
    )
    OR
    (
        mf.same_house_number
        AND mf.first_name_token
    )
    OR
    (
        mf.name_prefix_4
        AND pf.freq <= 50
    )
    OR
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
# Containment
# ---------------------------------------------------------

print("Calculating containment...")

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

# ---------------------------------------------------------
# Benchmark
# ---------------------------------------------------------

tests = [
    (
        "Token count difference <= 0",
        "abs(s1_token_count - s23_token_count) <= 0"
    ),
    (
        "Token count difference <= 1",
        "abs(s1_token_count - s23_token_count) <= 1"
    ),
    (
        "Token count difference <= 2",
        "abs(s1_token_count - s23_token_count) <= 2"
    ),
    (
        "Token count difference <= 3",
        "abs(s1_token_count - s23_token_count) <= 3"
    ),
]

print("\n" + "=" * 70)
print("RESULTS")
print("Base condition: containment + name length difference <= 4")
print("=" * 70)

for name, token_condition in tests:

    condition = f"""
        s1_first_in_s23
        AND abs(s1_name_len - s23_name_len) <= 4
        AND {token_condition}
    """

    count = con.execute(f"""
        SELECT COUNT(*)
        FROM remaining_test
        WHERE {condition}
    """).fetchone()[0]

    remaining_recall = count / remaining * 100

    overall_recall = (
        (7_638_365 - remaining + count)
        / 7_638_365
        * 100
    )

    print(f"\n{name}")
    print(f"  True pairs recovered : {count:,}")
    print(f"  Remaining recall     : {remaining_recall:.2f}%")
    print(f"  Overall pair recall  : {overall_recall:.2f}%")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

con.close()