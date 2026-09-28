import os
import gc
import xgboost as xgb
import polars as pl
import duckdb


# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

S1_FILE = os.path.join(
    BASE, "normalized_data", "test_source1_normalized.tsv"
)

S2_FILE = os.path.join(
    BASE, "normalized_data", "test_source2_normalized.tsv"
)

S3_FILE = os.path.join(
    BASE, "normalized_data", "test_source3_normalized.tsv"
)

CANDIDATE_FILE = os.path.join(
    BASE, "candidate_pairs_real.tsv"
)

MODEL_FILE = os.path.join(
    BASE, "model", "xgboost_v2.json"
)

THRESHOLD_FILE = os.path.join(
    BASE, "model", "best_threshold_v2.txt"
)

OUTPUT_DIR = os.path.join(BASE, "output")

MATCHING_FILE = os.path.join(
    OUTPUT_DIR, "matching_results.tsv"
)

SCORED_FILE = os.path.join(
    BASE, "test_scored_candidates.tsv"
)

CHUNK_SIZE = 50_000


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)

with open(THRESHOLD_FILE, "r") as f:
    threshold = float(f.read().strip())

print("Model:", MODEL_FILE)
print("Threshold:", threshold)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING NORMALIZED TEST DATA")
print("=" * 70)

s1 = pl.read_csv(
    S1_FILE,
    separator="\t",
    infer_schema_length=1000,
)

s2 = pl.read_csv(
    S2_FILE,
    separator="\t",
    infer_schema_length=1000,
)

s3 = pl.read_csv(
    S3_FILE,
    separator="\t",
    infer_schema_length=1000,
)

print("S1 rows:", len(s1))
print("S2 rows:", len(s2))
print("S3 rows:", len(s3))


# ============================================================
# STANDARDIZE
# ============================================================

s1 = s1.select([
    pl.col("entity_id").alias("source1_entity_id"),
    pl.col("business_name")
    .fill_null("")
    .cast(pl.Utf8)
    .alias("s1_name"),
    pl.col("business_address")
    .fill_null("")
    .cast(pl.Utf8)
    .alias("s1_address"),
    pl.col("country")
    .fill_null("")
    .cast(pl.Utf8)
    .alias("s1_country"),
])

s2 = s2.select([
    pl.col("entity_id").alias("matched_entity_id"),
    pl.col("business_name")
    .fill_null("")
    .cast(pl.Utf8)
    .alias("matched_name"),
    pl.col("business_address")
    .fill_null("")
    .cast(pl.Utf8)
    .alias("matched_address"),
    pl.col("country")
    .fill_null("")
    .cast(pl.Utf8)
    .alias("matched_country"),
])

s3 = s3.select([
    pl.col("entity_id").alias("matched_entity_id"),
    pl.col("business_name")
    .fill_null("")
    .cast(pl.Utf8)
    .alias("matched_name"),
    pl.col("business_address")
    .fill_null("")
    .cast(pl.Utf8)
    .alias("matched_address"),
    pl.col("country")
    .fill_null("")
    .cast(pl.Utf8)
    .alias("matched_country"),
])

s23 = pl.concat([s2, s3], how="vertical")

del s2, s3
gc.collect()

print("Combined S2+S3 rows:", len(s23))


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    # NAME
    "name_exact",
    "name_prefix2",
    "name_prefix3",
    "name_prefix4",
    "name_prefix5",
    "name_suffix3",
    "name_suffix4",
    "name_first_char_match",
    "name_last_char_match",
    "name_first_token_match",
    "name_last_token_match",
    "name_token_overlap",
    "name_jaccard",
    "name_len_s1",
    "name_len_match",
    "name_length_abs_diff",
    "name_length_diff",
    "name_tokens_s1",
    "name_tokens_match",
    "name_token_count_diff",

    # ADDRESS
    "address_exact",
    "address_prefix5",
    "address_prefix8",
    "address_suffix5",
    "address_suffix8",
    "address_len_s1",
    "address_len_match",
    "address_length_abs_diff",
    "address_length_diff",
    "address_jaccard",
    "address_numeric_overlap",
    "address_house_match",

    # OTHER
    "country_match",
    "country_missing",
    "source2_candidate",
    "source3_candidate",
]


# ============================================================
# FEATURE FUNCTION
# ============================================================

def make_features(df):

    name1 = pl.col("s1_name")
    name2 = pl.col("matched_name")

    addr1 = pl.col("s1_address")
    addr2 = pl.col("matched_address")

    # --------------------------------------------------------
    # NAME TOKENS
    # --------------------------------------------------------

    tokens1 = name1.str.split(" ")
    tokens2 = name2.str.split(" ")

    token_intersection = (
        tokens1
        .list.set_intersection(tokens2)
        .list.len()
    )

    token_union = (
        tokens1
        .list.concat(tokens2)
        .list.unique()
        .list.len()
    )

    # --------------------------------------------------------
    # ADDRESS NUMBERS
    # --------------------------------------------------------

    nums1 = addr1.str.extract_all(r"\d+")
    nums2 = addr2.str.extract_all(r"\d+")

    numeric_intersection = (
        nums1
        .list.set_intersection(nums2)
        .list.len()
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    out = df.with_columns([

        # ====================================================
        # NAME
        # ====================================================

        (name1 == name2)
        .cast(pl.Int8)
        .alias("name_exact"),

        (name1.str.slice(0, 2) == name2.str.slice(0, 2))
        .cast(pl.Int8)
        .alias("name_prefix2"),

        (name1.str.slice(0, 3) == name2.str.slice(0, 3))
        .cast(pl.Int8)
        .alias("name_prefix3"),

        (name1.str.slice(0, 4) == name2.str.slice(0, 4))
        .cast(pl.Int8)
        .alias("name_prefix4"),

        (name1.str.slice(0, 5) == name2.str.slice(0, 5))
        .cast(pl.Int8)
        .alias("name_prefix5"),

        (name1.str.slice(-3, 3) == name2.str.slice(-3, 3))
        .cast(pl.Int8)
        .alias("name_suffix3"),

        (name1.str.slice(-4, 4) == name2.str.slice(-4, 4))
        .cast(pl.Int8)
        .alias("name_suffix4"),

        (
            name1.str.slice(0, 1)
            ==
            name2.str.slice(0, 1)
        )
        .cast(pl.Int8)
        .alias("name_first_char_match"),

        (
            name1.str.slice(-1, 1)
            ==
            name2.str.slice(-1, 1)
        )
        .cast(pl.Int8)
        .alias("name_last_char_match"),

        (
            tokens1.list.first()
            ==
            tokens2.list.first()
        )
        .cast(pl.Int8)
        .alias("name_first_token_match"),

        (
            tokens1.list.last()
            ==
            tokens2.list.last()
        )
        .cast(pl.Int8)
        .alias("name_last_token_match"),

        (token_intersection > 0)
        .cast(pl.Int8)
        .alias("name_token_overlap"),

        (
            token_intersection
            /
            pl.when(token_union == 0)
            .then(1)
            .otherwise(token_union)
        )
        .alias("name_jaccard"),

        name1.str.len_chars()
        .alias("name_len_s1"),

        name2.str.len_chars()
        .alias("name_len_match"),

        (
            name1.str.len_chars()
            -
            name2.str.len_chars()
        )
        .abs()
        .alias("name_length_abs_diff"),

        (
            (
                name1.str.len_chars()
                -
                name2.str.len_chars()
            ).abs()
            /
            pl.when(name1.str.len_chars() == 0)
            .then(1)
            .otherwise(name1.str.len_chars())
        )
        .alias("name_length_diff"),

        tokens1.list.len()
        .alias("name_tokens_s1"),

        tokens2.list.len()
        .alias("name_tokens_match"),

        (
            tokens1.list.len()
            -
            tokens2.list.len()
        )
        .abs()
        .alias("name_token_count_diff"),

        # ====================================================
        # ADDRESS
        # ====================================================

        (addr1 == addr2)
        .cast(pl.Int8)
        .alias("address_exact"),

        (
            addr1.str.slice(0, 5)
            ==
            addr2.str.slice(0, 5)
        )
        .cast(pl.Int8)
        .alias("address_prefix5"),

        (
            addr1.str.slice(0, 8)
            ==
            addr2.str.slice(0, 8)
        )
        .cast(pl.Int8)
        .alias("address_prefix8"),

        (
            addr1.str.slice(-5, 5)
            ==
            addr2.str.slice(-5, 5)
        )
        .cast(pl.Int8)
        .alias("address_suffix5"),

        (
            addr1.str.slice(-8, 8)
            ==
            addr2.str.slice(-8, 8)
        )
        .cast(pl.Int8)
        .alias("address_suffix8"),

        addr1.str.len_chars()
        .alias("address_len_s1"),

        addr2.str.len_chars()
        .alias("address_len_match"),

        (
            addr1.str.len_chars()
            -
            addr2.str.len_chars()
        )
        .abs()
        .alias("address_length_abs_diff"),

        (
            (
                addr1.str.len_chars()
                -
                addr2.str.len_chars()
            ).abs()
            /
            pl.when(addr1.str.len_chars() == 0)
            .then(1)
            .otherwise(addr1.str.len_chars())
        )
        .alias("address_length_diff"),

        (
            addr1.str.split(" ")
            .list.set_intersection(
                addr2.str.split(" ")
            )
            .list.len()
            /
            pl.when(
                addr1.str.split(" ")
                .list.concat(addr2.str.split(" "))
                .list.unique()
                .list.len() == 0
            )
            .then(1)
            .otherwise(
                addr1.str.split(" ")
                .list.concat(addr2.str.split(" "))
                .list.unique()
                .list.len()
            )
        )
        .alias("address_jaccard"),

        numeric_intersection
        .alias("address_numeric_overlap"),

        (
            nums1.list.first()
            ==
            nums2.list.first()
        )
        .cast(pl.Int8)
        .alias("address_house_match"),

        # ====================================================
        # COUNTRY
        # ====================================================

        (
            pl.col("s1_country")
            ==
            pl.col("matched_country")
        )
        .cast(pl.Int8)
        .alias("country_match"),

        (
            (pl.col("s1_country") == "")
            |
            (pl.col("matched_country") == "")
        )
        .cast(pl.Int8)
        .alias("country_missing"),

        # ====================================================
        # SOURCE
        # ====================================================

        pl.col("matched_entity_id")
        .str.starts_with("S2-")
        .cast(pl.Int8)
        .alias("source2_candidate"),

        pl.col("matched_entity_id")
        .str.starts_with("S3-")
        .cast(pl.Int8)
        .alias("source3_candidate"),
    ])

    return out


# ============================================================
# READ CANDIDATES STREAMING
# ============================================================

print("\n" + "=" * 70)
print("SCORING CANDIDATES")
print("=" * 70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Delete previous temporary scores
if os.path.exists(SCORED_FILE):
    os.remove(SCORED_FILE)

header_written = False
total_pairs = 0
batch_no = 0

reader = pl.read_csv_batched(
    CANDIDATE_FILE,
    separator="\t",
    batch_size=CHUNK_SIZE,
    has_header=True,
    infer_schema_length=1000,
)


while True:

    batches = reader.next_batches(1)

    if not batches:
        break

    raw = batches[0]

    if raw.height == 0:
        continue

    batch_no += 1

    # ========================================================
    # EXPAND CANDIDATE LIST
    # ========================================================

    raw = raw.with_columns(
        pl.col("candidate_entity_ids")
        .fill_null("")
        .str.split(",")
        .alias("candidate_list")
    )

    expanded = (
        raw
        .explode("candidate_list")
        .rename({
            "candidate_list": "matched_entity_id"
        })
        .filter(
            pl.col("matched_entity_id") != ""
        )
        .select([
            "source1_entity_id",
            "matched_entity_id",
        ])
    )

    if expanded.height == 0:
        continue

    # ========================================================
    # JOIN SOURCE 1
    # ========================================================

    joined = expanded.join(
        s1,
        on="source1_entity_id",
        how="left",
    )

    # ========================================================
    # JOIN SOURCE 2 + SOURCE 3
    # ========================================================

    joined = joined.join(
        s23,
        on="matched_entity_id",
        how="left",
    )

    # ========================================================
    # GENERATE FEATURES
    # ========================================================

    features_df = make_features(joined)

    X = features_df.select(FEATURE_COLUMNS)

    X = X.with_columns(
        pl.all().cast(pl.Float32)
    )

    # ========================================================
    # PREDICT
    # ========================================================

    probabilities = model.predict_proba(
        X.to_pandas()
    )[:, 1]

    result = pl.DataFrame({
        "source1_entity_id":
            joined["source1_entity_id"],

        "matched_entity_id":
            joined["matched_entity_id"],

        "probability":
            probabilities,
    })

    # ========================================================
    # WRITE SCORES
    # ========================================================

    # IMPORTANT:
    # Your Polars version does not support append=True.
    # We therefore open the file in binary append/write mode.

    with open(
        SCORED_FILE,
        "ab" if header_written else "wb"
    ) as f:

        result.write_csv(
            f,
            separator="\t",
            include_header=not header_written,
        )

    header_written = True

    total_pairs += result.height

    print(
        f"Batch {batch_no:04d} | "
        f"Expanded/scored: {total_pairs:,}"
    )

    del raw
    del expanded
    del joined
    del features_df
    del X
    del result

    gc.collect()


# ============================================================
# SCORING COMPLETE
# ============================================================

print("\nCandidate scoring complete.")

print(
    "Total pair candidates scored:",
    f"{total_pairs:,}"
)


# ============================================================
# FINAL AGGREGATION
# ============================================================

print("\n" + "=" * 70)
print("BUILDING FINAL MATCHING RESULTS")
print("=" * 70)

con = duckdb.connect()

scored_path = SCORED_FILE.replace("\\", "/")
s1_path = S1_FILE.replace("\\", "/")

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

    s.source1_entity_id,

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

    ON s.source1_entity_id =
       a.source1_entity_id

ORDER BY s.source1_entity_id
"""

final_df = con.execute(query).pl()

final_df.write_csv(
    MATCHING_FILE,
    separator="\t",
)

con.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL MATCHING RESULTS CREATED")
print("=" * 70)

print(
    "Output:",
    MATCHING_FILE
)

print(
    "Rows:",
    f"{len(final_df):,}"
)

non_empty = final_df.filter(
    pl.col("matched_entity_ids") != ""
)

print(
    "S1 with predicted matches:",
    f"{len(non_empty):,}"
)

print(
    "S1 with no predicted matches:",
    f"{len(final_df) - len(non_empty):,}"
)

print("\nDONE.")