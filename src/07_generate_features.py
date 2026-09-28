from pathlib import Path
import polars as pl

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(".")
NORM_DIR = BASE_DIR / "normalized_data"
TRAINING_PAIRS = BASE_DIR / "training_data" / "training_pairs.tsv"

OUTPUT_DIR = BASE_DIR / "training_data" / "features_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 100_000

S1_FILE = NORM_DIR / "train_source1_normalized.tsv"
S2_FILE = NORM_DIR / "train_source2_normalized.tsv"
S3_FILE = NORM_DIR / "train_source3_normalized.tsv"

print("=" * 70)
print("FEATURE GENERATION V2")
print("=" * 70)


# ============================================================
# HELPERS
# ============================================================

def jaccard_expr(col1, col2):
    tokens1 = pl.col(col1).str.split(" ")
    tokens2 = pl.col(col2).str.split(" ")

    intersection = tokens1.list.set_intersection(tokens2).list.len()
    union = tokens1.list.set_union(tokens2).list.len()

    return (
        pl.when(union == 0)
        .then(0.0)
        .otherwise(
            intersection.cast(pl.Float64)
            / union.cast(pl.Float64)
        )
    )


def numeric_overlap_expr(col1, col2):
    nums1 = (
        pl.col(col1)
        .str.extract_all(r"\d+")
        .list.unique()
    )

    nums2 = (
        pl.col(col2)
        .str.extract_all(r"\d+")
        .list.unique()
    )

    intersection = nums1.list.set_intersection(nums2).list.len()

    return (
        pl.when(nums1.list.len() == 0)
        .then(0.0)
        .otherwise(
            intersection.cast(pl.Float64)
            / nums1.list.len().cast(pl.Float64)
        )
    )


def house_number_expr(col):
    return (
        pl.col(col)
        .str.extract(r"(\d+)", 1)
        .fill_null("")
    )


def token_count_expr(col):
    return (
        pl.col(col)
        .str.split(" ")
        .list.eval(
            pl.element().filter(pl.element() != "")
        )
        .list.len()
    )


# ============================================================
# LOAD SOURCE TABLES
# ============================================================

print("\nLoading normalized source files...")

s1 = pl.read_csv(
    S1_FILE,
    separator="\t",
    infer_schema_length=1000,
).select([
    "entity_id",
    "business_name",
    "business_address",
    "country",
])

s2 = pl.read_csv(
    S2_FILE,
    separator="\t",
    infer_schema_length=1000,
).select([
    "entity_id",
    "business_name",
    "business_address",
    "country",
])

s3 = pl.read_csv(
    S3_FILE,
    separator="\t",
    infer_schema_length=1000,
).select([
    "entity_id",
    "business_name",
    "business_address",
    "country",
])

print(f"S1 rows: {len(s1):,}")
print(f"S2 rows: {len(s2):,}")
print(f"S3 rows: {len(s3):,}")


# ============================================================
# RENAME
# ============================================================

s1 = s1.rename({
    "entity_id": "source1_entity_id",
    "business_name": "s1_name",
    "business_address": "s1_address",
    "country": "s1_country",
})

s2 = s2.rename({
    "entity_id": "matched_entity_id",
    "business_name": "matched_name",
    "business_address": "matched_address",
    "country": "matched_country",
})

s3 = s3.rename({
    "entity_id": "matched_entity_id",
    "business_name": "matched_name",
    "business_address": "matched_address",
    "country": "matched_country",
})

s23 = pl.concat([s2, s3], how="vertical")

del s2
del s3


# ============================================================
# LOAD TRAINING PAIRS
# ============================================================

print("\nLoading training pairs...")

pairs = pl.read_csv(
    TRAINING_PAIRS,
    separator="\t",
    schema_overrides={
        "source1_entity_id": pl.String,
        "matched_entity_id": pl.String,
        "label": pl.Int8,
    },
)

total_pairs = len(pairs)

print(f"Total training pairs: {total_pairs:,}")


# ============================================================
# PROCESS CHUNKS
# ============================================================

feature_files = []

for start in range(0, total_pairs, CHUNK_SIZE):

    end = min(start + CHUNK_SIZE, total_pairs)

    print(
        f"\nProcessing {start:,} -> {end:,}"
    )

    chunk = pairs.slice(
        start,
        end - start
    )

    # --------------------------------------------------------
    # JOIN S1
    # --------------------------------------------------------

    chunk = chunk.join(
        s1,
        on="source1_entity_id",
        how="left",
    )

    # --------------------------------------------------------
    # JOIN S23
    # --------------------------------------------------------

    chunk = chunk.join(
        s23,
        on="matched_entity_id",
        how="left",
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    string_columns = [
        "s1_name",
        "matched_name",
        "s1_address",
        "matched_address",
        "s1_country",
        "matched_country",
    ]

    chunk = chunk.with_columns([
        pl.col(c)
        .fill_null("")
        .cast(pl.String)
        for c in string_columns
    ])

    # ========================================================
    # BASIC DERIVED COLUMNS
    # ========================================================

    chunk = chunk.with_columns([

        # NAME TOKEN COUNTS
        token_count_expr("s1_name")
        .cast(pl.Float32)
        .alias("name_tokens_s1"),

        token_count_expr("matched_name")
        .cast(pl.Float32)
        .alias("name_tokens_match"),

        # ADDRESS TOKEN COUNTS
        token_count_expr("s1_address")
        .cast(pl.Float32)
        .alias("address_tokens_s1"),

        token_count_expr("matched_address")
        .cast(pl.Float32)
        .alias("address_tokens_match"),

        # HOUSE NUMBERS
        house_number_expr("s1_address")
        .alias("_s1_house"),

        house_number_expr("matched_address")
        .alias("_match_house"),
    ])

    # ========================================================
    # NAME FEATURES
    # ========================================================

    chunk = chunk.with_columns([

        # Exact
        (
            pl.col("s1_name")
            == pl.col("matched_name")
        )
        .cast(pl.Float32)
        .alias("name_exact"),

        # Prefixes
        (
            pl.col("s1_name").str.slice(0, 2)
            ==
            pl.col("matched_name").str.slice(0, 2)
        )
        .cast(pl.Float32)
        .alias("name_prefix2"),

        (
            pl.col("s1_name").str.slice(0, 3)
            ==
            pl.col("matched_name").str.slice(0, 3)
        )
        .cast(pl.Float32)
        .alias("name_prefix3"),

        (
            pl.col("s1_name").str.slice(0, 4)
            ==
            pl.col("matched_name").str.slice(0, 4)
        )
        .cast(pl.Float32)
        .alias("name_prefix4"),

        (
            pl.col("s1_name").str.slice(0, 5)
            ==
            pl.col("matched_name").str.slice(0, 5)
        )
        .cast(pl.Float32)
        .alias("name_prefix5"),

        # Suffixes
        (
            pl.col("s1_name").str.slice(-3, 3)
            ==
            pl.col("matched_name").str.slice(-3, 3)
        )
        .cast(pl.Float32)
        .alias("name_suffix3"),

        (
            pl.col("s1_name").str.slice(-4, 4)
            ==
            pl.col("matched_name").str.slice(-4, 4)
        )
        .cast(pl.Float32)
        .alias("name_suffix4"),

        # First character
        (
            pl.col("s1_name").str.slice(0, 1)
            ==
            pl.col("matched_name").str.slice(0, 1)
        )
        .cast(pl.Float32)
        .alias("name_first_char_match"),

        # Last character
        (
            pl.col("s1_name").str.slice(-1, 1)
            ==
            pl.col("matched_name").str.slice(-1, 1)
        )
        .cast(pl.Float32)
        .alias("name_last_char_match"),

        # Jaccard
        jaccard_expr(
            "s1_name",
            "matched_name"
        )
        .cast(pl.Float32)
        .alias("name_jaccard"),

        # First token
        (
            pl.col("s1_name").str.split(" ").list.get(0)
            ==
            pl.col("matched_name").str.split(" ").list.get(0)
        )
        .cast(pl.Float32)
        .alias("name_first_token_match"),

        # Last token
        (
            pl.col("s1_name").str.split(" ").list.last()
            ==
            pl.col("matched_name").str.split(" ").list.last()
        )
        .cast(pl.Float32)
        .alias("name_last_token_match"),

        # Token containment
        (
            pl.col("s1_name")
            .str.split(" ")
            .list.set_intersection(
                pl.col("matched_name").str.split(" ")
            )
            .list.len()
            > 0
        )
        .cast(pl.Float32)
        .alias("name_token_overlap"),

        # Length
        pl.col("s1_name")
        .str.len_chars()
        .cast(pl.Float32)
        .alias("name_len_s1"),

        pl.col("matched_name")
        .str.len_chars()
        .cast(pl.Float32)
        .alias("name_len_match"),
    ])

    # ========================================================
    # NAME DIFFERENCES
    # ========================================================

    chunk = chunk.with_columns([

        (
            pl.col("name_len_s1")
            -
            pl.col("name_len_match")
        )
        .abs()
        .cast(pl.Float32)
        .alias("name_length_abs_diff"),

        (
            (
                pl.col("name_len_s1")
                -
                pl.col("name_len_match")
            ).abs()
            /
            pl.max_horizontal(
                pl.col("name_len_s1"),
                pl.col("name_len_match"),
                pl.lit(1.0),
            )
        )
        .cast(pl.Float32)
        .alias("name_length_diff"),

        (
            pl.col("name_tokens_s1")
            -
            pl.col("name_tokens_match")
        )
        .abs()
        .cast(pl.Float32)
        .alias("name_token_count_diff"),
    ])

    # ========================================================
    # ADDRESS FEATURES
    # ========================================================

    chunk = chunk.with_columns([

        (
            pl.col("s1_address")
            ==
            pl.col("matched_address")
        )
        .cast(pl.Float32)
        .alias("address_exact"),

        # Prefixes
        (
            pl.col("s1_address").str.slice(0, 5)
            ==
            pl.col("matched_address").str.slice(0, 5)
        )
        .cast(pl.Float32)
        .alias("address_prefix5"),

        (
            pl.col("s1_address").str.slice(0, 8)
            ==
            pl.col("matched_address").str.slice(0, 8)
        )
        .cast(pl.Float32)
        .alias("address_prefix8"),

        # Suffixes
        (
            pl.col("s1_address").str.slice(-5, 5)
            ==
            pl.col("matched_address").str.slice(-5, 5)
        )
        .cast(pl.Float32)
        .alias("address_suffix5"),

        (
            pl.col("s1_address").str.slice(-8, 8)
            ==
            pl.col("matched_address").str.slice(-8, 8)
        )
        .cast(pl.Float32)
        .alias("address_suffix8"),

        # Length
        pl.col("s1_address")
        .str.len_chars()
        .cast(pl.Float32)
        .alias("address_len_s1"),

        pl.col("matched_address")
        .str.len_chars()
        .cast(pl.Float32)
        .alias("address_len_match"),

        # Jaccard
        jaccard_expr(
            "s1_address",
            "matched_address"
        )
        .cast(pl.Float32)
        .alias("address_jaccard"),

        # Numeric overlap
        numeric_overlap_expr(
            "s1_address",
            "matched_address"
        )
        .cast(pl.Float32)
        .alias("address_numeric_overlap"),

        # House number
        (
            (pl.col("_s1_house") != "")
            &
            (pl.col("_s1_house") == pl.col("_match_house"))
        )
        .cast(pl.Float32)
        .alias("address_house_match"),
    ])

    # ========================================================
    # ADDRESS DIFFERENCES
    # ========================================================

    chunk = chunk.with_columns([

        (
            pl.col("address_len_s1")
            -
            pl.col("address_len_match")
        )
        .abs()
        .cast(pl.Float32)
        .alias("address_length_abs_diff"),

        (
            (
                pl.col("address_len_s1")
                -
                pl.col("address_len_match")
            ).abs()
            /
            pl.max_horizontal(
                pl.col("address_len_s1"),
                pl.col("address_len_match"),
                pl.lit(1.0),
            )
        )
        .cast(pl.Float32)
        .alias("address_length_diff"),

        # Country
        (
            pl.col("s1_country")
            ==
            pl.col("matched_country")
        )
        .cast(pl.Float32)
        .alias("country_match"),

        (
            (pl.col("s1_country") == "")
            |
            (pl.col("matched_country") == "")
        )
        .cast(pl.Float32)
        .alias("country_missing"),

        (
            pl.col("matched_entity_id")
            .str.starts_with("S2-")
        )
        .cast(pl.Float32)
        .alias("source2_candidate"),

        (
            pl.col("matched_entity_id")
            .str.starts_with("S3-")
        )
        .cast(pl.Float32)
        .alias("source3_candidate"),
    ])

    # ========================================================
    # FINAL FEATURES
    # ========================================================

    feature_columns = [
        "source1_entity_id",
        "matched_entity_id",
        "label",

        # Name
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

        # Address
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

        # Country/source
        "country_match",
        "country_missing",
        "source2_candidate",
        "source3_candidate",
    ]

    features = chunk.select(feature_columns)

    # ========================================================
    # WRITE
    # ========================================================

    output_file = (
        OUTPUT_DIR
        / f"features_{start // CHUNK_SIZE:04d}.parquet"
    )

    features.write_parquet(
        output_file,
        compression="zstd",
    )

    print(f"Saved: {output_file}")

    del chunk
    del features


print("\n" + "=" * 70)
print("FEATURE GENERATION V2 COMPLETE")
print("=" * 70)

print(f"Output directory: {OUTPUT_DIR}")