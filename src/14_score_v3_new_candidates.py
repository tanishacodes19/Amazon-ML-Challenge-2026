import os
import duckdb
import polars as pl
import xgboost as xgb
import numpy as np

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

NORM = os.path.join(BASE, "normalized_data")
DB_FILE = os.path.join(BASE, "v3_block_evaluation.duckdb")

MODEL_FILE = os.path.join(
    BASE, "model", "xgboost_v2.json"
)

OUT_DIR = os.path.join(
    BASE, "training_data", "v3_new_features"
)

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("V3 NEW-CANDIDATE SCORING")
print("=" * 60)


# =========================================================
# 1. Load V3 NEW candidates
# =========================================================

con = duckdb.connect(DB_FILE)

con.execute("PRAGMA memory_limit='4GB'")
con.execute("PRAGMA threads=2")

print("\nLoading NEW V3 candidates...")

new_count = con.execute("""
SELECT COUNT(*)
FROM new_candidates
""").fetchone()[0]

print("New candidates:", new_count)


# =========================================================
# 2. Load normalized training sources
# =========================================================

def load_source(path):
    return pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=1000,
        low_memory=True,
    ).with_columns([
        pl.col("entity_id").cast(pl.Utf8),
        pl.col("business_name").fill_null("").cast(pl.Utf8),
        pl.col("business_address").fill_null("").cast(pl.Utf8),
        pl.col("country").fill_null("").cast(pl.Utf8),
    ])


print("\nLoading normalized sources...")

s1 = load_source(
    os.path.join(
        NORM,
        "train_source1_normalized.tsv"
    )
)

s2 = load_source(
    os.path.join(
        NORM,
        "train_source2_normalized.tsv"
    )
)

s3 = load_source(
    os.path.join(
        NORM,
        "train_source3_normalized.tsv"
    )
)

s23 = pl.concat([s2, s3], how="vertical")

print("S1:", len(s1))
print("S2+S3:", len(s23))


# =========================================================
# 3. Rename columns
# =========================================================

s1 = s1.rename({
    "entity_id": "source1_entity_id",
    "business_name": "s1_name",
    "business_address": "s1_address",
    "country": "s1_country",
})

s23 = s23.rename({
    "entity_id": "matched_entity_id",
    "business_name": "matched_name",
    "business_address": "matched_address",
    "country": "matched_country",
})


# =========================================================
# 4. Get NEW candidates in chunks
# =========================================================

CHUNK_SIZE = 100_000

total_chunks = (
    new_count + CHUNK_SIZE - 1
) // CHUNK_SIZE


# =========================================================
# 5. Feature functions
# =========================================================

def add_features(df):

    df = df.with_columns([

        # -------------------------
        # NAME
        # -------------------------

        (
            pl.col("s1_name") ==
            pl.col("matched_name")
        ).cast(pl.Int8).alias("name_exact"),

        (
            pl.col("s1_name").str.slice(0, 2) ==
            pl.col("matched_name").str.slice(0, 2)
        ).cast(pl.Int8).alias("name_prefix2"),

        (
            pl.col("s1_name").str.slice(0, 3) ==
            pl.col("matched_name").str.slice(0, 3)
        ).cast(pl.Int8).alias("name_prefix3"),

        (
            pl.col("s1_name").str.slice(0, 4) ==
            pl.col("matched_name").str.slice(0, 4)
        ).cast(pl.Int8).alias("name_prefix4"),

        (
            pl.col("s1_name").str.slice(0, 5) ==
            pl.col("matched_name").str.slice(0, 5)
        ).cast(pl.Int8).alias("name_prefix5"),

        (
            pl.col("s1_name").str.slice(-3, 3) ==
            pl.col("matched_name").str.slice(-3, 3)
        ).cast(pl.Int8).alias("name_suffix3"),

        (
            pl.col("s1_name").str.slice(-4, 4) ==
            pl.col("matched_name").str.slice(-4, 4)
        ).cast(pl.Int8).alias("name_suffix4"),

        (
            pl.col("s1_name").str.slice(0, 1) ==
            pl.col("matched_name").str.slice(0, 1)
        ).cast(pl.Int8).alias("name_first_char_match"),

        (
            pl.col("s1_name").str.slice(-1, 1) ==
            pl.col("matched_name").str.slice(-1, 1)
        ).cast(pl.Int8).alias("name_last_char_match"),

        (
            pl.col("s1_name").str.extract(
                r"^(\S+)", 1
            ).fill_null("")
            ==
            pl.col("matched_name").str.extract(
                r"^(\S+)", 1
            ).fill_null("")
        ).cast(pl.Int8).alias("name_first_token_match"),

        (
            pl.col("s1_name").str.extract(
                r"(\S+)$", 1
            ).fill_null("")
            ==
            pl.col("matched_name").str.extract(
                r"(\S+)$", 1
            ).fill_null("")
        ).cast(pl.Int8).alias("name_last_token_match"),

        # -------------------------
        # ADDRESS
        # -------------------------

        (
            pl.col("s1_address") ==
            pl.col("matched_address")
        ).cast(pl.Int8).alias("address_exact"),

        (
            pl.col("s1_address").str.slice(0, 5) ==
            pl.col("matched_address").str.slice(0, 5)
        ).cast(pl.Int8).alias("address_prefix5"),

        (
            pl.col("s1_address").str.slice(0, 8) ==
            pl.col("matched_address").str.slice(0, 8)
        ).cast(pl.Int8).alias("address_prefix8"),

        (
            pl.col("s1_address").str.slice(-5, 5) ==
            pl.col("matched_address").str.slice(-5, 5)
        ).cast(pl.Int8).alias("address_suffix5"),

        (
            pl.col("s1_address").str.slice(-8, 8) ==
            pl.col("matched_address").str.slice(-8, 8)
        ).cast(pl.Int8).alias("address_suffix8"),

        pl.col("s1_address").str.len_chars()
        .alias("address_len_s1"),

        pl.col("matched_address").str.len_chars()
        .alias("address_len_match"),

        (
            pl.col("s1_address").str.len_chars()
            -
            pl.col("matched_address").str.len_chars()
        ).abs().alias("address_length_abs_diff"),

        (
            (
                pl.col("s1_address").str.len_chars()
                -
                pl.col("matched_address").str.len_chars()
            ).abs()
            /
            pl.when(
                pl.col("s1_address").str.len_chars() > 0
            )
            .then(pl.col("s1_address").str.len_chars())
            .otherwise(1)
        )
        .alias("address_length_diff"),

        # -------------------------
        # COUNTRY
        # -------------------------

        (
            pl.col("s1_country") ==
            pl.col("matched_country")
        ).cast(pl.Int8).alias("country_match"),

        (
            (pl.col("s1_country") == "") |
            (pl.col("matched_country") == "")
        ).cast(pl.Int8).alias("country_missing"),

        # -------------------------
        # SOURCE
        # -------------------------

        pl.col("matched_entity_id")
        .str.starts_with("S2-")
        .cast(pl.Int8)
        .alias("source2_candidate"),

        pl.col("matched_entity_id")
        .str.starts_with("S3-")
        .cast(pl.Int8)
        .alias("source3_candidate"),
    ])

    return df


# =========================================================
# 6. Process chunks
# =========================================================

model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)

print("\nModel loaded:", MODEL_FILE)

feature_columns = [
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
    "address_exact",
    "address_prefix5",
    "address_prefix8",
    "address_suffix5",
    "address_suffix8",
    "address_len_s1",
    "address_len_match",
    "address_length_abs_diff",
    "address_length_diff",
    "country_match",
    "country_missing",
    "source2_candidate",
    "source3_candidate",
]


# =========================================================
# IMPORTANT:
# Get candidate chunks using DuckDB LIMIT/OFFSET
# =========================================================

for chunk_no in range(total_chunks):

    offset = chunk_no * CHUNK_SIZE

    print(
        f"\n[{chunk_no + 1}/{total_chunks}] "
        f"offset={offset}"
    )

    pairs = con.execute("""
        SELECT
            source1_entity_id,
            matched_entity_id
        FROM new_candidates
        LIMIT ? OFFSET ?
    """, [CHUNK_SIZE, offset]).pl()

    if len(pairs) == 0:
        break

    # Join S1
    df = pairs.join(
        s1,
        on="source1_entity_id",
        how="left"
    )

    # Join S2/S3
    df = df.join(
        s23,
        on="matched_entity_id",
        how="left"
    )

    # Features
    df = add_features(df)

    X = (
        df
        .select(feature_columns)
        .fill_null(0)
        .to_numpy()
        .astype(np.float32)
    )

    probabilities = model.predict_proba(X)[:, 1]

    df_out = pl.DataFrame({
        "source1_entity_id":
            df["source1_entity_id"],

        "matched_entity_id":
            df["matched_entity_id"],

        "label":
            probabilities,
    })

    out_file = os.path.join(
        OUT_DIR,
        f"v3_scores_{chunk_no:04d}.parquet"
    )

    df_out.write_parquet(out_file)

    print(
        "Scored:",
        len(df_out),
        "| positive >= 0.53:",
        int((probabilities >= 0.53).sum())
    )


print("\n======================================")
print("V3 SCORING COMPLETE")
print("======================================")

con.close()