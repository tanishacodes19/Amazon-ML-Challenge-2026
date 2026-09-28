import os
import polars as pl
import duckdb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORM = os.path.join(BASE, "normalized_data")
OUT = os.path.join(BASE, "training_data", "candidate_v3_blocks")

os.makedirs(OUT, exist_ok=True)

S1_FILE = os.path.join(NORM, "train_source1_normalized.tsv")
S2_FILE = os.path.join(NORM, "train_source2_normalized.tsv")
S3_FILE = os.path.join(NORM, "train_source3_normalized.tsv")


def load_source(path, source):
    return (
        pl.read_csv(
            path,
            separator="\t",
            infer_schema_length=1000,
            low_memory=True,
        )
        .with_columns([
            pl.col("entity_id").cast(pl.Utf8),
            pl.col("business_name").fill_null("").cast(pl.Utf8),
            pl.col("business_address").fill_null("").cast(pl.Utf8),
            pl.col("country").fill_null("").cast(pl.Utf8),
        ])
        .rename({
            "entity_id": "entity_id",
            "business_name": "name",
            "business_address": "address",
            "country": "country",
        })
        .with_columns(
            pl.lit(source).alias("source")
        )
    )


print("Loading normalized data...")

s1 = load_source(S1_FILE, "S1")
s2 = load_source(S2_FILE, "S2")
s3 = load_source(S3_FILE, "S3")

print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))


# ---------------------------------------------------------
# Combined candidate sources
# ---------------------------------------------------------

s23 = pl.concat([
    s2,
    s3
], how="vertical")

print("Combined S2+S3:", len(s23))


# ---------------------------------------------------------
# Useful derived columns
# ---------------------------------------------------------

def add_features(df):
    return df.with_columns([
        pl.col("name")
        .str.extract(r"^(\S{4})", 1)
        .fill_null("")
        .alias("name_prefix4"),

        pl.col("name")
        .str.extract(r"^(\S+)", 1)
        .fill_null("")
        .alias("first_token"),

        pl.col("name")
        .str.extract(r"(\S+)$", 1)
        .fill_null("")
        .alias("last_token"),

        pl.col("address")
        .str.extract(r"^(\d+)", 1)
        .fill_null("")
        .alias("house_number"),

        pl.col("name")
        .str.extract(r"^(\S{5})", 1)
        .fill_null("")
        .alias("name_prefix5"),
    ])


s1 = add_features(s1)
s23 = add_features(s23)


# =========================================================
# BLOCK 1
# country + name prefix4
# frequency capped at 50
# =========================================================

print("\n[1/4] Country + name_prefix4")

freq = (
    s23
    .filter(
        (pl.col("country") != "") &
        (pl.col("name_prefix4") != "")
    )
    .group_by(["country", "name_prefix4"])
    .len()
    .rename({"len": "freq"})
)

s1_b = s1.join(
    freq.filter(pl.col("freq") <= 50),
    on=["country", "name_prefix4"],
    how="inner"
)

s23_b = s23.join(
    freq.filter(pl.col("freq") <= 50)
    .select(["country", "name_prefix4"]),
    on=["country", "name_prefix4"],
    how="inner"
)

block1 = (
    s1_b.select([
        pl.col("entity_id").alias("source1_entity_id"),
        "country",
        "name_prefix4"
    ])
    .join(
        s23_b.select([
            pl.col("entity_id").alias("matched_entity_id"),
            "country",
            "name_prefix4"
        ]),
        on=["country", "name_prefix4"],
        how="inner"
    )
    .select([
        "source1_entity_id",
        "matched_entity_id"
    ])
    .unique()
)

path = os.path.join(OUT, "country_prefix4.tsv")
block1.write_csv(path, separator="\t")

print("Block 1:", len(block1))


# =========================================================
# BLOCK 2
# country + first token
# frequency <= 50
# =========================================================

print("\n[2/4] Country + first token")

freq = (
    s23
    .filter(
        (pl.col("country") != "") &
        (pl.col("first_token") != "")
    )
    .group_by(["country", "first_token"])
    .len()
    .rename({"len": "freq"})
)

valid = freq.filter(pl.col("freq") <= 50)

s1_b = s1.join(
    valid,
    on=["country", "first_token"],
    how="inner"
)

s23_b = s23.join(
    valid.select(["country", "first_token"]),
    on=["country", "first_token"],
    how="inner"
)

block2 = (
    s1_b.select([
        pl.col("entity_id").alias("source1_entity_id"),
        "country",
        "first_token"
    ])
    .join(
        s23_b.select([
            pl.col("entity_id").alias("matched_entity_id"),
            "country",
            "first_token"
        ]),
        on=["country", "first_token"],
        how="inner"
    )
    .select([
        "source1_entity_id",
        "matched_entity_id"
    ])
    .unique()
)

path = os.path.join(OUT, "country_first_token.tsv")
block2.write_csv(path, separator="\t")

print("Block 2:", len(block2))


# =========================================================
# BLOCK 3
# house number + name prefix4
# =========================================================

print("\n[3/4] House number + name prefix4")

freq = (
    s23
    .filter(
        (pl.col("house_number") != "") &
        (pl.col("name_prefix4") != "")
    )
    .group_by(["house_number", "name_prefix4"])
    .len()
    .rename({"len": "freq"})
)

valid = freq.filter(pl.col("freq") <= 50)

s1_b = s1.join(
    valid,
    on=["house_number", "name_prefix4"],
    how="inner"
)

s23_b = s23.join(
    valid.select(["house_number", "name_prefix4"]),
    on=["house_number", "name_prefix4"],
    how="inner"
)

block3 = (
    s1_b.select([
        pl.col("entity_id").alias("source1_entity_id"),
        "house_number",
        "name_prefix4"
    ])
    .join(
        s23_b.select([
            pl.col("entity_id").alias("matched_entity_id"),
            "house_number",
            "name_prefix4"
        ]),
        on=["house_number", "name_prefix4"],
        how="inner"
    )
    .select([
        "source1_entity_id",
        "matched_entity_id"
    ])
    .unique()
)

path = os.path.join(OUT, "house_prefix4.tsv")
block3.write_csv(path, separator="\t")

print("Block 3:", len(block3))


# =========================================================
# BLOCK 4
# house number + first token
# =========================================================

print("\n[4/4] House number + first token")

freq = (
    s23
    .filter(
        (pl.col("house_number") != "") &
        (pl.col("first_token") != "")
    )
    .group_by(["house_number", "first_token"])
    .len()
    .rename({"len": "freq"})
)

valid = freq.filter(pl.col("freq") <= 50)

s1_b = s1.join(
    valid,
    on=["house_number", "first_token"],
    how="inner"
)

s23_b = s23.join(
    valid.select(["house_number", "first_token"]),
    on=["house_number", "first_token"],
    how="inner"
)

block4 = (
    s1_b.select([
        pl.col("entity_id").alias("source1_entity_id"),
        "house_number",
        "first_token"
    ])
    .join(
        s23_b.select([
            pl.col("entity_id").alias("matched_entity_id"),
            "house_number",
            "first_token"
        ]),
        on=["house_number", "first_token"],
        how="inner"
    )
    .select([
        "source1_entity_id",
        "matched_entity_id"
    ])
    .unique()
)

path = os.path.join(OUT, "house_first_token.tsv")
block4.write_csv(path, separator="\t")

print("Block 4:", len(block4))


print("\n======================================")
print("V3 BLOCK GENERATION COMPLETE")
print("Output:", OUT)
print("======================================")