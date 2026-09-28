from pathlib import Path
import csv
import shutil

import numpy as np
import polars as pl
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(".")

NORM_DIR = BASE_DIR / "normalized_data"

MODEL_PATH = BASE_DIR / "model" / "xgboost_baseline.json"
THRESHOLD_PATH = BASE_DIR / "model" / "best_threshold.txt"

CANDIDATE_FILE = BASE_DIR / "candidate_pairs_real.tsv"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MATCHING_OUTPUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT = OUTPUT_DIR / "candidate_pairs.tsv"


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 50_000


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "name_exact",
    "name_len_s1",
    "name_len_match",
    "name_first_char_match",
    "name_jaccard",
    "name_length_diff",

    "address_exact",
    "address_len_s1",
    "address_len_match",
    "address_jaccard",
    "address_numeric_overlap",
    "address_length_diff",

    "country_match",
    "country_missing",

    "source2_candidate",
    "source3_candidate",
]


# ============================================================
# FEATURE HELPERS
# ============================================================

def jaccard_expr(col1, col2):
    tokens1 = pl.col(col1).str.split(" ")
    tokens2 = pl.col(col2).str.split(" ")

    intersection = (
        tokens1.list.set_intersection(tokens2).list.len()
    )

    union = (
        tokens1.list.set_union(tokens2).list.len()
    )

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

    intersection = (
        nums1.list.set_intersection(nums2).list.len()
    )

    return (
        pl.when(nums1.list.len() == 0)
        .then(0.0)
        .otherwise(
            intersection.cast(pl.Float64)
            / nums1.list.len().cast(pl.Float64)
        )
    )


# ============================================================
# FEATURE CREATION
# ============================================================

def create_features(chunk):

    # --------------------------------------------------------
    # JOIN S1
    # --------------------------------------------------------

    chunk = chunk.join(
        s1,
        on="source1_entity_id",
        how="left",
    )

    # --------------------------------------------------------
    # JOIN S2/S3
    # --------------------------------------------------------

    chunk = chunk.join(
        s23,
        on="matched_entity_id",
        how="left",
    )

    # --------------------------------------------------------
    # CLEAN STRINGS
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

    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    chunk = chunk.with_columns([

        (
            pl.col("s1_name")
            == pl.col("matched_name")
        )
        .cast(pl.Float32)
        .alias("name_exact"),

        pl.col("s1_name")
        .str.len_chars()
        .cast(pl.Float32)
        .alias("name_len_s1"),

        pl.col("matched_name")
        .str.len_chars()
        .cast(pl.Float32)
        .alias("name_len_match"),

        (
            pl.col("s1_name").str.slice(0, 1)
            ==
            pl.col("matched_name").str.slice(0, 1)
        )
        .cast(pl.Float32)
        .alias("name_first_char_match"),

        jaccard_expr(
            "s1_name",
            "matched_name"
        )
        .cast(pl.Float32)
        .alias("name_jaccard"),
    ])

    chunk = chunk.with_columns(
        (
            (
                pl.col("name_len_s1")
                - pl.col("name_len_match")
            ).abs()
            /
            pl.max_horizontal(
                pl.col("name_len_s1"),
                pl.col("name_len_match"),
                pl.lit(1.0),
            )
        )
        .cast(pl.Float32)
        .alias("name_length_diff")
    )

    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    chunk = chunk.with_columns([

        (
            pl.col("s1_address")
            ==
            pl.col("matched_address")
        )
        .cast(pl.Float32)
        .alias("address_exact"),

        pl.col("s1_address")
        .str.len_chars()
        .cast(pl.Float32)
        .alias("address_len_s1"),

        pl.col("matched_address")
        .str.len_chars()
        .cast(pl.Float32)
        .alias("address_len_match"),

        jaccard_expr(
            "s1_address",
            "matched_address"
        )
        .cast(pl.Float32)
        .alias("address_jaccard"),

        numeric_overlap_expr(
            "s1_address",
            "matched_address"
        )
        .cast(pl.Float32)
        .alias("address_numeric_overlap"),
    ])

    chunk = chunk.with_columns(
        (
            (
                pl.col("address_len_s1")
                - pl.col("address_len_match")
            ).abs()
            /
            pl.max_horizontal(
                pl.col("address_len_s1"),
                pl.col("address_len_match"),
                pl.lit(1.0),
            )
        )
        .cast(pl.Float32)
        .alias("address_length_diff")
    )

    # --------------------------------------------------------
    # COUNTRY / SOURCE FEATURES
    # --------------------------------------------------------

    chunk = chunk.with_columns([

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

    return chunk


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TEST PREDICTION")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading XGBoost model...")

model = xgb.XGBClassifier()

model.load_model(
    MODEL_PATH
)

threshold = float(
    THRESHOLD_PATH.read_text(
        encoding="utf-8"
    ).strip()
)

print(f"Model: {MODEL_PATH}")
print(f"Threshold: {threshold:.2f}")


# ============================================================
# COPY CANDIDATE FILE
# ============================================================

print("\nPreparing submission candidate file...")

shutil.copyfile(
    CANDIDATE_FILE,
    CANDIDATE_OUTPUT,
)

print(
    f"Candidate file copied to: "
    f"{CANDIDATE_OUTPUT}"
)


# ============================================================
# LOAD NORMALIZED SOURCES
# ============================================================

print("\nLoading normalized test sources...")

s1 = (
    pl.read_csv(
        NORM_DIR / "test_source1_normalized.tsv",
        separator="\t",
        infer_schema_length=1000,
    )
    .select([
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ])
    .rename({
        "entity_id": "source1_entity_id",
        "business_name": "s1_name",
        "business_address": "s1_address",
        "country": "s1_country",
    })
)

s2 = (
    pl.read_csv(
        NORM_DIR / "test_source2_normalized.tsv",
        separator="\t",
        infer_schema_length=1000,
    )
    .select([
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ])
    .rename({
        "entity_id": "matched_entity_id",
        "business_name": "matched_name",
        "business_address": "matched_address",
        "country": "matched_country",
    })
)

s3 = (
    pl.read_csv(
        NORM_DIR / "test_source3_normalized.tsv",
        separator="\t",
        infer_schema_length=1000,
    )
    .select([
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ])
    .rename({
        "entity_id": "matched_entity_id",
        "business_name": "matched_name",
        "business_address": "matched_address",
        "country": "matched_country",
    })
)

s23 = pl.concat(
    [s2, s3],
    how="vertical",
)

del s2
del s3

print(f"S1 rows: {len(s1):,}")
print(f"S2 + S3 rows: {len(s23):,}")


# ============================================================
# OPEN OUTPUT
# ============================================================

print("\nScoring test candidates...")

with open(
    CANDIDATE_FILE,
    "r",
    encoding="utf-8",
    newline="",
) as candidate_file, open(
    MATCHING_OUTPUT,
    "w",
    encoding="utf-8",
    newline="",
) as output_file:

    reader = csv.DictReader(
        candidate_file,
        delimiter="\t",
    )

    writer = csv.writer(
        output_file,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writerow([
        "source1_entity_id",
        "matched_entity_ids",
    ])

    batch = []

    processed_s1 = 0
    processed_candidates = 0
    matched_s1 = 0

    for row in reader:

        s1_id = row["source1_entity_id"]
        candidate_string = (
            row["candidate_entity_ids"] or ""
        )

        candidate_ids = [
            x.strip()
            for x in candidate_string.split(",")
            if x.strip()
        ]

        # No candidates
        if not candidate_ids:

            writer.writerow([
                s1_id,
                "",
            ])

            processed_s1 += 1

            continue

        for candidate_id in candidate_ids:

            batch.append({
                "source1_entity_id": s1_id,
                "matched_entity_id": candidate_id,
            })

        # ----------------------------------------------------
        # PROCESS BATCH
        # ----------------------------------------------------

        if len(batch) >= BATCH_SIZE:

            chunk = pl.DataFrame(batch)

            features = create_features(
                chunk
            )

            X = features.select(
                FEATURE_COLUMNS
            ).to_numpy()

            probabilities = model.predict_proba(
                X
            )[:, 1]

            features = features.with_columns(
                pl.Series(
                    "probability",
                    probabilities,
                )
            )

            # Group selected matches by S1
            selected = (
                features
                .filter(
                    pl.col("probability")
                    >= threshold
                )
                .group_by("source1_entity_id")
                .agg(
                    pl.col("matched_entity_id")
                    .alias("matches")
                )
            )

            selected_map = {
                row["source1_entity_id"]:
                    row["matches"]
                for row in selected.to_dicts()
            }

            # Write results for S1s represented in this batch
            batch_s1_ids = []

            for item in batch:
                if item["source1_entity_id"] not in batch_s1_ids:
                    batch_s1_ids.append(
                        item["source1_entity_id"]
                    )

            for current_s1 in batch_s1_ids:

                matches = selected_map.get(
                    current_s1,
                    [],
                )

                # Remove duplicates
                matches = list(
                    dict.fromkeys(matches)
                )

                if matches:
                    writer.writerow([
                        current_s1,
                        ",".join(matches),
                    ])

                    matched_s1 += 1

            processed_candidates += len(batch)

            processed_s1 += len(batch_s1_ids)

            print(
                f"Processed S1: "
                f"{processed_s1:,} | "
                f"Candidates: "
                f"{processed_candidates:,} | "
                f"S1 with matches: "
                f"{matched_s1:,}"
            )

            batch = []

    # ========================================================
    # FINAL BATCH
    # ========================================================

    if batch:

        chunk = pl.DataFrame(batch)

        features = create_features(
            chunk
        )

        X = features.select(
            FEATURE_COLUMNS
        ).to_numpy()

        probabilities = model.predict_proba(
            X
        )[:, 1]

        features = features.with_columns(
            pl.Series(
                "probability",
                probabilities,
            )
        )

        selected = (
            features
            .filter(
                pl.col("probability")
                >= threshold
            )
            .group_by("source1_entity_id")
            .agg(
                pl.col("matched_entity_id")
                .alias("matches")
            )
        )

        selected_map = {
            row["source1_entity_id"]:
                row["matches"]
            for row in selected.to_dicts()
        }

        batch_s1_ids = []

        for item in batch:
            if item["source1_entity_id"] not in batch_s1_ids:
                batch_s1_ids.append(
                    item["source1_entity_id"]
                )

        for current_s1 in batch_s1_ids:

            matches = selected_map.get(
                current_s1,
                [],
            )

            matches = list(
                dict.fromkeys(matches)
            )

            if matches:
                writer.writerow([
                    current_s1,
                    ",".join(matches),
                ])

                matched_s1 += 1

        processed_candidates += len(batch)
        processed_s1 += len(batch_s1_ids)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TEST PREDICTION COMPLETE")
print("=" * 70)

print(f"S1 entities processed : {processed_s1:,}")
print(f"Candidate pairs scored: {processed_candidates:,}")
print(f"S1 with matches       : {matched_s1:,}")

print(
    f"\nMatching output: "
    f"{MATCHING_OUTPUT}"
)

print(
    f"Candidate output: "
    f"{CANDIDATE_OUTPUT}"
)