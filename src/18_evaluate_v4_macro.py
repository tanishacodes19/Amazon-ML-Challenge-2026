import os
import numpy as np
import polars as pl
import xgboost as xgb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

DB_FILE = os.path.join(
    BASE,
    "v4_training.duckdb"
)

MODEL_FILE = os.path.join(
    BASE,
    "model",
    "xgboost_v4.json"
)

print("=" * 70)
print("FAST V4 S1 MACRO F0.5 EVALUATION")
print("=" * 70)


# =========================================================
# Load validation data
# =========================================================

import duckdb

con = duckdb.connect(DB_FILE)

FEATURES = [
    "name_exact",
    "name_contains",
    "name_prefix3",
    "name_prefix5",
    "name_suffix4",
    "name_first_token_match",
    "name_last_token_match",
    "name_shared_token_count",
    "name_token_jaccard",
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort",
    "name_token_set",
    "name_len_s1",
    "name_len_match",
    "name_len_abs_diff",
    "name_len_relative_diff",
    "address_exact",
    "address_contains",
    "address_prefix5",
    "address_prefix8",
    "address_suffix8",
    "house_match",
    "address_ratio",
    "address_partial_ratio",
    "address_token_sort",
    "address_token_set",
    "address_token_jaccard",
    "address_numeric_overlap",
    "address_len_s1",
    "address_len_match",
    "address_len_abs_diff",
    "address_len_relative_diff",
    "country_match",
    "country_missing",
    "source2",
    "source3",
    "name_address_ratio_mean",
    "name_address_ratio_product",
    "strong_name_address",
    "strong_name_house",
]

feature_sql = ", ".join(FEATURES)

print("\nLoading validation data...")

valid = con.execute(f"""
SELECT
    source1_entity_id,
    label,
    source2,
    source3,
    {feature_sql}
FROM valid_data
""").pl()

print("Validation rows:", len(valid))
print(
    "Validation S1 entities:",
    valid["source1_entity_id"].n_unique()
)


# =========================================================
# Load model
# =========================================================

print("\nLoading V4 model...")

model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)

X = (
    valid
    .select(FEATURES)
    .fill_null(0)
    .to_numpy()
    .astype(np.float32)
)

print("Generating probabilities...")

prob = model.predict_proba(X)[:, 1]

labels = valid["label"].to_numpy()

# Integer group ID for each S1
group_id = (
    valid
    .select(
        pl.col("source1_entity_id")
        .cast(pl.Categorical)
        .to_physical()
        .alias("group_id")
    )
    ["group_id"]
    .to_numpy()
)

num_groups = int(group_id.max()) + 1

print("Groups:", num_groups)


# =========================================================
# FAST MACRO F0.5
# =========================================================

def evaluate_global(threshold):

    pred = prob >= threshold

    tp = np.bincount(
        group_id,
        weights=(
            (labels == 1) & pred
        ).astype(np.int32),
        minlength=num_groups
    )

    fp = np.bincount(
        group_id,
        weights=(
            (labels == 0) & pred
        ).astype(np.int32),
        minlength=num_groups
    )

    fn = np.bincount(
        group_id,
        weights=(
            (labels == 1) & (~pred)
        ).astype(np.int32),
        minlength=num_groups
    )

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(
            tp,
            dtype=float
        ),
        where=(tp + fp) > 0
    )

    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(
            tp,
            dtype=float
        ),
        where=(tp + fn) > 0
    )

    # Important singleton/no-prediction behavior:
    # If an S1 has no predicted match and no true match,
    # it gets F0.5 = 1.
    no_true_no_pred = (
        (tp + fn == 0) &
        (tp + fp == 0)
    )

    score = np.divide(
        1.25 * precision * recall,
        0.25 * precision + recall,
        out=np.zeros_like(
            precision,
            dtype=float
        ),
        where=(0.25 * precision + recall) > 0
    )

    score[no_true_no_pred] = 1.0

    return score.mean()


# =========================================================
# GLOBAL SEARCH
# =========================================================

print("\n" + "=" * 70)
print("S1 MACRO F0.5 — GLOBAL THRESHOLD")
print("=" * 70)

best_score = -1
best_threshold = None

for threshold in np.arange(
    0.40,
    0.951,
    0.01
):

    score = evaluate_global(threshold)

    print(
        f"threshold={threshold:.2f} "
        f"Macro-F0.5={score:.6f}"
    )

    if score > best_score:
        best_score = score
        best_threshold = threshold


# =========================================================
# SOURCE-SPECIFIC
# =========================================================

print("\n" + "=" * 70)
print("S1 MACRO F0.5 — SOURCE SPECIFIC")
print("=" * 70)


s2_mask = (
    valid["source2"]
    .to_numpy()
    .astype(bool)
)

s3_mask = (
    valid["source3"]
    .to_numpy()
    .astype(bool)
)


def evaluate_source_specific(
    s2_threshold,
    s3_threshold
):

    pred = np.where(
        s2_mask,
        prob >= s2_threshold,
        prob >= s3_threshold
    )

    tp = np.bincount(
        group_id,
        weights=(
            (labels == 1) & pred
        ).astype(np.int32),
        minlength=num_groups
    )

    fp = np.bincount(
        group_id,
        weights=(
            (labels == 0) & pred
        ).astype(np.int32),
        minlength=num_groups
    )

    fn = np.bincount(
        group_id,
        weights=(
            (labels == 1) & (~pred)
        ).astype(np.int32),
        minlength=num_groups
    )

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(
            tp,
            dtype=float
        ),
        where=(tp + fp) > 0
    )

    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(
            tp,
            dtype=float
        ),
        where=(tp + fn) > 0
    )

    score = np.divide(
        1.25 * precision * recall,
        0.25 * precision + recall,
        out=np.zeros_like(
            precision,
            dtype=float
        ),
        where=(0.25 * precision + recall) > 0
    )

    no_true_no_pred = (
        (tp + fn == 0) &
        (tp + fp == 0)
    )

    score[no_true_no_pred] = 1.0

    return score.mean()


best_specific = (-1, None, None)

# Search around the thresholds found previously.
for s2_t in np.arange(
    0.65,
    0.901,
    0.01
):

    for s3_t in np.arange(
        0.70,
        0.951,
        0.01
    ):

        score = evaluate_source_specific(
            s2_t,
            s3_t
        )

        if score > best_specific[0]:

            best_specific = (
                score,
                s2_t,
                s3_t
            )


print(
    f"Best S2 threshold: "
    f"{best_specific[1]:.2f}"
)

print(
    f"Best S3 threshold: "
    f"{best_specific[2]:.2f}"
)

print(
    f"Source-specific Macro-F0.5: "
    f"{best_specific[0]:.6f}"
)


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("FINAL V4 VALIDATION")
print("=" * 70)

print(
    f"Best global threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best global Macro-F0.5: "
    f"{best_score:.6f}"
)

print(
    f"Best S2 threshold: "
    f"{best_specific[1]:.2f}"
)

print(
    f"Best S3 threshold: "
    f"{best_specific[2]:.2f}"
)

print(
    f"Best source-specific Macro-F0.5: "
    f"{best_specific[0]:.6f}"
)

print(
    "\nV2 baseline: 0.952940"
)

if best_specific[0] > best_score:
    print(
        "\nRESULT: SOURCE-SPECIFIC THRESHOLDS ARE BETTER."
    )
else:
    print(
        "\nRESULT: GLOBAL THRESHOLD IS BETTER."
    )

print("\nDONE.")

con.close()