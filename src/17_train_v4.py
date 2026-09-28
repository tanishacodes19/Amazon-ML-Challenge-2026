import os
import duckdb
import polars as pl
import numpy as np
import xgboost as xgb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

FEATURE_DIR = os.path.join(
    BASE, "training_data", "features_v4"
)

MODEL_DIR = os.path.join(
    BASE, "model"
)

os.makedirs(MODEL_DIR, exist_ok=True)

DB_FILE = os.path.join(
    BASE, "v4_training.duckdb"
)

print("=" * 70)
print("V4 XGBOOST TRAINING")
print("=" * 70)


# =========================================================
# 1. Connect DuckDB
# =========================================================

con = duckdb.connect(DB_FILE)

con.execute("PRAGMA memory_limit='4GB'")
con.execute("PRAGMA threads=2")

FEATURE_GLOB = os.path.join(
    FEATURE_DIR,
    "features_*.parquet"
).replace("\\", "/")


# =========================================================
# 2. Inspect dataset
# =========================================================

print("\nReading V4 feature files...")

counts = con.execute(f"""
SELECT
    label,
    COUNT(*) AS n
FROM read_parquet('{FEATURE_GLOB}')
GROUP BY label
ORDER BY label
""").fetchall()

for row in counts:
    print("label =", row[0], "| rows =", row[1])


total = con.execute(f"""
SELECT COUNT(*)
FROM read_parquet('{FEATURE_GLOB}')
""").fetchone()[0]

print("Total feature rows:", total)


# =========================================================
# 3. Select balanced 2M dataset
#
# IMPORTANT:
# Selection happens BEFORE entity split.
# Then entity split guarantees the same S1 never appears
# in both train and validation.
# =========================================================

print("\nSelecting balanced dataset...")
print("1,000,000 positives")
print("1,000,000 negatives")


con.execute(f"""
CREATE OR REPLACE TABLE sampled AS

WITH positives AS (

    SELECT *
    FROM read_parquet('{FEATURE_GLOB}')
    WHERE label = 1

),

negatives AS (

    SELECT *
    FROM read_parquet('{FEATURE_GLOB}')
    WHERE label = 0

),

p AS (

    SELECT *
    FROM positives
    USING SAMPLE reservoir (1000000 ROWS)
    REPEATABLE (42)

),

n AS (

    SELECT *
    FROM negatives
    USING SAMPLE reservoir (1000000 ROWS)
    REPEATABLE (43)

)

SELECT * FROM p

UNION ALL

SELECT * FROM n
""")


sample_count = con.execute(
    "SELECT COUNT(*) FROM sampled"
).fetchone()[0]

print("Sampled rows:", sample_count)


# =========================================================
# 4. Entity-level split
# =========================================================

print("\nCreating entity-level train/validation split...")

# 75% train / 25% validation.
#
# The split depends ONLY on source1_entity_id.
# Therefore all pairs belonging to one S1 entity stay
# entirely inside train OR validation.

con.execute("""
CREATE OR REPLACE TABLE train_data AS

SELECT *
FROM sampled

WHERE
    ABS(HASH(source1_entity_id)) % 4 != 0
""")


con.execute("""
CREATE OR REPLACE TABLE valid_data AS

SELECT *
FROM sampled

WHERE
    ABS(HASH(source1_entity_id)) % 4 = 0
""")


train_count = con.execute(
    "SELECT COUNT(*) FROM train_data"
).fetchone()[0]

valid_count = con.execute(
    "SELECT COUNT(*) FROM valid_data"
).fetchone()[0]

print("Training rows:", train_count)
print("Validation rows:", valid_count)


print("\nTraining labels:")

print(
    con.execute("""
    SELECT label, COUNT(*)
    FROM train_data
    GROUP BY label
    ORDER BY label
    """).fetchall()
)


print("\nValidation labels:")

print(
    con.execute("""
    SELECT label, COUNT(*)
    FROM valid_data
    GROUP BY label
    ORDER BY label
    """).fetchall()
)


# =========================================================
# 5. Feature columns
# =========================================================

FEATURES = [

    # NAME
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

    # ADDRESS
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

    # COUNTRY
    "country_match",
    "country_missing",

    # SOURCE
    "source2",
    "source3",

    # COMBINED
    "name_address_ratio_mean",
    "name_address_ratio_product",
    "strong_name_address",
    "strong_name_house",
]


feature_sql = ", ".join(FEATURES)


# =========================================================
# 6. Load train / validation into NumPy
# =========================================================

print("\nLoading training matrix...")

train_df = con.execute(f"""
SELECT
    {feature_sql},
    label
FROM train_data
""").pl()

valid_df = con.execute(f"""
SELECT
    {feature_sql},
    label,
    source2,
    source3
FROM valid_data
""").pl()


print("Train matrix:", train_df.shape)
print("Validation matrix:", valid_df.shape)


X_train = (
    train_df
    .select(FEATURES)
    .fill_null(0)
    .to_numpy()
    .astype(np.float32)
)

y_train = (
    train_df["label"]
    .to_numpy()
    .astype(np.int8)
)

X_valid = (
    valid_df
    .select(FEATURES)
    .fill_null(0)
    .to_numpy()
    .astype(np.float32)
)

y_valid = (
    valid_df["label"]
    .to_numpy()
    .astype(np.int8)
)


print("\nX_train:", X_train.shape)
print("X_valid:", X_valid.shape)


# =========================================================
# 7. Train XGBoost
# =========================================================

print("\nTraining XGBoost V4...")


model = xgb.XGBClassifier(

    n_estimators=700,

    max_depth=8,

    learning_rate=0.035,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_weight=3,

    gamma=0,

    reg_alpha=0.05,

    reg_lambda=1.5,

    objective="binary:logistic",

    eval_metric="logloss",

    tree_method="hist",

    max_bin=256,

    random_state=42,

    n_jobs=2,
)


model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_valid, y_valid)
    ],

    verbose=50,
)


# =========================================================
# 8. Predict validation
# =========================================================

print("\nGenerating validation probabilities...")

probs = model.predict_proba(X_valid)[:, 1]


# =========================================================
# 9. Macro F0.5
# =========================================================

def f05(tp, fp, fn):

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    if precision == 0 and recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
        /
        (0.25 * precision + recall)
    )


def macro_f05(y_true, y_prob, threshold):

    pred = y_prob >= threshold

    # Every row is one candidate pair.
    # This is NOT the official per-S1 Macro F0.5 yet.
    #
    # We use it only as a diagnostic for threshold selection.

    tp = np.sum(
        (y_true == 1) & (pred == 1)
    )

    fp = np.sum(
        (y_true == 0) & (pred == 1)
    )

    fn = np.sum(
        (y_true == 1) & (pred == 0)
    )

    return f05(tp, fp, fn), tp, fp, fn


# =========================================================
# 10. Global threshold search
# =========================================================

print("\n" + "=" * 70)
print("THRESHOLD SEARCH")
print("=" * 70)

best_score = -1
best_threshold = None

for threshold in np.arange(
    0.20,
    0.951,
    0.01
):

    score, tp, fp, fn = macro_f05(
        y_valid,
        probs,
        threshold
    )

    if score > best_score:

        best_score = score
        best_threshold = threshold

        print(
            f"threshold={threshold:.2f} "
            f"F0.5={score:.6f} "
            f"TP={tp:,} "
            f"FP={fp:,} "
            f"FN={fn:,}"
        )


# =========================================================
# 11. S2 / S3 threshold diagnostics
# =========================================================

print("\n" + "=" * 70)
print("SOURCE-SPECIFIC THRESHOLD SEARCH")
print("=" * 70)


source2_mask = (
    valid_df["source2"]
    .to_numpy()
    .astype(bool)
)

source3_mask = (
    valid_df["source3"]
    .to_numpy()
    .astype(bool)
)


def search_source(
    mask,
    source_name
):

    best = (-1, None)

    yt = y_valid[mask]
    yp = probs[mask]

    for threshold in np.arange(
        0.20,
        0.951,
        0.01
    ):

        score, tp, fp, fn = macro_f05(
            yt,
            yp,
            threshold
        )

        if score > best[0]:
            best = (
                score,
                threshold
            )

    print(
        f"{source_name}: "
        f"best_threshold={best[1]:.2f} "
        f"diagnostic_F0.5={best[0]:.6f}"
    )

    return best


s2_best = search_source(
    source2_mask,
    "S2"
)

s3_best = search_source(
    source3_mask,
    "S3"
)


# =========================================================
# 12. Save model
# =========================================================

model_path = os.path.join(
    MODEL_DIR,
    "xgboost_v4.json"
)

model.save_model(model_path)


threshold_path = os.path.join(
    MODEL_DIR,
    "v4_threshold.txt"
)

with open(
    threshold_path,
    "w"
) as f:

    f.write(
        str(best_threshold)
    )


# Save source thresholds

with open(
    os.path.join(
        MODEL_DIR,
        "v4_s2_threshold.txt"
    ),
    "w"
) as f:

    f.write(
        str(s2_best[1])
    )


with open(
    os.path.join(
        MODEL_DIR,
        "v4_s3_threshold.txt"
    ),
    "w"
) as f:

    f.write(
        str(s3_best[1])
    )


# =========================================================
# 13. Final report
# =========================================================

print("\n" + "=" * 70)
print("V4 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best global threshold : {best_threshold:.2f}"
)

print(
    f"Diagnostic F0.5       : {best_score:.6f}"
)

print(
    f"S2 threshold          : {s2_best[1]:.2f}"
)

print(
    f"S3 threshold          : {s3_best[1]:.2f}"
)

print(
    "Model:",
    model_path
)

print("\nIMPORTANT:")
print(
    "The F0.5 above is a candidate-level diagnostic."
)
print(
    "We still need the official per-S1 Macro F0.5 "
    "evaluation before deciding whether V4 is better."
)

con.close()