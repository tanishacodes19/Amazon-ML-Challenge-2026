import os
import duckdb
import polars as pl
import numpy as np
import xgboost as xgb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

FEATURE_DIR = os.path.join(
    BASE,
    "training_data",
    "features_v4"
)

MODEL_DIR = os.path.join(
    BASE,
    "model"
)

DB_FILE = os.path.join(
    BASE,
    "v4_training.duckdb"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("V5.5 HARD-NEGATIVE XGBOOST TRAINING")
print("=" * 70)

con = duckdb.connect(DB_FILE)

con.execute("PRAGMA memory_limit='3500MB'")
con.execute("PRAGMA threads=2")

FEATURE_GLOB = os.path.join(
    FEATURE_DIR,
    "features_*.parquet"
).replace("\\", "/")

# =========================================================
# FEATURES
# =========================================================

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

# =========================================================
# INSPECT
# =========================================================

print("\nReading feature dataset...")

counts = con.execute(f"""
SELECT
    label,
    COUNT(*) AS n
FROM read_parquet('{FEATURE_GLOB}')
GROUP BY label
ORDER BY label
""").fetchall()

for row in counts:
    print(
        f"label={row[0]} rows={row[1]:,}"
    )

# =========================================================
# POSITIVES
# =========================================================

print("\nSelecting positive pairs...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE positives AS

SELECT
    {feature_sql},
    label,
    source1_entity_id
FROM read_parquet('{FEATURE_GLOB}')
WHERE label = 1
""")

positive_count = con.execute("""
SELECT COUNT(*)
FROM positives
""").fetchone()[0]

print(
    f"Available positives: {positive_count:,}"
)

# Use up to 1M positives
positive_target = min(
    positive_count,
    1_000_000
)

con.execute(f"""
CREATE OR REPLACE TEMP TABLE pos_sample AS

SELECT *
FROM positives
USING SAMPLE reservoir
({positive_target} ROWS)
REPEATABLE (42)
""")

print(
    f"Positive training rows: "
    f"{positive_target:,}"
)

# =========================================================
# HARD NEGATIVES
# =========================================================

print("\nSelecting HARD NEGATIVES...")

# These are deliberately difficult candidate pairs.
#
# Priority:
#   1. strong name similarity
#   2. shared first token
#   3. containment
#   4. address similarity
#   5. house-number agreement
#
# We intentionally avoid random easy negatives.

hard_negative_query = f"""

SELECT
    {feature_sql},
    label,
    source1_entity_id
FROM read_parquet('{FEATURE_GLOB}')

WHERE label = 0

AND (

       name_contains = 1

    OR name_first_token_match = 1

    OR name_token_jaccard >= 0.30

    OR name_ratio >= 0.65

    OR name_partial_ratio >= 0.75

    OR address_contains = 1

    OR address_ratio >= 0.60

    OR house_match = 1

    OR strong_name_address = 1

    OR strong_name_house = 1
)

"""

con.execute(f"""
CREATE OR REPLACE TEMP TABLE hard_negatives AS

{hard_negative_query}
""")

hard_count = con.execute("""
SELECT COUNT(*)
FROM hard_negatives
""").fetchone()[0]

print(
    f"Hard-negative pool: "
    f"{hard_count:,}"
)

if hard_count == 0:

    print(
        "\nERROR: No hard negatives found."
    )

    con.close()
    raise SystemExit(1)

# =========================================================
# SAMPLE HARD NEGATIVES
# =========================================================

negative_target = min(
    hard_count,
    1_500_000
)

print(
    f"Sampling {negative_target:,} hard negatives..."
)

con.execute(f"""
CREATE OR REPLACE TEMP TABLE neg_sample AS

SELECT *
FROM hard_negatives
USING SAMPLE reservoir
({negative_target} ROWS)
REPEATABLE (43)
""")

# =========================================================
# COMBINE
# =========================================================

print("\nCombining training data...")

con.execute("""
CREATE OR REPLACE TEMP TABLE sampled AS

SELECT *
FROM pos_sample

UNION ALL

SELECT *
FROM neg_sample
""")

total = con.execute("""
SELECT COUNT(*)
FROM sampled
""").fetchone()[0]

print(
    f"Total training rows: {total:,}"
)

print(
    con.execute("""
    SELECT
        label,
        COUNT(*)
    FROM sampled
    GROUP BY label
    ORDER BY label
    """).fetchall()
)

# =========================================================
# ENTITY-LEVEL SPLIT
# =========================================================

print("\nCreating entity-level split...")

con.execute("""
CREATE OR REPLACE TEMP TABLE train_data AS

SELECT *
FROM sampled

WHERE ABS(HASH(source1_entity_id)) % 4 != 0
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE valid_data AS

SELECT *
FROM sampled

WHERE ABS(HASH(source1_entity_id)) % 4 = 0
""")

train_count = con.execute("""
SELECT COUNT(*)
FROM train_data
""").fetchone()[0]

valid_count = con.execute("""
SELECT COUNT(*)
FROM valid_data
""").fetchone()[0]

print(
    f"Train rows: {train_count:,}"
)

print(
    f"Valid rows: {valid_count:,}"
)

# =========================================================
# LOAD MATRICES
# =========================================================

print("\nLoading matrices...")

train_df = con.execute(f"""
SELECT
    {feature_sql},
    label
FROM train_data
""").pl()

valid_df = con.execute(f"""
SELECT
    {feature_sql},
    label
FROM valid_data
""").pl()

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

print(
    "X_train:",
    X_train.shape
)

print(
    "X_valid:",
    X_valid.shape
)

# =========================================================
# XGBOOST
# =========================================================

print("\nTraining HARD-NEGATIVE XGBoost...")

model = xgb.XGBClassifier(

    n_estimators=900,

    max_depth=8,

    learning_rate=0.03,

    subsample=0.85,

    colsample_bytree=0.90,

    min_child_weight=3,

    gamma=0,

    reg_alpha=0.10,

    reg_lambda=2.0,

    objective="binary:logistic",

    eval_metric="logloss",

    tree_method="hist",

    max_bin=256,

    random_state=42,

    n_jobs=2
)

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_valid, y_valid)
    ],

    verbose=50
)

# =========================================================
# SAVE
# =========================================================

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_v55_hard_negative.json"
)

model.save_model(
    MODEL_FILE
)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Model:",
    MODEL_FILE
)

print(
    f"Training rows: {len(train_df):,}"
)

print(
    f"Validation rows: {len(valid_df):,}"
)

print("=" * 70)

con.close()