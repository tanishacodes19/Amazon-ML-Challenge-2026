from pathlib import Path
import glob
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(".")
FEATURE_DIR = BASE_DIR / "training_data" / "features_v2"

MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgboost_v2.json"
THRESHOLD_PATH = MODEL_DIR / "best_threshold_v2.txt"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

POSITIVE_TARGET = 1_000_000
NEGATIVE_TARGET = 1_000_000

VALIDATION_PERCENT = 20


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # COUNTRY / SOURCE
    # --------------------------------------------------------

    "country_match",
    "country_missing",

    "source2_candidate",
    "source3_candidate",
]


# ============================================================
# F0.5
# ============================================================

def f05(precision, recall):

    denominator = (
        0.25 * precision
        + recall
    )

    if denominator == 0:
        return 0.0

    return (
        1.25
        * precision
        * recall
        / denominator
    )


# ============================================================
# MACRO F0.5
# ============================================================

def macro_f05(df, threshold):

    df = df.copy()

    df["pred"] = (
        df["probability"] >= threshold
    ).astype(np.int8)

    scores = []

    for _, group in df.groupby(
        "source1_entity_id",
        sort=False
    ):

        y_true = group["label"].to_numpy()
        y_pred = group["pred"].to_numpy()

        tp = np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )

        fp = np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )

        fn = np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        # Correct no-match entity
        if tp == 0 and fp == 0 and fn == 0:

            score = 1.0

        elif tp == 0:

            score = 0.0

        else:

            score = f05(
                precision,
                recall
            )

        scores.append(score)

    return float(np.mean(scores))


# ============================================================
# FIND FEATURES
# ============================================================

feature_files = sorted(
    glob.glob(
        str(FEATURE_DIR / "features_*.parquet")
    )
)

print("=" * 70)
print("XGBOOST V2 TRAINING")
print("=" * 70)

print(
    f"\nFeature files found: "
    f"{len(feature_files)}"
)

if not feature_files:

    raise RuntimeError(
        "No V2 feature files found."
    )

for file in feature_files[:5]:
    print(
        "  ✓",
        Path(file).name
    )

if len(feature_files) > 5:

    print(
        f"  ... and "
        f"{len(feature_files) - 5} more"
    )


# ============================================================
# COLLECT BALANCED DATA
# ============================================================

columns_needed = [
    "source1_entity_id",
    "matched_entity_id",
    "label",
] + FEATURE_COLUMNS

frames = []

positive_rows = 0
negative_rows = 0

print("\nCollecting balanced training data...")

for file in feature_files:

    df = pd.read_parquet(
        file,
        columns=columns_needed,
    )

    pos = df[
        df["label"] == 1
    ]

    neg = df[
        df["label"] == 0
    ]


    # --------------------------------------------------------
    # POSITIVES
    # --------------------------------------------------------

    if positive_rows < POSITIVE_TARGET:

        remaining = (
            POSITIVE_TARGET
            - positive_rows
        )

        pos_take = pos.iloc[
            :remaining
        ]

        if len(pos_take) > 0:

            frames.append(pos_take)

            positive_rows += len(
                pos_take
            )


    # --------------------------------------------------------
    # NEGATIVES
    # --------------------------------------------------------

    if negative_rows < NEGATIVE_TARGET:

        remaining = (
            NEGATIVE_TARGET
            - negative_rows
        )

        neg_take = neg.iloc[
            :remaining
        ]

        if len(neg_take) > 0:

            frames.append(neg_take)

            negative_rows += len(
                neg_take
            )


    print(
        f"{Path(file).name}: "
        f"positives={positive_rows:,}/"
        f"{POSITIVE_TARGET:,}, "
        f"negatives={negative_rows:,}/"
        f"{NEGATIVE_TARGET:,}"
    )


    if (
        positive_rows >= POSITIVE_TARGET
        and negative_rows >= NEGATIVE_TARGET
    ):

        break


# ============================================================
# COMBINE
# ============================================================

data = pd.concat(
    frames,
    ignore_index=True,
)

data = data.sample(
    frac=1.0,
    random_state=RANDOM_SEED,
).reset_index(drop=True)


print(
    f"\nBalanced dataset: "
    f"{len(data):,} rows"
)

print(
    f"Positive rows: "
    f"{(data['label'] == 1).sum():,}"
)

print(
    f"Negative rows: "
    f"{(data['label'] == 0).sum():,}"
)


# ============================================================
# ENTITY-LEVEL SPLIT
# ============================================================

print(
    "\nCreating entity-level "
    "train/validation split..."
)

unique_s1 = (
    data["source1_entity_id"]
    .drop_duplicates()
    .to_numpy()
)

rng = np.random.default_rng(
    RANDOM_SEED
)

rng.shuffle(unique_s1)

split_index = int(
    len(unique_s1)
    * (
        1
        - VALIDATION_PERCENT / 100
    )
)

train_s1 = set(
    unique_s1[:split_index]
)

valid_s1 = set(
    unique_s1[split_index:]
)

train_mask = (
    data["source1_entity_id"]
    .isin(train_s1)
)

valid_mask = (
    data["source1_entity_id"]
    .isin(valid_s1)
)

train = data.loc[
    train_mask
].copy()

valid = data.loc[
    valid_mask
].copy()


print(
    f"Training S1 entities   : "
    f"{len(train_s1):,}"
)

print(
    f"Validation S1 entities : "
    f"{len(valid_s1):,}"
)

print(
    f"Training rows          : "
    f"{len(train):,}"
)

print(
    f"Validation rows        : "
    f"{len(valid):,}"
)


# ============================================================
# X / Y
# ============================================================

X_train = train[
    FEATURE_COLUMNS
].astype(np.float32)

y_train = train[
    "label"
].astype(np.int8)

X_valid = valid[
    FEATURE_COLUMNS
].astype(np.float32)

y_valid = valid[
    "label"
].astype(np.int8)


print("\nClass distribution:")

print(
    f"Training positives: "
    f"{int(y_train.sum()):,}"
)

print(
    f"Training negatives: "
    f"{int((y_train == 0).sum()):,}"
)

print(
    f"Validation positives: "
    f"{int(y_valid.sum()):,}"
)

print(
    f"Validation negatives: "
    f"{int((y_valid == 0).sum()):,}"
)


# ============================================================
# XGBOOST
# ============================================================

print("\nTraining XGBoost V2...")

model = xgb.XGBClassifier(

    n_estimators=500,

    max_depth=7,

    learning_rate=0.04,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_weight=2,

    gamma=0,

    reg_alpha=0.0,

    reg_lambda=1.0,

    objective="binary:logistic",

    eval_metric="logloss",

    tree_method="hist",

    max_bin=256,

    random_state=RANDOM_SEED,

    n_jobs=2,
)


model.fit(
    X_train,
    y_train,

    eval_set=[
        (
            X_valid,
            y_valid
        )
    ],

    verbose=True,
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print(
    "\nGenerating validation "
    "probabilities..."
)

valid_probabilities = (
    model.predict_proba(
        X_valid
    )[:, 1]
)


valid_result = valid[
    [
        "source1_entity_id",
        "matched_entity_id",
        "label",
    ]
].copy()

valid_result[
    "probability"
] = valid_probabilities


# ============================================================
# FINE THRESHOLD SEARCH
# ============================================================

print(
    "\nSearching threshold "
    "for Macro F0.5..."
)

thresholds = np.arange(
    0.20,
    0.951,
    0.01,
)

results = []

best_threshold = None
best_score = -1

for threshold in thresholds:

    score = macro_f05(
        valid_result,
        threshold,
    )

    results.append(
        (
            threshold,
            score,
        )
    )

    if score > best_score:

        best_score = score
        best_threshold = threshold

    print(
        f"Threshold {threshold:.2f}"
        f" -> Macro F0.5 = "
        f"{score:.6f}"
    )


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST V2 VALIDATION RESULT")
print("=" * 70)

print(
    f"Best threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Best Macro F0.5 : "
    f"{best_score:.6f}"
)


# ============================================================
# SAVE
# ============================================================

model.save_model(
    MODEL_PATH
)

THRESHOLD_PATH.write_text(
    str(best_threshold),
    encoding="utf-8",
)

print(
    f"\nModel saved to: "
    f"{MODEL_PATH}"
)

print(
    f"Threshold saved to: "
    f"{THRESHOLD_PATH}"
)

print("\n" + "=" * 70)
print("XGBOOST V2 COMPLETE")
print("=" * 70)