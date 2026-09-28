# src/22_score_v5_new_candidates.py

import duckdb
import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb

from rapidfuzz.fuzz import (
    ratio,
    partial_ratio,
    token_sort_ratio,
    token_set_ratio
)

# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

V5_FILE = os.path.join(
    BASE,
    "training_candidate_pairs_v5.tsv"
)

BASELINE_FILE = os.path.join(
    BASE,
    "training_candidate_pairs.tsv"
)

GROUND_TRUTH_FILE = os.path.join(
    BASE,
    "ground_truth_pairs.tsv"
)

S1_FILE = os.path.join(
    BASE,
    "normalized_data",
    "train_source1_normalized.tsv"
)

S2_FILE = os.path.join(
    BASE,
    "normalized_data",
    "train_source2_normalized.tsv"
)

S3_FILE = os.path.join(
    BASE,
    "normalized_data",
    "train_source3_normalized.tsv"
)

MODEL_FILE = os.path.join(
    BASE,
    "model",
    "xgboost_v4.json"
)

OUTPUT_FILE = os.path.join(
    BASE,
    "v5_new_candidates_scored.tsv"
)

TMP_DIR = os.path.join(
    BASE,
    "duckdb_tmp"
)

os.makedirs(TMP_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 10000

# V4 validation indicated ~0.51 as the best global
# S1-level threshold on the sampled validation.
THRESHOLD = 0.51


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_ratio(a, b):
    if not a or not b:
        return 0.0

    return ratio(
        str(a),
        str(b)
    ) / 100.0


def safe_partial(a, b):
    if not a or not b:
        return 0.0

    return partial_ratio(
        str(a),
        str(b)
    ) / 100.0


def safe_token_sort(a, b):
    if not a or not b:
        return 0.0

    return token_sort_ratio(
        str(a),
        str(b)
    ) / 100.0


def safe_token_set(a, b):
    if not a or not b:
        return 0.0

    return token_set_ratio(
        str(a),
        str(b)
    ) / 100.0


def token_set(s):
    if not s:
        return set()

    return set(
        str(s).split()
    )


def numeric_tokens(s):
    if not s:
        return set()

    return {
        x
        for x in str(s).split()
        if any(c.isdigit() for c in x)
    }


def first_token(s):
    if not s:
        return ""

    return str(s).split()[0]


def last_token(s):
    if not s:
        return ""

    return str(s).split()[-1]


def house_number(s):
    if not s:
        return ""

    parts = str(s).split()

    if not parts:
        return ""

    token = parts[0]

    digits = ""

    for c in token:

        if c.isdigit():
            digits += c
        else:
            break

    return digits


def jaccard(a, b):

    sa = token_set(a)
    sb = token_set(b)

    if not sa and not sb:
        return 1.0

    if not sa or not sb:
        return 0.0

    return len(
        sa & sb
    ) / len(
        sa | sb
    )


def numeric_overlap(a, b):

    sa = numeric_tokens(a)
    sb = numeric_tokens(b)

    if not sa or not sb:
        return 0.0

    return len(
        sa & sb
    ) / len(
        sa | sb
    )


# ============================================================
# EXACT V4 FEATURE FUNCTION
# ============================================================

def make_features(df):

    n1 = df["name_normalized_s1"].fillna("").astype(str)
    n2 = df["name_normalized_s23"].fillna("").astype(str)

    a1 = df["address_normalized_s1"].fillna("").astype(str)
    a2 = df["address_normalized_s23"].fillna("").astype(str)

    c1 = df["country_s1"].fillna("").astype(str)
    c2 = df["country_s23"].fillna("").astype(str)

    result = pd.DataFrame(index=df.index)

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    result["name_exact"] = (
        (n1 == n2) &
        (n1 != "")
    ).astype(np.int8)

    result["name_contains"] = [
        int(
            bool(x) and bool(y) and
            (x in y or y in x)
        )
        for x, y in zip(n1, n2)
    ]

    result["name_prefix3"] = [
        int(
            len(x) >= 3 and
            len(y) >= 3 and
            x[:3] == y[:3]
        )
        for x, y in zip(n1, n2)
    ]

    result["name_prefix5"] = [
        int(
            len(x) >= 5 and
            len(y) >= 5 and
            x[:5] == y[:5]
        )
        for x, y in zip(n1, n2)
    ]

    result["name_suffix4"] = [
        int(
            len(x) >= 4 and
            len(y) >= 4 and
            x[-4:] == y[-4:]
        )
        for x, y in zip(n1, n2)
    ]

    result["first_token_match"] = [
        int(
            first_token(x) != "" and
            first_token(x) == first_token(y)
        )
        for x, y in zip(n1, n2)
    ]

    result["last_token_match"] = [
        int(
            last_token(x) != "" and
            last_token(x) == last_token(y)
        )
        for x, y in zip(n1, n2)
    ]

    result["shared_name_tokens"] = [
        len(
            token_set(x) &
            token_set(y)
        )
        for x, y in zip(n1, n2)
    ]

    result["name_token_jaccard"] = [
        jaccard(x, y)
        for x, y in zip(n1, n2)
    ]

    result["name_ratio"] = [
        safe_ratio(x, y)
        for x, y in zip(n1, n2)
    ]

    result["name_partial_ratio"] = [
        safe_partial(x, y)
        for x, y in zip(n1, n2)
    ]

    result["name_token_sort"] = [
        safe_token_sort(x, y)
        for x, y in zip(n1, n2)
    ]

    result["name_token_set"] = [
        safe_token_set(x, y)
        for x, y in zip(n1, n2)
    ]

    result["name_len_s1"] = [
        len(x)
        for x in n1
    ]

    result["name_len_match"] = [
        len(x)
        for x in n2
    ]

    result["name_len_absdiff"] = [
        abs(len(x) - len(y))
        for x, y in zip(n1, n2)
    ]

    result["name_len_reldiff"] = [
        abs(len(x) - len(y)) /
        max(len(x), len(y), 1)
        for x, y in zip(n1, n2)
    ]

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    result["address_exact"] = (
        (a1 == a2) &
        (a1 != "")
    ).astype(np.int8)

    result["address_contains"] = [
        int(
            bool(x) and bool(y) and
            (x in y or y in x)
        )
        for x, y in zip(a1, a2)
    ]

    result["address_prefix5"] = [
        int(
            len(x) >= 5 and
            len(y) >= 5 and
            x[:5] == y[:5]
        )
        for x, y in zip(a1, a2)
    ]

    result["address_prefix8"] = [
        int(
            len(x) >= 8 and
            len(y) >= 8 and
            x[:8] == y[:8]
        )
        for x, y in zip(a1, a2)
    ]

    result["address_suffix8"] = [
        int(
            len(x) >= 8 and
            len(y) >= 8 and
            x[-8:] == y[-8:]
        )
        for x, y in zip(a1, a2)
    ]

    result["house_match"] = [
        int(
            house_number(x) != "" and
            house_number(x) == house_number(y)
        )
        for x, y in zip(a1, a2)
    ]

    result["address_ratio"] = [
        safe_ratio(x, y)
        for x, y in zip(a1, a2)
    ]

    result["address_partial_ratio"] = [
        safe_partial(x, y)
        for x, y in zip(a1, a2)
    ]

    result["address_token_sort"] = [
        safe_token_sort(x, y)
        for x, y in zip(a1, a2)
    ]

    result["address_token_set"] = [
        safe_token_set(x, y)
        for x, y in zip(a1, a2)
    ]

    result["address_token_jaccard"] = [
        jaccard(x, y)
        for x, y in zip(a1, a2)
    ]

    result["address_numeric_overlap"] = [
        numeric_overlap(x, y)
        for x, y in zip(a1, a2)
    ]

    result["address_len_s1"] = [
        len(x)
        for x in a1
    ]

    result["address_len_match"] = [
        len(x)
        for x in a2
    ]

    result["address_len_absdiff"] = [
        abs(len(x) - len(y))
        for x, y in zip(a1, a2)
    ]

    result["address_len_reldiff"] = [
        abs(len(x) - len(y)) /
        max(len(x), len(y), 1)
        for x, y in zip(a1, a2)
    ]

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    result["country_match"] = (
        (c1 == c2) &
        (c1 != "") &
        (c2 != "")
    ).astype(np.int8)

    result["country_missing"] = (
        (c1 == "") |
        (c2 == "")
    ).astype(np.int8)

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    ids = (
        df["candidate_entity_id"]
        .fillna("")
        .astype(str)
    )

    result["source2_candidate"] = (
        ids.str.startswith("S2-")
    ).astype(np.int8)

    result["source3_candidate"] = (
        ids.str.startswith("S3-")
    ).astype(np.int8)

    # --------------------------------------------------------
    # COMBINED FEATURES
    # --------------------------------------------------------

    result["combined_mean"] = (
        result["name_ratio"] +
        result["address_ratio"]
    ) / 2.0

    result["combined_product"] = (
        result["name_ratio"] *
        result["address_ratio"]
    )

    result["strong_name_address"] = (
        (result["name_ratio"] >= 0.90) &
        (result["address_ratio"] >= 0.70)
    ).astype(np.int8)

    result["strong_name_house"] = (
        (result["name_ratio"] >= 0.90) &
        (result["house_match"] == 1)
    ).astype(np.int8)

    return result


# ============================================================
# START
# ============================================================

print("=" * 78)
print("V5 NEW-CANDIDATE V4 MODEL SCORING")
print("=" * 78)

print(
    f"\nV5 file: {V5_FILE}"
)

print(
    f"Model: {MODEL_FILE}"
)

print(
    f"Threshold: {THRESHOLD}"
)


# ============================================================
# DUCKDB LOAD
# ============================================================

con = duckdb.connect()

con.execute(
    "PRAGMA threads=2"
)

con.execute(
    "PRAGMA memory_limit='3000MB'"
)

con.execute(
    "PRAGMA preserve_insertion_order=false"
)

con.execute(
    f"SET temp_directory='{TMP_DIR.replace(chr(92), '/')}'"
)


# ============================================================
# LOAD BASELINE
# ============================================================

print("\nLoading baseline candidates...")

con.execute(f"""
    CREATE OR REPLACE TABLE baseline AS

    SELECT DISTINCT

        column0 AS source1_entity_id,
        column1 AS candidate_entity_id

    FROM read_csv(
        '{BASELINE_FILE.replace(chr(92), '/')}',

        delim='\\t',
        header=true,
        quote='',

        columns={{
            'column0':'VARCHAR',
            'column1':'VARCHAR'
        }}
    )
""")


# ============================================================
# LOAD V5
# ============================================================

print("Loading V5 candidates...")

con.execute(f"""
    CREATE OR REPLACE TABLE v5 AS

    SELECT DISTINCT

        column0 AS source1_entity_id,
        column1 AS candidate_entity_id

    FROM read_csv(
        '{V5_FILE.replace(chr(92), '/')}',

        delim='\\t',
        header=true,
        quote='',

        columns={{
            'column0':'VARCHAR',
            'column1':'VARCHAR'
        }}
    )
""")


# ============================================================
# EXTRACT ONLY NEW CANDIDATES
# ============================================================

print("Finding NEW V5 candidates...")

con.execute("""
    CREATE OR REPLACE TABLE new_candidates AS

    SELECT

        v.source1_entity_id,
        v.candidate_entity_id

    FROM v5 AS v

    LEFT JOIN baseline AS b

        ON v.source1_entity_id =
           b.source1_entity_id

        AND v.candidate_entity_id =
            b.candidate_entity_id

    WHERE b.candidate_entity_id IS NULL
""")


new_count = con.execute("""
    SELECT COUNT(*)
    FROM new_candidates
""").fetchone()[0]

print(
    f"NEW V5 candidates: "
    f"{new_count:,}"
)


# ============================================================
# LOAD S1
# ============================================================

print("\nLoading S1...")

con.execute(f"""
    CREATE OR REPLACE TABLE s1 AS

    SELECT

        entity_id,
        country,
        name_normalized,
        address_normalized

    FROM read_csv(
        '{S1_FILE.replace(chr(92), '/')}',

        delim='\\t',
        header=true,
        quote='',

        columns={{
            'entity_id':'VARCHAR',
            'business_name':'VARCHAR',
            'business_address':'VARCHAR',
            'country':'VARCHAR',
            'name_normalized':'VARCHAR',
            'address_normalized':'VARCHAR'
        }}
    )
""")


# ============================================================
# LOAD S2 + S3
# ============================================================

print("Loading S2 + S3...")

con.execute(f"""
    CREATE OR REPLACE TABLE s23 AS

    SELECT

        entity_id,
        country,
        name_normalized,
        address_normalized

    FROM read_csv(
        '{S2_FILE.replace(chr(92), '/')}',

        delim='\\t',
        header=true,
        quote='',

        columns={{
            'entity_id':'VARCHAR',
            'business_name':'VARCHAR',
            'business_address':'VARCHAR',
            'country':'VARCHAR',
            'name_normalized':'VARCHAR',
            'address_normalized':'VARCHAR'
        }}
    )

    UNION ALL

    SELECT

        entity_id,
        country,
        name_normalized,
        address_normalized

    FROM read_csv(
        '{S3_FILE.replace(chr(92), '/')}',

        delim='\\t',
        header=true,
        quote='',

        columns={{
            'entity_id':'VARCHAR',
            'business_name':'VARCHAR',
            'business_address':'VARCHAR',
            'country':'VARCHAR',
            'name_normalized':'VARCHAR',
            'address_normalized':'VARCHAR'
        }}
    )
""")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

con.execute(f"""
    CREATE OR REPLACE TABLE ground_truth AS

    SELECT DISTINCT

        column0 AS source1_entity_id,
        column1 AS matched_entity_id

    FROM read_csv(
        '{GROUND_TRUTH_FILE.replace(chr(92), '/')}',

        delim='\\t',
        header=true,
        quote='',

        columns={{
            'column0':'VARCHAR',
            'column1':'VARCHAR'
        }}
    )
""")


# ============================================================
# LOAD V4 MODEL
# ============================================================

print("\nLoading XGBoost V4 model...")

model = xgb.XGBClassifier()

model.load_model(
    MODEL_FILE
)

print("Model loaded.")


# ============================================================
# SCORE IN BATCHES
# ============================================================

print("\nScoring NEW V5 candidates...")

feature_columns = None

total_scored = 0

all_predictions = []

start_total = time.time()


while True:

    batch = con.execute(f"""
        SELECT

            n.source1_entity_id,

            n.candidate_entity_id,

            s1.country AS country_s1,
            s1.name_normalized AS name_normalized_s1,
            s1.address_normalized AS address_normalized_s1,

            s23.country AS country_s23,
            s23.name_normalized AS name_normalized_s23,
            s23.address_normalized AS address_normalized_s23

        FROM new_candidates AS n

        JOIN s1

            ON n.source1_entity_id =
               s1.entity_id

        JOIN s23

            ON n.candidate_entity_id =
               s23.entity_id

        LIMIT {BATCH_SIZE}
        OFFSET {total_scored}
    """).df()

    if len(batch) == 0:
        break

    feature_df = make_features(
        batch
    )

    if feature_columns is None:

        feature_columns = list(
            feature_df.columns
        )

        print(
            f"Features: "
            f"{len(feature_columns)}"
        )

    feature_df = feature_df[
        feature_columns
    ]

    probabilities = model.predict_proba(
        feature_df
    )[:, 1]

    batch["probability"] = probabilities

    batch["prediction"] = (
        probabilities >= THRESHOLD
    ).astype(np.int8)

    all_predictions.append(
        batch[
            [
                "source1_entity_id",
                "candidate_entity_id",
                "probability",
                "prediction"
            ]
        ]
    )

    total_scored += len(batch)

    if total_scored % 100000 < BATCH_SIZE:

        elapsed = (
            time.time() -
            start_total
        )

        print(
            f"  Scored "
            f"{total_scored:,} / "
            f"{new_count:,} "
            f"({elapsed:.1f}s)"
        )


# ============================================================
# COMBINE RESULTS
# ============================================================

scored = pd.concat(
    all_predictions,
    ignore_index=True
)

print(
    f"\nTotal scored: "
    f"{len(scored):,}"
)


# ============================================================
# SAVE SCORED NEW CANDIDATES
# ============================================================

scored.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

print(
    f"Saved scored candidates:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# MODEL PREDICTION SUMMARY
# ============================================================

predicted_count = int(
    scored["prediction"].sum()
)

print("\n")
print("=" * 78)
print("V5 NEW-CANDIDATE MODEL SUMMARY")
print("=" * 78)

print(
    f"NEW candidates       : "
    f"{len(scored):,}"
)

print(
    f"Predicted as matches : "
    f"{predicted_count:,}"
)


# ============================================================
# JOIN WITH GROUND TRUTH
# ============================================================

truth_df = con.execute("""
    SELECT

        source1_entity_id,
        matched_entity_id

    FROM ground_truth
""").df()


truth_df["is_true"] = 1


scored_eval = scored.merge(

    truth_df,

    left_on=[
        "source1_entity_id",
        "candidate_entity_id"
    ],

    right_on=[
        "source1_entity_id",
        "matched_entity_id"
    ],

    how="left"
)


scored_eval["is_true"] = (
    scored_eval["is_true"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# CONFUSION COUNTS
# ============================================================

tp = int(
    (
        (scored_eval["prediction"] == 1) &
        (scored_eval["is_true"] == 1)
    ).sum()
)

fp = int(
    (
        (scored_eval["prediction"] == 1) &
        (scored_eval["is_true"] == 0)
    ).sum()
)

fn = int(
    (
        (scored_eval["prediction"] == 0) &
        (scored_eval["is_true"] == 1)
    ).sum()
)


# ============================================================
# METRICS
# ============================================================

if tp + fp > 0:

    precision = (
        tp /
        (tp + fp)
    )

else:

    precision = 0.0


if tp + fn > 0:

    recall = (
        tp /
        (tp + fn)
    )

else:

    recall = 0.0


beta = 0.5

beta2 = beta * beta

denominator = (
    beta2 * precision +
    recall
)

if denominator > 0:

    f05 = (
        (1 + beta2) *
        precision *
        recall /
        denominator
    )

else:

    f05 = 0.0


# ============================================================
# PRINT
# ============================================================

print("\nConfusion counts:")

print(
    f"  TP : {tp:,}"
)

print(
    f"  FP : {fp:,}"
)

print(
    f"  FN : {fn:,}"
)

print("\nMetrics:")

print(
    f"  Precision : {precision:.6f}"
)

print(
    f"  Recall    : {recall:.6f}"
)

print(
    f"  F0.5      : {f05:.6f}"
)


# ============================================================
# THRESHOLD SWEEP
# ============================================================

print("\n")
print("=" * 78)
print("THRESHOLD SWEEP ON NEW V5 CANDIDATES")
print("=" * 78)

for threshold in [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.51,
    0.52,
    0.53,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]:

    pred = (
        scored_eval["probability"] >=
        threshold
    )

    tp_t = int(
        (
            pred &
            (scored_eval["is_true"] == 1)
        ).sum()
    )

    fp_t = int(
        (
            pred &
            (scored_eval["is_true"] == 0)
        ).sum()
    )

    fn_t = int(
        (
            (~pred) &
            (scored_eval["is_true"] == 1)
        ).sum()
    )

    if tp_t + fp_t > 0:

        p = tp_t / (
            tp_t + fp_t
        )

    else:

        p = 0.0

    if tp_t + fn_t > 0:

        r = tp_t / (
            tp_t + fn_t
        )

    else:

        r = 0.0

    den = (
        beta2 * p +
        r
    )

    if den > 0:

        score = (
            (1 + beta2) *
            p *
            r /
            den
        )

    else:

        score = 0.0

    print(
        f"{threshold:.2f} | "
        f"P={p:.6f} | "
        f"R={r:.6f} | "
        f"F0.5={score:.6f} | "
        f"pred={int(pred.sum()):,}"
    )


# ============================================================
# DONE
# ============================================================

elapsed_total = (
    time.time() -
    start_total
)

print("\n")
print("=" * 78)
print("DONE")
print("=" * 78)

print(
    f"Total scoring time: "
    f"{elapsed_total:.1f}s"
)

print(
    f"Output: "
    f"{OUTPUT_FILE}"
)

con.close()