# src/23_evaluate_v5_full.py

import os
import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
from rapidfuzz import fuzz

# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

GROUND_TRUTH = os.path.join(
    BASE, "ground_truth_pairs.tsv"
)

BASELINE_CANDIDATES = os.path.join(
    BASE, "training_candidate_pairs.tsv"
)

NEW_SCORED = os.path.join(
    BASE, "v5_new_candidates_scored.tsv"
)

MODEL_PATH = os.path.join(
    BASE,
    "model",
    "xgboost_v55_hard_negative.json"
)

TMP_DIR = os.path.join(
    BASE, "duckdb_tmp"
)

# IMPORTANT:
# Same baseline threshold used in our previous V4 evaluation.
BASELINE_THRESHOLD = 0.51

# Thresholds for ONLY the new V5 candidates.
NEW_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.82,
    0.85,
    0.87,
    0.90,
    0.92,
    0.95
]

# 50k keeps RAM reasonable on your 8 GB laptop.
N_S1 = 50_000

# Candidate scoring batch.
SCORE_BATCH = 10_000

os.makedirs(TMP_DIR, exist_ok=True)


# ============================================================
# V4 FEATURE FUNCTIONS
# EXACT DEFINITIONS USED BY src/16_generate_v4_features.py
# ============================================================

def safe_ratio(a, b):
    try:
        return fuzz.ratio(a, b) / 100.0
    except:
        return 0.0


def safe_partial(a, b):
    try:
        return fuzz.partial_ratio(a, b) / 100.0
    except:
        return 0.0


def safe_token_sort(a, b):
    try:
        return fuzz.token_sort_ratio(a, b) / 100.0
    except:
        return 0.0


def safe_token_set(a, b):
    try:
        return fuzz.token_set_ratio(a, b) / 100.0
    except:
        return 0.0


def token_set(s):
    if not s:
        return set()

    return set(
        x for x in str(s).split()
        if x
    )


def numeric_tokens(s):
    if not s:
        return set()

    return set(
        x for x in str(s).split()
        if any(c.isdigit() for c in x)
    )


def first_token(s):
    if not s:
        return ""

    parts = str(s).split()

    return parts[0] if parts else ""


def last_token(s):
    if not s:
        return ""

    parts = str(s).split()

    return parts[-1] if parts else ""


def house_number(s):
    if not s:
        return ""

    parts = str(s).split()

    for x in parts:

        if any(c.isdigit() for c in x):

            digits = ""

            for c in x:

                if c.isdigit():
                    digits += c

                else:
                    break

            if digits:
                return digits

    return ""


def jaccard(a, b):

    sa = token_set(a)
    sb = token_set(b)

    if not sa and not sb:
        return 1.0

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / len(sa | sb)


def numeric_overlap(a, b):

    sa = numeric_tokens(a)
    sb = numeric_tokens(b)

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / max(
        1,
        min(len(sa), len(sb))
    )


def make_features(df):

    n1 = df["name_normalized_s1"].fillna("").astype(str)
    n2 = df["name_normalized_s2"].fillna("").astype(str)

    a1 = df["address_normalized_s1"].fillna("").astype(str)
    a2 = df["address_normalized_s2"].fillna("").astype(str)

    c1 = df["country_s1"].fillna("").astype(str)
    c2 = df["country_s2"].fillna("").astype(str)

    h1 = a1.map(house_number)
    h2 = a2.map(house_number)

    f1 = n1.map(first_token)
    f2 = n2.map(first_token)

    l1 = n1.map(last_token)
    l2 = n2.map(last_token)

    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    name_ratio = np.array([
        safe_ratio(x, y)
        for x, y in zip(n1, n2)
    ])

    name_partial = np.array([
        safe_partial(x, y)
        for x, y in zip(n1, n2)
    ])

    name_token_sort = np.array([
        safe_token_sort(x, y)
        for x, y in zip(n1, n2)
    ])

    name_token_set = np.array([
        safe_token_set(x, y)
        for x, y in zip(n1, n2)
    ])

    name_jaccard = np.array([
        jaccard(x, y)
        for x, y in zip(n1, n2)
    ])

    shared_name = np.array([
        len(token_set(x) & token_set(y))
        for x, y in zip(n1, n2)
    ])

    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    addr_ratio = np.array([
        safe_ratio(x, y)
        for x, y in zip(a1, a2)
    ])

    addr_partial = np.array([
        safe_partial(x, y)
        for x, y in zip(a1, a2)
    ])

    addr_token_sort = np.array([
        safe_token_sort(x, y)
        for x, y in zip(a1, a2)
    ])

    addr_token_set = np.array([
        safe_token_set(x, y)
        for x, y in zip(a1, a2)
    ])

    addr_jaccard = np.array([
        jaccard(x, y)
        for x, y in zip(a1, a2)
    ])

    addr_numeric_overlap = np.array([
        numeric_overlap(x, y)
        for x, y in zip(a1, a2)
    ])

    # --------------------------------------------------------
    # LENGTHS
    # --------------------------------------------------------

    nlen1 = n1.str.len().to_numpy()
    nlen2 = n2.str.len().to_numpy()

    alen1 = a1.str.len().to_numpy()
    alen2 = a2.str.len().to_numpy()

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    out = pd.DataFrame({

        # NAME
        "name_exact":
            (n1 == n2).astype(np.int8),

        "name_contains":
            [
                int(
                    bool(x)
                    and bool(y)
                    and (x in y or y in x)
                )
                for x, y in zip(n1, n2)
            ],

        "name_prefix3":
            [
                int(
                    len(x) >= 3
                    and len(y) >= 3
                    and x[:3] == y[:3]
                )
                for x, y in zip(n1, n2)
            ],

        "name_prefix5":
            [
                int(
                    len(x) >= 5
                    and len(y) >= 5
                    and x[:5] == y[:5]
                )
                for x, y in zip(n1, n2)
            ],

        "name_suffix4":
            [
                int(
                    len(x) >= 4
                    and len(y) >= 4
                    and x[-4:] == y[-4:]
                )
                for x, y in zip(n1, n2)
            ],

        "name_first_token":
            [
                int(bool(x) and bool(y) and x == y)
                for x, y in zip(f1, f2)
            ],

        "name_last_token":
            [
                int(bool(x) and bool(y) and x == y)
                for x, y in zip(l1, l2)
            ],

        "shared_name_tokens":
            shared_name,

        "name_token_jaccard":
            name_jaccard,

        "name_ratio":
            name_ratio,

        "name_partial":
            name_partial,

        "name_token_sort":
            name_token_sort,

        "name_token_set":
            name_token_set,

        "name_len_s1":
            nlen1,

        "name_len_s2":
            nlen2,

        "name_len_absdiff":
            np.abs(nlen1 - nlen2),

        "name_len_reldiff":
            np.abs(nlen1 - nlen2)
            /
            np.maximum(
                1,
                np.maximum(nlen1, nlen2)
            ),

        # ADDRESS
        "address_exact":
            (a1 == a2).astype(np.int8),

        "address_contains":
            [
                int(
                    bool(x)
                    and bool(y)
                    and (x in y or y in x)
                )
                for x, y in zip(a1, a2)
            ],

        "address_prefix5":
            [
                int(
                    len(x) >= 5
                    and len(y) >= 5
                    and x[:5] == y[:5]
                )
                for x, y in zip(a1, a2)
            ],

        "address_prefix8":
            [
                int(
                    len(x) >= 8
                    and len(y) >= 8
                    and x[:8] == y[:8]
                )
                for x, y in zip(a1, a2)
            ],

        "address_suffix8":
            [
                int(
                    len(x) >= 8
                    and len(y) >= 8
                    and x[-8:] == y[-8:]
                )
                for x, y in zip(a1, a2)
            ],

        "house_match":
            [
                int(
                    bool(x)
                    and bool(y)
                    and x == y
                )
                for x, y in zip(h1, h2)
            ],

        "address_ratio":
            addr_ratio,

        "address_partial":
            addr_partial,

        "address_token_sort":
            addr_token_sort,

        "address_token_set":
            addr_token_set,

        "address_jaccard":
            addr_jaccard,

        "address_numeric_overlap":
            addr_numeric_overlap,

        "address_len_s1":
            alen1,

        "address_len_s2":
            alen2,

        "address_len_absdiff":
            np.abs(alen1 - alen2),

        "address_len_reldiff":
            np.abs(alen1 - alen2)
            /
            np.maximum(
                1,
                np.maximum(alen1, alen2)
            ),

        # COUNTRY / SOURCE
        "country_match":
            [
                int(
                    bool(x)
                    and bool(y)
                    and x == y
                )
                for x, y in zip(c1, c2)
            ],

        "country_missing":
            [
                int(not x or not y)
                for x, y in zip(c1, c2)
            ],

        "source2_candidate":
            df[
                "candidate_entity_id"
            ].str.startswith("S2-").astype(np.int8),

        "source3_candidate":
            df[
                "candidate_entity_id"
            ].str.startswith("S3-").astype(np.int8),

        # COMBINED
        "combined_mean":
            (
                name_ratio +
                addr_ratio
            ) / 2.0,

        "combined_product":
            name_ratio *
            addr_ratio,

        "strong_name_address":
            (
                (name_ratio >= 0.90) &
                (addr_ratio >= 0.70)
            ).astype(np.int8),

        "strong_name_house":
            (
                (name_ratio >= 0.90) &
                (h1 != "") &
                (h1 == h2)
            ).astype(np.int8)
    })

    return out


# ============================================================
# F0.5
# ============================================================

def f05(tp, fp, fn):

    if tp == 0:
        if fp == 0 and fn == 0:
            return 1.0

        return 0.0

    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    beta2 = 0.25

    return (
        1.25 *
        precision *
        recall
        /
        (beta2 * precision + recall)
    )


# ============================================================
# START
# ============================================================

print("=" * 78)
print("V5 FULL S1-LEVEL EVALUATION")
print("=" * 78)

print(
    f"\nEvaluation S1 sample : {N_S1:,}"
)

print(
    f"Baseline threshold   : {BASELINE_THRESHOLD}"
)

print(
    f"New thresholds       : {NEW_THRESHOLDS}"
)


# ============================================================
# FILE CHECK
# ============================================================

for path in [
    GROUND_TRUTH,
    BASELINE_CANDIDATES,
    NEW_SCORED,
    MODEL_PATH
]:

    if not os.path.exists(path):

        print(
            "\nMISSING:"
        )

        print(path)

        raise SystemExit(1)

    print(
        "OK:",
        os.path.basename(path)
    )


# ============================================================
# DUCKDB
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
# LOAD GROUND TRUTH
# ============================================================

print(
    "\nLoading ground truth..."
)

gt = con.execute(f"""
    SELECT
        source1_entity_id,
        matched_entity_id

    FROM read_csv(
        '{GROUND_TRUTH.replace(chr(92), '/')}',
        delim='\\t',
        header=true,
        quote='',
        columns={{
            'source1_entity_id':'VARCHAR',
            'matched_entity_id':'VARCHAR'
        }}
    )
""").df()

gt[
    "source1_entity_id"
] = gt[
    "source1_entity_id"
].astype(str)

gt[
    "matched_entity_id"
] = gt[
    "matched_entity_id"
].astype(str)


truth_groups = (
    gt
    .groupby("source1_entity_id")
    ["matched_entity_id"]
    .apply(set)
    .to_dict()
)

all_s1 = np.array(
    list(truth_groups.keys())
)

print(
    f"Ground truth pairs: {len(gt):,}"
)

print(
    f"S1 entities: {len(all_s1):,}"
)


# ============================================================
# SELECT DETERMINISTIC 50K S1
# ============================================================

# Sort by ID so repeated runs use the same evaluation sample.
all_s1_sorted = np.sort(all_s1)

heldout_s1 = set(
    all_s1_sorted[:N_S1]
)

print(
    f"\nHeld-out S1: {len(heldout_s1):,}"
)


# ============================================================
# LOAD NORMALIZED TRAIN DATA
# ============================================================

print(
    "\nLoading normalized S1/S2/S3..."
)

s1_path = os.path.join(
    BASE,
    "normalized_data",
    "train_source1_normalized.tsv"
)

s2_path = os.path.join(
    BASE,
    "normalized_data",
    "train_source2_normalized.tsv"
)

s3_path = os.path.join(
    BASE,
    "normalized_data",
    "train_source3_normalized.tsv"
)


# Only load S1 rows belonging to our sample.
heldout_list = list(heldout_s1)

heldout_df = pd.DataFrame({
    "source1_entity_id":
        heldout_list
})

con.register(
    "heldout_s1",
    heldout_df
)


s1 = con.execute(f"""
    SELECT
        entity_id AS source1_entity_id,
        name_normalized AS name_normalized_s1,
        address_normalized AS address_normalized_s1,
        country AS country_s1

    FROM read_csv(
        '{s1_path.replace(chr(92), '/')}',
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
    ) s

    INNER JOIN heldout_s1 h
    ON s.entity_id =
       h.source1_entity_id
""").df()


print(
    f"Held-out S1 rows loaded: {len(s1):,}"
)


# ============================================================
# LOAD ONLY REQUIRED S2/S3 RECORDS
# ============================================================

print(
    "\nLoading baseline candidates for held-out S1..."
)

baseline_candidates = con.execute(f"""
    SELECT
        source1_entity_id,
        candidate_entity_id

    FROM read_csv(
        '{BASELINE_CANDIDATES.replace(chr(92), '/')}',
        delim='\\t',
        header=true,
        quote='',
        columns={{
            'source1_entity_id':'VARCHAR',
            'candidate_entity_id':'VARCHAR'
        }}
    )

    WHERE source1_entity_id IN (
        SELECT source1_entity_id
        FROM heldout_s1
    )
""").df()


print(
    f"Baseline candidates for sample: "
    f"{len(baseline_candidates):,}"
)


# ============================================================
# LOAD S2/S3 DATA
# ============================================================

candidate_ids = (
    baseline_candidates[
        "candidate_entity_id"
    ]
    .drop_duplicates()
    .tolist()
)

candidate_id_df = pd.DataFrame({
    "candidate_entity_id":
        candidate_ids
})

con.register(
    "candidate_ids",
    candidate_id_df
)


print(
    "Loading matching S2/S3 records..."
)

s2s3 = con.execute(f"""
    SELECT
        entity_id AS candidate_entity_id,
        name_normalized AS name_normalized_s2,
        address_normalized AS address_normalized_s2,
        country AS country_s2

    FROM read_csv(
        '{s2_path.replace(chr(92), '/')}',
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

    WHERE entity_id IN (
        SELECT candidate_entity_id
        FROM candidate_ids
    )

    UNION ALL

    SELECT
        entity_id AS candidate_entity_id,
        name_normalized AS name_normalized_s2,
        address_normalized AS address_normalized_s2,
        country AS country_s2

    FROM read_csv(
        '{s3_path.replace(chr(92), '/')}',
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

    WHERE entity_id IN (
        SELECT candidate_entity_id
        FROM candidate_ids
    )
""").df()


print(
    f"S2/S3 records loaded: {len(s2s3):,}"
)


# ============================================================
# JOIN BASELINE CANDIDATES
# ============================================================

print(
    "\nJoining candidate records..."
)

pairs = baseline_candidates.merge(
    s1,
    on="source1_entity_id",
    how="inner"
)

pairs = pairs.merge(
    s2s3,
    on="candidate_entity_id",
    how="inner"
)

print(
    f"Joined baseline pairs: "
    f"{len(pairs):,}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading XGBoost V4 model..."
)

model = xgb.XGBClassifier()

model.load_model(
    MODEL_PATH
)

print(
    "Model loaded."
)


# ============================================================
# SCORE BASELINE
# ============================================================

print(
    "\nGenerating exact V4 features..."
)

feature_parts = []

total = len(pairs)

for start in range(
    0,
    total,
    SCORE_BATCH
):

    end = min(
        start + SCORE_BATCH,
        total
    )

    chunk = pairs.iloc[
        start:end
    ].copy()

    features = make_features(
        chunk
    )

    feature_parts.append(
        features
    )

    if (
        start == 0
        or
        end == total
        or
        start % 100_000 == 0
    ):

        print(
            f"Features: "
            f"{end:,}/{total:,}"
        )


X_baseline = pd.concat(
    feature_parts,
    ignore_index=True
)

print(
    f"Feature matrix: "
    f"{X_baseline.shape}"
)


# ============================================================
# ALIGN FEATURES TO V4 MODEL
# ============================================================

# The saved XGBoost model does not contain feature-name
# metadata, so get_booster().feature_names can be None.
#
# V4 was trained using the exact feature order below.
# Keep this order identical to the V4 feature generator.

V4_FEATURE_ORDER = [

    # NAME
    "name_exact",
    "name_contains",
    "name_prefix3",
    "name_prefix5",
    "name_suffix4",
    "name_first_token",
    "name_last_token",
    "shared_name_tokens",
    "name_token_jaccard",
    "name_ratio",
    "name_partial",
    "name_token_sort",
    "name_token_set",
    "name_len_s1",
    "name_len_s2",
    "name_len_absdiff",
    "name_len_reldiff",

    # ADDRESS
    "address_exact",
    "address_contains",
    "address_prefix5",
    "address_prefix8",
    "address_suffix8",
    "house_match",
    "address_ratio",
    "address_partial",
    "address_token_sort",
    "address_token_set",
    "address_jaccard",
    "address_numeric_overlap",
    "address_len_s1",
    "address_len_s2",
    "address_len_absdiff",
    "address_len_reldiff",

    # COUNTRY / SOURCE
    "country_match",
    "country_missing",
    "source2_candidate",
    "source3_candidate",

    # COMBINED
    "combined_mean",
    "combined_product",
    "strong_name_address",
    "strong_name_house"
]

print(
    f"\nChecking V4 feature count: "
    f"{len(V4_FEATURE_ORDER)}"
)

missing = [
    x
    for x in V4_FEATURE_ORDER
    if x not in X_baseline.columns
]

if missing:

    print(
        "\nERROR: Missing V4 features:"
    )

    for x in missing:
        print(
            "  ",
            x
        )

    raise SystemExit(1)


extra = [
    x
    for x in X_baseline.columns
    if x not in V4_FEATURE_ORDER
]

if extra:

    print(
        "\nWARNING: Extra features found:"
    )

    print(extra)


X_baseline = X_baseline[
    V4_FEATURE_ORDER
].astype(np.float32)

print(
    "V4 feature matrix aligned:"
)

print(
    X_baseline.shape
)

# ============================================================
# BASELINE PREDICTIONS
# ============================================================

print(
    "\nScoring baseline..."
)

baseline_prob = model.predict_proba(
    X_baseline
)[:, 1]


baseline_scored = pairs[
    [
        "source1_entity_id",
        "candidate_entity_id"
    ]
].copy()

baseline_scored[
    "probability"
] = baseline_prob


print(
    f"Baseline probabilities generated: "
    f"{len(baseline_scored):,}"
)


# Free large objects.
del X_baseline
del feature_parts
del pairs


# ============================================================
# LOAD NEW V5 SCORES
# ============================================================

print(
    "\nLoading already-scored V5 additions..."
)

new_scored = pd.read_csv(
    NEW_SCORED,
    sep="\t"
)

new_scored[
    "source1_entity_id"
] = new_scored[
    "source1_entity_id"
].astype(str)

new_scored[
    "candidate_entity_id"
] = new_scored[
    "candidate_entity_id"
].astype(str)

new_scored[
    "probability"
] = pd.to_numeric(
    new_scored[
        "probability"
    ],
    errors="coerce"
)


# Only our held-out S1 sample.
new_scored = new_scored[
    new_scored[
        "source1_entity_id"
    ].isin(
        heldout_s1
    )
].copy()


print(
    f"New V5 candidates in sample: "
    f"{len(new_scored):,}"
)


# ============================================================
# BUILD GROUPS
# ============================================================

truth_sample = {
    s1_id: truth_groups.get(
        s1_id,
        set()
    )
    for s1_id in heldout_s1
}


baseline_groups = (
    baseline_scored
    .groupby(
        "source1_entity_id"
    )[
        "candidate_entity_id"
    ]
    .apply(set)
    .to_dict()
)


# ============================================================
# BASELINE MACRO
# ============================================================

def evaluate_groups(
    prediction_groups
):

    scores = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_predictions = 0

    for s1_id in heldout_s1:

        predicted = prediction_groups.get(
            s1_id,
            set()
        )

        actual = truth_sample.get(
            s1_id,
            set()
        )

        tp = len(
            predicted & actual
        )

        fp = len(
            predicted - actual
        )

        fn = len(
            actual - predicted
        )

        score = f05(
            tp,
            fp,
            fn
        )

        scores.append(score)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        total_predictions += len(
            predicted
        )

    precision = (
        total_tp /
        max(
            1,
            total_tp + total_fp
        )
    )

    recall = (
        total_tp /
        max(
            1,
            total_tp + total_fn
        )
    )

    return {
        "macro_f05":
            float(np.mean(scores)),

        "precision":
            precision,

        "recall":
            recall,

        "tp":
            total_tp,

        "fp":
            total_fp,

        "fn":
            total_fn,

        "predictions":
            total_predictions
    }


baseline_result = evaluate_groups(
    baseline_groups
)


# ============================================================
# PRINT BASELINE
# ============================================================

print("\n")
print("=" * 78)
print("BASELINE RESULT")
print("=" * 78)

for k, v in baseline_result.items():

    if isinstance(v, float):

        print(
            f"{k:15s}: {v:.6f}"
        )

    else:

        print(
            f"{k:15s}: {v:,}"
        )


# ============================================================
# V5 THRESHOLD SWEEP
# ============================================================

results = []


print("\n")
print("=" * 78)
print("V5 S1-LEVEL THRESHOLD SWEEP")
print("=" * 78)


for threshold in NEW_THRESHOLDS:

    new_pred = new_scored[
        new_scored[
            "probability"
        ] >= threshold
    ]

    new_groups = (
        new_pred
        .groupby(
            "source1_entity_id"
        )[
            "candidate_entity_id"
        ]
        .apply(set)
        .to_dict()
    )

    # Combine baseline + new candidates.
    combined_groups = {}

    for s1_id in heldout_s1:

        base_set = baseline_groups.get(
            s1_id,
            set()
        )

        extra_set = new_groups.get(
            s1_id,
            set()
        )

        combined_groups[
            s1_id
        ] = base_set | extra_set


    result = evaluate_groups(
        combined_groups
    )

    result[
        "new_threshold"
    ] = threshold

    result[
        "new_predictions"
    ] = len(new_pred)

    result[
        "improvement"
    ] = (
        result["macro_f05"]
        -
        baseline_result["macro_f05"]
    )

    results.append(
        result
    )

    print(
        f"\nThreshold = {threshold:.2f}"
    )

    print(
        f"New predictions : "
        f"{len(new_pred):,}"
    )

    print(
        f"Combined        : "
        f"{result['predictions']:,}"
    )

    print(
        f"TP={result['tp']:,} "
        f"FP={result['fp']:,} "
        f"FN={result['fn']:,}"
    )

    print(
        f"Precision={result['precision']:.6f}"
    )

    print(
        f"Recall={result['recall']:.6f}"
    )

    print(
        f"Macro F0.5={result['macro_f05']:.6f}"
    )

    print(
        f"Change={result['improvement']:+.6f}"
    )


# ============================================================
# FINAL TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n")
print("=" * 78)
print("FINAL COMPARISON")
print("=" * 78)

print(
    results_df[
        [
            "new_threshold",
            "new_predictions",
            "predictions",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "macro_f05",
            "improvement"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# BEST V5
# ============================================================

best = results_df.loc[
    results_df[
        "macro_f05"
    ].idxmax()
]


print("\n")
print("=" * 78)
print("V5 EVALUATION SUMMARY")
print("=" * 78)

print(
    f"Baseline Macro F0.5 : "
    f"{baseline_result['macro_f05']:.6f}"
)

print(
    f"V5 new threshold    : "
    f"{best['new_threshold']:.2f}"
)

print(
    f"V5 Macro F0.5       : "
    f"{best['macro_f05']:.6f}"
)

print(
    f"Improvement         : "
    f"{best['improvement']:+.6f}"
)


# ============================================================
# SAVE
# ============================================================

output = os.path.join(
    BASE,
    "v5_full_s1_evaluation.csv"
)

results_df.to_csv(
    output,
    index=False
)

print(
    "\nSaved results:"
)

print(
    output
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 78)

if best["improvement"] > 0:

    print(
        "V5 IMPROVES THE S1-LEVEL MACRO F0.5 "
        "ON THIS 50K-S1 HELD-OUT SAMPLE."
    )

else:

    print(
        "V5 DOES NOT IMPROVE THE S1-LEVEL "
        "MACRO F0.5 ON THIS 50K-S1 SAMPLE."
    )

print("=" * 78)

con.close()