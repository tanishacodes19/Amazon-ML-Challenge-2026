import os
import gc

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from rapidfuzz.fuzz import (
    ratio,
    partial_ratio,
    token_sort_ratio,
    token_set_ratio
)


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CANDIDATE_FILE = os.path.join(
    BASE,
    "training_candidate_pairs.tsv"
)

GROUND_TRUTH = os.path.join(
    BASE,
    "ground_truth_pairs.tsv"
)

MODEL_PATH = os.path.join(
    BASE,
    "model",
    "xgboost_v4.json"
)

NORM_DIR = os.path.join(
    BASE,
    "normalized_data"
)


# ============================================================
# SETTINGS
# ============================================================

S1_SAMPLE_SIZE = 50_000

CANDIDATE_CHUNK = 100_000

SCORE_BATCH = 10_000

THRESHOLD = 0.51


print("=" * 75)
print("V4 EXACT FULL-CANDIDATE VALIDATION")
print("=" * 75)


# ============================================================
# FILE CHECK
# ============================================================

for path in [
    CANDIDATE_FILE,
    GROUND_TRUTH,
    MODEL_PATH
]:

    if not os.path.exists(path):
        raise FileNotFoundError(path)


# ============================================================
# EXACT V4 FEATURE FUNCTIONS
# ============================================================

def safe_ratio(a, b):

    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


def safe_partial(a, b):

    if not a or not b:
        return 0.0

    return partial_ratio(a, b) / 100.0


def safe_token_sort(a, b):

    if not a or not b:
        return 0.0

    return token_sort_ratio(a, b) / 100.0


def safe_token_set(a, b):

    if not a or not b:
        return 0.0

    return token_set_ratio(a, b) / 100.0


def token_set(s):

    if not s:
        return set()

    return set(
        s.split()
    )


def numeric_tokens(s):

    if not s:
        return set()

    return set(
        token
        for token in s.split()
        if any(
            char.isdigit()
            for char in token
        )
    )


def first_token(s):

    if not s:
        return ""

    return s.split()[0]


def last_token(s):

    if not s:
        return ""

    return s.split()[-1]


def house_number(s):

    if not s:
        return ""

    tokens = s.split()

    for token in tokens:

        if any(
            char.isdigit()
            for char in token
        ):

            return token

    return ""


def jaccard(a, b):

    a = token_set(a)
    b = token_set(b)

    if not a and not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


def numeric_overlap(a, b):

    a = numeric_tokens(a)
    b = numeric_tokens(b)

    if not a and not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


# ============================================================
# EXACT V4 FEATURES
# ============================================================

FEATURE_NAMES = [

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
    "strong_name_house"
]


def make_feature_row(
    s1_row,
    matched_row
):

    n1 = s1_row["name_normalized"] or ""
    n2 = matched_row["name_normalized"] or ""

    a1 = s1_row["address_normalized"] or ""
    a2 = matched_row["address_normalized"] or ""

    nt1 = token_set(n1)
    nt2 = token_set(n2)

    h1 = house_number(a1)
    h2 = house_number(a2)

    name_len1 = len(n1)
    name_len2 = len(n2)

    addr_len1 = len(a1)
    addr_len2 = len(a2)

    name_ratio = safe_ratio(
        n1,
        n2
    )

    addr_ratio = safe_ratio(
        a1,
        a2
    )

    name_partial = safe_partial(
        n1,
        n2
    )

    addr_partial = safe_partial(
        a1,
        a2
    )

    name_sort = safe_token_sort(
        n1,
        n2
    )

    name_set = safe_token_set(
        n1,
        n2
    )

    addr_sort = safe_token_sort(
        a1,
        a2
    )

    addr_set = safe_token_set(
        a1,
        a2
    )

    name_j = jaccard(
        n1,
        n2
    )

    addr_j = jaccard(
        a1,
        a2
    )

    num_j = numeric_overlap(
        a1,
        a2
    )

    shared_name = len(
        nt1 & nt2
    )

    return [

        int(
            n1 == n2 and
            n1 != ""
        ),

        int(
            n1 != "" and
            n2 != "" and
            (
                n1 in n2 or
                n2 in n1
            )
        ),

        int(
            n1[:3] == n2[:3] and
            len(n1) >= 3 and
            len(n2) >= 3
        ),

        int(
            n1[:5] == n2[:5] and
            len(n1) >= 5 and
            len(n2) >= 5
        ),

        int(
            n1[-4:] == n2[-4:] and
            len(n1) >= 4 and
            len(n2) >= 4
        ),

        int(
            first_token(n1) ==
            first_token(n2)
            and
            first_token(n1) != ""
        ),

        int(
            last_token(n1) ==
            last_token(n2)
            and
            last_token(n1) != ""
        ),

        shared_name,

        name_j,

        name_ratio,

        name_partial,

        name_sort,

        name_set,

        name_len1,

        name_len2,

        abs(
            name_len1 -
            name_len2
        ),

        abs(
            name_len1 -
            name_len2
        ) / max(
            name_len1,
            1
        ),

        int(
            a1 == a2 and
            a1 != ""
        ),

        int(
            a1 != "" and
            a2 != "" and
            (
                a1 in a2 or
                a2 in a1
            )
        ),

        int(
            a1[:5] == a2[:5] and
            len(a1) >= 5 and
            len(a2) >= 5
        ),

        int(
            a1[:8] == a2[:8] and
            len(a1) >= 8 and
            len(a2) >= 8
        ),

        int(
            a1[-8:] == a2[-8:] and
            len(a1) >= 8 and
            len(a2) >= 8
        ),

        int(
            h1 != "" and
            h1 == h2
        ),

        addr_ratio,

        addr_partial,

        addr_sort,

        addr_set,

        addr_j,

        num_j,

        addr_len1,

        addr_len2,

        abs(
            addr_len1 -
            addr_len2
        ),

        abs(
            addr_len1 -
            addr_len2
        ) / max(
            addr_len1,
            1
        ),

        int(
            s1_row["country"] != "" and
            s1_row["country"] ==
            matched_row["country"]
        ),

        int(
            s1_row["country"] == "" or
            matched_row["country"] == ""
        ),

        int(
            matched_row["entity_id"].startswith(
                "S2-"
            )
        ),

        int(
            matched_row["entity_id"].startswith(
                "S3-"
            )
        ),

        (
            name_ratio +
            addr_ratio
        ) / 2.0,

        name_ratio *
        addr_ratio,

        int(
            name_ratio >= 0.90 and
            addr_ratio >= 0.70
        ),

        int(
            name_ratio >= 0.90 and
            h1 != "" and
            h1 == h2
        )
    ]


# ============================================================
# LOAD TRAIN SOURCES
# ============================================================

print("\nLoading TRAIN normalized data...")

SOURCE_COLS = [
    "entity_id",
    "name_normalized",
    "address_normalized",
    "country"
]


s1 = pl.read_csv(
    os.path.join(
        NORM_DIR,
        "train_source1_normalized.tsv"
    ),
    separator="\t",
    columns=SOURCE_COLS,
    infer_schema_length=1000
)

s2 = pl.read_csv(
    os.path.join(
        NORM_DIR,
        "train_source2_normalized.tsv"
    ),
    separator="\t",
    columns=SOURCE_COLS,
    infer_schema_length=1000
)

s3 = pl.read_csv(
    os.path.join(
        NORM_DIR,
        "train_source3_normalized.tsv"
    ),
    separator="\t",
    columns=SOURCE_COLS,
    infer_schema_length=1000
)


print(
    "S1:",
    f"{len(s1):,}"
)

print(
    "S2:",
    f"{len(s2):,}"
)

print(
    "S3:",
    f"{len(s3):,}"
)


# ============================================================
# BUILD LOOKUPS
# ============================================================

print("\nBuilding lookups...")

s1_lookup = {
    row["entity_id"]: row
    for row in s1.iter_rows(
        named=True
    )
}

s23 = pl.concat(
    [s2, s3]
)

s23_lookup = {
    row["entity_id"]: row
    for row in s23.iter_rows(
        named=True
    )
}

del s1
del s2
del s3
del s23

gc.collect()


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str
)

print(
    "Ground truth pairs:",
    f"{len(gt):,}"
)


# ============================================================
# FIXED RANDOM HELD-OUT S1 SAMPLE
# ============================================================

rng = np.random.default_rng(
    20260926
)

all_s1_ids = np.array(
    list(s1_lookup.keys()),
    dtype=object
)

rng.shuffle(
    all_s1_ids
)

heldout_ids = set(
    all_s1_ids[
        :S1_SAMPLE_SIZE
    ]
)

print(
    "Held-out S1:",
    f"{len(heldout_ids):,}"
)


# ============================================================
# GROUND TRUTH FOR HELD-OUT S1
# ============================================================

gt = gt[
    gt["source1_entity_id"].isin(
        heldout_ids
    )
]

true_pairs = set(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_id"]
    )
)

true_by_s1 = {}

for s1_id, matched_id in true_pairs:

    true_by_s1.setdefault(
        s1_id,
        set()
    ).add(
        matched_id
    )

print(
    "Held-out true pairs:",
    f"{len(true_pairs):,}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading V4 model...")

model = xgb.XGBClassifier()

model.load_model(
    MODEL_PATH
)


# ============================================================
# READ CANDIDATES
# ============================================================

print("\nReading candidate universe...")

candidate_pairs = []

for chunk_no, chunk in enumerate(
    pd.read_csv(
        CANDIDATE_FILE,
        sep="\t",
        dtype=str,
        chunksize=CANDIDATE_CHUNK
    )
):

    chunk = chunk[
        chunk["source1_entity_id"].isin(
            heldout_ids
        )
    ]

    if len(chunk):

        candidate_pairs.extend(
            zip(
                chunk["source1_entity_id"],
                chunk["candidate_entity_id"]
            )
        )

    if chunk_no % 20 == 0:

        print(
            f"Candidate chunks: "
            f"{chunk_no:,} | "
            f"pairs: "
            f"{len(candidate_pairs):,}"
        )


candidate_pairs = list(
    set(candidate_pairs)
)

print(
    "\nUnique candidates:",
    f"{len(candidate_pairs):,}"
)


# ============================================================
# CANDIDATE RECALL
# ============================================================

candidate_set = set(
    candidate_pairs
)

recovered = sum(
    pair in candidate_set
    for pair in true_pairs
)

pair_recall = (
    recovered /
    len(true_pairs)
)

print()
print("=" * 75)
print("CANDIDATE RECALL")
print("=" * 75)

print(
    "True pairs:",
    f"{len(true_pairs):,}"
)

print(
    "Recovered:",
    f"{recovered:,}"
)

print(
    f"Pair recall: "
    f"{pair_recall:.6%}"
)


# ============================================================
# SCORE CANDIDATES
# ============================================================

print()
print("=" * 75)
print("V4 SCORING")
print("=" * 75)


predicted_pairs = set()

total = len(
    candidate_pairs
)


for start in range(
    0,
    total,
    SCORE_BATCH
):

    batch = candidate_pairs[
        start:
        start + SCORE_BATCH
    ]

    features = []

    valid_pairs = []

    for s1_id, candidate_id in batch:

        s1_row = s1_lookup.get(
            s1_id
        )

        matched_row = s23_lookup.get(
            candidate_id
        )

        if (
            s1_row is None or
            matched_row is None
        ):
            continue

        features.append(
            make_feature_row(
                s1_row,
                matched_row
            )
        )

        valid_pairs.append(
            (
                s1_id,
                candidate_id
            )
        )

    if features:

        X = np.asarray(
            features,
            dtype=np.float32
        )

        probabilities = (
            model
            .predict_proba(X)[:, 1]
        )

        for pair, probability in zip(
            valid_pairs,
            probabilities
        ):

            if probability >= THRESHOLD:

                predicted_pairs.add(
                    pair
                )

    if (
        start == 0 or
        (start // SCORE_BATCH) % 10 == 0
    ):

        print(
            f"Scored "
            f"{min(start + SCORE_BATCH, total):,}"
            f" / "
            f"{total:,}"
        )

    del features
    del valid_pairs

    gc.collect()


# ============================================================
# S1 MACRO F0.5
# ============================================================

print()
print("=" * 75)
print("FULL-CANDIDATE V4 RESULT")
print("=" * 75)


pred_by_s1 = {}

for s1_id, candidate_id in predicted_pairs:

    pred_by_s1.setdefault(
        s1_id,
        set()
    ).add(
        candidate_id
    )


scores = []


for s1_id in heldout_ids:

    true_set = true_by_s1.get(
        s1_id,
        set()
    )

    pred_set = pred_by_s1.get(
        s1_id,
        set()
    )

    tp = len(
        true_set &
        pred_set
    )

    fp = len(
        pred_set -
        true_set
    )

    fn = len(
        true_set -
        pred_set
    )

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


    if (
        precision == 0 and
        recall == 0
    ):

        if (
            not true_set and
            not pred_set
        ):

            f05 = 1.0

        else:

            f05 = 0.0

    else:

        f05 = (
            1.25 *
            precision *
            recall
        ) / (
            0.25 *
            precision +
            recall
        )

    scores.append(
        f05
    )


macro_f05 = np.mean(
    scores
)


# ============================================================
# RESULTS
# ============================================================

print(
    "Held-out S1 entities:",
    f"{len(heldout_ids):,}"
)

print(
    "True pairs:",
    f"{len(true_pairs):,}"
)

print(
    "Candidate pairs:",
    f"{len(candidate_pairs):,}"
)

print(
    f"Candidate recall: "
    f"{pair_recall:.6%}"
)

print(
    "Predicted matches:",
    f"{len(predicted_pairs):,}"
)

print(
    f"V4 threshold: "
    f"{THRESHOLD}"
)

print(
    f"EXACT V4 FULL-CANDIDATE "
    f"S1 MACRO F0.5: "
    f"{macro_f05:.6f}"
)

print()
print("=" * 75)
print("DONE")
print("=" * 75)