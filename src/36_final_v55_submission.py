import os
import csv
import gc
import polars as pl
import numpy as np
import xgboost as xgb
from rapidfuzz import fuzz


# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

S1_FILE = os.path.join(
    BASE, "normalized_data", "test_source1_normalized.tsv"
)

S2_FILE = os.path.join(
    BASE, "normalized_data", "test_source2_normalized.tsv"
)

S3_FILE = os.path.join(
    BASE, "normalized_data", "test_source3_normalized.tsv"
)

# Existing UNIQUE test candidate pairs
CANDIDATE_FILE = os.path.join(
    BASE, "test_scored_candidates.tsv"
)

MODEL_FILE = os.path.join(
    BASE, "model", "xgboost_v55_hard_negative.json"
)

OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MATCHING_FILE = os.path.join(
    OUTPUT_DIR, "matching_results.tsv"
)

CANDIDATE_OUTPUT = os.path.join(
    OUTPUT_DIR, "candidate_pairs.tsv"
)

THRESHOLD = 0.95

# Keep this reasonably small for the 8 GB machine
CHUNK_SIZE = 25_000


# ============================================================
# EXACT 41 FEATURES USED BY V5.5
# ============================================================

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


# ============================================================
# HELPERS
# ============================================================

def safe_ratio(a, b):
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def safe_partial(a, b):
    if not a or not b:
        return 0.0
    return fuzz.partial_ratio(a, b) / 100.0


def safe_token_sort(a, b):
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def safe_token_set(a, b):
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def token_set(s):
    if not s:
        return set()
    return set(s.split())


def numeric_tokens(s):
    if not s:
        return set()

    return {
        x for x in s.split()
        if any(ch.isdigit() for ch in x)
    }


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

    for token in s.split():
        if token and token[0].isdigit():
            return token

    return ""


def jaccard(a, b):
    A = token_set(a)
    B = token_set(b)

    if not A and not B:
        return 1.0

    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


def numeric_overlap(a, b):
    A = numeric_tokens(a)
    B = numeric_tokens(b)

    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


# ============================================================
# LOAD TEST SOURCES
# ============================================================

print("=" * 70)
print("V5.5 FINAL TEST INFERENCE")
print("=" * 70)

print("\nLoading S1...")

s1 = pl.read_csv(
    S1_FILE,
    separator="\t",
    infer_schema_length=1000,
).select([
    pl.col("entity_id").cast(pl.Utf8).alias("source1_entity_id"),
    pl.col("business_name")
      .fill_null("")
      .cast(pl.Utf8)
      .alias("s1_name"),
    pl.col("business_address")
      .fill_null("")
      .cast(pl.Utf8)
      .alias("s1_address"),
    pl.col("country")
      .fill_null("")
      .cast(pl.Utf8)
      .alias("s1_country"),
])

print("S1:", len(s1))


print("Loading S2...")

s2 = pl.read_csv(
    S2_FILE,
    separator="\t",
    infer_schema_length=1000,
).select([
    pl.col("entity_id").cast(pl.Utf8).alias("matched_entity_id"),
    pl.col("business_name")
      .fill_null("")
      .cast(pl.Utf8)
      .alias("matched_name"),
    pl.col("business_address")
      .fill_null("")
      .cast(pl.Utf8)
      .alias("matched_address"),
    pl.col("country")
      .fill_null("")
      .cast(pl.Utf8)
      .alias("matched_country"),
])


print("Loading S3...")

s3 = pl.read_csv(
    S3_FILE,
    separator="\t",
    infer_schema_length=1000,
).select([
    pl.col("entity_id").cast(pl.Utf8).alias("matched_entity_id"),
    pl.col("business_name")
      .fill_null("")
      .cast(pl.Utf8)
      .alias("matched_name"),
    pl.col("business_address")
      .fill_null("")
      .cast(pl.Utf8)
      .alias("matched_address"),
    pl.col("country")
      .fill_null("")
      .cast(pl.Utf8)
      .alias("matched_country"),
])


s23 = pl.concat([s2, s3], how="vertical")

del s2, s3
gc.collect()

print("S2 + S3:", len(s23))


# ============================================================
# LOOKUP DICTIONARIES
# ============================================================

print("\nCreating lookup tables...")

s1_lookup = s1
s23_lookup = s23


# ============================================================
# LOAD V5.5 MODEL
# ============================================================

print("\nLoading V5.5 model...")

model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)

print("Model:", MODEL_FILE)
print("Threshold:", THRESHOLD)
print("Features:", len(FEATURES))


# ============================================================
# FEATURE CREATION
# ============================================================

def make_features(df):

    rows = []

    for row in df.iter_rows(named=True):

        n1 = row["s1_name"] or ""
        n2 = row["matched_name"] or ""

        a1 = row["s1_address"] or ""
        a2 = row["matched_address"] or ""

        nt1 = token_set(n1)
        nt2 = token_set(n2)

        h1 = house_number(a1)
        h2 = house_number(a2)

        name_len1 = len(n1)
        name_len2 = len(n2)

        addr_len1 = len(a1)
        addr_len2 = len(a2)

        name_ratio = safe_ratio(n1, n2)
        addr_ratio = safe_ratio(a1, a2)

        name_partial = safe_partial(n1, n2)
        addr_partial = safe_partial(a1, a2)

        name_sort = safe_token_sort(n1, n2)
        name_set = safe_token_set(n1, n2)

        addr_sort = safe_token_sort(a1, a2)
        addr_set = safe_token_set(a1, a2)

        name_j = jaccard(n1, n2)
        addr_j = jaccard(a1, a2)

        num_j = numeric_overlap(a1, a2)

        shared_name = len(nt1 & nt2)

        rows.append({

            # NAME

            "name_exact":
                int(n1 == n2 and n1 != ""),

            "name_contains":
                int(
                    n1 != "" and
                    n2 != "" and
                    (n1 in n2 or n2 in n1)
                ),

            "name_prefix3":
                int(
                    n1[:3] == n2[:3] and
                    len(n1) >= 3 and
                    len(n2) >= 3
                ),

            "name_prefix5":
                int(
                    n1[:5] == n2[:5] and
                    len(n1) >= 5 and
                    len(n2) >= 5
                ),

            "name_suffix4":
                int(
                    n1[-4:] == n2[-4:] and
                    len(n1) >= 4 and
                    len(n2) >= 4
                ),

            "name_first_token_match":
                int(
                    first_token(n1) == first_token(n2)
                    and first_token(n1) != ""
                ),

            "name_last_token_match":
                int(
                    last_token(n1) == last_token(n2)
                    and last_token(n1) != ""
                ),

            "name_shared_token_count":
                shared_name,

            "name_token_jaccard":
                name_j,

            "name_ratio":
                name_ratio,

            "name_partial_ratio":
                name_partial,

            "name_token_sort":
                name_sort,

            "name_token_set":
                name_set,

            "name_len_s1":
                name_len1,

            "name_len_match":
                name_len2,

            "name_len_abs_diff":
                abs(name_len1 - name_len2),

            "name_len_relative_diff":
                abs(name_len1 - name_len2)
                / max(name_len1, 1),

            # ADDRESS

            "address_exact":
                int(a1 == a2 and a1 != ""),

            "address_contains":
                int(
                    a1 != "" and
                    a2 != "" and
                    (a1 in a2 or a2 in a1)
                ),

            "address_prefix5":
                int(
                    a1[:5] == a2[:5] and
                    len(a1) >= 5 and
                    len(a2) >= 5
                ),

            "address_prefix8":
                int(
                    a1[:8] == a2[:8] and
                    len(a1) >= 8 and
                    len(a2) >= 8
                ),

            "address_suffix8":
                int(
                    a1[-8:] == a2[-8:] and
                    len(a1) >= 8 and
                    len(a2) >= 8
                ),

            "house_match":
                int(
                    h1 != "" and
                    h1 == h2
                ),

            "address_ratio":
                addr_ratio,

            "address_partial_ratio":
                addr_partial,

            "address_token_sort":
                addr_sort,

            "address_token_set":
                addr_set,

            "address_token_jaccard":
                addr_j,

            "address_numeric_overlap":
                num_j,

            "address_len_s1":
                addr_len1,

            "address_len_match":
                addr_len2,

            "address_len_abs_diff":
                abs(addr_len1 - addr_len2),

            "address_len_relative_diff":
                abs(addr_len1 - addr_len2)
                / max(addr_len1, 1),

            # COUNTRY

            "country_match":
                int(
                    row["s1_country"] != ""
                    and
                    row["s1_country"]
                    ==
                    row["matched_country"]
                ),

            "country_missing":
                int(
                    row["s1_country"] == ""
                    or
                    row["matched_country"] == ""
                ),

            # SOURCE

            "source2":
                int(
                    row["matched_entity_id"].startswith("S2-")
                ),

            "source3":
                int(
                    row["matched_entity_id"].startswith("S3-")
                ),

            # COMBINED

            "name_address_ratio_mean":
                (name_ratio + addr_ratio) / 2.0,

            "name_address_ratio_product":
                name_ratio * addr_ratio,

            "strong_name_address":
                int(
                    name_ratio >= 0.90
                    and
                    addr_ratio >= 0.70
                ),

            "strong_name_house":
                int(
                    name_ratio >= 0.90
                    and
                    h1 != ""
                    and
                    h1 == h2
                ),
        })

    return pl.DataFrame(rows)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\nPreparing outputs...")

# Candidate submission = EXACT candidate set used for scoring.
with open(
    CANDIDATE_OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f,
        delimiter="\t",
        lineterminator="\n"
    )

    writer.writerow([
        "source1_entity_id",
        "matched_entity_id"
    ])

    # Copy unique candidate pairs.
    candidate_scan = pl.read_csv(
        CANDIDATE_FILE,
        separator="\t",
        columns=[
            "source1_entity_id",
            "matched_entity_id"
        ],
        batch_size=CHUNK_SIZE,
    )

    candidate_scan.write_csv(
        f,
        separator="\t",
        include_header=False
    )


# ============================================================
# MATCHING OUTPUT
# ============================================================

# We first collect only selected matches.
#
# Because there can be many selected candidates,
# write them to a temporary file and aggregate afterwards.

TEMP_MATCHES = os.path.join(
    OUTPUT_DIR,
    "_v55_selected_pairs.tsv"
)

with open(
    TEMP_MATCHES,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f,
        delimiter="\t",
        lineterminator="\n"
    )

    writer.writerow([
        "source1_entity_id",
        "matched_entity_id"
    ])

    total_candidates = 0
    total_selected = 0

    print("\nScoring candidates...")

    # Use Polars streaming batches.
    reader = pl.read_csv_batched(
        CANDIDATE_FILE,
        separator="\t",
        batch_size=CHUNK_SIZE,
        has_header=True,
    )

    batch_no = 0

    while True:

        batches = reader.next_batches(1)

        if not batches:
            break

        chunk = batches[0]

        batch_no += 1

        total_candidates += len(chunk)

        # Join S1
        chunk = chunk.join(
            s1_lookup,
            on="source1_entity_id",
            how="left"
        )

        # Join S2/S3
        chunk = chunk.join(
            s23_lookup,
            on="matched_entity_id",
            how="left"
        )

        # Clean missing strings
        string_cols = [
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
            .cast(pl.Utf8)
            for c in string_cols
        ])

        # Exact V5.5 feature construction
        feature_df = make_features(chunk)

        X = (
            feature_df
            .select(FEATURES)
            .fill_null(0)
            .to_numpy()
            .astype(np.float32)
        )

        probabilities = model.predict_proba(X)[:, 1]

        selected_mask = probabilities >= THRESHOLD

        if np.any(selected_mask):

            selected_ids = chunk.select([
                "source1_entity_id",
                "matched_entity_id"
            ]).to_numpy()[selected_mask]

            for source1_id, matched_id in selected_ids:

                writer.writerow([
                    source1_id,
                    matched_id
                ])

            total_selected += int(
                np.sum(selected_mask)
            )

        if batch_no % 20 == 0:

            print(
                f"Batch {batch_no:,} | "
                f"Candidates {total_candidates:,} | "
                f"Selected {total_selected:,}"
            )

        del chunk
        del feature_df
        del X
        del probabilities

        gc.collect()


print("\nScoring complete.")

print(
    f"Candidates scored: {total_candidates:,}"
)

print(
    f"Pairs selected: {total_selected:,}"
)


# ============================================================
# BUILD MATCHING_RESULTS
# ============================================================

print("\nBuilding matching_results.tsv...")

selected = pl.read_csv(
    TEMP_MATCHES,
    separator="\t",
    infer_schema_length=1000,
)

selected = (
    selected
    .unique(
        subset=[
            "source1_entity_id",
            "matched_entity_id"
        ]
    )
)


# Aggregate selected candidates per S1
grouped = (
    selected
    .group_by("source1_entity_id")
    .agg(
        pl.col("matched_entity_id")
        .alias("matches")
    )
)


# IMPORTANT:
# Start from EVERY S1 entity.
# This preserves S1s with zero matches.

result = (
    s1
    .select("source1_entity_id")
    .join(
        grouped,
        on="source1_entity_id",
        how="left"
    )
    .with_columns(
        pl.col("matches")
        .list.join(",")
        .fill_null("")
        .alias("matched_entity_ids")
    )
    .select([
        "source1_entity_id",
        "matched_entity_ids"
    ])
)


result.write_csv(
    MATCHING_FILE,
    separator="\t",
    include_header=True
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print(
    "Test S1 entities:",
    len(s1)
)

print(
    "Matching result rows:",
    len(result)
)

print(
    "Candidate pairs scored:",
    total_candidates
)

print(
    "Selected matches:",
    total_selected
)

nonempty = (
    result
    .filter(
        pl.col("matched_entity_ids") != ""
    )
    .height
)

empty = len(result) - nonempty

print(
    "S1 with >=1 match:",
    nonempty
)

print(
    "S1 with 0 matches:",
    empty
)

if len(result) != len(s1):

    raise RuntimeError(
        "ERROR: matching_results.tsv does NOT contain every S1!"
    )

if total_candidates != 25_438_438:

    print(
        "\nWARNING:"
    )

    print(
        "Expected 25,438,438 candidates, "
        "but scored:",
        total_candidates
    )


print("\nFiles created:")

print(
    MATCHING_FILE
)

print(
    CANDIDATE_OUTPUT
)

print("\nDONE.")


# ============================================================
# CLEAN TEMP
# ============================================================

try:
    os.remove(TEMP_MATCHES)
except OSError:
    pass

gc.collect()