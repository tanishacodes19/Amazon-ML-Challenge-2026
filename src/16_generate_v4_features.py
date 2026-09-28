import os
import polars as pl
import numpy as np

from rapidfuzz import fuzz

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

TRAIN_PAIRS = os.path.join(
    BASE, "training_data", "training_pairs.tsv"
)

NORM = os.path.join(
    BASE, "normalized_data"
)

OUT = os.path.join(
    BASE, "training_data", "features_v4"
)

os.makedirs(OUT, exist_ok=True)

CHUNK_SIZE = 100_000

print("=" * 60)
print("V4 FEATURE GENERATION")
print("=" * 60)


# ---------------------------------------------------------
# Load source data
# ---------------------------------------------------------

def load_source(path):
    return pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=1000,
        low_memory=True,
    ).with_columns([
        pl.col("entity_id").cast(pl.Utf8),
        pl.col("business_name").fill_null("").cast(pl.Utf8),
        pl.col("business_address").fill_null("").cast(pl.Utf8),
        pl.col("country").fill_null("").cast(pl.Utf8),
    ])


print("\nLoading sources...")

s1 = load_source(
    os.path.join(NORM, "train_source1_normalized.tsv")
)

s2 = load_source(
    os.path.join(NORM, "train_source2_normalized.tsv")
)

s3 = load_source(
    os.path.join(NORM, "train_source3_normalized.tsv")
)

s23 = pl.concat([s2, s3], how="vertical")

print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))


# ---------------------------------------------------------
# Rename
# ---------------------------------------------------------

s1 = s1.rename({
    "entity_id": "source1_entity_id",
    "business_name": "s1_name",
    "business_address": "s1_address",
    "country": "s1_country",
})

s23 = s23.rename({
    "entity_id": "matched_entity_id",
    "business_name": "matched_name",
    "business_address": "matched_address",
    "country": "matched_country",
})


# ---------------------------------------------------------
# Convert lookup tables
# ---------------------------------------------------------

s1_lookup = s1

s23_lookup = s23


# ---------------------------------------------------------
# RapidFuzz helpers
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Build features
# ---------------------------------------------------------

def make_features(df):

    rows = []

    for row in df.iter_rows(named=True):

        n1 = row["s1_name"] or ""
        n2 = row["matched_name"] or ""

        a1 = row["s1_address"] or ""
        a2 = row["matched_address"] or ""

        nt1 = token_set(n1)
        nt2 = token_set(n2)

        at1 = token_set(a1)
        at2 = token_set(a2)

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
            # -------------------------------
            # ID / LABEL
            # -------------------------------

            "source1_entity_id":
                row["source1_entity_id"],

            "matched_entity_id":
                row["matched_entity_id"],

            "label":
                row["label"],

            # -------------------------------
            # NAME EXACT / STRUCTURAL
            # -------------------------------

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
                int(first_token(n1) == first_token(n2)
                    and first_token(n1) != ""),

            "name_last_token_match":
                int(last_token(n1) == last_token(n2)
                    and last_token(n1) != ""),

            "name_shared_token_count":
                shared_name,

            "name_token_jaccard":
                name_j,

            # -------------------------------
            # NAME FUZZY
            # -------------------------------

            "name_ratio":
                name_ratio,

            "name_partial_ratio":
                name_partial,

            "name_token_sort":
                name_sort,

            "name_token_set":
                name_set,

            # -------------------------------
            # NAME LENGTH
            # -------------------------------

            "name_len_s1":
                name_len1,

            "name_len_match":
                name_len2,

            "name_len_abs_diff":
                abs(name_len1 - name_len2),

            "name_len_relative_diff":
                abs(name_len1 - name_len2)
                / max(name_len1, 1),

            # -------------------------------
            # ADDRESS EXACT / STRUCTURAL
            # -------------------------------

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
                int(h1 != "" and h1 == h2),

            # -------------------------------
            # ADDRESS FUZZY
            # -------------------------------

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

            # -------------------------------
            # ADDRESS LENGTH
            # -------------------------------

            "address_len_s1":
                addr_len1,

            "address_len_match":
                addr_len2,

            "address_len_abs_diff":
                abs(addr_len1 - addr_len2),

            "address_len_relative_diff":
                abs(addr_len1 - addr_len2)
                / max(addr_len1, 1),

            # -------------------------------
            # COUNTRY
            # -------------------------------

            "country_match":
                int(
                    row["s1_country"] != "" and
                    row["s1_country"] ==
                    row["matched_country"]
                ),

            "country_missing":
                int(
                    row["s1_country"] == "" or
                    row["matched_country"] == ""
                ),

            # -------------------------------
            # SOURCE
            # -------------------------------

            "source2":
                int(
                    row["matched_entity_id"].startswith("S2-")
                ),

            "source3":
                int(
                    row["matched_entity_id"].startswith("S3-")
                ),

            # -------------------------------
            # COMBINED SIGNALS
            # -------------------------------

            "name_address_ratio_mean":
                (name_ratio + addr_ratio) / 2.0,

            "name_address_ratio_product":
                name_ratio * addr_ratio,

            "strong_name_address":
                int(
                    name_ratio >= 0.90 and
                    addr_ratio >= 0.70
                ),

            "strong_name_house":
                int(
                    name_ratio >= 0.90 and
                    h1 != "" and
                    h1 == h2
                ),
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------
# Read training pairs in chunks
# ---------------------------------------------------------

print("\nReading training pairs...")

total = (
    pl.scan_csv(
        TRAIN_PAIRS,
        separator="\t"
    )
    .select(pl.len())
    .collect()
    .item()
)

print("Training pairs:", total)

total_chunks = (
    total + CHUNK_SIZE - 1
) // CHUNK_SIZE


# ---------------------------------------------------------
# Process
# ---------------------------------------------------------

for chunk_no in range(total_chunks):

    offset = chunk_no * CHUNK_SIZE

    print(
        f"\n[{chunk_no + 1}/{total_chunks}] "
        f"offset={offset}"
    )

    if chunk_no == 0:
        pairs = pl.read_csv(
            TRAIN_PAIRS,
            separator="\t",
            n_rows=CHUNK_SIZE,
            infer_schema_length=1000,
        )
    else:
        pairs = pl.read_csv(
            TRAIN_PAIRS,
            separator="\t",
            skip_rows=offset + 1,
            has_header=False,
            new_columns=[
                "source1_entity_id",
                "matched_entity_id",
                "label",
            ],
            n_rows=CHUNK_SIZE,
            infer_schema_length=1000,
        )

    pairs = pairs.with_columns([
        pl.col("source1_entity_id").cast(pl.Utf8),
        pl.col("matched_entity_id").cast(pl.Utf8),
    ])

    df = pairs.join(
        s1_lookup,
        on="source1_entity_id",
        how="left"
    )

    df = df.join(
        s23_lookup,
        on="matched_entity_id",
        how="left"
    )

    features = make_features(df)

    output = os.path.join(
        OUT,
        f"features_{chunk_no:04d}.parquet"
    )

    features.write_parquet(output)

    print(
        "Written:",
        len(features),
        "rows"
    )


print("\n" + "=" * 60)
print("V4 FEATURE GENERATION COMPLETE")
print("=" * 60)