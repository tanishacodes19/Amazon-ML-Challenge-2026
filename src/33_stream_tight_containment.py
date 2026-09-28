import os
import csv
import time
from collections import defaultdict

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORM = os.path.join(BASE, "normalized_data")

S1_FILE = os.path.join(
    NORM,
    "train_source1_normalized.tsv"
)

S2_FILE = os.path.join(
    NORM,
    "train_source2_normalized.tsv"
)

S3_FILE = os.path.join(
    NORM,
    "train_source3_normalized.tsv"
)

OUT_DIR = os.path.join(
    BASE,
    "training_data",
    "v5_candidate_blocks"
)

OUTPUT_FILE = os.path.join(
    OUT_DIR,
    "containment_length4.tsv"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

MAX_LENGTH_DIFF = 4

# Prevent extremely common first tokens from exploding.
# This is deliberately conservative.
MAX_TOKEN_FREQUENCY = 50


print("=" * 70)
print("STREAMING TIGHT CONTAINMENT BLOCK")
print("=" * 70)

print(f"\nLength difference <= {MAX_LENGTH_DIFF}")
print(f"First-token frequency <= {MAX_TOKEN_FREQUENCY}")


# ============================================================
# 1. CALCULATE S1 FIRST-TOKEN FREQUENCIES
# ============================================================

print("\n[1/4] Calculating S1 first-token frequencies...")

s1_freq = defaultdict(int)

start = time.time()

with open(
    S1_FILE,
    "r",
    encoding="utf-8",
    errors="replace"
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    for row in reader:

        name = (
            row.get("business_name") or ""
        ).strip().lower()

        country = (
            row.get("country") or ""
        ).strip().lower()

        if not name or not country:
            continue

        first_token = name.split()[0]

        s1_freq[
            (country, first_token)
        ] += 1


allowed = {
    key
    for key, count in s1_freq.items()
    if count <= MAX_TOKEN_FREQUENCY
}

print(
    f"Unique country/token keys: "
    f"{len(s1_freq):,}"
)

print(
    f"Allowed keys <= {MAX_TOKEN_FREQUENCY}: "
    f"{len(allowed):,}"
)

print(
    f"Time: {time.time() - start:.1f}s"
)


# ============================================================
# 2. BUILD STREAMING S23 INDEX
# ============================================================

print("\n[2/4] Building S23 containment index...")

start = time.time()

# key:
#     (country, token)
#
# value:
#     list of (entity_id, name_length)
#
# This lets us apply the length filter BEFORE writing candidates.

token_index = defaultdict(list)

total_s23 = 0
indexed_s23 = 0
indexed_entries = 0


for s23_file in [S2_FILE, S3_FILE]:

    print(
        f"Reading {os.path.basename(s23_file)}..."
    )

    with open(
        s23_file,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            total_s23 += 1

            entity_id = (
                row.get("entity_id") or ""
            ).strip()

            name = (
                row.get("business_name") or ""
            ).strip().lower()

            country = (
                row.get("country") or ""
            ).strip().lower()

            if not entity_id or not name or not country:
                continue

            name_length = len(
                "".join(name.split())
            )

            # Only unique tokens within one business name.
            tokens = set(name.split())

            usable = False

            for token in tokens:

                key = (
                    country,
                    token
                )

                if key in allowed:

                    token_index[key].append(
                        (
                            entity_id,
                            name_length
                        )
                    )

                    indexed_entries += 1
                    usable = True

            if usable:
                indexed_s23 += 1

print(
    f"S23/S3 rows read: "
    f"{total_s23:,}"
)

print(
    f"S23 rows indexed: "
    f"{indexed_s23:,}"
)

print(
    f"Index entries: "
    f"{indexed_entries:,}"
)

print(
    f"Unique token keys: "
    f"{len(token_index):,}"
)

print(
    f"Index time: "
    f"{time.time() - start:.1f}s"
)


# ============================================================
# 3. STREAM S1 -> CANDIDATES
# ============================================================

print("\n[3/4] Generating tight containment candidates...")

start = time.time()

raw_candidates = 0
s1_processed = 0
s1_with_candidates = 0

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as out:

    writer = csv.writer(
        out,
        delimiter="\t",
        lineterminator="\n"
    )

    writer.writerow([
        "source1_entity_id",
        "candidate_entity_id"
    ])

    with open(
        S1_FILE,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            s1_processed += 1

            s1_id = (
                row.get("entity_id") or ""
            ).strip()

            name = (
                row.get("business_name") or ""
            ).strip().lower()

            country = (
                row.get("country") or ""
            ).strip().lower()

            if not s1_id or not name or not country:
                continue

            first_token = name.split()[0]

            key = (
                country,
                first_token
            )

            if key not in allowed:
                continue

            s1_name_length = len(
                "".join(name.split())
            )

            candidates = token_index.get(
                key,
                []
            )

            found_for_s1 = False

            for candidate_id, candidate_length in candidates:

                # ------------------------------------------------
                # IMPORTANT FILTER
                # ------------------------------------------------

                if abs(
                    s1_name_length -
                    candidate_length
                ) > MAX_LENGTH_DIFF:
                    continue

                writer.writerow([
                    s1_id,
                    candidate_id
                ])

                raw_candidates += 1
                found_for_s1 = True

            if found_for_s1:
                s1_with_candidates += 1

            if s1_processed % 100_000 == 0:

                elapsed = time.time() - start

                print(
                    f"Processed S1: "
                    f"{s1_processed:,} | "
                    f"Candidates: "
                    f"{raw_candidates:,} | "
                    f"Time: "
                    f"{elapsed:.1f}s"
                )


print(
    f"\nS1 processed: "
    f"{s1_processed:,}"
)

print(
    f"S1 with candidates: "
    f"{s1_with_candidates:,}"
)

print(
    f"Raw candidates: "
    f"{raw_candidates:,}"
)

print(
    f"Generation time: "
    f"{time.time() - start:.1f}s"
)


# ============================================================
# 4. DEDUPLICATE
# ============================================================

print("\n[4/4] Deduplicating candidate block...")

import duckdb

db = duckdb.connect()

db.execute(
    "PRAGMA threads=2"
)

db.execute(
    "PRAGMA memory_limit='1500MB'"
)

tmp_dir = os.path.join(
    BASE,
    "duckdb_tmp"
)

os.makedirs(
    tmp_dir,
    exist_ok=True
)

db.execute(
    f"SET temp_directory='{tmp_dir.replace(chr(92), '/')}'"
)

dedup_file = os.path.join(
    OUT_DIR,
    "containment_length4_dedup.tsv"
)

dedup_file_sql = dedup_file.replace(
    "\\",
    "/"
)

output_sql = OUTPUT_FILE.replace(
    "\\",
    "/"
)

db.execute(f"""
COPY (

    SELECT DISTINCT
        source1_entity_id,
        candidate_entity_id

    FROM read_csv(
        '{output_sql}',
        delim='\\t',
        header=true,
        columns={{
            'source1_entity_id':'VARCHAR',
            'candidate_entity_id':'VARCHAR'
        }}
    )

)

TO '{dedup_file_sql}'

(
    FORMAT CSV,
    DELIMITER '\\t',
    HEADER
)
""")

unique_count = db.execute(f"""
SELECT COUNT(*)

FROM read_csv(
    '{dedup_file_sql}',
    delim='\\t',
    header=true,
    columns={{
        'source1_entity_id':'VARCHAR',
        'candidate_entity_id':'VARCHAR'
    }}
)
""").fetchone()[0]

db.close()


# Replace original raw file with deduplicated file
os.replace(
    dedup_file,
    OUTPUT_FILE
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("TIGHT CONTAINMENT BLOCK COMPLETE")
print("=" * 70)

print(
    f"Raw candidates       : "
    f"{raw_candidates:,}"
)

print(
    f"Unique candidates    : "
    f"{unique_count:,}"
)

print(
    f"Candidates / S1      : "
    f"{unique_count / max(s1_processed, 1):,.2f}"
)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)

print("=" * 70)