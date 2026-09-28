import duckdb
import os
import csv
from collections import defaultdict

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
NORM = os.path.join(BASE, "normalized_data")

TMP = os.path.join(BASE, "containment_tmp")
os.makedirs(TMP, exist_ok=True)

OUTPUT_RAW = os.path.join(
    TMP,
    "containment_s1_to_s23_raw.tsv"
)

OUTPUT_FINAL = os.path.join(
    BASE,
    "containment_s1_to_s23_freq50.tsv"
)

print("=" * 70)
print("STREAMING CONTAINMENT BLOCK")
print("=" * 70)


# =========================================================
# 1. Build S1 first-token frequency
# =========================================================

print("\n[1/5] Calculating S1 first-token frequencies...")

s1_path = os.path.join(
    NORM,
    "train_source1_normalized.tsv"
)

freq = defaultdict(int)

with open(
    s1_path,
    "r",
    encoding="utf-8",
    errors="replace"
) as f:

    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:

        name = (row.get("business_name") or "").strip().lower()
        country = (row.get("country") or "").strip().lower()

        if not name:
            continue

        first = name.split()[0]

        freq[(country, first)] += 1


print(f"Unique country/token combinations: {len(freq):,}")


# Keep only <=50 frequency tokens.
allowed = {
    key
    for key, count in freq.items()
    if count <= 50
}

print(
    f"Allowed tokens (frequency <=50): "
    f"{len(allowed):,}"
)


# =========================================================
# 2. Build S23 inverted index
# =========================================================

print("\n[2/5] Building S23 token index...")

s23_files = [
    os.path.join(
        NORM,
        "train_source2_normalized.tsv"
    ),
    os.path.join(
        NORM,
        "train_source3_normalized.tsv"
    )
]

# token_index[(country, token)] = list of S23 IDs
token_index = defaultdict(list)

total_s23 = 0

for path in s23_files:

    print(f"Reading: {os.path.basename(path)}")

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            total_s23 += 1

            sid = (row.get("entity_id") or "").strip()
            name = (row.get("business_name") or "").strip().lower()
            country = (row.get("country") or "").strip().lower()

            if not sid or not name:
                continue

            # Unique tokens within each name.
            tokens = set(name.split())

            for token in tokens:

                key = (country, token)

                # Only index tokens that can actually
                # be used by the S1 frequency <=50 block.
                if key in allowed:
                    token_index[key].append(sid)

    print(
        f"Indexed so far: {total_s23:,} S23/S3 rows"
    )


print(
    f"Unique usable token keys: "
    f"{len(token_index):,}"
)


# =========================================================
# 3. Stream S1 -> S23 candidates
# =========================================================

print("\n[3/5] Generating candidates...")

raw_count = 0

with open(
    OUTPUT_RAW,
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
        s1_path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            s1_id = (
                row.get("entity_id") or ""
            ).strip()

            name = (
                row.get("business_name") or ""
            ).strip().lower()

            country = (
                row.get("country") or ""
            ).strip().lower()

            if not s1_id or not name:
                continue

            first = name.split()[0]

            key = (country, first)

            if key not in allowed:
                continue

            # S23 names containing S1 first token
            candidates = token_index.get(key, [])

            for s23_id in candidates:

                writer.writerow([
                    s1_id,
                    s23_id
                ])

                raw_count += 1

                if raw_count % 5_000_000 == 0:

                    print(
                        f"Generated "
                        f"{raw_count:,} candidates..."
                    )


print(
    f"\nRaw candidates generated: "
    f"{raw_count:,}"
)


# =========================================================
# 4. Deduplicate with DuckDB
# =========================================================

print("\n[4/5] Deduplicating...")

DB = duckdb.connect()

DB.execute("PRAGMA threads=2")
DB.execute("PRAGMA memory_limit='1500MB'")
DB.execute(
    f"PRAGMA temp_directory='{BASE}\\duckdb_tmp'"
)

DB.execute(f"""
COPY (
    SELECT DISTINCT
        source1_entity_id,
        candidate_entity_id
    FROM read_csv_auto(
        '{OUTPUT_RAW}',
        delim='\t',
        header=true
    )
)
TO '{OUTPUT_FINAL}'
WITH (
    FORMAT CSV,
    DELIMITER '\t',
    HEADER
)
""")


final_count = DB.execute(f"""
SELECT COUNT(*)
FROM read_csv_auto(
    '{OUTPUT_FINAL}',
    delim='\t',
    header=true
)
""").fetchone()[0]


print(
    f"Unique containment candidates: "
    f"{final_count:,}"
)


# =========================================================
# 5. Check true-pair recovery
# =========================================================

print("\n[5/5] Checking true-pair recovery...")

DB.execute("""
CREATE TEMP TABLE containment AS
SELECT
    source1_entity_id AS s1_id,
    candidate_entity_id AS s23_id
FROM read_csv_auto(
    ?,
    delim='\t',
    header=true
)
""", [OUTPUT_FINAL])


DB.execute("""
CREATE TEMP TABLE truth AS
SELECT
    source1_entity_id AS s1_id,
    matched_entity_id AS s23_id
FROM read_csv_auto(
    'ground_truth_pairs.tsv',
    delim='\t',
    header=true
)
""")


recovered = DB.execute("""
SELECT COUNT(*)
FROM containment c
INNER JOIN truth t
    ON c.s1_id = t.s1_id
   AND c.s23_id = t.s23_id
""").fetchone()[0]


print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Raw candidates       : {raw_count:,}"
)

print(
    f"Unique candidates    : {final_count:,}"
)

print(
    f"True pairs recovered : {recovered:,}"
)

if final_count > 0:

    print(
        f"True pairs / 1M candidates : "
        f"{recovered / final_count * 1_000_000:,.1f}"
    )


print()
print(f"Output: {OUTPUT_FINAL}")

DB.close()

print("\nDone.")