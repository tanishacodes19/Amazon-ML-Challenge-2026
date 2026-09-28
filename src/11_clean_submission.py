import os
import csv
import tempfile

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"

MATCHING = os.path.join(
    BASE, "output", "matching_results.tsv"
)

CANDIDATE = os.path.join(
    BASE, "output", "candidate_pairs.tsv"
)


def clean_tsv(path, id_column, value_column):
    temp_path = path + ".tmp"

    print("\nCleaning:", path)

    rows = 0
    empty = 0

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as fin, open(
        temp_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as fout:

        reader = csv.DictReader(
            fin,
            delimiter="\t"
        )

        writer = csv.writer(
            fout,
            delimiter="\t",
            lineterminator="\n"
        )

        writer.writerow([
            id_column,
            value_column
        ])

        for row in reader:

            entity_id = row[id_column].strip()
            value = row[value_column].strip()

            # Convert literal "" into genuinely empty field
            if value == '""':
                value = ""

            # Also handle whitespace-only values
            if not value:
                empty += 1
                value = ""

            writer.writerow([
                entity_id,
                value
            ])

            rows += 1

            if rows % 250000 == 0:
                print(f"Processed: {rows:,}")

    os.replace(temp_path, path)

    print("Rows :", f"{rows:,}")
    print("Empty:", f"{empty:,}")
    print("DONE:", path)


# ============================================================
# CLEAN MATCHING RESULTS
# ============================================================

clean_tsv(
    MATCHING,
    "source1_entity_id",
    "matched_entity_ids"
)


# ============================================================
# CLEAN CANDIDATE PAIRS
# ============================================================

clean_tsv(
    CANDIDATE,
    "source1_entity_id",
    "candidate_entity_ids"
)


print("\n" + "=" * 70)
print("SUBMISSION FILES CLEANED")
print("=" * 70)