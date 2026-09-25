from pathlib import Path
import csv
from collections import Counter


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(
    r"C:\Users\Admin\Documents\Amazon-ML-Dataset\student_resource"
)

TRAIN_DIR = DATA_DIR / "dataset" / "train"

GROUND_TRUTH_FILE = TRAIN_DIR / "train_ground_truth.tsv"

# Local working file.
# This stays outside GitHub because *.tsv is in .gitignore.
OUTPUT_FILE = Path("ground_truth_pairs.tsv")


# ============================================================
# PROCESS GROUND TRUTH
# ============================================================

def process_ground_truth():

    print("=" * 70)
    print("GROUND TRUTH PROCESSING")
    print("=" * 70)

    total_s1 = 0
    total_positive_pairs = 0

    match_count_distribution = Counter()

    print(f"\nReading:")
    print(GROUND_TRUTH_FILE)

    with open(
        GROUND_TRUTH_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as infile, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as outfile:

        reader = csv.DictReader(
            infile,
            delimiter="\t"
        )

        writer = csv.writer(
            outfile,
            delimiter="\t"
        )

        # Header for our positive-pair file
        writer.writerow([
            "source1_entity_id",
            "matched_entity_id"
        ])

        for row in reader:

            source1_id = row["source1_entity_id"].strip()

            matched_ids_raw = row["matched_entity_ids"].strip()

            total_s1 += 1

            # Empty match list = singleton/no-match
            if not matched_ids_raw:
                match_count_distribution[0] += 1
                continue

            matched_ids = [
                entity_id.strip()
                for entity_id in matched_ids_raw.split(",")
                if entity_id.strip()
            ]

            match_count = len(matched_ids)

            match_count_distribution[match_count] += 1

            for matched_id in matched_ids:

                writer.writerow([
                    source1_id,
                    matched_id
                ])

                total_positive_pairs += 1


    # ========================================================
    # PRINT STATISTICS
    # ========================================================

    print("\n" + "=" * 70)
    print("GROUND TRUTH STATISTICS")
    print("=" * 70)

    print(f"\nTotal Source 1 entities: {total_s1:,}")

    print(
        f"Total positive S1 → S2/S3 pairs: "
        f"{total_positive_pairs:,}"
    )

    print(
        f"\nS1 match-count distribution:"
    )

    for match_count, entity_count in sorted(
        match_count_distribution.items()
    ):

        print(
            f"  {match_count} match(es): "
            f"{entity_count:,} S1 entities"
        )

    print(f"\nPositive pairs saved to:")
    print(OUTPUT_FILE.resolve())

    print("\nDONE.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    process_ground_truth()