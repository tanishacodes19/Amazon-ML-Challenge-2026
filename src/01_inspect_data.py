from pathlib import Path
import pandas as pd


# ============================================================
# DATASET PATH
# ============================================================

DATA_DIR = Path(
    r"C:\Users\Admin\Documents\Amazon-ML-Dataset\student_resource"
)

TRAIN_DIR = DATA_DIR / "dataset" / "train"
TEST_DIR = DATA_DIR / "dataset" / "test"


# ============================================================
# FUNCTION TO INSPECT A TSV FILE
# ============================================================

def inspect_file(file_path, nrows=5):

    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    df = pd.read_csv(
        file_path,
        sep="\t",
        encoding="utf-8",
        nrows=nrows
    )

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample shape:")
    print(df.shape)

    print("\nSample data:")
    print(df.to_string(index=False))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())


# ============================================================
# TRAINING FILES
# ============================================================

print("\n\n========== TRAINING DATA ==========")

inspect_file(
    TRAIN_DIR / "train_source1.tsv"
)

inspect_file(
    TRAIN_DIR / "train_source2.tsv"
)

inspect_file(
    TRAIN_DIR / "train_source3.tsv"
)

inspect_file(
    TRAIN_DIR / "train_ground_truth.tsv"
)


# ============================================================
# TEST FILES
# ============================================================

print("\n\n========== TEST DATA ==========")

inspect_file(
    TEST_DIR / "test_source1.tsv"
)

inspect_file(
    TEST_DIR / "test_source2.tsv"
)

inspect_file(
    TEST_DIR / "test_source3.tsv"
)