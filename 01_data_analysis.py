import polars as pl
from pathlib import Path

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATASET_DIR = Path(
    r"C:\Users\viper\Documents\student_resource\student_resource\dataset"
)

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"


# ---------------------------------------------------------
# Files to analyze
# ---------------------------------------------------------

FILES = {
    "train_source1": TRAIN_DIR / "train_source1.tsv",
    "train_source2": TRAIN_DIR / "train_source2.tsv",
    "train_source3": TRAIN_DIR / "train_source3.tsv",
    "train_ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
    "test_source1": TEST_DIR / "test_source1.tsv",
    "test_source2": TEST_DIR / "test_source2.tsv",
    "test_source3": TEST_DIR / "test_source3.tsv",
}


# ---------------------------------------------------------
# Analyze one source file
# ---------------------------------------------------------

def analyze_source_file(name, path):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # Lazy scan: Polars does not immediately load the whole file.
    df = pl.scan_csv(
        path,
        separator="\t",
        infer_schema_length=1000,
        null_values=["", "NULL", "null", "None"],
    )

    # Schema
    print("\nColumns and data types:")
    print(df.collect_schema())

    # Row count
    row_count = df.select(
        pl.len().alias("rows")
    ).collect().item()

    print(f"\nRows: {row_count:,}")

    # Missing values
    missing = (
        df.select(
            [
                pl.col(column).null_count().alias(column)
                for column in df.collect_schema().names()
            ]
        )
        .collect()
    )

    print("\nMissing values:")
    print(missing)

    # Country distribution for source files
    if "country" in df.collect_schema().names():
        country_counts = (
            df.group_by("country")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .collect()
        )

        print("\nCountry distribution:")
        print(country_counts)

    # Show a few records
    print("\nSample rows:")
    print(df.head(3).collect())


# ---------------------------------------------------------
# Analyze ground truth
# ---------------------------------------------------------

def analyze_ground_truth(path):
    print("\n" + "=" * 70)
    print("train_ground_truth")
    print("=" * 70)

    df = pl.scan_csv(
        path,
        separator="\t",
        infer_schema_length=1000,
        null_values=["", "NULL", "null", "None"],
    )

    print("\nColumns and data types:")
    print(df.collect_schema())

    row_count = df.select(
        pl.len().alias("rows")
    ).collect().item()

    print(f"\nRows: {row_count:,}")

    print("\nSample rows:")
    print(df.head(3).collect())


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("Business Entity Resolution - Data Analysis")
    print("=" * 70)

    for name, path in FILES.items():
        if not path.exists():
            print(f"\nWARNING: File not found: {path}")
            continue

        if name == "train_ground_truth":
            analyze_ground_truth(path)
        else:
            analyze_source_file(name, path)

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
