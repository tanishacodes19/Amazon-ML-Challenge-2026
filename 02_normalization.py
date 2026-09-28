from pathlib import Path
import polars as pl


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\viper\Documents\student_resource\student_resource\dataset"
)

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = Path("normalized_data")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# NORMALIZATION EXPRESSIONS
# ============================================================

def normalized_name_expr():
    """
    Normalize business names while preserving Unicode text.
    """

    return (
        pl.col("business_name")
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def normalized_address_expr():
    """
    Normalize addresses while preserving Unicode text.
    """

    return (
        pl.col("business_address")
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(input_path: Path, output_path: Path):
    print()
    print("=" * 70)
    print(f"Processing: {input_path.name}")
    print("=" * 70)

    df = (
        pl.scan_csv(
            input_path,
            separator="\t",
            infer_schema_length=1000,
            null_values=["", "NULL", "null", "None"],
        )
        .with_columns(
            [
                normalized_name_expr().alias("name_normalized"),
                normalized_address_expr().alias("address_normalized"),
            ]
        )
    )

    print("Writing normalized file...")

    df.sink_csv(
        output_path,
        separator="\t",
    )

    print(f"Saved: {output_path}")
    print("Done.")


# ============================================================
# TRAIN DATA
# ============================================================

process_file(
    TRAIN_DIR / "train_source1.tsv",
    OUTPUT_DIR / "train_source1_normalized.tsv",
)

process_file(
    TRAIN_DIR / "train_source2.tsv",
    OUTPUT_DIR / "train_source2_normalized.tsv",
)

process_file(
    TRAIN_DIR / "train_source3.tsv",
    OUTPUT_DIR / "train_source3_normalized.tsv",
)


# ============================================================
# TEST DATA
# ============================================================

process_file(
    TEST_DIR / "test_source1.tsv",
    OUTPUT_DIR / "test_source1_normalized.tsv",
)

process_file(
    TEST_DIR / "test_source2.tsv",
    OUTPUT_DIR / "test_source2_normalized.tsv",
)

process_file(
    TEST_DIR / "test_source3.tsv",
    OUTPUT_DIR / "test_source3_normalized.tsv",
)


print()
print("=" * 70)
print("NORMALIZATION COMPLETE")
print("=" * 70)