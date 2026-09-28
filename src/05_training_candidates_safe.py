from pathlib import Path
import polars as pl


BASE_DIR = Path(".")
NORM_DIR = BASE_DIR / "normalized_data"
OUTPUT_DIR = BASE_DIR / "training_candidate_blocks"
OUTPUT_DIR.mkdir(exist_ok=True)

FINAL_OUTPUT = BASE_DIR / "training_candidate_pairs.tsv"


# ------------------------------------------------------------
# LOAD TRAINING DATA
# ------------------------------------------------------------

print("=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

s1 = pl.read_csv(
    NORM_DIR / "train_source1_normalized.tsv",
    separator="\t",
)

s2 = pl.read_csv(
    NORM_DIR / "train_source2_normalized.tsv",
    separator="\t",
)

s3 = pl.read_csv(
    NORM_DIR / "train_source3_normalized.tsv",
    separator="\t",
)

s23 = pl.concat([s2, s3])

print(f"S1 rows:     {s1.height:,}")
print(f"S2+S3 rows: {s23.height:,}")


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def house_number(expr):
    return expr.str.extract(r"^(\d+)", 1)


def name_prefix(expr):
    return expr.str.slice(0, 8)


def address_prefix(expr):
    return expr.str.slice(0, 12)


def address_suffix(expr):
    return expr.str.slice(-12)


def save_block(name, df):
    """
    Save one blocking strategy to disk.
    """

    print("\n" + "-" * 70)
    print(f"BLOCK: {name}")
    print("-" * 70)

    df = (
        df
        .select([
            "source1_entity_id",
            "candidate_entity_id",
        ])
        .unique()
    )

    print(f"Candidate pairs: {df.height:,}")

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "")
        .replace("<", "")
        .replace(">", "")
    )

    path = OUTPUT_DIR / f"{filename}.tsv"

    df.write_csv(
        path,
        separator="\t",
    )

    print(f"Saved: {path}")

    # Free this block from memory.
    del df


# ------------------------------------------------------------
# BLOCK 1
# EXACT NORMALIZED NAME
# ------------------------------------------------------------

print("\nPreparing exact-name block...")

s1_name = (
    s1
    .select(["entity_id", "name_normalized"])
    .filter(pl.col("name_normalized") != "")
)

s23_name = (
    s23
    .select(["entity_id", "name_normalized"])
    .filter(pl.col("name_normalized") != "")
)

candidates = (
    s1_name
    .join(
        s23_name,
        on="name_normalized",
        how="inner",
        suffix="_candidate",
    )
    .select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
    ])
)

save_block("exact_normalized_name", candidates)


# ------------------------------------------------------------
# BLOCK 2
# HOUSE NUMBER + NAME PREFIX
# ------------------------------------------------------------

print("\nPreparing house-number + name-prefix block...")

s1_hn = (
    s1
    .select([
        "entity_id",
        "name_normalized",
        "address_normalized",
    ])
    .with_columns([
        house_number(
            pl.col("address_normalized")
        ).alias("house_number"),

        name_prefix(
            pl.col("name_normalized")
        ).alias("name_prefix"),
    ])
    .filter(
        (pl.col("house_number") != "")
        & (pl.col("name_prefix") != "")
    )
)

s23_hn = (
    s23
    .select([
        "entity_id",
        "name_normalized",
        "address_normalized",
    ])
    .with_columns([
        house_number(
            pl.col("address_normalized")
        ).alias("house_number"),

        name_prefix(
            pl.col("name_normalized")
        ).alias("name_prefix"),
    ])
    .filter(
        (pl.col("house_number") != "")
        & (pl.col("name_prefix") != "")
    )
)

candidates = (
    s1_hn
    .join(
        s23_hn,
        on=["house_number", "name_prefix"],
        how="inner",
        suffix="_candidate",
    )
    .select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
    ])
)

save_block("house_number_name_prefix", candidates)


# ------------------------------------------------------------
# BLOCK 3
# RARE FIRST NAME TOKEN <=25
# ------------------------------------------------------------

print("\nPreparing rare first-name block...")


def first_token(expr):
    return (
        expr
        .str.split(" ")
        .list.get(0)
    )


s1_first = (
    s1
    .select(["entity_id", "name_normalized"])
    .with_columns(
        first_token(
            pl.col("name_normalized")
        ).alias("token")
    )
    .filter(pl.col("token") != "")
)

s23_first = (
    s23
    .select(["entity_id", "name_normalized"])
    .with_columns(
        first_token(
            pl.col("name_normalized")
        ).alias("token")
    )
    .filter(pl.col("token") != "")
)

first_counts = (
    s23_first
    .group_by("token")
    .len()
    .rename({"len": "frequency"})
)

rare_first = (
    first_counts
    .filter(pl.col("frequency") <= 25)
    .select("token")
)

s1_first = s1_first.join(
    rare_first,
    on="token",
    how="inner",
)

s23_first = s23_first.join(
    rare_first,
    on="token",
    how="inner",
)

candidates = (
    s1_first
    .join(
        s23_first,
        on="token",
        how="inner",
        suffix="_candidate",
    )
    .select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
    ])
)

save_block("rare_first_name_25", candidates)


# ------------------------------------------------------------
# BLOCK 4
# RARE ADDRESS PREFIX <=5
# ------------------------------------------------------------

print("\nPreparing rare address-prefix block...")

s1_addr = (
    s1
    .select(["entity_id", "address_normalized"])
    .with_columns(
        address_prefix(
            pl.col("address_normalized")
        ).alias("address_prefix")
    )
    .filter(pl.col("address_prefix") != "")
)

s23_addr = (
    s23
    .select(["entity_id", "address_normalized"])
    .with_columns(
        address_prefix(
            pl.col("address_normalized")
        ).alias("address_prefix")
    )
    .filter(pl.col("address_prefix") != "")
)

addr_counts = (
    s23_addr
    .group_by("address_prefix")
    .len()
    .rename({"len": "frequency"})
)

rare_addr = (
    addr_counts
    .filter(pl.col("frequency") <= 5)
    .select("address_prefix")
)

s1_addr = s1_addr.join(
    rare_addr,
    on="address_prefix",
    how="inner",
)

s23_addr = s23_addr.join(
    rare_addr,
    on="address_prefix",
    how="inner",
)

candidates = (
    s1_addr
    .join(
        s23_addr,
        on="address_prefix",
        how="inner",
        suffix="_candidate",
    )
    .select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
    ])
)

save_block("rare_address_prefix_5", candidates)


# ------------------------------------------------------------
# BLOCK 5
# RARE ADDRESS SUFFIX <=5
# ------------------------------------------------------------

print("\nPreparing rare address-suffix block...")

s1_suffix = (
    s1
    .select(["entity_id", "address_normalized"])
    .with_columns(
        address_suffix(
            pl.col("address_normalized")
        ).alias("address_suffix")
    )
    .filter(pl.col("address_suffix") != "")
)

s23_suffix = (
    s23
    .select(["entity_id", "address_normalized"])
    .with_columns(
        address_suffix(
            pl.col("address_normalized")
        ).alias("address_suffix")
    )
    .filter(pl.col("address_suffix") != "")
)

suffix_counts = (
    s23_suffix
    .group_by("address_suffix")
    .len()
    .rename({"len": "frequency"})
)

rare_suffix = (
    suffix_counts
    .filter(pl.col("frequency") <= 5)
    .select("address_suffix")
)

s1_suffix = s1_suffix.join(
    rare_suffix,
    on="address_suffix",
    how="inner",
)

s23_suffix = s23_suffix.join(
    rare_suffix,
    on="address_suffix",
    how="inner",
)

candidates = (
    s1_suffix
    .join(
        s23_suffix,
        on="address_suffix",
        how="inner",
        suffix="_candidate",
    )
    .select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
    ])
)

save_block("rare_address_suffix_5", candidates)


# ------------------------------------------------------------
# BLOCK 6
# RARE SECOND NAME TOKEN <=25
# ------------------------------------------------------------

print("\nPreparing rare second-name block...")


def second_token(expr):
    return (
        expr
        .str.split(" ")
        .list.get(
            1,
            null_on_oob=True,
        )
    )


s1_second = (
    s1
    .select(["entity_id", "name_normalized"])
    .with_columns(
        second_token(
            pl.col("name_normalized")
        ).alias("token")
    )
    .filter(
        pl.col("token").is_not_null()
        & (pl.col("token") != "")
    )
)

s23_second = (
    s23
    .select(["entity_id", "name_normalized"])
    .with_columns(
        second_token(
            pl.col("name_normalized")
        ).alias("token")
    )
    .filter(
        pl.col("token").is_not_null()
        & (pl.col("token") != "")
    )
)

second_counts = (
    s23_second
    .group_by("token")
    .len()
    .rename({"len": "frequency"})
)

rare_second = (
    second_counts
    .filter(pl.col("frequency") <= 25)
    .select("token")
)

s1_second = s1_second.join(
    rare_second,
    on="token",
    how="inner",
)

s23_second = s23_second.join(
    rare_second,
    on="token",
    how="inner",
)

candidates = (
    s1_second
    .join(
        s23_second,
        on="token",
        how="inner",
        suffix="_candidate",
    )
    .select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
    ])
)

save_block("rare_second_name_25", candidates)


# ------------------------------------------------------------
# BLOCK 7
# RARE EXACT ADDRESS <=10
# ------------------------------------------------------------

print("\nPreparing rare exact-address block...")

s1_exact_addr = (
    s1
    .select(["entity_id", "address_normalized"])
    .filter(pl.col("address_normalized") != "")
)

s23_exact_addr = (
    s23
    .select(["entity_id", "address_normalized"])
    .filter(pl.col("address_normalized") != "")
)

exact_addr_counts = (
    s23_exact_addr
    .group_by("address_normalized")
    .len()
    .rename({"len": "frequency"})
)

rare_exact_addr = (
    exact_addr_counts
    .filter(pl.col("frequency") <= 10)
    .select("address_normalized")
)

s1_exact_addr = s1_exact_addr.join(
    rare_exact_addr,
    on="address_normalized",
    how="inner",
)

s23_exact_addr = s23_exact_addr.join(
    rare_exact_addr,
    on="address_normalized",
    how="inner",
)

candidates = (
    s1_exact_addr
    .join(
        s23_exact_addr,
        on="address_normalized",
        how="inner",
        suffix="_candidate",
    )
    .select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
    ])
)

save_block("rare_exact_address_10", candidates)


# ------------------------------------------------------------
# BLOCK 8
# RARE LAST NAME TOKEN <=25
# ------------------------------------------------------------

print("\nPreparing rare last-name block...")


def last_token(expr):
    return (
        expr
        .str.split(" ")
        .list.last()
    )


s1_last = (
    s1
    .select(["entity_id", "name_normalized"])
    .with_columns(
        last_token(
            pl.col("name_normalized")
        ).alias("token")
    )
    .filter(pl.col("token") != "")
)

s23_last = (
    s23
    .select(["entity_id", "name_normalized"])
    .with_columns(
        last_token(
            pl.col("name_normalized")
        ).alias("token")
    )
    .filter(pl.col("token") != "")
)

last_counts = (
    s23_last
    .group_by("token")
    .len()
    .rename({"len": "frequency"})
)

rare_last = (
    last_counts
    .filter(pl.col("frequency") <= 25)
    .select("token")
)

s1_last = s1_last.join(
    rare_last,
    on="token",
    how="inner",
)

s23_last = s23_last.join(
    rare_last,
    on="token",
    how="inner",
)

candidates = (
    s1_last
    .join(
        s23_last,
        on="token",
        how="inner",
        suffix="_candidate",
    )
    .select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
    ])
)

save_block("rare_last_name_25", candidates)


# ------------------------------------------------------------
# BLOCK 9
# INFORMATIVE NAME TOKEN <=25
# ------------------------------------------------------------

print("\nPreparing informative-name block...")

generic_tokens = {
    "the",
    "private",
    "limited",
    "ltd",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "pvt",
    "plc",
    "llp",
    "group",
    "holdings",
    "holding",
    "services",
    "service",
    "solutions",
    "international",
    "india",
    "usa",
    "us",
}


def informative_tokens(expr):
    return (
        expr
        .str.split(" ")
        .list.eval(
            pl.element().filter(
                ~pl.element().is_in(generic_tokens)
            )
        )
    )


s1_info = (
    s1
    .select(["entity_id", "name_normalized"])
    .with_columns(
        informative_tokens(
            pl.col("name_normalized")
        ).alias("tokens")
    )
    .explode("tokens")
    .rename({"tokens": "token"})
    .filter(pl.col("token") != "")
)

s23_info = (
    s23
    .select(["entity_id", "name_normalized"])
    .with_columns(
        informative_tokens(
            pl.col("name_normalized")
        ).alias("tokens")
    )
    .explode("tokens")
    .rename({"tokens": "token"})
    .filter(pl.col("token") != "")
)

info_counts = (
    s23_info
    .group_by("token")
    .len()
    .rename({"len": "frequency"})
)

rare_info = (
    info_counts
    .filter(pl.col("frequency") <= 25)
    .select("token")
)

s1_info = s1_info.join(
    rare_info,
    on="token",
    how="inner",
)

s23_info = s23_info.join(
    rare_info,
    on="token",
    how="inner",
)

candidates = (
    s1_info
    .join(
        s23_info,
        on="token",
        how="inner",
        suffix="_candidate",
    )
    .select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
    ])
)

save_block("informative_name_25", candidates)


# ------------------------------------------------------------
# COMBINE BLOCKS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COMBINING BLOCKS")
print("=" * 70)

block_files = list(
    OUTPUT_DIR.glob("*.tsv")
)

print(f"Block files: {len(block_files)}")

if not block_files:
    raise RuntimeError("No candidate blocks were generated.")

frames = []

for path in block_files:
    print(f"Reading: {path.name}")

    df = pl.read_csv(
        path,
        separator="\t",
        schema_overrides={
            "source1_entity_id": pl.String,
            "candidate_entity_id": pl.String,
        },
    )

    frames.append(df)


combined = pl.concat(frames).unique()

print(
    f"Unique candidate pairs: "
    f"{combined.height:,}"
)

# ------------------------------------------------------------
# SAVE LONG-FORM CANDIDATES
# ------------------------------------------------------------

combined.write_csv(
    FINAL_OUTPUT,
    separator="\t",
)

print(f"Saved: {FINAL_OUTPUT}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)