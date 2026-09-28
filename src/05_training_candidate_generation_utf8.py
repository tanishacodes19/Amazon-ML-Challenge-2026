import polars as pl
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
NORM_DIR = BASE_DIR / "normalized_data"
TEST_DIR = (
    BASE_DIR.parent
    / "student_resource"
    / "student_resource"
    / "dataset"
    / "test"
)

OUTPUT = BASE_DIR / "training_candidate_pairs.tsv"

print("=" * 70)
print("FINAL CANDIDATE GENERATION")
print("=" * 70)

# ------------------------------------------------------------
# LOAD NORMALIZED TEST DATA
# ------------------------------------------------------------

print("\nLoading normalized test data...")

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
# FINAL CANDIDATE STORAGE
# ------------------------------------------------------------

# Dictionary:
# S1 entity_id -> set of candidate S2/S3 entity_ids
#
# We process one block at a time and immediately merge
# candidates into this dictionary.

candidate_map = defaultdict(set)


def add_candidates(df):
    """
    Add candidate pairs from a Polars DataFrame to candidate_map.
    Expected columns:
        source1_entity_id
        candidate_entity_id
    """

    if df.height == 0:
        return

    for row in df.iter_rows():
        source_id = row[0]
        candidate_id = row[1]

        candidate_map[source_id].add(candidate_id)


def run_block(name, candidates):
    """
    Run one blocking strategy and merge its candidates.
    """

    print("\n" + "-" * 70)
    print(f"BLOCK: {name}")
    print("-" * 70)

    candidates = (
        candidates
        .select([
            "source1_entity_id",
            "candidate_entity_id",
        ])
        .unique()
    )

    print(
        f"Unique candidate pairs: "
        f"{candidates.height:,}"
    )

    add_candidates(candidates)

    print(
        f"S1 entities with candidates so far: "
        f"{len(candidate_map):,}"
    )


# ------------------------------------------------------------
# COMMON HELPERS
# ------------------------------------------------------------

def house_number(expr):
    return expr.str.extract(
        r"^(\d+)",
        1,
    )


def name_prefix(expr):
    return expr.str.slice(0, 8)


def address_prefix(expr):
    return expr.str.slice(0, 12)


def address_suffix(expr):
    return expr.str.slice(-12)


# ------------------------------------------------------------
# BLOCK 1
# EXACT NORMALIZED NAME
# ------------------------------------------------------------

print("\nPreparing exact-name block...")

s1_name = (
    s1
    .select([
        "entity_id",
        "name_normalized",
    ])
    .filter(
        pl.col("name_normalized") != ""
    )
)

s23_name = (
    s23
    .select([
        "entity_id",
        "name_normalized",
    ])
    .filter(
        pl.col("name_normalized") != ""
    )
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
        pl.col("entity_id")
            .alias("source1_entity_id"),
        pl.col("entity_id_candidate")
            .alias("candidate_entity_id"),
    ])
)

run_block(
    "Exact normalized name",
    candidates,
)


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
        &
        (pl.col("name_prefix") != "")
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
        &
        (pl.col("name_prefix") != "")
    )
)

candidates = (
    s1_hn
    .join(
        s23_hn,
        on=[
            "house_number",
            "name_prefix",
        ],
        how="inner",
        suffix="_candidate",
    )
    .select([
        pl.col("entity_id")
            .alias("source1_entity_id"),
        pl.col("entity_id_candidate")
            .alias("candidate_entity_id"),
    ])
)

run_block(
    "House number + name prefix",
    candidates,
)


# ------------------------------------------------------------
# BLOCK 3
# RARE FIRST NAME TOKEN <= 25
# ------------------------------------------------------------

print("\nPreparing rare first-name token block...")

def first_token(expr):
    return (
        expr
        .str.split(" ")
        .list.get(0)
    )


s1_first = (
    s1
    .select([
        "entity_id",
        "name_normalized",
    ])
    .with_columns(
        first_token(
            pl.col("name_normalized")
        ).alias("token")
    )
    .filter(
        pl.col("token") != ""
    )
)

s23_first = (
    s23
    .select([
        "entity_id",
        "name_normalized",
    ])
    .with_columns(
        first_token(
            pl.col("name_normalized")
        ).alias("token")
    )
    .filter(
        pl.col("token") != ""
    )
)

first_counts = (
    s23_first
    .group_by("token")
    .len()
    .rename({"len": "frequency"})
)

rare_first = (
    first_counts
    .filter(
        pl.col("frequency") <= 25
    )
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
        pl.col("entity_id")
            .alias("source1_entity_id"),
        pl.col("entity_id_candidate")
            .alias("candidate_entity_id"),
    ])
)

run_block(
    "Rare first-name token <=25",
    candidates,
)


# ------------------------------------------------------------
# BLOCK 4
# RARE ADDRESS PREFIX <= 5
# ------------------------------------------------------------

print("\nPreparing rare address-prefix block...")

s1_addr = (
    s1
    .select([
        "entity_id",
        "address_normalized",
    ])
    .with_columns(
        address_prefix(
            pl.col("address_normalized")
        ).alias("address_prefix")
    )
    .filter(
        pl.col("address_prefix") != ""
    )
)

s23_addr = (
    s23
    .select([
        "entity_id",
        "address_normalized",
    ])
    .with_columns(
        address_prefix(
            pl.col("address_normalized")
        ).alias("address_prefix")
    )
    .filter(
        pl.col("address_prefix") != ""
    )
)

addr_counts = (
    s23_addr
    .group_by("address_prefix")
    .len()
    .rename({"len": "frequency"})
)

rare_addr = (
    addr_counts
    .filter(
        pl.col("frequency") <= 5
    )
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
        pl.col("entity_id")
            .alias("source1_entity_id"),
        pl.col("entity_id_candidate")
            .alias("candidate_entity_id"),
    ])
)

run_block(
    "Rare address prefix <=5",
    candidates,
)


# ------------------------------------------------------------
# BLOCK 5
# RARE ADDRESS SUFFIX <= 5
# ------------------------------------------------------------

print("\nPreparing rare address-suffix block...")

s1_suffix = (
    s1
    .select([
        "entity_id",
        "address_normalized",
    ])
    .with_columns(
        address_suffix(
            pl.col("address_normalized")
        ).alias("address_suffix")
    )
    .filter(
        pl.col("address_suffix") != ""
    )
)

s23_suffix = (
    s23
    .select([
        "entity_id",
        "address_normalized",
    ])
    .with_columns(
        address_suffix(
            pl.col("address_normalized")
        ).alias("address_suffix")
    )
    .filter(
        pl.col("address_suffix") != ""
    )
)

suffix_counts = (
    s23_suffix
    .group_by("address_suffix")
    .len()
    .rename({"len": "frequency"})
)

rare_suffix = (
    suffix_counts
    .filter(
        pl.col("frequency") <= 5
    )
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
        pl.col("entity_id")
            .alias("source1_entity_id"),
        pl.col("entity_id_candidate")
            .alias("candidate_entity_id"),
    ])
)

run_block(
    "Rare address suffix <=5",
    candidates,
)


# ------------------------------------------------------------
# BLOCK 6
# RARE SECOND NAME TOKEN <=25
# ------------------------------------------------------------

print("\nPreparing rare second-name token block...")


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
    .select([
        "entity_id",
        "name_normalized",
    ])
    .with_columns(
        second_token(
            pl.col("name_normalized")
        ).alias("token")
    )
    .filter(
        pl.col("token").is_not_null()
        &
        (pl.col("token") != "")
    )
)

s23_second = (
    s23
    .select([
        "entity_id",
        "name_normalized",
    ])
    .with_columns(
        second_token(
            pl.col("name_normalized")
        ).alias("token")
    )
    .filter(
        pl.col("token").is_not_null()
        &
        (pl.col("token") != "")
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
    .filter(
        pl.col("frequency") <= 25
    )
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
        pl.col("entity_id")
            .alias("source1_entity_id"),
        pl.col("entity_id_candidate")
            .alias("candidate_entity_id"),
    ])
)

run_block(
    "Rare second-name token <=25",
    candidates,
)


# ------------------------------------------------------------
# BLOCK 7
# RARE EXACT ADDRESS <=10
# ------------------------------------------------------------

print("\nPreparing rare exact-address block...")

s1_exact_addr = (
    s1
    .select([
        "entity_id",
        "address_normalized",
    ])
    .filter(
        pl.col("address_normalized") != ""
    )
)

s23_exact_addr = (
    s23
    .select([
        "entity_id",
        "address_normalized",
    ])
    .filter(
        pl.col("address_normalized") != ""
    )
)

exact_addr_counts = (
    s23_exact_addr
    .group_by("address_normalized")
    .len()
    .rename({"len": "frequency"})
)

rare_exact_addr = (
    exact_addr_counts
    .filter(
        pl.col("frequency") <= 10
    )
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
        pl.col("entity_id")
            .alias("source1_entity_id"),
        pl.col("entity_id_candidate")
            .alias("candidate_entity_id"),
    ])
)

run_block(
    "Rare exact address <=10",
    candidates,
)


# ------------------------------------------------------------
# BLOCK 8
# RARE LAST NAME TOKEN <=25
# ------------------------------------------------------------

print("\nPreparing rare last-name token block...")


def last_token(expr):
    return (
        expr
        .str.split(" ")
        .list.last()
    )


s1_last = (
    s1
    .select([
        "entity_id",
        "name_normalized",
    ])
    .with_columns(
        last_token(
            pl.col("name_normalized")
        ).alias("token")
    )
    .filter(
        pl.col("token") != ""
    )
)

s23_last = (
    s23
    .select([
        "entity_id",
        "name_normalized",
    ])
    .with_columns(
        last_token(
            pl.col("name_normalized")
        ).alias("token")
    )
    .filter(
        pl.col("token") != ""
    )
)

last_counts = (
    s23_last
    .group_by("token")
    .len()
    .rename({"len": "frequency"})
)

rare_last = (
    last_counts
    .filter(
        pl.col("frequency") <= 25
    )
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
        pl.col("entity_id")
            .alias("source1_entity_id"),
        pl.col("entity_id_candidate")
            .alias("candidate_entity_id"),
    ])
)

run_block(
    "Rare last-name token <=25",
    candidates,
)


# ------------------------------------------------------------
# BLOCK 9
# INFORMATIVE NAME TOKEN <=25
# ------------------------------------------------------------

print("\nPreparing informative-token block...")

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
    .select([
        "entity_id",
        "name_normalized",
    ])
    .with_columns(
        informative_tokens(
            pl.col("name_normalized")
        ).alias("tokens")
    )
    .explode("tokens")
    .rename({
        "tokens": "token"
    })
    .filter(
        pl.col("token") != ""
    )
)

s23_info = (
    s23
    .select([
        "entity_id",
        "name_normalized",
    ])
    .with_columns(
        informative_tokens(
            pl.col("name_normalized")
        ).alias("tokens")
    )
    .explode("tokens")
    .rename({
        "tokens": "token"
    })
    .filter(
        pl.col("token") != ""
    )
)

info_counts = (
    s23_info
    .group_by("token")
    .len()
    .rename({"len": "frequency"})
)

rare_info = (
    info_counts
    .filter(
        pl.col("frequency") <= 25
    )
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
        pl.col("entity_id")
            .alias("source1_entity_id"),
        pl.col("entity_id_candidate")
            .alias("candidate_entity_id"),
    ])
)

run_block(
    "Informative name token <=25",
    candidates,
)


# ------------------------------------------------------------
# BUILD FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BUILDING FINAL training_candidate_pairs.tsv")
print("=" * 70)

print(
    f"\nS1 entities with at least one candidate: "
    f"{len(candidate_map):,}"
)

rows = []

for entity_id in s1["entity_id"].to_list():

    candidates = candidate_map.get(
        entity_id,
        set(),
    )

    candidate_string = ",".join(
        sorted(candidates)
    )

    rows.append({
        "source1_entity_id": entity_id,
        "candidate_entity_ids": candidate_string,
    })


final_df = pl.DataFrame(
    rows,
    schema={
        "source1_entity_id": pl.String,
        "candidate_entity_ids": pl.String,
    },
)

print(
    f"Final output rows: "
    f"{final_df.height:,}"
)

print(
    f"Expected S1 rows: "
    f"{s1.height:,}"
)

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\nVALIDATION")

assert (
    final_df.height
    == s1.height
), "Wrong number of S1 rows"

assert (
    final_df["source1_entity_id"].n_unique()
    == s1["entity_id"].n_unique()
), "Duplicate S1 entity IDs"

# Check candidate IDs are from S2+S3
valid_ids = set(
    s23["entity_id"].to_list()
)

bad_count = 0

for value in final_df["candidate_entity_ids"].to_list():

    if not value:
        continue

    ids = value.split(",")

    if len(ids) != len(set(ids)):
        bad_count += 1
        break

    for candidate_id in ids:
        if candidate_id not in valid_ids:
            bad_count += 1
            break

    if bad_count:
        break

assert (
    bad_count == 0
), "Invalid or duplicate candidate IDs found"

print("Γ£ô Correct number of S1 rows")
print("Γ£ô No duplicate S1 entity IDs")
print("Γ£ô No duplicate candidate IDs")
print("Γ£ô All candidate IDs belong to S2/S3")

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

final_df.write_csv(
    OUTPUT,
    separator="\t",
)

print(
    f"\nSaved: {OUTPUT}"
)

print("=" * 70)
print("DONE")
print("=" * 70)
