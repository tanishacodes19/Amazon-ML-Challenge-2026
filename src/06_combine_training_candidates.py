from pathlib import Path
import duckdb

BASE_DIR = Path(".")
BLOCK_DIR = BASE_DIR / "training_candidate_blocks"
OUTPUT = BASE_DIR / "training_candidate_pairs.tsv"
DB_FILE = BASE_DIR / "training_candidates.duckdb"

blocks = list(BLOCK_DIR.glob("*.tsv"))

print("=" * 70)
print("DUCKDB: COMBINING TRAINING CANDIDATE BLOCKS")
print("=" * 70)

print(f"Found {len(blocks)} block files.")

if len(blocks) != 9:
    raise RuntimeError(
        f"Expected 9 block files, found {len(blocks)}"
    )

for block in sorted(blocks):
    print(f"  ✓ {block.name}")


# ------------------------------------------------------------
# CREATE DUCKDB DATABASE
# ------------------------------------------------------------

con = duckdb.connect(str(DB_FILE))

# Keep DuckDB from using all available RAM.
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=2")

print("\nDuckDB memory limit: 2 GB")
print("Threads: 2")


# ------------------------------------------------------------
# COMBINE + DEDUPLICATE
# ------------------------------------------------------------

print("\nCombining and deduplicating...")

files_sql = ", ".join(
    f"'{str(p.resolve()).replace(chr(92), '/')}'"
    for p in blocks
)

query = f"""
COPY (
    SELECT DISTINCT
        source1_entity_id,
        candidate_entity_id
    FROM read_csv(
        [{files_sql}],
        delim='\\t',
        header=true,
        union_by_name=true,
        columns={{
            'source1_entity_id': 'VARCHAR',
            'candidate_entity_id': 'VARCHAR'
        }}
    )
)
TO '{str(OUTPUT.resolve()).replace(chr(92), '/')}'
(
    DELIMITER '\\t',
    HEADER true
)
"""

con.execute(query)


# ------------------------------------------------------------
# COUNT RESULT
# ------------------------------------------------------------

result = con.execute(
    f"""
    SELECT COUNT(*)
    FROM read_csv(
        '{str(OUTPUT.resolve()).replace(chr(92), '/')}',
        delim='\\t',
        header=true
    )
    """
).fetchone()

count = result[0]

print("\n" + "=" * 70)
print("COMBINATION COMPLETE")
print("=" * 70)

print(f"Unique training candidate pairs: {count:,}")
print(f"Saved: {OUTPUT}")


con.close()