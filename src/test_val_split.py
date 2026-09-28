import duckdb
import numpy as np
import pandas as pd
import polars as pl
import os

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
GT_PAIRS = os.path.join(BASE, "ground_truth_pairs.tsv")
CANDIDATES = os.path.join(BASE, "training_candidate_pairs.tsv")
S1_FILE = os.path.join(BASE, "normalized_data", "train_source1_normalized.tsv")

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3GB'")

# Sample 20,000 validation S1 entities using entity hash
val_s1 = con.execute(f"""
SELECT entity_id AS source1_entity_id
FROM read_csv('{S1_FILE.replace(chr(92), '/')}', delim='\\t', header=true)
WHERE ABS(HASH(entity_id)) % 4 = 0
ORDER BY entity_id
LIMIT 25000
""").df()

con.register("val_s1", val_s1)

val_gt = con.execute(f"""
SELECT g.source1_entity_id, g.matched_entity_id
FROM read_csv('{GT_PAIRS.replace(chr(92), '/')}', delim='\\t', header=true) g
JOIN val_s1 v ON g.source1_entity_id = v.source1_entity_id
""").df()

val_cands = con.execute(f"""
SELECT c.source1_entity_id, c.candidate_entity_id AS matched_entity_id
FROM read_csv('{CANDIDATES.replace(chr(92), '/')}', delim='\\t', header=true) c
JOIN val_s1 v ON c.source1_entity_id = v.source1_entity_id
""").df()

print(f"Validation S1 count: {len(val_s1):,}")
print(f"Validation Ground Truth pairs: {len(val_gt):,}")
print(f"Validation Candidate pairs: {len(val_cands):,}")

# Blocking recall on validation set
con.register("val_gt", val_gt)
con.register("val_cands", val_cands)

recovered = con.execute("""
SELECT COUNT(*) FROM val_gt g
JOIN val_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]

print(f"Recovered true matches: {recovered:,} / {len(val_gt):,} ({recovered/len(val_gt)*100:.2f}%)")
print(f"Average candidates per S1: {len(val_cands)/len(val_s1):.2f}")
