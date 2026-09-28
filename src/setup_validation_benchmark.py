import duckdb
import os
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
from rapidfuzz import fuzz

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
os.makedirs(VAL_DIR, exist_ok=True)

GT_PAIRS = os.path.join(BASE, "ground_truth_pairs.tsv")
CANDIDATES = os.path.join(BASE, "training_candidate_pairs.tsv")
S1_FILE = os.path.join(BASE, "normalized_data", "train_source1_normalized.tsv")
S2_FILE = os.path.join(BASE, "normalized_data", "train_source2_normalized.tsv")
S3_FILE = os.path.join(BASE, "normalized_data", "train_source3_normalized.tsv")

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3GB'")

print("Selecting 25,000 validation S1 entities...")
val_s1 = con.execute(f"""
SELECT 
    entity_id AS source1_entity_id,
    business_name,
    business_address,
    country,
    name_normalized AS name_normalized_s1,
    address_normalized AS address_normalized_s1
FROM read_csv('{S1_FILE.replace(chr(92), '/')}', delim='\\t', header=true)
WHERE ABS(HASH(entity_id)) % 4 = 0
ORDER BY entity_id
LIMIT 25000
""").df()

val_s1.to_parquet(os.path.join(VAL_DIR, "val_s1.parquet"), index=False)
con.register("val_s1", val_s1)

print("Selecting ground truth for validation S1...")
val_gt = con.execute(f"""
SELECT g.source1_entity_id, g.matched_entity_id
FROM read_csv('{GT_PAIRS.replace(chr(92), '/')}', delim='\\t', header=true) g
JOIN val_s1 v ON g.source1_entity_id = v.source1_entity_id
""").df()
val_gt.to_parquet(os.path.join(VAL_DIR, "val_gt.parquet"), index=False)

print("Selecting baseline candidates for validation S1...")
val_cands = con.execute(f"""
SELECT c.source1_entity_id, c.candidate_entity_id AS matched_entity_id
FROM read_csv('{CANDIDATES.replace(chr(92), '/')}', delim='\\t', header=true) c
JOIN val_s1 v ON c.source1_entity_id = v.source1_entity_id
""").df()
val_cands.to_parquet(os.path.join(VAL_DIR, "val_cands.parquet"), index=False)

# Collect all matched IDs needed
con.register("val_cands", val_cands)
con.register("val_gt", val_gt)

val_match_ids = con.execute("""
SELECT DISTINCT matched_entity_id FROM val_cands
UNION
SELECT DISTINCT matched_entity_id FROM val_gt
""").df()
con.register("val_match_ids", val_match_ids)

print("Selecting S2 and S3 rows for candidate pool...")
val_s23 = con.execute(f"""
WITH all_s23 AS (
    SELECT entity_id AS matched_entity_id, business_name, business_address, country, name_normalized AS matched_name, address_normalized AS matched_address
    FROM read_csv('{S2_FILE.replace(chr(92), '/')}', delim='\\t', header=true)
    UNION ALL
    SELECT entity_id AS matched_entity_id, business_name, business_address, country, name_normalized AS matched_name, address_normalized AS matched_address
    FROM read_csv('{S3_FILE.replace(chr(92), '/')}', delim='\\t', header=true)
)
SELECT a.* FROM all_s23 a
JOIN val_match_ids m ON a.matched_entity_id = m.matched_entity_id
""").df()
val_s23.to_parquet(os.path.join(VAL_DIR, "val_s23.parquet"), index=False)

print("\nBenchmark dataset saved:")
print(f"  val_s1: {len(val_s1):,}")
print(f"  val_gt: {len(val_gt):,}")
print(f"  val_cands: {len(val_cands):,}")
print(f"  val_s23: {len(val_s23):,}")
