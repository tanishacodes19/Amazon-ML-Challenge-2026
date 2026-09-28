import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import time
import numpy as np
import polars as pl

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
sys.path.append(os.path.join(BASE, "src"))
from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED

VAL_DIR = os.path.join(BASE, "validation_benchmark")

print("=" * 70)
print("EXTRACTING 52 FEATURES ON V16 CANDIDATES (623k PAIRS)")
print("=" * 70)

t0 = time.time()
v16_cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_cands.parquet"))
s1 = pl.read_parquet(os.path.join(VAL_DIR, "val_s1_v13_clean.parquet"))
s23 = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet"))

print(f"Loaded {len(v16_cands):,} candidate pairs in {time.time()-t0:.1f}s.")

# Join with text attributes
pairs_df = v16_cands.join(
    s1.select(["source1_entity_id", "business_name", "business_address", "country", "clean_name", "clean_addr"]),
    on="source1_entity_id"
).join(
    s23.select(["matched_entity_id", "business_name", "business_address", "clean_name", "clean_addr"]),
    on="matched_entity_id"
)

# Extract house and postal fields
def extract_hp(df_in, addr_col, prefix):
    return df_in.with_columns([
        pl.col(addr_col).str.extract(r'(?:^|#|\bno\.?|\bplot\.?|\bflat\.?|\bunit\s*)\s*(\d+[a-z]?)', 1).str.strip_chars_start('0').fill_null('').alias(f'{prefix}_house'),
        pl.col(addr_col).str.extract(r'\b(\d{5,6})\b', 1).fill_null('').alias(f'{prefix}_postal')
    ])

pairs_df = extract_hp(pairs_df, "clean_addr", "s1")
pairs_df = extract_hp(pairs_df, "clean_addr_right", "matched")

pairs_df = pairs_df.rename({
    "clean_name": "s1_name_norm",
    "clean_name_right": "matched_name_norm",
    "clean_addr": "s1_addr_norm",
    "clean_addr_right": "matched_addr_norm",
    "country": "s1_country"
}).with_columns(pl.col("s1_country").alias("matched_country"))

print(f"Prepared joined DataFrame ({len(pairs_df):,} rows). Extracting 52 features...")
t_feat = time.time()

# Chunked extraction to keep memory low
CHUNK_SZ = 100000
chunks_X = []
n_chunks = (len(pairs_df) + CHUNK_SZ - 1) // CHUNK_SZ

for i in range(n_chunks):
    sub_df = pairs_df.slice(i * CHUNK_SZ, CHUNK_SZ)
    feat_sub = extract_features_df(sub_df)
    X_sub = feat_sub.select(V5_FEATURES_EXPANDED).to_numpy()
    chunks_X.append(X_sub)
    print(f"  Chunk {i+1}/{n_chunks} ({len(sub_df):,} rows) extracted in {time.time()-t_feat:.1f}s.")
    t_feat = time.time()

X_all = np.vstack(chunks_X)
print(f"Extraction complete! Matrix shape: {X_all.shape}")

# Save X and metadata
out_X = os.path.join(VAL_DIR, "val_v16_features52_X.npy")
out_meta = os.path.join(VAL_DIR, "val_v16_features52_df.parquet")

np.save(out_X, X_all)
pairs_df.select([
    "source1_entity_id", "matched_entity_id"
]).with_columns([
    pl.Series("name_ratio", X_all[:, V5_FEATURES_EXPANDED.index("name_ratio")]),
    pl.Series("address_ratio", X_all[:, V5_FEATURES_EXPANDED.index("address_ratio")]),
    pl.Series("house_match", X_all[:, V5_FEATURES_EXPANDED.index("house_match")].astype(int))
]).write_parquet(out_meta)

print(f"Saved {out_X} ({os.path.getsize(out_X)/(1024*1024):.1f} MB)")
print(f"Saved {out_meta} ({os.path.getsize(out_meta)/(1024*1024):.1f} MB)")
print(f"TOTAL PIPELINE COMPLETED IN {time.time()-t0:.1f}s.")
