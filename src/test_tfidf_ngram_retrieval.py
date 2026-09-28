import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from eval_framework import load_benchmark
from normalizer import normalize_business_name

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

print("=" * 70)
print("CHANNEL G: CHARACTER N-GRAM TF-IDF SPARSE RETRIEVAL")
print("=" * 70)

t0 = time.time()
s1, gt, _, s23 = load_benchmark()
v7_path = os.path.join(VAL_DIR, "val_v7_cands.parquet")
v7_cands = pl.read_parquet(v7_path)

# Extract missed ground truth pairs in V7
v7_pairs = set(zip(v7_cands["source1_entity_id"], v7_cands["matched_entity_id"]))
gt_pairs = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
missed_pairs = gt_pairs - v7_pairs
print(f"Total Ground Truth:  {len(gt_pairs):,}")
print(f"V7 Recovered:        {len(gt_pairs) - len(missed_pairs):,} ({(len(gt_pairs) - len(missed_pairs))/len(gt_pairs)*100:.2f}%)")
print(f"Remaining Missed:    {len(missed_pairs):,} ({len(missed_pairs)/len(gt_pairs)*100:.2f}%)")

# Precompute clean normalized names
print("\nNormalizing names...")
s1_names = [normalize_business_name(x) for x in s1["business_name"].to_list()]
s23_names = [normalize_business_name(x) for x in s23["business_name"].to_list()]

s1_ids = s1["source1_entity_id"].to_list()
s23_ids = s23["matched_entity_id"].to_list()
s1_countries = s1["country"].to_list()
s23_countries = s23["country"].to_list()

# Group by country to avoid cross-country false matches
for target_country in ["US", "India"]:
    print(f"\nProcessing Country: {target_country}...")
    s1_idx = [i for i, c in enumerate(s1_countries) if c == target_country and len(s1_names[i]) >= 4]
    s23_idx = [i for i, c in enumerate(s23_countries) if c == target_country and len(s23_names[i]) >= 4]
    
    if not s1_idx or not s23_idx:
        continue
        
    s1_sub_names = [s1_names[i] for i in s1_idx]
    s23_sub_names = [s23_names[i] for i in s23_idx]
    
    print(f"  S1 count: {len(s1_sub_names):,}, S23 count: {len(s23_sub_names):,}")
    
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_df=0.2)
    vec.fit(s1_sub_names + s23_sub_names)
    
    X1 = vec.transform(s1_sub_names)
    X2 = vec.transform(s23_sub_names)
    
    # Chunked sparse matrix multiplication to stay under memory
    CHUNK = 5000
    recovered_in_country = 0
    cands_in_country = 0
    
    COSINE_THRESHOLD = 0.75
    TOP_K = 10
    
    new_pairs = []
    
    for start in range(0, len(s1_idx), CHUNK):
        end = min(start + CHUNK, len(s1_idx))
        sub_X1 = X1[start:end]
        
        sim = sub_X1.dot(X2.T) # Sparse dot product
        
        # Extract pairs exceeding threshold
        cx = sim.tocoo()
        for r_i, c_j, val in zip(cx.row, cx.col, cx.data):
            if val >= COSINE_THRESHOLD:
                orig_s1_id = s1_ids[s1_idx[start + r_i]]
                orig_s23_id = s23_ids[s23_idx[c_j]]
                new_pairs.append((orig_s1_id, orig_s23_id))
                
    new_pair_set = set(new_pairs)
    rec_missed = len(new_pair_set & missed_pairs)
    print(f"  {target_country} N-gram Cosine Candidates: {len(new_pair_set):,} -> Recovered Missed: {rec_missed:,} ({(rec_missed/max(1, len(new_pair_set)))*100:.2f}% eff)")

print(f"\nCompleted in {time.time()-t0:.1f}s")
