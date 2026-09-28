import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, duckdb, polars as pl
import numpy as np

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))
s1 = pl.read_parquet(os.path.join(VAL_DIR, "val_s1_v13_clean.parquet"))
s23 = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet"))

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# Count how many times each normalized address appears in s1
s1_addr_counts = s1.filter(pl.col("clean_addr") != "")["clean_addr"].value_counts()
addr_count_map = dict(zip(s1_addr_counts["clean_addr"], s1_addr_counts["count"]))

# Get s1 address for each candidate pair
s1_addr_dict = dict(zip(s1["source1_entity_id"], s1["clean_addr"]))
s23_addr_dict = dict(zip(s23["matched_entity_id"], s23["clean_addr"]))

from feature_engine_v2 import V5_FEATURES_EXPANDED
feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
addr_ex = feat_dict["address_exact"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
name_r = feat_dict["name_ratio"]

# S1 address exclusivity
s1_addrs_for_pairs = np.array([s1_addr_dict.get(sid, "") for sid in s1_ids])
s1_addr_freq = np.array([addr_count_map.get(a, 999) for a in s1_addrs_for_pairs])

print("=== ADDRESS EXCLUSIVITY ANALYSIS ===")
# Where S1 address is globally UNIQUE in S1 (freq == 1)
is_unique_s1_addr = (s1_addr_freq == 1) & (s1_addrs_for_pairs != "")

for min_ar in [1.0, 0.95, 0.90]:
    for max_nr in [0.40, 0.50, 0.60]:
        m = is_unique_s1_addr & (addr_r >= min_ar) & (name_r < max_nr)
        tp = (m & is_gt).sum()
        fp = (m & ~is_gt).sum()
        pur = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"Unique S1 Addr & addr_r >= {min_ar:.2f} & name_r < {max_nr:.2f}: TP={tp:,}, FP={fp:,}, Purity={pur:.2%}")
