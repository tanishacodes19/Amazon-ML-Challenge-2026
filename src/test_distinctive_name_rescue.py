import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, numpy as np, polars as pl
from rapidfuzz import fuzz

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

from feature_engine_v2 import V5_FEATURES_EXPANDED
feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
c_match = feat_dict["country_match"]
name_ex = feat_dict["name_exact"]
name_tok_s = feat_dict["name_token_set"]

# Clean names
s1_name_dict = dict(zip(s1["source1_entity_id"], s1["clean_name"]))
s23_name_dict = dict(zip(s23["matched_entity_id"], s23["clean_name"]))

s1_names = np.array([s1_name_dict.get(sid, "") for sid in s1_ids])
m_names = np.array([s23_name_dict.get(mid, "") for mid in m_ids])

# Distinctiveness: word count >= 3, and min char length >= 15
s1_n_words = np.array([len(n.split()) for n in s1_names])
s1_n_chars = np.array([len(n) for n in s1_names])

# S1 name frequency in validation set
s1_name_counts = s1["clean_name"].value_counts()
name_freq_map = dict(zip(s1_name_counts["clean_name"], s1_name_counts["count"]))
s1_name_freq = np.array([name_freq_map.get(n, 999) for n in s1_names])

print("=== DISTINCTIVE NAME RESCUE WHEN ADDRESS IS MISSING ===")
for min_words in [2, 3, 4]:
    for min_chars in [12, 15, 18, 22]:
        m = (c_match == 1) & (addr_r < 0.25) & (name_ex == 1) & (s1_n_words >= min_words) & (s1_n_chars >= min_chars) & (s1_name_freq == 1)
        tp = (m & is_gt).sum()
        fp = (m & ~is_gt).sum()
        pur = tp / (tp + fp) if (tp + fp) > 0 else 0
        if tp >= 50:
            print(f"words >= {min_words}, chars >= {min_chars}, name_freq=1 & name_ex=1: TP={tp:4,d}, FP={fp:3,d}, Purity={pur:.2%}")
