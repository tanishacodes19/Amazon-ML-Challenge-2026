import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import os, numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_ex = feat_dict["name_exact"]
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
postal_e = feat_dict["postal_exact"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]
c_match = feat_dict["country_match"]
addr_ex = feat_dict["address_exact"]

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

print("=== 1. MISSING ADDRESS + EXACT NAME ===")
for min_nr in [1.0, 0.95, 0.90]:
    m = (name_r >= min_nr) & (c_match == 1) & (addr_r < 0.25)
    tp = (m & is_gt).sum()
    fp = (m & ~is_gt).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"name_r >= {min_nr:.2f} & c_match=1 & addr_r < 0.25: TP={tp:,}, FP={fp:,}, Purity={p:.2%}")

print("\n=== 2. EXACT/VERY HIGH ADDRESS + ACRONYM ===")
for min_ar in [0.90, 0.80, 0.70]:
    m = (is_acronym == 1) & (addr_r >= min_ar)
    tp = (m & is_gt).sum()
    fp = (m & ~is_gt).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"is_acronym=1 & addr_r >= {min_ar:.2f}: TP={tp:,}, FP={fp:,}, Purity={p:.2%}")

print("\n=== 3. EXACT ADDRESS + HOUSE MATCH + HIGH NAME ===")
for min_nr in [0.70, 0.60, 0.50, 0.40]:
    m = (addr_ex == 1) & (house_m == 1) & (name_r >= min_nr)
    tp = (m & is_gt).sum()
    fp = (m & ~is_gt).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"addr_ex=1 & house_m=1 & name_r >= {min_nr:.2f}: TP={tp:,}, FP={fp:,}, Purity={p:.2%}")

print("\n=== 4. HIGH ADDRESS (0.80+) + HIGH NAME (0.80+) ===")
for min_thresh in [0.80, 0.82, 0.85]:
    m = (name_r >= min_thresh) & (addr_r >= min_thresh)
    tp = (m & is_gt).sum()
    fp = (m & ~is_gt).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"name_r >= {min_thresh:.2f} & addr_r >= {min_thresh:.2f}: TP={tp:,}, FP={fp:,}, Purity={p:.2%}")
