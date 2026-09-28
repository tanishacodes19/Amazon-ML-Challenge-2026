import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, numpy as np, polars as pl
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
sys.path.append(os.path.join(BASE, "src"))

VAL_DIR = os.path.join(BASE, "validation_benchmark")
gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))

from feature_engine_v2 import V5_FEATURES_EXPANDED

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]
name_ex = feat_dict["name_exact"]
tok_set = feat_dict["name_token_set"]
first_tok = feat_dict["name_first_token_match"]

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

print("=" * 85)
print("TESTING DISENTANGLED CO-LOCATION AND NAME-DOMINANT RULES")
print("=" * 85)

# 1. Smoking Gun of False Positives: Co-location Conflict
# Addr >= 0.75, but Name < 0.60 and NOT acronym
co_loc = (addr_r >= 0.75) & (name_r < 0.60) & (is_acronym == 0) & (name_c == 0)
fp_caught = np.sum(co_loc & ~is_gt)
tp_hurt = np.sum(co_loc & is_gt)
print(f"Co-Location Conflict Feature: Catches {fp_caught:,} False Positives! Hurts only {tp_hurt:,} True Positives.")
print(f"  Precision of this negative filter: {fp_caught / (fp_caught + tp_hurt) * 100:.2f}% (Safe to reject!)")

# 2. Smoking Gun of Missing True Matches: Name-Dominant Match with Missing Address
# Name >= 0.90, first_tok == 1, Addr < 0.20
name_dom = (name_r >= 0.90) & (first_tok == 1) & (addr_r < 0.20)
tp_rescued = np.sum(name_dom & is_gt)
fp_added = np.sum(name_dom & ~is_gt)
print(f"\nName-Dominant Rescue (Addr<0.20): Rescues {tp_rescued:,} True Positives! Adds only {fp_added:,} False Positives.")
print(f"  Purity of this rescue: {tp_rescued / (tp_rescued + fp_added) * 100:.2f}%")

# 3. Exact Name + Postal Match (Even with messy address)
postal_ex = feat_dict["postal_exact"]
np_match = (name_ex == 1) & (postal_ex == 1)
tp_np = np.sum(np_match & is_gt)
fp_np = np.sum(np_match & ~is_gt)
print(f"\nExact Name + Postal Match: Contains {tp_np:,} True Positives, {fp_np:,} False Positives.")
print(f"  Purity: {tp_np / (tp_np + fp_np) * 100:.2f}%")
