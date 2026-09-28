import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import os, numpy as np, polars as pl
from rapidfuzz import fuzz
from normalizer import extract_structured_fields, normalize_business_name

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))
s1 = pl.read_parquet(os.path.join(VAL_DIR, "val_s1_v13_clean.parquet"))
s23 = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet"))

print("Loading models...")
import xgboost as xgb, lightgbm as lgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))
p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

from feature_engine_v2 import V5_FEATURES_EXPANDED
feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
postal_e = feat_dict["postal_exact"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]

# Base gate
base_g = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
current_gate = ((base_g | rescue1) & (name_r >= 0.40)) | ((p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)) | ((p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40))

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# Build s1 and s23 house mappings directly using extract_structured_fields
print("Extracting structured fields for s1 and s23...")
s1_addr_dict = dict(zip(s1["source1_entity_id"], s1["business_address"].fill_null("")))
s23_addr_dict = dict(zip(s23["matched_entity_id"], s23["business_address"].fill_null("")))

unique_addrs = set(s1["business_address"].fill_null("").to_list()) | set(s23["business_address"].fill_null("").to_list())
addr_struct_map = {a: extract_structured_fields(a) for a in unique_addrs}

s1_house_map = {sid: addr_struct_map[addr]["house_number"] for sid, addr in s1_addr_dict.items()}
s23_house_map = {mid: addr_struct_map[addr]["house_number"] for mid, addr in s23_addr_dict.items()}

h1_arr = np.array([s1_house_map.get(sid, "") for sid in s1_ids])
h2_arr = np.array([s23_house_map.get(mid, "") for mid in m_ids])
both_have_house = (h1_arr != "") & (h2_arr != "")
house_conflict = both_have_house & (h1_arr != h2_arr)

# Test house conflict precision
print("\n=== HOUSE CONFLICT ANALYSIS ===")
total_hc = house_conflict.sum()
hc_in_gt = (house_conflict & is_gt).sum()
hc_in_current_gate = (house_conflict & current_gate).sum()
hc_in_current_gate_gt = (house_conflict & current_gate & is_gt).sum()
hc_in_current_gate_fp = (house_conflict & current_gate & ~is_gt).sum()
print(f"Total candidate pairs with house conflict: {total_hc:,}")
print(f"House conflict pairs that are true GT: {hc_in_gt:,}")
print(f"In current gate: {hc_in_current_gate:,} (TP: {hc_in_current_gate_gt:,}, FP: {hc_in_current_gate_fp:,})")

# Look at 1-to-many cardinality in Ground Truth
print("\n=== GROUND TRUTH CARDINALITY ===")
gt_s1_counts = gt["source1_entity_id"].value_counts()
print(f"GT S1 entities: {len(gt_s1_counts):,}")
print(f"GT S1 with exactly 1 match: {(gt_s1_counts['count'] == 1).sum():,} ({(gt_s1_counts['count'] == 1).mean():.2%})")
print(f"GT S1 with >1 matches: {(gt_s1_counts['count'] > 1).sum():,} ({(gt_s1_counts['count'] > 1).mean():.2%})")
print(f"GT S1 max matches: {gt_s1_counts['count'].max()}")

gt_m_counts = gt["matched_entity_id"].value_counts()
print(f"GT S23 entities: {len(gt_m_counts):,}")
print(f"GT S23 with exactly 1 match: {(gt_m_counts['count'] == 1).sum():,} ({(gt_m_counts['count'] == 1).mean():.2%})")
print(f"GT S23 with >1 matches: {(gt_m_counts['count'] > 1).sum():,} ({(gt_m_counts['count'] > 1).mean():.2%})")
print(f"GT S23 max matches: {gt_m_counts['count'].max()}")

# Inspect Current Gate Performance
tp_orig = (current_gate & is_gt).sum()
fp_orig = (current_gate & ~is_gt).sum()
p_orig = tp_orig / (tp_orig + fp_orig)
r_orig = tp_orig / 86275
f05_orig = 1.25 * p_orig * r_orig / (0.25 * p_orig + r_orig)
print(f"\nCurrent Gate (V15 Calibrated):")
print(f"  TP: {tp_orig:,} | FP: {fp_orig:,} | P: {p_orig:.4%} | R: {r_orig:.4%} | Global F0.5: {f05_orig:.6f}")

# What if we eliminate house conflicts WHERE name_ratio < 0.90?
gate_no_hc = current_gate & ~(house_conflict & (name_r < 0.90))
tp_hc = (gate_no_hc & is_gt).sum()
fp_hc = (gate_no_hc & ~is_gt).sum()
p_hc = tp_hc / (tp_hc + fp_hc)
r_hc = tp_hc / 86275
f05_hc = 1.25 * p_hc * r_hc / (0.25 * p_hc + r_hc)
print(f"\nAfter filtering house conflicts (name_ratio < 0.90):")
print(f"  TP: {tp_hc:,} ({tp_hc - tp_orig:+d}) | FP: {fp_hc:,} ({fp_hc - fp_orig:+d}) | P: {p_hc:.4%} | R: {r_hc:.4%} | Global F0.5: {f05_hc:.6f}")

# What if we test Greedy 1-to-1 or Top-k Assignment?
# Sort by p_ens descending
order = np.argsort(-p_ens)
for max_matches_per_s1 in [1, 2, 3]:
    seen_s1 = {}
    gate_assigned = np.zeros(len(pairs_df), dtype=bool)
    for idx in order:
        if not gate_no_hc[idx]:
            continue
        sid = s1_ids[idx]
        c = seen_s1.get(sid, 0)
        if c < max_matches_per_s1:
            gate_assigned[idx] = True
            seen_s1[sid] = c + 1
            
    tp_a = (gate_assigned & is_gt).sum()
    fp_a = (gate_assigned & ~is_gt).sum()
    p_a = tp_a / (tp_a + fp_a)
    r_a = tp_a / 86275
    f05_a = 1.25 * p_a * r_a / (0.25 * p_a + r_a)
    print(f"\nMax {max_matches_per_s1} match(es) per S1 (with HC filter):")
    print(f"  TP: {tp_a:,} ({tp_a - tp_orig:+d}) | FP: {fp_a:,} ({fp_a - fp_orig:+d}) | P: {p_a:.4%} | R: {r_a:.4%} | Global F0.5: {f05_a:.6f}")
