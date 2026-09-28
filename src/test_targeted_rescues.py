import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

print("1. Loading validation benchmark data...")
gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')

import xgboost as xgb, lightgbm as lgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('model/xgboost_v13_52features.json')
lgb_model = lgb.Booster(model_file='model/lightgbm_v13_52features.txt')
p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]
addr_c = feat_dict["address_contains"]
addr_e = feat_dict["address_exact"]
addr_tok = feat_dict["address_token_set"]
name_tok = feat_dict["name_token_set"]
addr_jacc = feat_dict["address_token_jaccard"]
addr_c3 = feat_dict["address_char3_jaccard"]
name_pfx = feat_dict["name_prefix5"]
name_fst = feat_dict["name_first_token_match"]

# Baseline V16 champion gate
base_gate = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
co_location = (name_r >= 0.40)
r_contain = (p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)
r_acronym = (p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40)
v16_mask = ((base_gate | rescue1) & co_location) | r_contain | r_acronym

# Apply greedy injective constraint on candidate mask
def apply_injective(mask, scores):
    cands = np.where(mask)[0]
    s_ord = cands[np.argsort(-scores[cands])]
    s23_seen = set()
    accepted = []
    for i in s_ord:
        mid = m_ids[i]
        if mid not in s23_seen:
            s23_seen.add(mid)
            accepted.append(i)
    out = np.zeros(len(mask), dtype=bool)
    out[accepted] = True
    return out

v16_injective = apply_injective(v16_mask, p_ens)
tp_base = (v16_injective & is_gt).sum()
fp_base = (v16_injective & ~is_gt).sum()
p_base = tp_base / (tp_base + fp_base)
r_base = tp_base / len(gt_set)
f05_base = 1.25 * p_base * r_base / (0.25 * p_base + r_base)
print(f"V16 Baseline Injective: TP={tp_base:,} | FP={fp_base:,} | P={p_base*100:.2f}% | R={r_base*100:.2f}% | F0.5={f05_base:.4f}")

# Now test potential high-purity rescue channels from unaccepted pairs
unacc = ~v16_injective

print("\n--- Testing Individual Rescue Channel Purity on Unaccepted Pairs ---")
rescue_candidates = {
    "R1_addr_exact_name80": unacc & (addr_e == 1) & (name_tok >= 0.80),
    "R2_addr_exact_name60": unacc & (addr_e == 1) & (name_tok >= 0.60),
    "R3_addr_exact_name40": unacc & (addr_e == 1) & (name_tok >= 0.40),
    "R4_addr_contain_name80": unacc & (addr_c == 1) & (name_tok >= 0.80),
    "R5_addr_contain_name90": unacc & (addr_c == 1) & (name_tok >= 0.90),
    "R6_addr95_name70": unacc & (addr_r >= 0.95) & (name_tok >= 0.70) & (house_m == 1),
    "R7_addr90_name80": unacc & (addr_r >= 0.90) & (name_tok >= 0.80) & (house_m == 1),
    "R8_addr85_name85": unacc & (addr_r >= 0.85) & (name_tok >= 0.85) & (house_m == 1),
    "R9_p_ens85_name60": unacc & (p_ens >= 0.85) & (name_tok >= 0.60) & (addr_r >= 0.70),
    "R10_p_ens80_name70": unacc & (p_ens >= 0.80) & (name_tok >= 0.70) & (addr_r >= 0.80),
    "R11_addr_jacc50_name70": unacc & (addr_jacc >= 0.50) & (name_tok >= 0.70),
    "R12_addr_c3_50_name70": unacc & (addr_c3 >= 0.50) & (name_tok >= 0.70),
}

for name, rmask in rescue_candidates.items():
    tp = (rmask & is_gt).sum()
    fp = (rmask & ~is_gt).sum()
    purity = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"{name:<25} | TP={tp:>5,} | FP={fp:>5,} | Purity={purity*100:>6.2f}% | Total={rmask.sum():>5,}")
