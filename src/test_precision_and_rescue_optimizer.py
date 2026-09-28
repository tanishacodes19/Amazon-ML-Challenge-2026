import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import os, numpy as np, polars as pl
from feature_engine_v2 import V5_FEATURES_EXPANDED

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

print("1. Loading validation benchmark data...")
gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))

print("2. Predicting with XGBoost + LightGBM...")
import xgboost as xgb, lightgbm as lgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))
p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
postal_e = feat_dict["postal_exact"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]
name_ex = feat_dict["name_exact"]
addr_ex = feat_dict["address_exact"]
addr_tok_s = feat_dict["address_token_set"]

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)
TOTAL_GT = 86275

def evaluate_gate(mask, name="Gate", apply_s23_injective=True):
    if apply_s23_injective:
        # Sort candidates passing mask by p_ens descending
        cand_indices = np.where(mask)[0]
        sorted_order = cand_indices[np.argsort(-p_ens[cand_indices])]
        seen_s23 = set()
        final_mask = np.zeros(len(mask), dtype=bool)
        for idx in sorted_order:
            mid = m_ids[idx]
            if mid not in seen_s23:
                seen_s23.add(mid)
                final_mask[idx] = True
        mask_to_eval = final_mask
    else:
        mask_to_eval = mask
        
    tp = (mask_to_eval & is_gt).sum()
    fp = (mask_to_eval & ~is_gt).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / TOTAL_GT
    f05 = 1.25 * p * r / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0
    print(f"[{name}] (Inj={apply_s23_injective}): TP={tp:,} | FP={fp:,} | P={p:.4%} | R={r:.4%} | F0.5={f05:.6f}")
    return f05, tp, fp, p, r, mask_to_eval

print("\n--- BASELINE CHECKS ---")
# Current V15 Calibrated Gate
base_g = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
v15_gate = ((base_g | rescue1) & (name_r >= 0.40)) | ((p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)) | ((p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40))

evaluate_gate(v15_gate, "V15 Original (No Inj)", apply_s23_injective=False)
evaluate_gate(v15_gate, "V15 + S23 Injective", apply_s23_injective=True)

print("\n--- TESTING INDIVIDUAL RESCUE CHANNELS PURITY ---")
rules = {
    "R1_ExactName_ExactPostal": (name_ex == 1) & (postal_e == 1),
    "R2_ExactName_HouseMatch": (name_ex == 1) & (house_m == 1),
    "R3_ExactAddress_NameRatio70": (addr_ex == 1) & (name_r >= 0.70),
    "R4_NameRatio85_ExactPostal": (name_r >= 0.85) & (postal_e == 1),
    "R5_NameRatio75_House_Postal": (name_r >= 0.75) & (house_m == 1) & (postal_e == 1),
    "R6_NameRatio90_House": (name_r >= 0.90) & (house_m == 1),
    "R7_Acronym_AddrRatio60": (is_acronym == 1) & (addr_r >= 0.60),
    "R8_Acronym_Postal": (is_acronym == 1) & (postal_e == 1),
    "R9_Contains_AddrRatio80": (name_c == 1) & (addr_r >= 0.80),
}
for r_name, r_mask in rules.items():
    tp = (r_mask & is_gt).sum()
    fp = (r_mask & ~is_gt).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    print(f"  {r_name:30s}: TP={tp:5,d}, FP={fp:4,d}, Purity={p:.2%}")

print("\n--- SWEEPING COMPREHENSIVE COMBINATIONS ---")
best_f05 = 0.0
best_config = None

for base_tau in [0.93, 0.94, 0.95, 0.96, 0.965, 0.970, 0.972, 0.975]:
    for min_name_base in [0.35, 0.40, 0.45]:
        for rescue_tau in [0.85, 0.88, 0.90, 0.92]:
            b_gate = (p_ens >= base_tau) & (name_r >= min_name_base) & ((addr_r >= 0.35) | (house_m == 1) | (postal_e == 1))
            
            # High-confidence rescue channels
            r_channels = (
                # ML rescue with balanced name/addr
                ((p_ens >= rescue_tau) & (name_r >= 0.55) & (addr_r >= 0.55)) |
                # Exact name + postal or house
                ((name_ex == 1) & ((postal_e == 1) | (house_m == 1))) |
                # Very high name + postal
                ((name_r >= 0.85) & (postal_e == 1)) |
                # Very high name + house
                ((name_r >= 0.88) & (house_m == 1)) |
                # Containment + high address
                ((name_c == 1) & (addr_r >= 0.70) & (p_ens >= 0.80)) |
                # Acronym + address
                ((is_acronym == 1) & (addr_r >= 0.50) & (p_ens >= 0.75))
            )
            
            combined = b_gate | r_channels
            
            # Fast eval with S23 injective
            cand_indices = np.where(combined)[0]
            sorted_order = cand_indices[np.argsort(-p_ens[cand_indices])]
            seen_s23 = set()
            final_mask = np.zeros(len(combined), dtype=bool)
            for idx in sorted_order:
                mid = m_ids[idx]
                if mid not in seen_s23:
                    seen_s23.add(mid)
                    final_mask[idx] = True
            
            tp = (final_mask & is_gt).sum()
            fp = (final_mask & ~is_gt).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / TOTAL_GT
            f05 = 1.25 * p * r / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0
            
            if f05 > best_f05:
                best_f05 = f05
                best_config = (base_tau, min_name_base, rescue_tau, tp, fp, p, r, f05)
                print(f"NEW BEST! base_tau={base_tau}, min_name={min_name_base}, rescue_tau={rescue_tau} -> F0.5={f05:.6f} (P={p:.4%}, R={r:.4%}, TP={tp:,}, FP={fp:,})")

print("\n" + "="*70)
print(f"OVERALL BEST CONFIG:")
print(f"  base_tau: {best_config[0]}")
print(f"  min_name_base: {best_config[1]}")
print(f"  rescue_tau: {best_config[2]}")
print(f"  TP: {best_config[3]:,}")
print(f"  FP: {best_config[4]:,}")
print(f"  Precision: {best_config[5]:.4%}")
print(f"  Recall: {best_config[6]:.4%}")
print(f"  Global F0.5: {best_config[7]:.6f}")
print("="*70)
