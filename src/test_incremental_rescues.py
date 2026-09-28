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
name_tok_j = feat_dict["name_token_jaccard"]
c_match = feat_dict["country_match"]

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# Base gate
base_g = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
v15_base = ((base_g | rescue1) & (name_r >= 0.40)) | ((p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)) | ((p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40))

# Test individual pure additions
candidate_rescues = [
    ("Name85+Postal", (name_r >= 0.85) & (postal_e == 1) & (c_match == 1)),
    ("NameExact+Postal", (name_ex == 1) & (postal_e == 1)),
    ("Name80+House+Postal", (name_r >= 0.80) & (house_m == 1) & (postal_e == 1)),
    ("Acronym+Addr70", (is_acronym == 1) & (addr_r >= 0.70)),
    ("NameContains+Addr90+Prob85", (name_c == 1) & (addr_r >= 0.90) & (p_ens >= 0.85)),
    ("Name95+Addr70+Prob80", (name_r >= 0.95) & (addr_r >= 0.70) & (p_ens >= 0.80)),
    ("Name90+Addr85+Prob80", (name_r >= 0.90) & (addr_r >= 0.85) & (p_ens >= 0.80)),
    ("AddrExact+House+Name75", (addr_ex == 1) & (house_m == 1) & (name_r >= 0.75)),
    ("Prob92+Name70+Addr70", (p_ens >= 0.92) & (name_r >= 0.70) & (addr_r >= 0.70)),
    ("Prob88+Name80+Addr80", (p_ens >= 0.88) & (name_r >= 0.80) & (addr_r >= 0.80)),
]

print("=== INCREMENTAL RESCUE PURITY EVALUATION ===")
curr_gate = v15_base.copy()

def eval_injective(m):
    cands = np.where(m)[0]
    s_ord = cands[np.argsort(-p_ens[cands])]
    s23_seen = set()
    m_eval = np.zeros(len(m), dtype=bool)
    for i in s_ord:
        mid = m_ids[i]
        if mid not in s23_seen:
            s23_seen.add(mid)
            m_eval[i] = True
    tp = (m_eval & is_gt).sum()
    fp = (m_eval & ~is_gt).sum()
    p = tp / (tp + fp)
    r = tp / 86275
    f05 = 1.25 * p * r / (0.25 * p + r)
    return tp, fp, p, r, f05, m_eval

tp0, fp0, p0, r0, f0, _ = eval_injective(curr_gate)
print(f"Base Gate: TP={tp0:,}, FP={fp0:,}, P={p0:.4%}, R={r0:.4%}, F0.5={f0:.6f}\n")

for label, r_rule in candidate_rescues:
    # See how many NEW pairs this adds
    new_pairs = r_rule & ~curr_gate
    new_tp = (new_pairs & is_gt).sum()
    new_fp = (new_pairs & ~is_gt).sum()
    new_purity = new_tp / (new_tp + new_fp) if (new_tp + new_fp) > 0 else 0
    
    test_gate = curr_gate | r_rule
    tp_t, fp_t, pt, rt, f_t, _ = eval_injective(test_gate)
    delta_tp = tp_t - tp0
    delta_fp = fp_t - fp0
    delta_f = f_t - f0
    print(f"{label:25s}: New TP={new_tp:4,d}, New FP={new_fp:3,d} (Purity={new_purity:.2%}) | Net dTP={delta_tp:+4d}, dFP={delta_fp:+3d} -> F0.5={f_t:.6f} ({delta_f:+.6f})")
