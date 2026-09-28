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
xgb_v13 = xgb.XGBClassifier()
xgb_v13.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_v13 = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))
p_v13 = 0.50 * xgb_v13.predict_proba(X_val)[:, 1] + 0.50 * lgb_v13.predict(X_val)

xgb_v17 = xgb.XGBClassifier()
xgb_v17.load_model(os.path.join(BASE, "model", "xgboost_v17_52features.json"))
lgb_v17 = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v17_52features.txt"))
p_v17 = 0.50 * xgb_v17.predict_proba(X_val)[:, 1] + 0.50 * lgb_v17.predict(X_val)

p_blend = 0.80 * p_v13 + 0.20 * p_v17

feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
postal_e = feat_dict["postal_exact"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]
name_ex = feat_dict["name_exact"]
c_match = feat_dict["country_match"]

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)
TOTAL_GT = 86275

# Base Gate
base_g = (p_blend >= 0.972) & (name_r >= 0.40) & ((addr_r >= 0.30) | (house_m == 1) | (postal_e == 1))
rescue_std = (p_blend >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95)) & (name_r >= 0.40)
core_gate = base_g | rescue_std

print("=== TESTING LOWER-PROB RESCUE TIERS WITH HIGH TEXT AGREEMENT ===")
tiers = [
    ("Name80+Addr85+Prob50", (name_r >= 0.80) & (addr_r >= 0.85) & (p_blend >= 0.50)),
    ("Name80+Addr90+Prob40", (name_r >= 0.80) & (addr_r >= 0.90) & (p_blend >= 0.40)),
    ("Name85+Addr80+Prob50", (name_r >= 0.85) & (addr_r >= 0.80) & (p_blend >= 0.50)),
    ("Name85+Addr85+Prob40", (name_r >= 0.85) & (addr_r >= 0.85) & (p_blend >= 0.40)),
    ("Name90+Addr75+Prob50", (name_r >= 0.90) & (addr_r >= 0.75) & (p_blend >= 0.50)),
    ("Name90+Addr80+Prob40", (name_r >= 0.90) & (addr_r >= 0.80) & (p_blend >= 0.40)),
    ("Name92+Addr70+Prob40", (name_r >= 0.92) & (addr_r >= 0.70) & (p_blend >= 0.40)),
    ("Name95+Addr60+Prob40", (name_r >= 0.95) & (addr_r >= 0.60) & (p_blend >= 0.40)),
    ("Name90+House1+Prob50", (name_r >= 0.90) & (house_m == 1) & (p_blend >= 0.50)),
    ("Name95+House1+Prob40", (name_r >= 0.95) & (house_m == 1) & (p_blend >= 0.40)),
]

def eval_with_rescue(extra_mask):
    combined = core_gate | extra_mask
    cands = np.where(combined)[0]
    s_ord = cands[np.argsort(-p_blend[cands])]
    s23_seen = set()
    m_eval = np.zeros(len(combined), dtype=bool)
    for i in s_ord:
        mid = m_ids[i]
        if mid not in s23_seen:
            s23_seen.add(mid)
            m_eval[i] = True
    tp = (m_eval & is_gt).sum()
    fp = (m_eval & ~is_gt).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / TOTAL_GT
    f05 = 1.25 * p * r / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
    return tp, fp, p, r, f05

tp0, fp0, p0, r0, f0 = eval_with_rescue(np.zeros(len(core_gate), dtype=bool))
print(f"Base: TP={tp0:,}, FP={fp0:,}, P={p0:.4%}, R={r0:.4%}, F0.5={f0:.6f}\n")

for label, mask in tiers:
    new_p = mask & ~core_gate
    tp_n = (new_p & is_gt).sum()
    fp_n = (new_p & ~is_gt).sum()
    pur = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0
    tp, fp, p, r, f05 = eval_with_rescue(mask)
    dtp = tp - tp0
    dfp = fp - fp0
    df = f05 - f0
    print(f"{label:22s}: New TP={tp_n:4,d}, FP={fp_n:3,d} (Purity={pur:.2%}) | Net dTP={dtp:+4d}, dFP={dfp:+3d} -> F0.5={f05:.6f} ({df:+.6f})")
