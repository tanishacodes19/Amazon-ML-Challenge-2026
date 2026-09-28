import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, numpy as np, polars as pl
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
sys.path.append(os.path.join(BASE, "src"))

VAL_DIR = os.path.join(BASE, "validation_benchmark")
gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))
s1 = pl.read_parquet(os.path.join(VAL_DIR, "val_s1_v13_clean.parquet"))
s23 = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet"))

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

base_g = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
gate = (base_g | rescue1) & (name_r >= 0.40)

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))

s1_dict = {r["source1_entity_id"]: r for r in s1.to_dicts()}
s23_dict = {r["matched_entity_id"]: r for r in s23.to_dicts()}

missed_in_pool = []
for idx in range(len(s1_ids)):
    sid = s1_ids[idx]; mid = m_ids[idx]
    if (sid, mid) in gt_set and not gate[idx]:
        missed_in_pool.append(idx)

print(f"Total True Matches in Pool Missed by Gate: {len(missed_in_pool):,}")

# Group by probability bins
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.90, 0.95, 0.972]
counts, _ = np.histogram(p_ens[missed_in_pool], bins=bins)
print("\nProbability Distribution of Missed True Matches in Candidate Pool:")
for b_low, b_high, cnt in zip(bins[:-1], bins[1:], counts):
    print(f"  Score [{b_low:.2f} - {b_high:.2f}): {cnt:,} pairs ({cnt/len(missed_in_pool)*100:.1f}%)")

print("\nSample 20 Missed True Matches with Score >= 0.50:")
samples = [idx for idx in missed_in_pool if p_ens[idx] >= 0.50][:20]
for idx in samples:
    sid = s1_ids[idx]; mid = m_ids[idx]
    r1 = s1_dict.get(sid, {})
    r2 = s23_dict.get(mid, {})
    print(f"\n[Score: {p_ens[idx]:.4f} | NameRat: {name_r[idx]:.2f} | AddrRat: {addr_r[idx]:.2f} | House: {house_m[idx]}]")
    print(f"  S1 : {r1.get('business_name')}  ||  {r1.get('business_address')}")
    print(f"  S23: {r2.get('business_name')}  ||  {r2.get('business_address')}")
