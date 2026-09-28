import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, numpy as np, polars as pl
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
sys.path.append(os.path.join(BASE, "src"))

VAL_DIR = os.path.join(BASE, "validation_benchmark")
gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v16_features52_X.npy"))
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

import re
indic_re = re.compile(r'[\u0900-\u0DFF]')
s23_dict = {r["matched_entity_id"]: r for r in s23.to_dicts()}

indic_tp_predicted = 0
indic_fp_predicted = 0
for idx in np.where(gate)[0]:
    sid = s1_ids[idx]; mid = m_ids[idx]
    r2 = s23_dict.get(mid, {})
    if indic_re.search(str(r2.get("business_name", ""))):
        if (sid, mid) in gt_set:
            indic_tp_predicted += 1
        else:
            indic_fp_predicted += 1

print(f"Indic pairs predicted by current gate: TP = {indic_tp_predicted:,} / 4,991 ({indic_tp_predicted/4991*100:.1f}%), FP = {indic_fp_predicted:,}")
