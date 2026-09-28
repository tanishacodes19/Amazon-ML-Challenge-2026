import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import polars as pl
import numpy as np

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

gt_val = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
s1_val = pl.read_parquet(os.path.join(VAL_DIR, "val_s1.parquet"))
s23_val = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v14_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v14_features52_X.npy"))

import xgboost as xgb
import lightgbm as lgb

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))

p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
name_ratios = pairs_df["name_ratio"].to_numpy()
addr_ratios = pairs_df["address_ratio"].to_numpy()
house_matches = pairs_df["house_match"].to_numpy()

from feature_engine_v2 import V5_FEATURES_EXPANDED
postal_exact = X_val[:, V5_FEATURES_EXPANDED.index("postal_exact")]
name_exact = X_val[:, V5_FEATURES_EXPANDED.index("name_exact")]
name_first_tok = X_val[:, V5_FEATURES_EXPANDED.index("name_first_token_match")]

# Candidates selected by model
mask = (p_ens >= 0.950) & ((name_ratios >= 0.20) | (addr_ratios >= 0.40) | (house_matches == 1))
gt_set = set(zip(gt_val["source1_entity_id"], gt_val["matched_entity_id"]))

fps = []
for i in np.where(mask)[0]:
    pair = (s1_ids[i], m_ids[i])
    if pair not in gt_set:
        fps.append({
            "idx": i,
            "s1_id": s1_ids[i],
            "m_id": m_ids[i],
            "score": p_ens[i],
            "name_ratio": name_ratios[i],
            "addr_ratio": addr_ratios[i],
            "house_match": house_matches[i],
            "postal_exact": postal_exact[i],
            "name_exact": name_exact[i],
            "name_first_tok": name_first_tok[i]
        })

fps_pl = pl.DataFrame(fps)
print(f"Total False Positives Analyzed: {len(fps_pl):,}")

# Breakdown by features
c1 = len(fps_pl.filter(pl.col("name_ratio") < 0.50))
c2 = len(fps_pl.filter((pl.col("name_ratio") >= 0.50) & (pl.col("name_first_tok") == 0)))
c3 = len(fps_pl.filter(pl.col("addr_ratio") < 0.25))
c4 = len(fps_pl.filter((pl.col("name_ratio") >= 0.85) & (pl.col("addr_ratio") < 0.35)))
c5 = len(fps_pl.filter((pl.col("house_match") == 0) & (pl.col("postal_exact") == 0) & (pl.col("addr_ratio") < 0.50)))

print(f"1. Low Name Ratio (< 0.50, different company in same building): {c1:,} ({c1/len(fps_pl)*100:.1f}%)")
print(f"2. First Token Mismatch (different brand prefix):                 {c2:,} ({c2/len(fps_pl)*100:.1f}%)")
print(f"3. Near-Zero Address Ratio (< 0.25, missing/unrelated location):  {c3:,} ({c3/len(fps_pl)*100:.1f}%)")
print(f"4. High Name (>=0.85) but Low Addr (<0.35, different city branch):{c4:,} ({c4/len(fps_pl)*100:.1f}%)")
print(f"5. No House, No Postal, and Low Addr (< 0.50):                    {c5:,} ({c5/len(fps_pl)*100:.1f}%)")
