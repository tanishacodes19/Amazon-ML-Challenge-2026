import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
sys.path.insert(0, 'src')
import os
import duckdb
import numpy as np
import polars as pl
from eval_framework import load_benchmark

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

s1_val, gt_val, _, s23_val = load_benchmark()
v14_cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v14_cands.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v14_features52_X.npy"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v14_features52_df.parquet"))

import xgboost as xgb
import lightgbm as lgb

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v13_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v13_52features.txt"))

p_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_lgb = lgb_model.predict(X_val)
p_ens = 0.50 * p_xgb + 0.50 * p_lgb

pairs_df = pairs_df.with_columns(pl.Series("score", p_ens))

# Join with ground truth to find candidate pairs that are TRUE matches but scored < 0.950
gt_val = gt_val.with_columns(pl.lit(1).alias("is_gt"))

scored_pairs = pairs_df.join(
    gt_val, on=["source1_entity_id", "matched_entity_id"], how="inner"
)

missed_scored = scored_pairs.filter(pl.col("score") < 0.950)
print(f"Total True Matches in V14 Candidates: {len(scored_pairs):,} / {len(gt_val):,} (95.51%)")
print(f"True Matches Scored < 0.950: {len(missed_scored):,} pairs")

# Score distribution of the filtered true matches
score_ranges = [
    (0.90, 0.95),
    (0.80, 0.90),
    (0.70, 0.80),
    (0.50, 0.70),
    (0.20, 0.50),
    (0.00, 0.20)
]

print("\nScore Distribution of Candidate True Matches Scored < 0.950:")
for low, high in score_ranges:
    cnt = len(missed_scored.filter((pl.col("score") >= low) & (pl.col("score") < high)))
    print(f"  Score [{low:.2f} - {high:.2f}): {cnt:,} pairs ({cnt/len(missed_scored)*100:.1f}%)")

# Inspect sample missed true matches with scores between 0.80 and 0.95
sample = missed_scored.filter((pl.col("score") >= 0.80) & (pl.col("score") < 0.95)).head(10)

s1_lookup = {r[0]: (r[1], r[2], r[3]) for r in s1_val.select(["source1_entity_id", "business_name", "business_address", "country"]).iter_rows()}
s23_lookup = {r[0]: (r[1], r[2], r[3]) for r in s23_val.select(["matched_entity_id", "business_name", "business_address", "country"]).iter_rows()}

print("\nSample True Matches Scored between 0.80 and 0.95:")
for row in sample.iter_rows(named=True):
    s1_id = row["source1_entity_id"]
    m_id = row["matched_entity_id"]
    score = row["score"]
    n_rat = row["name_ratio"]
    a_rat = row["address_ratio"]
    h_mat = row["house_match"]
    s1_info = s1_lookup.get(s1_id, ("", "", ""))
    m_info = s23_lookup.get(m_id, ("", "", ""))
    print(f"\n--- Score: {score:.4f} | NameRat: {n_rat:.2f} | AddrRat: {a_rat:.2f} | House: {h_mat} ---")
    print(f"S1:  {s1_info[0]} | {s1_info[1]}")
    print(f"S23: {m_info[0]} | {m_info[1]}")
