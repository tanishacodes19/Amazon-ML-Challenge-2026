import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
sys.path.insert(0, 'src')
import os
import duckdb
import numpy as np
import polars as pl
from eval_framework import load_benchmark, compute_macro_f05
from normalizer import normalize_business_name, normalize_address

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

print("Generating V14 ensemble probabilities...")
p_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_lgb = lgb_model.predict(X_val)
p_ens = 0.50 * p_xgb + 0.50 * p_lgb

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
name_ratios = pairs_df["name_ratio"].to_numpy()
addr_ratios = pairs_df["address_ratio"].to_numpy()
house_matches = pairs_df["house_match"].to_numpy()

# Base V14 predictions at tau=0.950
tau = 0.950
gate = (p_ens >= tau) & ((name_ratios >= 0.20) | (addr_ratios >= 0.40) | (house_matches == 1))
base_pred = pl.DataFrame({
    "source1_entity_id": s1_ids[gate],
    "matched_entity_id": m_ids[gate],
    "score": p_ens[gate]
})

res_base = compute_macro_f05(base_pred.select(["source1_entity_id", "matched_entity_id"]), gt_val, s1_val)
print(f"Base V14 @ tau={tau:.3f}: Macro F0.5 = {res_base['macro_f05']:.4f} | Prec: {res_base['precision']*100:.2f}% | Rec: {res_base['recall']*100:.2f}% | TP: {res_base['tp']:,} | FP: {res_base['fp']:,} | Singl FP: {res_base['singleton_fp']}")

# ============================================================
# TECHNIQUE 1: Transitive Graph Closure (S1 -> S2 <-> S3)
# ============================================================
print("\n" + "=" * 70)
print("TESTING TRIPARTITE GRAPH CLOSURE (S1 -> S2 <-> S3)")
print("=" * 70)

# Build S2 <-> S3 equivalence map using clean entities
s23_df = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet")).to_pandas()
con = duckdb.connect()
con.register("s23", s23_df)
con.register("pred", base_pred.to_pandas())
con.register("gt", gt_val.to_pandas())

# Find identical/high-confidence links between S2 and S3 (same country, exact clean name & exact clean address)
con.execute("""
CREATE OR REPLACE TEMP TABLE s2_entities AS
SELECT matched_entity_id, country, clean_name, clean_addr
FROM s23 WHERE matched_entity_id LIKE 'S2-%';

CREATE OR REPLACE TEMP TABLE s3_entities AS
SELECT matched_entity_id, country, clean_name, clean_addr
FROM s23 WHERE matched_entity_id LIKE 'S3-%';

CREATE OR REPLACE TEMP TABLE s2_to_s3 AS
SELECT s2.matched_entity_id AS s2_id, s3.matched_entity_id AS s3_id
FROM s2_entities s2
JOIN s3_entities s3 
  ON s2.country = s3.country 
 AND s2.clean_name = s3.clean_name 
 AND s2.clean_addr = s3.clean_addr
WHERE LENGTH(s2.clean_name) >= 4 AND LENGTH(s2.clean_addr) >= 6;
""")

eq_count = con.execute("SELECT COUNT(*) FROM s2_to_s3").fetchone()[0]
print(f"Identified {eq_count:,} identical entity links between S2 and S3")

# Transitive expansion: if S1 -> S2, add S1 -> S3
con.execute("""
CREATE OR REPLACE TEMP TABLE transitive_links AS
SELECT p.source1_entity_id, eq.s3_id AS matched_entity_id
FROM pred p
JOIN s2_to_s3 eq ON p.matched_entity_id = eq.s2_id
UNION
SELECT p.source1_entity_id, eq.s2_id AS matched_entity_id
FROM pred p
JOIN s2_to_s3 eq ON p.matched_entity_id = eq.s3_id;
""")

trans_count = con.execute("SELECT COUNT(*) FROM transitive_links").fetchone()[0]
new_links = con.execute("""
SELECT COUNT(*) FROM transitive_links t
LEFT JOIN pred p ON t.source1_entity_id = p.source1_entity_id AND t.matched_entity_id = p.matched_entity_id
WHERE p.matched_entity_id IS NULL;
""").fetchone()[0]
new_tp = con.execute("""
SELECT COUNT(*) FROM transitive_links t
LEFT JOIN pred p ON t.source1_entity_id = p.source1_entity_id AND t.matched_entity_id = p.matched_entity_id
JOIN gt g ON t.source1_entity_id = g.source1_entity_id AND t.matched_entity_id = g.matched_entity_id
WHERE p.matched_entity_id IS NULL;
""").fetchone()[0]

print(f"Transitive links found: {trans_count:,} (NEW pairs not in pred: {new_links:,}, of which TRUE matches: {new_tp:,})")

# Evaluate with transitive closure
con.execute("""
CREATE OR REPLACE TEMP TABLE pred_closure AS
SELECT source1_entity_id, matched_entity_id FROM pred
UNION
SELECT source1_entity_id, matched_entity_id FROM transitive_links;
""")
pred_closure_pl = pl.from_arrow(con.execute("SELECT * FROM pred_closure").to_arrow_table())
res_closure = compute_macro_f05(pred_closure_pl, gt_val, s1_val)
print(f"With Transitive Closure: Macro F0.5 = {res_closure['macro_f05']:.4f} | Prec: {res_closure['precision']*100:.2f}% | Rec: {res_closure['recall']*100:.2f}% | TP: {res_closure['tp']:,} | FP: {res_closure['fp']:,}")

# ============================================================
# TECHNIQUE 2: Singleton Shield Calibration
# ============================================================
print("\n" + "=" * 70)
print("TESTING SINGLETON SHIELD OPTIMIZATION")
print("=" * 70)

# For entities with exactly 1 prediction, if the prediction is low-margin or weak, drop it to protect singleton score
con.execute("""
CREATE OR REPLACE TEMP TABLE pred_with_metrics AS
SELECT p.source1_entity_id, p.matched_entity_id, p.score,
       COUNT(*) OVER (PARTITION BY p.source1_entity_id) as n_preds
FROM pred p;
""")

for min_margin in [0.950, 0.955, 0.960, 0.965, 0.970]:
    for drop_weak in [True, False]:
        if drop_weak:
            filt = f"WHERE NOT (n_preds = 1 AND score < {min_margin})"
        else:
            filt = ""
        pred_shield = con.execute(f"SELECT source1_entity_id, matched_entity_id FROM pred_with_metrics {filt}").to_arrow_table()
        res_shield = compute_macro_f05(pl.from_arrow(pred_shield), gt_val, s1_val)
        tag = f"drop singles < {min_margin:.3f}" if drop_weak else "no shield"
        print(f"Shield [{tag:<25}]: Macro F0.5 = {res_shield['macro_f05']:.4f} | TP: {res_shield['tp']:,} | FP: {res_shield['fp']:,} | Singl FP: {res_shield['singleton_fp']}")
