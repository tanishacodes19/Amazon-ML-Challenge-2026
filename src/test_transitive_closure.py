import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import time
import numpy as np
import polars as pl
from eval_framework import load_benchmark, compute_macro_f05
import duckdb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

s1_val, gt_val, _, s23_val = load_benchmark()
cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v12_cands.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v12_features52_df.parquet"))
X_val = np.load(os.path.join(VAL_DIR, "val_v12_features52_X.npy"))

import xgboost as xgb
import lightgbm as lgb

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, "model", "xgboost_v11_52features.json"))
lgb_model = lgb.Booster(model_file=os.path.join(BASE, "model", "lightgbm_v11_52features.txt"))

p_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_lgb = lgb_model.predict(X_val)
p_ens = 0.60 * p_xgb + 0.40 * p_lgb

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
name_ratios = pairs_df["name_ratio"].to_numpy()
addr_ratios = pairs_df["address_ratio"].to_numpy()
house_matches = pairs_df["house_match"].to_numpy()

# Baseline matches at tau=0.920
mask = (p_ens >= 0.920) & ((name_ratios >= 0.25) | (addr_ratios >= 0.50) | (house_matches == 1))
base_matches = pl.DataFrame({
    "source1_entity_id": s1_ids[mask],
    "matched_entity_id": m_ids[mask],
    "prob": p_ens[mask]
})

res_base = compute_macro_f05(base_matches.select(["source1_entity_id", "matched_entity_id"]), gt_val, s1_val)
print(f"Base Macro F0.5: {res_base['macro_f05']:.4f} | Prec: {res_base['precision']*100:.2f}% | Rec: {res_base['recall']*100:.2f}% | TP: {res_base['tp']:,} | FP: {res_base['fp']:,}")

print("\n" + "=" * 70)
print("TESTING CONSERVATIVE S2-S3 CO-OCCURRENCE / TRANSITIVE CLOSURE")
print("=" * 70)

con = duckdb.connect()
con.register("base_m", base_matches.to_pandas())
con.register("s23", s23_val.to_pandas())

# High-confidence anchor matches (prob >= 0.95)
# Find pairs in s23 that share exact normalized business name and exact normalized address in the same country
con.execute("""
CREATE TEMP TABLE s23_identical AS
SELECT 
    a.matched_entity_id as entity_a,
    b.matched_entity_id as entity_b
FROM s23 a
JOIN s23 b 
  ON a.country = b.country
 AND a.business_name = b.business_name
 AND a.business_address = b.business_address
WHERE a.matched_entity_id < b.matched_entity_id
  AND LENGTH(a.business_name) >= 6;
""")

print("Identical S2/S3 entity pairs found:", con.execute("SELECT COUNT(*) FROM s23_identical").fetchone()[0])

# Now check transitive matches: if S1 matches A (P >= 0.95) and A is identical to B, add S1 -> B
for p_anchor in [0.95, 0.97, 0.99]:
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE transitive_links AS
    WITH high_conf AS (
        SELECT source1_entity_id, matched_entity_id
        FROM base_m
        WHERE prob >= {p_anchor}
    ),
    trans_a AS (
        SELECT h.source1_entity_id, i.entity_b AS matched_entity_id
        FROM high_conf h
        JOIN s23_identical i ON h.matched_entity_id = i.entity_a
    ),
    trans_b AS (
        SELECT h.source1_entity_id, i.entity_a AS matched_entity_id
        FROM high_conf h
        JOIN s23_identical i ON h.matched_entity_id = i.entity_b
    )
    SELECT * FROM trans_a UNION SELECT * FROM trans_b;
    """)
    
    trans_df = pl.from_arrow(con.execute("""
    SELECT source1_entity_id, matched_entity_id FROM base_m
    UNION
    SELECT source1_entity_id, matched_entity_id FROM transitive_links
    """).arrow())
    
    res_trans = compute_macro_f05(trans_df, gt_val, s1_val)
    print(f"Anchor P >= {p_anchor:.2f} -> Transitive links: {len(trans_df)-len(base_matches):,} | Macro F0.5: {res_trans['macro_f05']:.4f} | Prec: {res_trans['precision']*100:.2f}% | Rec: {res_trans['recall']*100:.2f}% | TP: {res_trans['tp']:,} | FP: {res_trans['fp']:,}")
