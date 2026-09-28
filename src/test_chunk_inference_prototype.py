import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import time
import duckdb
import pandas as pd
import polars as pl
import numpy as np
import xgboost as xgb
import lightgbm as lgb

from normalizer import normalize_business_name, extract_structured_fields
from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED

BASE_DIR = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
MODEL_DIR = os.path.join(BASE_DIR, "model")
XGB_PATH = os.path.join(MODEL_DIR, "xgboost_v11_52features.json")
LGB_PATH = os.path.join(MODEL_DIR, "lightgbm_v11_52features.txt")

S1_NORM = os.path.join(BASE_DIR, "normalized_data", "test_source1_normalized.tsv").replace("\\", "/")
S2_NORM = os.path.join(BASE_DIR, "normalized_data", "test_source2_normalized.tsv").replace("\\", "/")
S3_NORM = os.path.join(BASE_DIR, "normalized_data", "test_source3_normalized.tsv").replace("\\", "/")
CANDS = os.path.join(BASE_DIR, "test_scored_candidates.tsv").replace("\\", "/")

print("=" * 70)
print("TESTING CHUNKED INFERENCE PROTOTYPE (50,000 PAIRS)")
print("=" * 70)

t0 = time.time()
con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

print("1. Loading in-memory normalized lookup tables in DuckDB...")
con.execute(f"CREATE TEMP TABLE s1_tbl AS SELECT entity_id, business_name, business_address, country FROM read_csv('{S1_NORM}', delim='\\t', header=true)")
con.execute(f"""
CREATE TEMP TABLE s23_tbl AS 
SELECT entity_id, business_name, business_address, country FROM read_csv('{S2_NORM}', delim='\\t', header=true)
UNION ALL
SELECT entity_id, business_name, business_address, country FROM read_csv('{S3_NORM}', delim='\\t', header=true)
""")
print(f"Loaded in {time.time()-t0:.1f}s.")

print("2. Loading Trained Ensemble Models...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_PATH)

lgb_model = lgb.Booster(model_file=LGB_PATH)

print("3. Querying 50,000 candidates with probability >= 0.05...")
t_q = time.time()
con.execute(f"""
CREATE TEMP TABLE chunk_cands AS
SELECT source1_entity_id, matched_entity_id
FROM read_csv('{CANDS}', delim='\\t', header=true)
WHERE probability >= 0.05
LIMIT 50000;
""")

joined = con.execute("""
SELECT p.source1_entity_id, p.matched_entity_id,
       COALESCE(s1.business_name,'') as s1_name_raw,
       COALESCE(s1.business_address,'') as s1_addr_raw,
       COALESCE(s1.country,'') as s1_country,
       COALESCE(s23.business_name,'') as m_name_raw,
       COALESCE(s23.business_address,'') as m_addr_raw,
       COALESCE(s23.country,'') as m_country
FROM chunk_cands p
JOIN s1_tbl s1 ON p.source1_entity_id = s1.entity_id
JOIN s23_tbl s23 ON p.matched_entity_id = s23.entity_id
""").df()
print(f"Joined 50k rows in {time.time()-t_q:.2f}s.")

print("4. Normalizing unique entities...")
t_norm = time.time()
un_names = set(joined["s1_name_raw"].unique()).union(set(joined["m_name_raw"].unique()))
un_addrs = set(joined["s1_addr_raw"].unique()).union(set(joined["m_addr_raw"].unique()))

name_map = {x: normalize_business_name(x) for x in un_names if x}
addr_map = {x: extract_structured_fields(x) for x in un_addrs if x}
print(f"Normalized {len(un_names):,} names & {len(un_addrs):,} addrs in {time.time()-t_norm:.2f}s.")

print("5. Extracting 52 features...")
t_feat = time.time()
rows = []
for _, r in joined.iterrows():
    s1_a = addr_map.get(r["s1_addr_raw"], {"address_normalized":"", "house_number":"", "postal_code":""})
    m_a = addr_map.get(r["m_addr_raw"], {"address_normalized":"", "house_number":"", "postal_code":""})
    rows.append({
        "source1_entity_id": r["source1_entity_id"],
        "matched_entity_id": r["matched_entity_id"],
        "s1_name_norm": name_map.get(r["s1_name_raw"], ""),
        "matched_name_norm": name_map.get(r["m_name_raw"], ""),
        "s1_addr_norm": s1_a["address_normalized"],
        "matched_addr_norm": m_a["address_normalized"],
        "s1_house": s1_a["house_number"],
        "matched_house": m_a["house_number"],
        "s1_postal": s1_a["postal_code"],
        "matched_postal": m_a["postal_code"],
        "s1_country": r["s1_country"] or "",
        "matched_country": r["m_country"] or "",
    })

df_pl = pl.DataFrame(rows)
feat_pl = extract_features_df(df_pl)
X_chunk = feat_pl.select(V5_FEATURES_EXPANDED).to_numpy()
print(f"Features extracted in {time.time()-t_feat:.2f}s.")

print("6. Predicting with Ensemble...")
t_pred = time.time()
p_xgb = xgb_model.predict_proba(X_chunk)[:, 1]
p_lgb = lgb_model.predict(X_chunk)
p_ens = 0.60 * p_xgb + 0.40 * p_lgb
print(f"Predicted in {time.time()-t_pred:.2f}s.")

name_ratios = feat_pl["name_ratio"].to_numpy()
addr_ratios = feat_pl["address_ratio"].to_numpy()
house_matches = feat_pl["house_match"].to_numpy()

gate = (p_ens >= 0.900) & ((name_ratios >= 0.25) | (addr_ratios >= 0.50) | (house_matches == 1))
survived = gate.sum()
print(f"Matched pairs (prob >= 0.900): {survived:,} / {len(joined):,} ({survived/len(joined)*100:.2f}%)")
print(f"Total Prototype Time: {time.time()-t0:.1f}s")
