import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.append('src')

import duckdb
import polars as pl
import xgboost as xgb
import lightgbm as lgb
from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED

con = duckdb.connect()

print("Checking features with real pipeline extraction...")
df = con.execute("""
WITH s1 AS (
    SELECT entity_id as source1_entity_id, name_normalized as s1_name_norm, address_normalized as s1_addr_norm, country as s1_country,
           LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS s1_house,
           COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS s1_postal
    FROM read_csv('normalized_data/test_source1_normalized.tsv', delim='\t', header=true)
    WHERE entity_id='S1-897975128'
),
s2 AS (
    SELECT entity_id as matched_entity_id, name_normalized as matched_name_norm, address_normalized as matched_addr_norm, country as matched_country,
           LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS matched_house,
           COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS matched_postal
    FROM read_csv('normalized_data/test_source2_normalized.tsv', delim='\t', header=true)
    WHERE entity_id='S2-269643891'
)
SELECT s1.*, s2.* FROM s1, s2;
""").pl()

feat = extract_features_df(df)
X = feat.select(V5_FEATURES_EXPANDED).to_numpy()

xgb_m = xgb.XGBClassifier()
xgb_m.load_model('model/xgboost_v13_52features.json')
lgb_m = lgb.Booster(model_file='model/lightgbm_v13_52features.txt')

p_xgb = xgb_m.predict_proba(X)[0, 1]
p_lgb = lgb_m.predict(X)[0]
p_b = 0.5 * p_xgb + 0.5 * p_lgb

print(f"p_xgb: {p_xgb:.4f}")
print(f"p_lgb: {p_lgb:.4f}")
print(f"p_blend: {p_b:.4f}")
for col in V5_FEATURES_EXPANDED:
    print(f"  {col}: {feat[col][0]}")
