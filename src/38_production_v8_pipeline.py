import sys
import os
import pandas as pd
import polars as pl
import xgboost as xgb
import duckdb
import time

sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

# Ensure src in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from normalizer import normalize_business_name, extract_structured_fields
from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CANDIDATES_FILE = os.path.join(BASE_DIR, 'test_scored_candidates.tsv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'xgboost_v8_52features.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

NORM_DIR = os.path.join(BASE_DIR, 'normalized_data')
S1_NORM = os.path.join(NORM_DIR, 'test_source1_normalized.tsv')
S2_NORM = os.path.join(NORM_DIR, 'test_source2_normalized.tsv')
S3_NORM = os.path.join(NORM_DIR, 'test_source3_normalized.tsv')
S1_RAW = r"D:\student_resource\student_resource\dataset\test\test_source1.tsv"

THRESHOLD = 0.900
CHUNK_SIZE = 50000

print("Setting up DuckDB...")
con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

con.execute(f"CREATE VIEW s1_data AS SELECT entity_id, business_name, business_address, country FROM read_csv('{S1_NORM}', delim='\\t', header=true)")
con.execute(f"""
CREATE VIEW s23_data AS 
SELECT entity_id, business_name, business_address, country FROM read_csv('{S2_NORM}', delim='\\t', header=true)
UNION ALL
SELECT entity_id, business_name, business_address, country FROM read_csv('{S3_NORM}', delim='\\t', header=true)
""")

print("Loading XGBoost model...")
bst = xgb.Booster()
bst.load_model(MODEL_PATH)

print("Processing candidates in chunks...")

name_cache = {}
def get_norm_name(name):
    if name not in name_cache:
        name_cache[name] = normalize_business_name(name)
    return name_cache[name]

addr_cache = {}
def get_norm_addr(addr):
    if addr not in addr_cache:
        addr_cache[addr] = extract_structured_fields(addr)
    return addr_cache[addr]

chunk_iter = pd.read_csv(CANDIDATES_FILE, sep='\t', chunksize=CHUNK_SIZE)
total_processed = 0

all_results = []

for i, chunk in enumerate(chunk_iter):
    # Register chunk_df
    con.register('chunk_df', chunk)
    joined_df = con.execute("""
    SELECT p.source1_entity_id, p.matched_entity_id,
           COALESCE(s1.business_name,'') as s1_name_raw,
           COALESCE(s1.business_address,'') as s1_addr_raw,
           COALESCE(s1.country,'') as s1_country,
           COALESCE(s23.business_name,'') as m_name_raw,
           COALESCE(s23.business_address,'') as m_addr_raw,
           COALESCE(s23.country,'') as m_country
    FROM chunk_df p
    JOIN s1_data s1 ON p.source1_entity_id = s1.entity_id
    JOIN s23_data s23 ON p.matched_entity_id = s23.entity_id
    """).df()
    
    if joined_df.empty:
        continue
        
    # Process norms
    joined_df['s1_name_norm'] = joined_df['s1_name_raw'].apply(get_norm_name)
    joined_df['matched_name_norm'] = joined_df['m_name_raw'].apply(get_norm_name)
    
    s1_addr = joined_df['s1_addr_raw'].apply(get_norm_addr)
    joined_df['s1_addr_norm'] = [a['address_normalized'] for a in s1_addr]
    joined_df['s1_house'] = [a['house_number'] for a in s1_addr]
    joined_df['s1_postal'] = [a['postal_code'] for a in s1_addr]
    
    m_addr = joined_df['m_addr_raw'].apply(get_norm_addr)
    joined_df['matched_addr_norm'] = [a['address_normalized'] for a in m_addr]
    joined_df['matched_house'] = [a['house_number'] for a in m_addr]
    joined_df['matched_postal'] = [a['postal_code'] for a in m_addr]
    
    joined_df['matched_country'] = joined_df['m_country']
    
    # Extract features
    pl_df = pl.DataFrame(joined_df)
    features_df = extract_features_df(pl_df)
    
    X = features_df[V5_FEATURES_EXPANDED].to_pandas()
    dmatrix = xgb.DMatrix(X)
    probs = bst.predict(dmatrix)
    
    joined_df['probability'] = probs
    
    # Filter
    filtered = joined_df[joined_df['probability'] >= THRESHOLD]
    if not filtered.empty:
        all_results.append(filtered[['source1_entity_id', 'matched_entity_id', 'probability']])
    
    total_processed += len(chunk)
    if total_processed % 500000 == 0:
        print(f"Processed {total_processed} pairs...")

print("Concatenating results...")
if all_results:
    final_matches = pd.concat(all_results, ignore_index=True)
else:
    final_matches = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_id', 'probability'])

final_matches_path = os.path.join(OUTPUT_DIR, 'tmp_matches.csv')
final_matches.to_csv(final_matches_path, index=False)

print("Generating candidate_pairs.tsv using DuckDB...")
con.execute(f"""
COPY (
    SELECT s1.entity_id AS source1_entity_id,
           COALESCE(STRING_AGG(c.matched_entity_id, ','), '') AS candidate_entity_ids
    FROM read_csv('{S1_RAW}', delim='\\t', header=true) s1
    LEFT JOIN read_csv('{CANDIDATES_FILE}', delim='\\t', header=true) c
      ON s1.entity_id = c.source1_entity_id
    GROUP BY s1.entity_id
) TO '{CANDIDATE_PAIRS}' (HEADER, DELIMITER '\\t', QUOTE '');
""")

print("Generating matching_results.tsv using DuckDB...")
con.execute(f"""
COPY (
    SELECT s1.entity_id AS source1_entity_id,
           COALESCE(STRING_AGG(m.matched_entity_id, ','), '') AS matched_entity_ids
    FROM read_csv('{S1_RAW}', delim='\\t', header=true) s1
    LEFT JOIN read_csv('{final_matches_path}', header=true) m
      ON s1.entity_id = m.source1_entity_id
    GROUP BY s1.entity_id
) TO '{MATCHING_RESULTS}' (HEADER, DELIMITER '\\t', QUOTE '');
""")

print("Done! Validating...")
import subprocess
val_cmd = [
    sys.executable,
    r"D:\student_resource\student_resource\utils\validate_submission.py",
    "--matching", MATCHING_RESULTS,
    "--candidate", CANDIDATE_PAIRS,
    "--test-dir", r"D:\student_resource\student_resource\dataset\test"
]
subprocess.run(val_cmd)
