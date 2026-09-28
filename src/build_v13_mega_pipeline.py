"""
V13 MEGA PIPELINE: Full rebuild with upgraded normalizer (unidecode + zero-strip)
- Builds V13 candidate set with expanded blocking channels
- Re-extracts all features with upgraded normalizer
- Evaluates on validation benchmark
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
sys.path.insert(0, 'src')
import os
import time
import duckdb
import numpy as np
import polars as pl
from eval_framework import load_benchmark, compute_macro_f05
from normalizer import normalize_business_name, normalize_address, extract_structured_fields
from feature_engine_v2 import extract_features_df, V5_FEATURES_EXPANDED

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
MODEL_DIR = os.path.join(BASE, "model")

print("=" * 70)
print("V13 MEGA PIPELINE: BLOCKING + FEATURES + EVALUATION")
print("  Normalizer: unidecode transliteration + leading zero strip + legal sort")
print("=" * 70)

t0 = time.time()

# ===================================================================
# STEP 0: Delete ALL stale caches
# ===================================================================
print("\n[STEP 0] Deleting stale cached features (normalizer changed)...")
stale_files = [
    os.path.join(VAL_DIR, "val_v12_features52_X.npy"),
    os.path.join(VAL_DIR, "val_v12_features52_df.parquet"),
    os.path.join(VAL_DIR, "val_v13_features52_X.npy"),
    os.path.join(VAL_DIR, "val_v13_features52_df.parquet"),
    os.path.join(MODEL_DIR, "train_v11_X.npy"),
    os.path.join(MODEL_DIR, "train_v11_y.npy"),
]
for f in stale_files:
    if os.path.exists(f):
        os.remove(f)
        print(f"  Deleted: {os.path.basename(f)}")

# ===================================================================
# STEP 1: Load benchmark data
# ===================================================================
print("\n[STEP 1] Loading benchmark data...")
s1, gt, _, s23 = load_benchmark()
total_gt = len(gt)
print(f"  S1 entities: {len(s1):,} | GT pairs: {total_gt:,} | S23 entities: {len(s23):,}")

# ===================================================================
# STEP 2: Apply upgraded normalizer to all entities
# ===================================================================
print("\n[STEP 2] Applying V13 normalizer (unidecode + zero-strip) to all entities...")
t_norm = time.time()
s1_clean_path = os.path.join(VAL_DIR, "val_s1_v13_clean.parquet")
s23_clean_path = os.path.join(VAL_DIR, "val_s23_v13_clean.parquet")

if os.path.exists(s1_clean_path) and os.path.exists(s23_clean_path):
    print("  Loading cached V13 normalized entities...")
    s1_df = pl.read_parquet(s1_clean_path).to_pandas()
    s23_df = pl.read_parquet(s23_clean_path).to_pandas()
else:
    s1_df = s1.to_pandas()
    s23_df = s23.to_pandas()
    s1_df["clean_name"] = s1_df["business_name"].apply(normalize_business_name)
    s23_df["clean_name"] = s23_df["business_name"].apply(normalize_business_name)
    s1_df["clean_addr"] = s1_df["business_address"].apply(lambda x: normalize_address(x) if x else "")
    s23_df["clean_addr"] = s23_df["business_address"].apply(lambda x: normalize_address(x) if x else "")
    pl.from_pandas(s1_df).write_parquet(s1_clean_path)
    pl.from_pandas(s23_df).write_parquet(s23_clean_path)

print(f"  Normalized entities ready in {time.time()-t_norm:.1f}s")

# ===================================================================
# STEP 3: Build V13 candidate set with 8+ blocking channels
# ===================================================================
print("\n[STEP 3] Building V13 candidate set with expanded blocking channels...")

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

con.register("s1", s1_df)
con.register("gt", gt.to_pandas())
con.register("s23", s23_df)

STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of', 'international', 'technology', 'management', 'incorporated', 'corporation', 'association', 'department', 'manufacturing')"

# CHANNEL 1: Brand First Token (4+ chars, non-stopword) + Country
print("  [Ch1] Brand First Token + Country...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE ch1_s1 AS
WITH base AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS}) AS words
    FROM s1
)
SELECT source1_entity_id, country, words[1] AS tok
FROM base WHERE LEN(words) >= 1;

CREATE OR REPLACE TEMP TABLE ch1_s23 AS
WITH base AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS}) AS words
    FROM s23
)
SELECT matched_entity_id, country, words[1] AS tok
FROM base WHERE LEN(words) >= 1;

CREATE OR REPLACE TEMP TABLE ch1 AS
SELECT source1_entity_id, matched_entity_id FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM ch1_s1 s1 JOIN ch1_s23 s23 ON s1.country = s23.country AND s1.tok = s23.tok
    WHERE LENGTH(s1.tok) >= 4
) WHERE cnt <= 20;
""")
ch1_n = con.execute("SELECT COUNT(*) FROM ch1").fetchone()[0]
ch1_rec = con.execute("SELECT COUNT(*) FROM gt g JOIN ch1 c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"    Ch1: {ch1_n:,} candidates | Recovers: {ch1_rec:,} ({ch1_rec/total_gt*100:.2f}%)")

# CHANNEL 2: Brand Second Token (if exists) + Country
print("  [Ch2] Brand Second Token + Country...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE ch2_s1 AS
WITH base AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS}) AS words
    FROM s1
)
SELECT source1_entity_id, country, words[2] AS tok
FROM base WHERE LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE ch2_s23 AS
WITH base AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_WORDS}) AS words
    FROM s23
)
SELECT matched_entity_id, country, words[2] AS tok
FROM base WHERE LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE ch2 AS
SELECT source1_entity_id, matched_entity_id FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM ch2_s1 s1 JOIN ch2_s23 s23 ON s1.country = s23.country AND s1.tok = s23.tok
    WHERE LENGTH(s1.tok) >= 4
) WHERE cnt <= 20;
""")
ch2_n = con.execute("SELECT COUNT(*) FROM ch2").fetchone()[0]
ch2_rec = con.execute("SELECT COUNT(*) FROM gt g JOIN ch2 c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"    Ch2: {ch2_n:,} candidates | Recovers: {ch2_rec:,} ({ch2_rec/total_gt*100:.2f}%)")

# CHANNEL 3: Sorted 2-Token Key (first two non-stop tokens sorted alphabetically) + Country
print("  [Ch3] Sorted 2-Token Key + Country...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE ch3_s1 AS
WITH base AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) AS words
    FROM s1
)
SELECT source1_entity_id, country,
       CASE WHEN words[1] < words[2] THEN words[1] || '_' || words[2] ELSE words[2] || '_' || words[1] END AS key2
FROM base WHERE LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE ch3_s23 AS
WITH base AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) AS words
    FROM s23
)
SELECT matched_entity_id, country,
       CASE WHEN words[1] < words[2] THEN words[1] || '_' || words[2] ELSE words[2] || '_' || words[1] END AS key2
FROM base WHERE LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE ch3 AS
SELECT source1_entity_id, matched_entity_id FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM ch3_s1 s1 JOIN ch3_s23 s23 ON s1.country = s23.country AND s1.key2 = s23.key2
) WHERE cnt <= 15;
""")
ch3_n = con.execute("SELECT COUNT(*) FROM ch3").fetchone()[0]
ch3_rec = con.execute("SELECT COUNT(*) FROM gt g JOIN ch3 c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"    Ch3: {ch3_n:,} candidates | Recovers: {ch3_rec:,} ({ch3_rec/total_gt*100:.2f}%)")

# CHANNEL 4: House Number + Street Token (first 5+ char address word) + Country
print("  [Ch4] House Number (zero-stripped) + Street Token + Country...")
con.execute("""
CREATE OR REPLACE TEMP TABLE ch4_s1 AS
WITH parsed AS (
    SELECT source1_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> x SIMILAR TO '[0-9]+.*') AS nums,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT SIMILAR TO '[0-9].*') AS words
    FROM s1
)
SELECT source1_entity_id, country,
       LTRIM(nums[1], '0') AS house, words[1] AS street_tok
FROM parsed WHERE LEN(nums) >= 1 AND LEN(words) >= 1 AND LENGTH(LTRIM(nums[1], '0')) >= 1;

CREATE OR REPLACE TEMP TABLE ch4_s23 AS
WITH parsed AS (
    SELECT matched_entity_id, country,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> x SIMILAR TO '[0-9]+.*') AS nums,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT SIMILAR TO '[0-9].*') AS words
    FROM s23
)
SELECT matched_entity_id, country,
       LTRIM(nums[1], '0') AS house, words[1] AS street_tok
FROM parsed WHERE LEN(nums) >= 1 AND LEN(words) >= 1 AND LENGTH(LTRIM(nums[1], '0')) >= 1;

CREATE OR REPLACE TEMP TABLE ch4 AS
SELECT source1_entity_id, matched_entity_id FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM ch4_s1 s1 JOIN ch4_s23 s23 
      ON s1.country = s23.country AND s1.house = s23.house AND s1.street_tok = s23.street_tok
) WHERE cnt <= 15;
""")
ch4_n = con.execute("SELECT COUNT(*) FROM ch4").fetchone()[0]
ch4_rec = con.execute("SELECT COUNT(*) FROM gt g JOIN ch4 c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"    Ch4: {ch4_n:,} candidates | Recovers: {ch4_rec:,} ({ch4_rec/total_gt*100:.2f}%)")

# CHANNEL 5: Postal Code + Brand First Char + Country
print("  [Ch5] Postal Code + Brand First Char + Country...")
con.execute("""
CREATE OR REPLACE TEMP TABLE ch5_s1 AS
SELECT source1_entity_id, country,
       REGEXP_EXTRACT(clean_addr, '(\d{5,6})', 1) AS postal,
       SUBSTR(clean_name, 1, 3) AS brand3
FROM s1
WHERE REGEXP_EXTRACT(clean_addr, '(\d{5,6})', 1) IS NOT NULL AND LENGTH(clean_name) >= 3;

CREATE OR REPLACE TEMP TABLE ch5_s23 AS
SELECT matched_entity_id, country,
       REGEXP_EXTRACT(clean_addr, '(\d{5,6})', 1) AS postal,
       SUBSTR(clean_name, 1, 3) AS brand3
FROM s23
WHERE REGEXP_EXTRACT(clean_addr, '(\d{5,6})', 1) IS NOT NULL AND LENGTH(clean_name) >= 3;

CREATE OR REPLACE TEMP TABLE ch5 AS
SELECT source1_entity_id, matched_entity_id FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM ch5_s1 s1 JOIN ch5_s23 s23 
      ON s1.country = s23.country AND s1.postal = s23.postal AND s1.brand3 = s23.brand3
) WHERE cnt <= 20;
""")
ch5_n = con.execute("SELECT COUNT(*) FROM ch5").fetchone()[0]
ch5_rec = con.execute("SELECT COUNT(*) FROM gt g JOIN ch5 c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"    Ch5: {ch5_n:,} candidates | Recovers: {ch5_rec:,} ({ch5_rec/total_gt*100:.2f}%)")

# CHANNEL 6: Street Bigram + Brand First Char + Country (from V12)
print("  [Ch6] Street Bigram + Brand First Char + Country...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE ch6_s1 AS
WITH parsed AS (
    SELECT source1_entity_id, country,
           SUBSTR(clean_name, 1, 1) as brand_c1,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 3 AND x NOT SIMILAR TO '[0-9].*') AS words
    FROM s1
)
SELECT source1_entity_id, country, brand_c1,
       words[1] || '_' || words[2] as street2
FROM parsed
WHERE brand_c1 != '' AND LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE ch6_s23 AS
WITH parsed AS (
    SELECT matched_entity_id, country,
           SUBSTR(clean_name, 1, 1) as brand_c1,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 3 AND x NOT SIMILAR TO '[0-9].*') AS words
    FROM s23
)
SELECT matched_entity_id, country, brand_c1,
       words[1] || '_' || words[2] as street2
FROM parsed
WHERE brand_c1 != '' AND LEN(words) >= 2;

CREATE OR REPLACE TEMP TABLE ch6 AS
SELECT source1_entity_id, matched_entity_id FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM ch6_s1 s1 JOIN ch6_s23 s23 
      ON s1.country = s23.country AND s1.street2 = s23.street2 AND s1.brand_c1 = s23.brand_c1
    WHERE LENGTH(s1.street2) >= 8
) WHERE cnt <= 12;
""")
ch6_n = con.execute("SELECT COUNT(*) FROM ch6").fetchone()[0]
ch6_rec = con.execute("SELECT COUNT(*) FROM gt g JOIN ch6 c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"    Ch6: {ch6_n:,} candidates | Recovers: {ch6_rec:,} ({ch6_rec/total_gt*100:.2f}%)")

# CHANNEL 7: Unit/Plot + Address Word (from V12)
print("  [Ch7] Unit/Plot + Address Word + Country...")
con.execute("""
CREATE OR REPLACE TEMP TABLE ch7_s1 AS
WITH parsed AS (
    SELECT source1_entity_id, country,
           REGEXP_EXTRACT(clean_addr, '(?:unit|plot|flat|shop|room|floor)\s+([a-z0-9]+)', 1) as unit,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT SIMILAR TO '[0-9].*') AS words
    FROM s1
)
SELECT source1_entity_id, country, unit, words[1] AS first_word
FROM parsed
WHERE unit IS NOT NULL AND unit != '' AND LENGTH(unit) >= 1 AND LEN(words) >= 1;

CREATE OR REPLACE TEMP TABLE ch7_s23 AS
WITH parsed AS (
    SELECT matched_entity_id, country,
           REGEXP_EXTRACT(clean_addr, '(?:unit|plot|flat|shop|room|floor)\s+([a-z0-9]+)', 1) as unit,
           LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT SIMILAR TO '[0-9].*') AS words
    FROM s23
)
SELECT matched_entity_id, country, unit, words[1] AS first_word
FROM parsed
WHERE unit IS NOT NULL AND unit != '' AND LENGTH(unit) >= 1 AND LEN(words) >= 1;

CREATE OR REPLACE TEMP TABLE ch7 AS
SELECT source1_entity_id, matched_entity_id FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM ch7_s1 s1 JOIN ch7_s23 s23 
      ON s1.country = s23.country AND s1.unit = s23.unit AND s1.first_word = s23.first_word
) WHERE cnt <= 12;
""")
ch7_n = con.execute("SELECT COUNT(*) FROM ch7").fetchone()[0]
ch7_rec = con.execute("SELECT COUNT(*) FROM gt g JOIN ch7 c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"    Ch7: {ch7_n:,} candidates | Recovers: {ch7_rec:,} ({ch7_rec/total_gt*100:.2f}%)")

# CHANNEL 8: Rare name token (5+ chars, non-stop) inverted index
print("  [Ch8] Rare Name Token Inverted Index + Country...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE raw_tokens AS
SELECT UNNEST(STR_SPLIT(clean_name, ' ')) AS tok
FROM (SELECT clean_name FROM s1 UNION ALL SELECT clean_name FROM s23);

CREATE OR REPLACE TEMP TABLE all_name_tokens AS
SELECT tok, COUNT(*) AS freq
FROM raw_tokens
WHERE LENGTH(tok) >= 5 AND tok NOT IN {STOP_WORDS}
GROUP BY tok
HAVING COUNT(*) BETWEEN 2 AND 150;

CREATE OR REPLACE TEMP TABLE ch8_s1 AS
SELECT t.source1_entity_id, t.country, t.tok
FROM (
    SELECT source1_entity_id, country, UNNEST(STR_SPLIT(clean_name, ' ')) AS tok
    FROM s1
) t
JOIN all_name_tokens f ON t.tok = f.tok;

CREATE OR REPLACE TEMP TABLE ch8_s23 AS
SELECT t.matched_entity_id, t.country, t.tok
FROM (
    SELECT matched_entity_id, country, UNNEST(STR_SPLIT(clean_name, ' ')) AS tok
    FROM s23
) t
JOIN all_name_tokens f ON t.tok = f.tok;

CREATE OR REPLACE TEMP TABLE ch8 AS
SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
    SELECT s1.source1_entity_id, s23.matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.source1_entity_id) as cnt
    FROM ch8_s1 s1 JOIN ch8_s23 s23 ON s1.country = s23.country AND s1.tok = s23.tok
) WHERE cnt <= 15;
""")
ch8_n = con.execute("SELECT COUNT(*) FROM ch8").fetchone()[0]
ch8_rec = con.execute("SELECT COUNT(*) FROM gt g JOIN ch8 c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"    Ch8: {ch8_n:,} candidates | Recovers: {ch8_rec:,} ({ch8_rec/total_gt*100:.2f}%)")

# Unify ALL channels into V13
v12_path = os.path.join(VAL_DIR, "val_v12_cands.parquet").replace("\\", "/")
print("\n  [Unifying] Building V13 candidate set from V12 baseline + 8 channels...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE v13_candidates AS
SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
    SELECT source1_entity_id, matched_entity_id FROM read_parquet('{v12_path}')
    UNION ALL SELECT * FROM ch1
    UNION ALL SELECT * FROM ch2
    UNION ALL SELECT * FROM ch3
    UNION ALL SELECT * FROM ch4
    UNION ALL SELECT * FROM ch5
    UNION ALL SELECT * FROM ch6
    UNION ALL SELECT * FROM ch7
    UNION ALL SELECT * FROM ch8
);
""")

v13_total = con.execute("SELECT COUNT(*) FROM v13_candidates").fetchone()[0]
v13_rec = con.execute("""
SELECT COUNT(*) FROM gt g
JOIN v13_candidates c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
""").fetchone()[0]
v13_recall = (v13_rec / total_gt) * 100
print(f"\n{'=' * 70}")
print(f"V13 CANDIDATE SET METRICS:")
print(f"  Total Candidate Pairs:    {v13_total:,} ({v13_total/25000:.2f} cands/S1)")
print(f"  Recovered Ground Truth:   {v13_rec:,} / {total_gt:,} ({v13_recall:.2f}%)")
print(f"  MISSED:                   {total_gt - v13_rec:,}")
print(f"{'=' * 70}")

out_path = os.path.join(VAL_DIR, "val_v13_cands.parquet")
con.execute("SELECT source1_entity_id, matched_entity_id FROM v13_candidates").df().to_parquet(out_path, index=False)
print(f"Saved V13 Candidates to: {out_path}")

# ===================================================================
# STEP 4: Extract features for V13 candidates with upgraded normalizer
# ===================================================================
print(f"\n[STEP 4] Extracting 52 features for {v13_total:,} V13 candidates...")
t_feat = time.time()

cands = pl.read_parquet(out_path)
s1_val = pl.from_pandas(s1_df[["source1_entity_id", "business_name", "business_address", "country"]])
s23_val = pl.from_pandas(s23_df[["matched_entity_id", "business_name", "business_address", "country"]])

v13_pairs = cands.join(
    s1_val.select([
        pl.col("source1_entity_id"),
        pl.col("business_name").alias("s1_name_raw"),
        pl.col("business_address").alias("s1_addr_raw"),
        pl.col("country").alias("s1_country"),
    ]), on="source1_entity_id", how="left"
).join(
    s23_val.select([
        pl.col("matched_entity_id"),
        pl.col("business_name").alias("m_name_raw"),
        pl.col("business_address").alias("m_addr_raw"),
        pl.col("country").alias("matched_country"),
    ]), on="matched_entity_id", how="left"
)

# Pre-compute normalizations
val_un_names = set(v13_pairs["s1_name_raw"].unique().drop_nulls()).union(set(v13_pairs["m_name_raw"].unique().drop_nulls()))
val_un_addrs = set(v13_pairs["s1_addr_raw"].unique().drop_nulls()).union(set(v13_pairs["m_addr_raw"].unique().drop_nulls()))

print(f"  Normalizing {len(val_un_names):,} unique names & {len(val_un_addrs):,} unique addresses...")
val_n_map = {x: normalize_business_name(x) for x in val_un_names if x}
val_a_map = {x: extract_structured_fields(x) for x in val_un_addrs if x}

print(f"  Building feature vectors...")
val_rows = []
for r in v13_pairs.iter_rows(named=True):
    s1_a = val_a_map.get(r["s1_addr_raw"], {"address_normalized":"", "house_number":"", "postal_code":""})
    m_a = val_a_map.get(r["m_addr_raw"], {"address_normalized":"", "house_number":"", "postal_code":""})
    val_rows.append({
        "source1_entity_id": r["source1_entity_id"],
        "matched_entity_id": r["matched_entity_id"],
        "s1_name_norm": val_n_map.get(r["s1_name_raw"], ""),
        "matched_name_norm": val_n_map.get(r["m_name_raw"], ""),
        "s1_addr_norm": s1_a["address_normalized"],
        "matched_addr_norm": m_a["address_normalized"],
        "s1_house": s1_a["house_number"],
        "matched_house": m_a["house_number"],
        "s1_postal": s1_a["postal_code"],
        "matched_postal": m_a["postal_code"],
        "s1_country": r["s1_country"] or "",
        "matched_country": r["matched_country"] or "",
    })

pairs_pl = pl.DataFrame(val_rows)
feat_pl = extract_features_df(pairs_pl)
print(f"  Features extracted in {time.time()-t_feat:.1f}s")

X_val = feat_pl.select(V5_FEATURES_EXPANDED).to_numpy()
np.save(os.path.join(VAL_DIR, "val_v13_features52_X.npy"), X_val)

pairs_df = pl.DataFrame({
    "source1_entity_id": cands["source1_entity_id"],
    "matched_entity_id": cands["matched_entity_id"],
    "name_ratio": feat_pl["name_ratio"],
    "address_ratio": feat_pl["address_ratio"],
    "house_match": feat_pl["house_match"]
})
pairs_df.write_parquet(os.path.join(VAL_DIR, "val_v13_features52_df.parquet"))

# ===================================================================
# STEP 5: Score with existing ensemble (old models, new features)
# ===================================================================
print(f"\n[STEP 5] Scoring V13 candidates with existing ensemble models...")
import xgboost as xgb
import lightgbm as lgb

XGB_PATH = os.path.join(MODEL_DIR, "xgboost_v11_52features.json")
LGB_PATH = os.path.join(MODEL_DIR, "lightgbm_v11_52features.txt")

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_PATH)
lgb_model = lgb.Booster(model_file=LGB_PATH)

p_xgb = xgb_model.predict_proba(X_val)[:, 1]
p_lgb = lgb_model.predict(X_val)
p_ens = 0.60 * p_xgb + 0.40 * p_lgb

s1_ids = pairs_df["source1_entity_id"].to_numpy()
m_ids = pairs_df["matched_entity_id"].to_numpy()
name_ratios = pairs_df["name_ratio"].to_numpy()
addr_ratios = pairs_df["address_ratio"].to_numpy()
house_matches = pairs_df["house_match"].to_numpy()

# Load ground truth for evaluation
gt_val = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
s1_val_ids = pl.read_parquet(os.path.join(VAL_DIR, "val_s1.parquet"))

models_eval = [
    ("XGBoost V11", p_xgb),
    ("LightGBM V11", p_lgb),
    ("Ensemble 60/40", p_ens)
]

print(f"\n{'=' * 80}")
for m_name, probs in models_eval:
    print(f"\n--- {m_name} on V13 Candidates ({v13_recall:.2f}% blocking recall) ---")
    print(f"{'Threshold':>10} | {'Macro F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'TP':>7} | {'FP':>6} | {'Singl FP':>8}")
    print("-" * 80)
    best_f = 0.0
    best_t = 0.0
    for tau in [0.50, 0.60, 0.70, 0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]:
        gate = (probs >= tau) & ((name_ratios >= 0.20) | (addr_ratios >= 0.40) | (house_matches == 1))
        pred_df = pl.DataFrame({
            "source1_entity_id": s1_ids[gate],
            "matched_entity_id": m_ids[gate]
        })
        res = compute_macro_f05(pred_df, gt_val, s1_val_ids)
        if res["macro_f05"] > best_f:
            best_f = res["macro_f05"]
            best_t = tau
        print(f"{tau:>10.3f} | {res['macro_f05']:>11.4f} | {res['precision']*100:>9.2f}% | {res['recall']*100:>7.2f}% | {res['tp']:>7,} | {res['fp']:>6,} | {res['singleton_fp']:>8,}")
    print(f"*** PEAK for {m_name}: {best_f:.4f} @ tau={best_t:.3f} ***")

print(f"\n{'=' * 80}")
print(f"V13 MEGA PIPELINE COMPLETED in {time.time()-t0:.1f}s")
print(f"{'=' * 80}")
