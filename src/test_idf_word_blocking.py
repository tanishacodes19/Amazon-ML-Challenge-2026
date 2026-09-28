import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, duckdb, polars as pl
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
v16_cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_cands.parquet"))
s1 = pl.read_parquet(os.path.join(VAL_DIR, "val_s1_v13_clean.parquet"))
s23 = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet"))

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")
con.register("gt", gt.to_pandas())
con.register("v16_cands", v16_cands.to_pandas())
con.register("s1", s1.to_pandas())
con.register("s23", s23.to_pandas())

con.execute("""
CREATE TEMP TABLE missed_v16 AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN v16_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")
total_missed = con.execute("SELECT COUNT(*) FROM missed_v16").fetchone()[0]
print(f"Total Missed before IDF Blocking: {total_missed:,}")

LEGAL_STOP = "('private', 'limited', 'incorporated', 'corporation', 'company', 'services', 'enterprises', 'associates', 'consultants', 'industries', 'international', 'solutions', 'technologies', 'holdings', 'management', 'development', 'commercial', 'general', 'medical', 'dental', 'health', 'center', 'national', 'global')"

# Channel 17: Rare Name Word for NULL/Empty Addresses
print("\nTesting Channel 17 (Rare Name Word for Null/Empty S23 Addresses)...")
con.execute(f"""
CREATE TEMP TABLE ch17_rare_name AS
WITH s1_words AS (
    SELECT source1_entity_id, country, UNNEST(LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {LEGAL_STOP})) as w
    FROM s1
),
s23_words AS (
    SELECT matched_entity_id, country, UNNEST(LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {LEGAL_STOP})) as w
    FROM s23
    WHERE clean_addr IS NULL OR clean_addr = '' OR clean_addr = 'none' OR clean_addr = 'nan'
),
word_freq AS (
    SELECT country, w, COUNT(*) as freq
    FROM (SELECT country, w FROM s1_words UNION ALL SELECT country, w FROM s23_words)
    GROUP BY country, w
    HAVING freq >= 2 AND freq <= 25
)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_words s1
JOIN word_freq wf ON s1.country = wf.country AND s1.w = wf.w
JOIN s23_words s23 ON s1.country = s23.country AND s1.w = s23.w
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")

rec_17 = con.execute("""
SELECT COUNT(DISTINCT (m.source1_entity_id || '_' || m.matched_entity_id))
FROM missed_v16 m JOIN ch17_rare_name c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id;
""").fetchone()[0]
pairs_17 = con.execute("SELECT COUNT(*) FROM ch17_rare_name").fetchone()[0]
print(f"  Channel 17 recovered +{rec_17:,} missed GT (added {pairs_17:,} pairs)")

# Channel 18: Rare Address Landmark Word (length >= 7, frequency <= 25)
ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"
print("\nTesting Channel 18 (Rare Address Word len>=7, freq<=25)...")
con.execute(f"""
CREATE TEMP TABLE ch18_rare_addr AS
WITH s1_awords AS (
    SELECT source1_entity_id, country, UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 7 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as w
    FROM s1
),
s23_awords AS (
    SELECT matched_entity_id, country, UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 7 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as w
    FROM s23
),
aword_freq AS (
    SELECT country, w, COUNT(*) as freq
    FROM (SELECT country, w FROM s1_awords UNION ALL SELECT country, w FROM s23_awords)
    GROUP BY country, w
    HAVING freq >= 2 AND freq <= 20
)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_awords s1
JOIN aword_freq wf ON s1.country = wf.country AND s1.w = wf.w
JOIN s23_awords s23 ON s1.country = s23.country AND s1.w = s23.w
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")

rec_18 = con.execute("""
SELECT COUNT(DISTINCT (m.source1_entity_id || '_' || m.matched_entity_id))
FROM missed_v16 m JOIN ch18_rare_addr c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id;
""").fetchone()[0]
pairs_18 = con.execute("SELECT COUNT(*) FROM ch18_rare_addr").fetchone()[0]
print(f"  Channel 18 recovered +{rec_18:,} missed GT (added {pairs_18:,} pairs)")

# Total combined recovery
con.execute("""
CREATE TEMP TABLE total_new AS
SELECT source1_entity_id, matched_entity_id FROM ch17_rare_name
UNION
SELECT source1_entity_id, matched_entity_id FROM ch18_rare_addr;
""")

total_rec = con.execute("""
SELECT COUNT(DISTINCT (m.source1_entity_id || '_' || m.matched_entity_id))
FROM missed_v16 m JOIN total_new c ON m.source1_entity_id = c.source1_entity_id AND m.matched_entity_id = c.matched_entity_id;
""").fetchone()[0]
total_pairs = con.execute("SELECT COUNT(*) FROM total_new").fetchone()[0]

print("=" * 80)
print(f"TOTAL RECOVERED BY CH17 + CH18: +{total_rec:,} / {total_missed:,} ({total_rec/total_missed*100:.1f}%)")
new_cov = 83484 + total_rec
print(f"NEW CANDIDATE POOL COVERAGE: {new_cov:,} / {len(gt):,} ({new_cov/len(gt)*100:.2f}%)")
print(f"Total Candidate Pairs Added: {total_pairs:,}")
print("=" * 80)
