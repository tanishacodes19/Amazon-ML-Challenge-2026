import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import os
import duckdb
import polars as pl

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
v14_cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v14_cands.parquet"))
s1 = pl.read_parquet(os.path.join(VAL_DIR, "val_s1_v13_clean.parquet")).to_pandas()
s23 = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet")).to_pandas()

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")
con.register("s1", s1)
con.register("gt", gt.to_pandas())
con.register("s23", s23)
con.register("v14_cands", v14_cands.to_pandas())

total_gt = len(gt)
print(f"Total Reference Ground Truth Pairs: {total_gt:,}")

ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"

# Channel 15: Door Number + Locality Token (length >= 6)
print("\nGenerating Channel 15 (Door Number + Distinctive Locality)...")
con.execute(f"""
CREATE TEMP TABLE ch15_door_locality AS
WITH s1_dl AS (
    SELECT source1_entity_id, country,
           house_number || '_' || word as k
    FROM (
        SELECT source1_entity_id, country,
               LTRIM(COALESCE(REGEXP_EXTRACT(clean_addr, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') as house_number,
               UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 6 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as word
        FROM s1
    )
    WHERE LENGTH(house_number) >= 1
),
s23_dl AS (
    SELECT matched_entity_id, country,
           house_number || '_' || word as k
    FROM (
        SELECT matched_entity_id, country,
               LTRIM(COALESCE(REGEXP_EXTRACT(clean_addr, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') as house_number,
               UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 6 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as word
        FROM s23
    )
    WHERE LENGTH(house_number) >= 1
)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_dl s1
JOIN s23_dl s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
cnt_15 = con.execute("SELECT COUNT(*) FROM ch15_door_locality;").fetchone()[0]
print(f"  Channel 15 generated {cnt_15:,} candidate pairs.")

# Channel 16: Sorted 2 Longest Distinctive Address Words
print("\nGenerating Channel 16 (Sorted 2 Longest Distinctive Address Words)...")
con.execute(f"""
CREATE TEMP TABLE ch16_sorted_addr2 AS
WITH s1_top2 AS (
    SELECT source1_entity_id, country,
           CASE WHEN sorted_words[1] < sorted_words[2] 
                THEN sorted_words[1] || '_' || sorted_words[2] 
                ELSE sorted_words[2] || '_' || sorted_words[1] END as k
    FROM (
        SELECT source1_entity_id, country,
               LIST_SLICE(LIST_REVERSE_SORT(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')), 1, 2) as sorted_words
        FROM s1
    )
    WHERE LEN(sorted_words) >= 2
),
s23_top2 AS (
    SELECT matched_entity_id, country,
           CASE WHEN sorted_words[1] < sorted_words[2] 
                THEN sorted_words[1] || '_' || sorted_words[2] 
                ELSE sorted_words[2] || '_' || sorted_words[1] END as k
    FROM (
        SELECT matched_entity_id, country,
               LIST_SLICE(LIST_REVERSE_SORT(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')), 1, 2) as sorted_words
        FROM s23
    )
    WHERE LEN(sorted_words) >= 2
)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_top2 s1
JOIN s23_top2 s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
cnt_16 = con.execute("SELECT COUNT(*) FROM ch16_sorted_addr2;").fetchone()[0]
print(f"  Channel 16 generated {cnt_16:,} candidate pairs.")

# Union with V14 candidates
print("\nMerging into V16 Unified Candidate Pool...")
con.execute("""
CREATE TEMP TABLE v16_cands_tbl AS
SELECT source1_entity_id, matched_entity_id FROM v14_cands
UNION
SELECT source1_entity_id, matched_entity_id FROM ch15_door_locality
UNION
SELECT source1_entity_id, matched_entity_id FROM ch16_sorted_addr2;
""")

total_v16_cands = con.execute("SELECT COUNT(*) FROM v16_cands_tbl;").fetchone()[0]
cov_gt = con.execute("""
SELECT COUNT(*) 
FROM gt g
JOIN v16_cands_tbl c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id;
""").fetchone()[0]

print("=" * 80)
print("V16 CANDIDATE POOL BENCHMARK RESULTS:")
print(f"  Total Candidate Pairs:      {total_v16_cands:,} (Avg {total_v16_cands/len(s1):.1f} pairs/entity)")
print(f"  V14 GT Coverage:            82,403 / {total_gt:,} (95.51%)")
print(f"  V16 GT Coverage:            {cov_gt:,} / {total_gt:,} ({cov_gt/total_gt*100:.2f}%)")
print(f"  NET TRUE MATCHES RECOVERED: +{cov_gt - 82403:,} true matches!")
print("=" * 80)

# Save to parquet for downstream feature extraction & evaluation
out_parquet = os.path.join(VAL_DIR, "val_v16_cands.parquet")
con.execute(f"COPY v16_cands_tbl TO '{out_parquet.replace(chr(92), '/')}' (FORMAT PARQUET);")
print(f"Saved V16 candidates to {out_parquet} ({os.path.getsize(out_parquet):,} bytes).")
