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
con.register("s1", s1)
con.register("gt", gt.to_pandas())
con.register("s23", s23)
con.register("v14_cands", v14_cands.to_pandas())

con.execute("""
CREATE TEMP TABLE missed_gt AS
SELECT g.source1_entity_id, g.matched_entity_id
FROM gt g
LEFT JOIN v14_cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")

total_missed = con.execute("SELECT COUNT(*) FROM missed_gt;").fetchone()[0]
print(f"Total Missed GT pairs: {total_missed:,} (Current Blocking Recall: {(len(gt)-total_missed)/len(gt)*100:.2f}%)")

# Test 1: Door Number + Locality Token (length >= 6)
ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"

con.execute(f"""
CREATE TEMP TABLE ch_door_locality AS
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

rec_1 = con.execute("""
SELECT COUNT(DISTINCT (m.source1_entity_id || '_' || m.matched_entity_id))
FROM missed_gt m JOIN ch_door_locality t ON m.source1_entity_id = t.source1_entity_id AND m.matched_entity_id = t.matched_entity_id;
""").fetchone()[0]
pairs_1 = con.execute("SELECT COUNT(*) FROM ch_door_locality;").fetchone()[0]
print(f"Test 1 (Door Number + Locality Token len>=6): Recovers +{rec_1:,} missed GT | Added {pairs_1:,} pairs")

# Test 2: Sorted Two Longest Non-Stop Address Words
con.execute(f"""
CREATE TEMP TABLE ch_sorted_addr2 AS
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

rec_2 = con.execute("""
SELECT COUNT(DISTINCT (m.source1_entity_id || '_' || m.matched_entity_id))
FROM missed_gt m JOIN ch_sorted_addr2 t ON m.source1_entity_id = t.source1_entity_id AND m.matched_entity_id = t.matched_entity_id;
""").fetchone()[0]
pairs_2 = con.execute("SELECT COUNT(*) FROM ch_sorted_addr2;").fetchone()[0]
print(f"Test 2 (Sorted 2 Longest Distinctive Address Words): Recovers +{rec_2:,} missed GT | Added {pairs_2:,} pairs")

# Test 3: Fuzzy Name Soundex / 3-char Brand Trigram
con.execute(f"""
CREATE TEMP TABLE ch_brand_tri AS
WITH s1_tri AS (
    SELECT source1_entity_id, country,
           SUBSTRING(REGEXP_REPLACE(clean_name, '[aeiou]', '', 'g'), 1, 4) as cons_stem
    FROM s1
    WHERE LENGTH(clean_name) >= 3
),
s23_tri AS (
    SELECT matched_entity_id, country,
           SUBSTRING(REGEXP_REPLACE(clean_name, '[aeiou]', '', 'g'), 1, 4) as cons_stem
    FROM s23
    WHERE LENGTH(clean_name) >= 3
),
cons_counts AS (
    SELECT country, cons_stem, COUNT(*) as freq
    FROM (SELECT country, cons_stem FROM s1_tri UNION ALL SELECT country, cons_stem FROM s23_tri)
    GROUP BY country, cons_stem
    HAVING freq >= 2 AND freq <= 20
)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_tri s1
JOIN cons_counts cc ON s1.country = cc.country AND s1.cons_stem = cc.cons_stem
JOIN s23_tri s23 ON s1.country = s23.country AND s1.cons_stem = cc.cons_stem
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")

rec_3 = con.execute("""
SELECT COUNT(DISTINCT (m.source1_entity_id || '_' || m.matched_entity_id))
FROM missed_gt m JOIN ch_brand_tri t ON m.source1_entity_id = t.source1_entity_id AND m.matched_entity_id = t.matched_entity_id;
""").fetchone()[0]
pairs_3 = con.execute("SELECT COUNT(*) FROM ch_brand_tri;").fetchone()[0]
print(f"Test 3 (Consonant Skeleton Stem freq 2-20): Recovers +{rec_3:,} missed GT | Added {pairs_3:,} pairs")

# Combined New Blocking Channels
con.execute("""
CREATE TEMP TABLE new_channels_combined AS
SELECT source1_entity_id, matched_entity_id FROM ch_door_locality
UNION
SELECT source1_entity_id, matched_entity_id FROM ch_sorted_addr2
UNION
SELECT source1_entity_id, matched_entity_id FROM ch_brand_tri;
""")

rec_total = con.execute("""
SELECT COUNT(DISTINCT (m.source1_entity_id || '_' || m.matched_entity_id))
FROM missed_gt m JOIN new_channels_combined t ON m.source1_entity_id = t.source1_entity_id AND m.matched_entity_id = t.matched_entity_id;
""").fetchone()[0]
pairs_total = con.execute("SELECT COUNT(*) FROM new_channels_combined;").fetchone()[0]

print("=" * 80)
print(f"TOTAL MISSED GT RECOVERED: +{rec_total:,} / {total_missed:,} ({rec_total/total_missed*100:.1f}%)")
new_cov = len(gt) - total_missed + rec_total
print(f"NEW CANDIDATE POOL COVERAGE: {new_cov:,} / {len(gt):,} ({new_cov/len(gt)*100:.2f}%)")
print(f"Total New Candidate Pairs Added: {pairs_total:,}")
print("=" * 80)
