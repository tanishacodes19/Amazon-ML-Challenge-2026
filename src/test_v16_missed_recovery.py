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
print(f"Total Missed GT pairs to recover: {total_missed:,} (Current Recall: {(len(gt)-total_missed)/len(gt)*100:.2f}%)")

# Test Channel A: Any shared distinctive address word (len >= 7) with low frequency (cap <= 10)
con.execute("""
CREATE TEMP TABLE test_addr_word AS
WITH s1_words AS (
    SELECT source1_entity_id, country, UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 7)) as w
    FROM s1
),
s23_words AS (
    SELECT matched_entity_id, country, UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 7)) as w
    FROM s23
),
word_counts AS (
    SELECT country, w, COUNT(*) as freq
    FROM (SELECT country, w FROM s1_words UNION ALL SELECT country, w FROM s23_words)
    GROUP BY country, w
    HAVING freq >= 2 AND freq <= 15
)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_words s1
JOIN word_counts wc ON s1.country = wc.country AND s1.w = wc.w
JOIN s23_words s23 ON s1.country = s23.country AND s1.w = s23.w
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")

rec_a = con.execute("""
SELECT COUNT(DISTINCT (m.source1_entity_id || '_' || m.matched_entity_id))
FROM missed_gt m JOIN test_addr_word t ON m.source1_entity_id = t.source1_entity_id AND m.matched_entity_id = t.matched_entity_id;
""").fetchone()[0]
pairs_a = con.execute("SELECT COUNT(*) FROM test_addr_word;").fetchone()[0]
print(f"Channel A (Distinctive Address Word len>=7, freq 2-15): Recovers +{rec_a:,} missed GT ({rec_a/total_missed*100:.1f}%) | Total pairs added: {pairs_a:,}")

# Test Channel B: Phonetic Soundex / Metaphone on Indic words or First Token 3-prefix
con.execute("""
CREATE TEMP TABLE test_indic_prefix AS
WITH s1_p AS (
    SELECT source1_entity_id, country, SUBSTRING(clean_name, 1, 4) as pfx
    FROM s1 WHERE LENGTH(clean_name) >= 4
),
s23_p AS (
    SELECT matched_entity_id, country, SUBSTRING(clean_name, 1, 4) as pfx
    FROM s23 WHERE LENGTH(clean_name) >= 4
),
pfx_counts AS (
    SELECT country, pfx, COUNT(*) as freq
    FROM (SELECT country, pfx FROM s1_p UNION ALL SELECT country, pfx FROM s23_p)
    GROUP BY country, pfx
    HAVING freq >= 2 AND freq <= 20
)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_p s1
JOIN pfx_counts pc ON s1.country = pc.country AND s1.pfx = pc.pfx
JOIN s23_p s23 ON s1.country = s23.country AND s1.pfx = pc.pfx
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
rec_b = con.execute("""
SELECT COUNT(DISTINCT (m.source1_entity_id || '_' || m.matched_entity_id))
FROM missed_gt m JOIN test_indic_prefix t ON m.source1_entity_id = t.source1_entity_id AND m.matched_entity_id = t.matched_entity_id;
""").fetchone()[0]
pairs_b = con.execute("SELECT COUNT(*) FROM test_indic_prefix;").fetchone()[0]
print(f"Channel B (4-char Name Prefix, freq 2-20): Recovers +{rec_b:,} missed GT ({rec_b/total_missed*100:.1f}%) | Total pairs added: {pairs_b:,}")

# Test Channel C: Door/Unit Number + Postal Code (Physical Address Identity)
con.execute("""
CREATE TEMP TABLE test_door_postal AS
WITH s1_dp AS (
    SELECT source1_entity_id, country, house_number, postal_code
    FROM s1
    WHERE LENGTH(house_number) >= 2 AND LENGTH(postal_code) >= 4
),
s23_dp AS (
    SELECT matched_entity_id, country, house_number, postal_code
    FROM s23
    WHERE LENGTH(house_number) >= 2 AND LENGTH(postal_code) >= 4
)
SELECT s1.source1_entity_id, s23.matched_entity_id
FROM s1_dp s1
JOIN s23_dp s23 ON s1.country = s23.country AND s1.postal_code = s23.postal_code AND s1.house_number = s23.house_number
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 15;
""")
rec_c = con.execute("""
SELECT COUNT(DISTINCT (m.source1_entity_id || '_' || m.matched_entity_id))
FROM missed_gt m JOIN test_door_postal t ON m.source1_entity_id = t.source1_entity_id AND m.matched_entity_id = t.matched_entity_id;
""").fetchone()[0]
pairs_c = con.execute("SELECT COUNT(*) FROM test_door_postal;").fetchone()[0]
print(f"Channel C (Door Number + Postal Code): Recovers +{rec_c:,} missed GT ({rec_c/total_missed*100:.1f}%) | Total pairs added: {pairs_c:,}")

# Combined A + B + C
con.execute("""
CREATE TEMP TABLE combined_recovery AS
SELECT source1_entity_id, matched_entity_id FROM test_addr_word
UNION
SELECT source1_entity_id, matched_entity_id FROM test_indic_prefix
UNION
SELECT source1_entity_id, matched_entity_id FROM test_door_postal;
""")
rec_all = con.execute("""
SELECT COUNT(DISTINCT (m.source1_entity_id || '_' || m.matched_entity_id))
FROM missed_gt m JOIN combined_recovery t ON m.source1_entity_id = t.source1_entity_id AND m.matched_entity_id = t.matched_entity_id;
""").fetchone()[0]
pairs_all = con.execute("SELECT COUNT(*) FROM combined_recovery;").fetchone()[0]
print("-" * 80)
print(f"TOTAL RECOVERED BY A+B+C: +{rec_all:,} / {total_missed:,} ({rec_all/total_missed*100:.1f}%)")
print(f"NEW POTENTIAL BLOCKING RECALL: {(len(gt)-total_missed+rec_all)/len(gt)*100:.2f}% (recovering {len(gt)-total_missed+rec_all:,} / {len(gt):,})")
