import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, time

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")

print("Testing Rule D (House No + Name Word len >= 5) purity on Training Ground Truth...", flush=True)

STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"

t0 = time.time()
res = con.execute(f"""
WITH s1_w AS (
    SELECT entity_id as s1_id, country,
           LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_number,
           (LIST_FILTER(STR_SPLIT(name_normalized, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {STOP_WORDS}))[1] as w1
    FROM read_csv('normalized_data/train_source1_normalized.tsv', delim='\\t', header=true)
    WHERE LENGTH(LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0')) >= 1
),
s23_w AS (
    SELECT entity_id as m_id, country,
           LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_number,
           (LIST_FILTER(STR_SPLIT(name_normalized, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {STOP_WORDS}))[1] as w1
    FROM (
        SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/train_source2_normalized.tsv', delim='\\t', header=true)
        UNION ALL
        SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/train_source3_normalized.tsv', delim='\\t', header=true)
    )
    WHERE LENGTH(LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0')) >= 1
),
pairs AS (
    SELECT s1.s1_id, s23.m_id
    FROM s1_w s1 JOIN s23_w s23 ON s1.country = s23.country AND s1.house_number = s23.house_number AND s1.w1 = s23.w1
    WHERE s1.w1 IS NOT NULL AND s23.w1 IS NOT NULL
    QUALIFY COUNT(*) OVER (PARTITION BY s1.s1_id) <= 5
)
SELECT COUNT(*) as total_preds,
       SUM(CASE WHEN g.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END) as true_positives,
       SUM(CASE WHEN g.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as precision
FROM pairs p
LEFT JOIN read_csv('ground_truth_pairs.tsv', delim='\\t', header=true) g 
  ON p.s1_id = g.source1_entity_id AND p.m_id = g.matched_entity_id;
""").fetchall()[0]

print(f"Rule D: {res[0]:,} pairs | {res[1]:,} TP | Precision = {res[2]:.2f}% in {time.time()-t0:.1f}s", flush=True)
