import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
sys.path.insert(0, 'src')
import duckdb
from eval_framework import load_benchmark
from normalizer import normalize_business_name

s1, gt, _, s23 = load_benchmark()
s1_df = s1.to_pandas()
s23_df = s23.to_pandas()
s1_df["clean_name"] = s1_df["business_name"].apply(normalize_business_name)
s23_df["clean_name"] = s23_df["business_name"].apply(normalize_business_name)

con = duckdb.connect()
con.register("s1", s1_df)
con.register("gt", gt.to_pandas())
con.register("s23", s23_df)

STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"

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

ch8_cnt = con.execute("SELECT COUNT(*) FROM ch8").fetchone()[0]
ch8_rec = con.execute("SELECT COUNT(*) FROM gt g JOIN ch8 c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id").fetchone()[0]
print(f"Ch8: {ch8_cnt:,} pairs | Recovered GT: {ch8_rec:,} ({ch8_rec/len(gt)*100:.2f}%)")
