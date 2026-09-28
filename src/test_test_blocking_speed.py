import sys
import duckdb
import time

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

S1_NORM = 'C:/Users/Admin/Documents/Amazon-ML-Challenge-2026/normalized_data/test_source1_normalized.tsv'
S2_NORM = 'C:/Users/Admin/Documents/Amazon-ML-Challenge-2026/normalized_data/test_source2_normalized.tsv'
S3_NORM = 'C:/Users/Admin/Documents/Amazon-ML-Challenge-2026/normalized_data/test_source3_normalized.tsv'

t0 = time.time()
print("1. Creating test views...")
con.execute(f"CREATE TEMP TABLE s1_tbl AS SELECT entity_id, business_name, business_address, country FROM read_csv('{S1_NORM}', delim='\\t', header=true)")
con.execute(f"""
CREATE TEMP TABLE s23_tbl AS 
SELECT entity_id, business_name, business_address, country FROM read_csv('{S2_NORM}', delim='\\t', header=true)
UNION ALL
SELECT entity_id, business_name, business_address, country FROM read_csv('{S3_NORM}', delim='\\t', header=true)
""")
print(f"Loaded tables in {time.time()-t0:.1f}s.")

STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"

print("\n2. Testing Channel 1 (Exact Two Brand Tokens) on full test set...")
t_ch1 = time.time()
con.execute(f"""
CREATE TEMP TABLE s1_two_tokens AS
WITH base AS (
    SELECT entity_id, country,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) AS words
    FROM s1_tbl
)
SELECT entity_id, country,
       CASE WHEN LEN(words) >= 2 THEN words[1] || '_' || words[2] ELSE '' END AS two_tok_key
FROM base WHERE LEN(words) >= 2;

CREATE TEMP TABLE s23_two_tokens AS
WITH base AS (
    SELECT entity_id, country,
           LIST_FILTER(STR_SPLIT(LOWER(TRIM(REGEXP_REPLACE(business_name, '[^a-zA-Z0-9]+', ' ', 'g'))), ' '), x -> LENGTH(x) >= 3 AND x NOT IN {STOP_WORDS}) AS words
    FROM s23_tbl
)
SELECT entity_id, country,
       CASE WHEN LEN(words) >= 2 THEN words[1] || '_' || words[2] ELSE '' END AS two_tok_key
FROM base WHERE LEN(words) >= 2;

CREATE TEMP TABLE ch1_pairs AS
SELECT source1_entity_id, matched_entity_id
FROM (
    SELECT s1.entity_id AS source1_entity_id, s23.entity_id AS matched_entity_id,
           COUNT(*) OVER (PARTITION BY s1.entity_id) as cnt
    FROM s1_two_tokens s1
    JOIN s23_two_tokens s23 ON s1.country = s23.country AND s1.two_tok_key = s23.two_tok_key
    WHERE s1.two_tok_key != '' AND LENGTH(s1.two_tok_key) >= 7
) sub WHERE cnt <= 15;
""")

ch1_cnt = con.execute("SELECT COUNT(*) FROM ch1_pairs").fetchone()[0]
print(f"Generated {ch1_cnt:,} Channel 1 pairs across entire test set in {time.time()-t_ch1:.1f}s!")
