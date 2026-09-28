import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")

print("Inspecting sample of remaining empty S1 entities...", flush=True)

con.execute("""
CREATE TEMP TABLE p_v20 AS
SELECT source1_entity_id, UNNEST(STR_SPLIT(matched_entity_ids, ',')) as matched_entity_id, 0.999 as prob
FROM read_csv('output/matching_results.tsv', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';

CREATE TEMP TABLE p_scored AS
SELECT source1_entity_id, matched_entity_id, probability as prob
FROM read_csv('test_scored_candidates.tsv', delim='\t', header=true)
WHERE probability >= 0.60;

CREATE TEMP TABLE s1_all AS
SELECT entity_id as s1_id, country, name_normalized as n1, address_normalized as a1
FROM read_csv('normalized_data/test_source1_normalized.tsv', delim='\t', header=true);

CREATE TEMP TABLE covered_s1 AS
SELECT DISTINCT source1_entity_id FROM p_v20
UNION
SELECT DISTINCT source1_entity_id FROM p_scored;

CREATE TEMP TABLE still_empty_s1 AS
SELECT s.* FROM s1_all s
LEFT JOIN covered_s1 c ON s.s1_id = c.source1_entity_id
WHERE c.source1_entity_id IS NULL;
""")

print("Sample 10 still-empty S1 entities:")
samples = con.execute("SELECT s1_id, n1, a1, country FROM still_empty_s1 LIMIT 10;").fetchall()
for s in samples:
    print(f"  {s[0]} | {s[1]} | {s[2]} | {s[3]}")
