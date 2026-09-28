import os
import sys
import duckdb
import time

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
OUTPUT_DIR = os.path.join(BASE, "output")
TEST_S1 = r"C:\Users\Admin\Documents\Amazon-ML-Dataset\student_resource\dataset\test\test_source1.tsv".replace("\\", "/")

orig_matching = os.path.join(OUTPUT_DIR, "matching_results.tsv").replace("\\", "/")
orig_candidates = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv").replace("\\", "/")

fixed_matching = os.path.join(OUTPUT_DIR, "matching_results_fixed.tsv")
fixed_candidates = os.path.join(OUTPUT_DIR, "candidate_pairs_fixed.tsv")

print("=" * 70)
print("TESTING FAST FORMAT CORRECTION FOR VALIDATOR COMPLIANCE")
print("=" * 70)

t0 = time.time()
con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3GB'")

print("1. Aggregating candidates per S1 and ensuring every S1 is present...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE test_s1 AS
SELECT entity_id AS source1_entity_id FROM read_csv('{TEST_S1}', delim='\\t', header=true);

CREATE OR REPLACE TEMP TABLE cand_agg AS
SELECT 
    source1_entity_id,
    STRING_AGG(matched_entity_id, ',') AS candidate_entity_ids
FROM (
    SELECT DISTINCT source1_entity_id, matched_entity_id
    FROM read_csv('{orig_candidates}', delim='\\t', header=true)
)
GROUP BY source1_entity_id;

CREATE OR REPLACE TEMP TABLE final_candidates AS
SELECT 
    s.source1_entity_id,
    COALESCE(c.candidate_entity_ids, '') AS candidate_entity_ids
FROM test_s1 s
LEFT JOIN cand_agg c ON s.source1_entity_id = c.source1_entity_id
ORDER BY s.source1_entity_id;
""")

print("Writing fixed candidate_pairs.tsv...")
con.execute(f"""
COPY final_candidates TO '{fixed_candidates.replace('\\', '/')}' (DELIMITER '\t', HEADER true, QUOTE '');
""")
print(f"Candidates written in {time.time()-t0:.1f}s")

print("\n2. Fixing matching_results.tsv quotes on singletons...")
t1 = time.time()
con.execute(f"""
CREATE OR REPLACE TEMP TABLE match_raw AS
SELECT 
    source1_entity_id,
    CASE 
        WHEN matched_entity_ids = '""' OR matched_entity_ids IS NULL THEN ''
        ELSE REPLACE(matched_entity_ids, '"', '')
    END AS matched_entity_ids
FROM read_csv('{orig_matching}', delim='\\t', header=true);

CREATE OR REPLACE TEMP TABLE final_matches AS
SELECT 
    s.source1_entity_id,
    COALESCE(m.matched_entity_ids, '') AS matched_entity_ids
FROM test_s1 s
LEFT JOIN match_raw m ON s.source1_entity_id = m.source1_entity_id
ORDER BY s.source1_entity_id;
""")

print("Writing fixed matching_results.tsv...")
con.execute(f"""
COPY final_matches TO '{fixed_matching.replace('\\', '/')}' (DELIMITER '\t', HEADER true, QUOTE '');
""")
print(f"Matches written in {time.time()-t1:.1f}s")
print(f"Total time: {time.time()-t0:.1f}s")
