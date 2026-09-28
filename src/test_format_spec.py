import os
import sys
import duckdb

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
TEST_S1 = r"C:\Users\Admin\Documents\Amazon-ML-Dataset\student_resource\dataset\test\test_source1.tsv"
TMP_DIR = os.path.join(BASE, "test_validator_check")
os.makedirs(TMP_DIR, exist_ok=True)

match_path = os.path.join(TMP_DIR, "matching_results.tsv")
cand_path = os.path.join(TMP_DIR, "candidate_pairs.tsv")

print("Reading first 100 S1 IDs from test_source1.tsv...")
con = duckdb.connect()
s1_ids = con.execute(f"SELECT entity_id FROM read_csv('{TEST_S1}', delim='\\t', header=true) LIMIT 100").fetchall()
s1_ids = [x[0] for x in s1_ids]

print("Writing mock files with proper format...")
with open(match_path, "w", newline="", encoding="utf-8") as f_m, \
     open(cand_path, "w", newline="", encoding="utf-8") as f_c:
    f_m.write("source1_entity_id\tmatched_entity_ids\n")
    f_c.write("source1_entity_id\tcandidate_entity_ids\n")
    
    for i, sid in enumerate(s1_ids):
        if i % 2 == 0:
            # Has match
            f_m.write(f"{sid}\tS2-{i}000,S3-{i}001\n")
            f_c.write(f"{sid}\tS2-{i}000,S3-{i}001,S2-{i}002\n")
        else:
            # Singleton (empty match, but may have candidates)
            f_m.write(f"{sid}\t\n")
            f_c.write(f"{sid}\tS2-{i}002\n")

print(f"Mock files created at {TMP_DIR}")
