import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, zipfile, subprocess
import duckdb
from collections import defaultdict

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
OUTPUT_DIR = os.path.join(BASE, "output")
MATCHING_TSV = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATES_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
V20_MATCH = os.path.join(BASE, "scratch", "v20_extracted", "matching_results.tsv")
V20_CANDS = os.path.join(BASE, "scratch", "v20_extracted", "candidate_pairs.tsv")
S1_RAW = r"D:\student_resource\student_resource\dataset\test\test_source1.tsv"
SUBMISSION_ZIP = os.path.join(BASE, "team_antigravity_final_v28_submission.zip")
VALIDATOR_PY = r"D:\student_resource\student_resource\utils\validate_submission.py"
TEST_DIR = r"D:\student_resource\student_resource\dataset\test"

os.makedirs(r"D:\duckdb_temp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='8GB'")
con.execute("SET temp_directory='D:/duckdb_temp'")
con.execute("PRAGMA max_temp_directory_size='500GiB'")

print("=" * 90, flush=True)
print("V28 — MAXIMUM RECALL EXPANSION: prob>=0.90 + All Ultra Rules", flush=True)
print("Strategy: V20 + Ultra Rules + ALL new ML pairs at prob>=0.90", flush=True)
print("(1,134,391 new pairs, 272,031 new S1 entities — widest net yet)", flush=True)
print("=" * 90, flush=True)

t0 = time.time()

# Step 1: V20 baseline
print("\n[Step 1] Loading V20 baseline...", flush=True)
con.execute(f"""
CREATE TEMP TABLE p_v20 AS
SELECT source1_entity_id, UNNEST(STR_SPLIT(matched_entity_ids, ',')) AS matched_entity_id, 0.9999 AS prob
FROM read_csv('{V20_MATCH.replace(chr(92), "/")}', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';
""")
n_v20 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_v20").fetchone()
print(f"  V20: {n_v20[0]:,} pairs, {n_v20[1]:,} S1 ({time.time()-t0:.1f}s)", flush=True)

# Step 2: Load normalized test data
t2 = time.time()
print("\n[Step 2] Loading normalized test data...", flush=True)
con.execute("""
CREATE TEMP TABLE s1_tbl AS
SELECT entity_id AS s1_id, name_normalized AS n1, address_normalized AS a1, country
FROM read_csv('normalized_data/test_source1_normalized.tsv', delim='\t', header=true);

CREATE TEMP TABLE s23_tbl AS
SELECT entity_id AS m_id, name_normalized AS n2, address_normalized AS a2, country
FROM (
    SELECT entity_id, name_normalized, address_normalized, country
    FROM read_csv('normalized_data/test_source2_normalized.tsv', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, name_normalized, address_normalized, country
    FROM read_csv('normalized_data/test_source3_normalized.tsv', delim='\t', header=true)
);
""")
print(f"  Tables loaded in {time.time()-t2:.1f}s.", flush=True)

# Step 3: Ultra-precision rules
t3 = time.time()
print("\n[Step 3] Certified ultra-precision rules (>=99.37%)...", flush=True)
con.execute("""
CREATE TEMP TABLE p_ultra AS
SELECT s1.s1_id AS source1_entity_id, s23.m_id AS matched_entity_id, 0.9995 AS prob
FROM s1_tbl s1 JOIN s23_tbl s23
  ON s1.country = s23.country AND s1.n1 = s23.n2 AND s1.a1 = s23.a2
WHERE LENGTH(s1.n1) >= 4 AND LENGTH(s1.a1) >= 4
UNION ALL
SELECT s1.s1_id, s23.m_id, 0.9990 AS prob
FROM s1_tbl s1 JOIN s23_tbl s23
  ON s1.country = s23.country AND s1.n1 = s23.n2
     AND SUBSTRING(s1.a1, 1, 8) = SUBSTRING(s23.a2, 1, 8)
WHERE LENGTH(s1.n1) >= 4 AND LENGTH(s1.a1) >= 8
UNION ALL
SELECT s1.s1_id, s23.m_id, 0.9992 AS prob
FROM s1_tbl s1 JOIN s23_tbl s23
  ON s1.country = s23.country AND s1.a1 = s23.a2
     AND SUBSTRING(s1.n1, 1, 8) = SUBSTRING(s23.n2, 1, 8)
WHERE LENGTH(s1.a1) >= 8 AND LENGTH(s1.n1) >= 8
UNION ALL
SELECT s1.s1_id, s23.m_id, 0.9988 AS prob
FROM s1_tbl s1 JOIN s23_tbl s23
  ON s1.country = s23.country AND s1.a1 = s23.a2
     AND SUBSTRING(s1.n1, 1, 3) = SUBSTRING(s23.n2, 1, 3)
WHERE LENGTH(s1.a1) >= 12 AND LENGTH(s1.n1) >= 3;
""")
u = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_ultra").fetchone()
print(f"  Ultra: {u[0]:,} pairs, {u[1]:,} S1 ({time.time()-t3:.1f}s)", flush=True)

# Step 4: ALL NEW ML scored pairs at prob>=0.90 not in V20
t4 = time.time()
print("\n[Step 4] Loading new ML pairs at prob>=0.90 (not in V20)...", flush=True)
con.execute(f"""
CREATE TEMP TABLE v20_set AS
SELECT source1_entity_id, UNNEST(STR_SPLIT(matched_entity_ids, ',')) AS matched_entity_id
FROM read_csv('{V20_MATCH.replace(chr(92), "/")}', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';

CREATE TEMP TABLE p_new_90 AS
SELECT s.source1_entity_id, s.matched_entity_id, s.probability AS prob
FROM read_csv('test_scored_candidates.tsv', delim='\t', header=true) s
LEFT JOIN v20_set v ON s.source1_entity_id = v.source1_entity_id
                   AND s.matched_entity_id = v.matched_entity_id
WHERE s.probability >= 0.90
  AND v.source1_entity_id IS NULL;
""")
n90 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_new_90").fetchone()
print(f"  New pairs (prob>=0.90, not in V20): {n90[0]:,} pairs, {n90[1]:,} S1 ({time.time()-t4:.1f}s)", flush=True)

# Step 5: Merge + injective resolution
t5 = time.time()
print("\n[Step 5] Merging and injective resolution...", flush=True)
con.execute("""
CREATE TEMP TABLE all_candidates AS
SELECT source1_entity_id, matched_entity_id, MAX(prob) AS prob
FROM (
    SELECT source1_entity_id, matched_entity_id, prob FROM p_v20
    UNION ALL SELECT source1_entity_id, matched_entity_id, prob FROM p_ultra
    UNION ALL SELECT source1_entity_id, matched_entity_id, prob FROM p_new_90
)
GROUP BY source1_entity_id, matched_entity_id;

CREATE TEMP TABLE final_matches AS
WITH ranked AS (
    SELECT source1_entity_id, matched_entity_id, prob,
           ROW_NUMBER() OVER (PARTITION BY matched_entity_id ORDER BY prob DESC) AS rk
    FROM all_candidates
)
SELECT source1_entity_id, matched_entity_id, prob FROM ranked WHERE rk = 1;
""")
fin = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM final_matches").fetchone()
matched_s1 = fin[1]
empty_s1 = 1732544 - matched_s1
print(f"  Done in {time.time()-t5:.1f}s:", flush=True)
print(f"    Match Pairs:  {fin[0]:,}  (V20: {n_v20[0]:,}, +{fin[0]-n_v20[0]:,})", flush=True)
print(f"    Matched S1:   {matched_s1:,} ({matched_s1/1732544*100:.2f}%)  (+{matched_s1-n_v20[1]:,})", flush=True)
print(f"    Empty:        {empty_s1:,} ({empty_s1/1732544*100:.2f}%)", flush=True)
print(f"    Avg/S1:       {fin[0]/matched_s1:.3f}  (GT: ~3.666)", flush=True)

# Step 6: Write output
t6 = time.time()
print("\n[Step 6] Writing output files...", flush=True)
row_order = []
with open(S1_RAW, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        row_order.append(line.split("\t")[0].strip())

s1_matches = defaultdict(list)
for sid, mid, prob in con.execute("SELECT source1_entity_id, matched_entity_id, prob FROM final_matches ORDER BY prob DESC").fetchall():
    s1_matches[sid].append(mid)

non_empty = empty = 0
with open(MATCHING_TSV, "w", encoding="utf-8") as f_out:
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in row_order:
        matches = s1_matches.get(sid, [])
        if matches:
            f_out.write(f"{sid}\t{','.join(matches)}\n")
            non_empty += 1
        else:
            f_out.write(f"{sid}\t\n")
            empty += 1
print(f"  matching_results.tsv: {non_empty:,} matched, {empty:,} empty ({time.time()-t6:.1f}s)", flush=True)

# Sync candidate_pairs
t7 = time.time()
all_cand_pairs = defaultdict(set)
with open(V20_CANDS, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0]
        cands = [x for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
        all_cand_pairs[sid].update(cands)
for sid, mid, _ in con.execute("SELECT source1_entity_id, matched_entity_id, prob FROM p_ultra").fetchall():
    all_cand_pairs[sid].add(mid)
for sid, mid, _ in con.execute("SELECT source1_entity_id, matched_entity_id, prob FROM p_new_90").fetchall():
    all_cand_pairs[sid].add(mid)
for sid, matches in s1_matches.items():
    for mid in matches:
        all_cand_pairs[sid].add(mid)

TEMP_CAND = os.path.join(OUTPUT_DIR, "candidate_pairs_v28_temp.tsv")
with open(TEMP_CAND, "w", encoding="utf-8") as f_out:
    f_out.write("source1_entity_id\tcandidate_entity_ids\n")
    for sid in row_order:
        cands = list(all_cand_pairs.get(sid, set()))
        if cands:
            f_out.write(f"{sid}\t{','.join(cands)}\n")
        else:
            f_out.write(f"{sid}\t\n")
os.replace(TEMP_CAND, CANDIDATES_TSV)
print(f"  candidate_pairs.tsv: {os.path.getsize(CANDIDATES_TSV)/(1024*1024):.1f} MB ({time.time()-t7:.1f}s)", flush=True)

# Package and validate
with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(MATCHING_TSV, arcname="matching_results.tsv")
    zf.write(CANDIDATES_TSV, arcname="candidate_pairs.tsv")
zip_size = os.path.getsize(SUBMISSION_ZIP) / (1024 * 1024)
print(f"\n  Zip: {zip_size:.1f} MB", flush=True)

print("\n" + "-" * 90, flush=True)
print("RUNNING OFFICIAL VALIDATOR...", flush=True)
cmd = [sys.executable, VALIDATOR_PY,
       "--matching", MATCHING_TSV, "--candidate", CANDIDATES_TSV,
       "--test-dir", TEST_DIR, "--check-ids"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout, flush=True)
if r.stderr: print("STDERR:", r.stderr, flush=True)

print("=" * 90, flush=True)
print(f"V28 COMPLETED in {time.time()-t0:.1f}s! Submission: {SUBMISSION_ZIP}", flush=True)
print("=" * 90, flush=True)
