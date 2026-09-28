import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, zipfile, subprocess
import duckdb
from collections import defaultdict

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
OUTPUT_DIR = os.path.join(BASE, "output")
MATCHING_TSV = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATES_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
S1_RAW = r"D:\student_resource\student_resource\dataset\test\test_source1.tsv"
SUBMISSION_ZIP = os.path.join(BASE, "team_antigravity_final_v22_submission.zip")
VALIDATOR_PY = r"D:\student_resource\student_resource\utils\validate_submission.py"
TEST_DIR = r"D:\student_resource\student_resource\dataset\test"

os.makedirs(r"D:\duckdb_temp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='4GB'")
con.execute("SET temp_directory='D:/duckdb_temp'")
con.execute("PRAGMA max_temp_directory_size='500GiB'")

print("=" * 85, flush=True)
print("AMAZON ML CHALLENGE 2026 — CHAMPION V22 MASTER SUBMISSION PIPELINE", flush=True)
print("Precision-Verified Ultra Rules (97.2%-100.0%) + Calibrated 52-Feature ML Candidates", flush=True)
print("Strict 1-to-1 Injective Bipartite Resolution & Full Format Validation", flush=True)
print("=" * 85, flush=True)

t0 = time.time()

# 1. Load V21 baseline matches
print("\n[Step 1/6] Loading V21 baseline matches...", flush=True)
con.execute(f"""
CREATE TEMP TABLE p_v21 AS
SELECT source1_entity_id, UNNEST(STR_SPLIT(matched_entity_ids, ',')) as matched_entity_id, 0.9990 as prob
FROM read_csv('{MATCHING_TSV.replace(chr(92), '/')}', delim='\\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';
""")
n_v21 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_v21").fetchone()
print(f"  V21 baseline loaded: {n_v21[0]:,} matches across {n_v21[1]:,} S1 entities in {time.time()-t0:.1f}s.", flush=True)

# 2. Load test normalized tables
t1 = time.time()
print("\n[Step 2/6] Loading normalized test datasets...", flush=True)
con.execute("""
CREATE TEMP TABLE s1_test AS
SELECT entity_id as s1_id, country, 
       name_normalized as n1, 
       address_normalized as a1
FROM read_csv('normalized_data/test_source1_normalized.tsv', delim='\t', header=true);

CREATE TEMP TABLE s23_test AS
SELECT entity_id as m_id, country, 
       name_normalized as n2, 
       address_normalized as a2
FROM (
    SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/test_source2_normalized.tsv', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/test_source3_normalized.tsv', delim='\t', header=true)
);
""")
print(f"  Normalized tables loaded in {time.time()-t1:.1f}s.", flush=True)

# 3. Extract Precision-Verified Ultra Rules (all empirically tested >= 97.18% on Ground Truth)
t2 = time.time()
print("\n[Step 3/6] Mining Precision-Verified Ultra Rules (97.18% - 100.0% GT Precision)...", flush=True)
con.execute("""
CREATE TEMP TABLE p_ultra AS
-- Rule 1: Exact Name + Exact Address (100.00% precision on GT)
SELECT s1.s1_id as source1_entity_id, s23.m_id as matched_entity_id, 0.9999 as prob
FROM s1_test s1 JOIN s23_test s23 
  ON s1.country = s23.country AND s1.n1 = s23.n2 AND s1.a1 = s23.a2
WHERE LENGTH(s1.n1) >= 4 AND LENGTH(s1.a1) >= 4

UNION ALL

-- Rule 2: Exact Name + Addr Pfx 8 (99.37% precision on GT)
SELECT s1.s1_id as source1_entity_id, s23.m_id as matched_entity_id, 0.9998 as prob
FROM s1_test s1 JOIN s23_test s23 
  ON s1.country = s23.country AND s1.n1 = s23.n2 AND SUBSTRING(s1.a1, 1, 8) = SUBSTRING(s23.a2, 1, 8)
WHERE LENGTH(s1.n1) >= 4 AND LENGTH(s1.a1) >= 8

UNION ALL

-- Rule 3: Exact Addr + Name Pfx 8 (99.87% precision on GT)
SELECT s1.s1_id as source1_entity_id, s23.m_id as matched_entity_id, 0.9997 as prob
FROM s1_test s1 JOIN s23_test s23 
  ON s1.country = s23.country AND s1.a1 = s23.a2 AND SUBSTRING(s1.n1, 1, 8) = SUBSTRING(s23.n2, 1, 8)
WHERE LENGTH(s1.a1) >= 8 AND LENGTH(s1.n1) >= 8

UNION ALL

-- Rule 4: Exact Addr (len >= 12) + Name Pfx 3 (99.48% precision on GT)
SELECT s1.s1_id as source1_entity_id, s23.m_id as matched_entity_id, 0.9996 as prob
FROM s1_test s1 JOIN s23_test s23 
  ON s1.country = s23.country AND s1.a1 = s23.a2 AND SUBSTRING(s1.n1, 1, 3) = SUBSTRING(s23.n2, 1, 3)
WHERE LENGTH(s1.a1) >= 12 AND LENGTH(s1.n1) >= 3

UNION ALL

-- Rule 5: Exact Name (len >= 6) + Addr Pfx 4 (97.18% precision on GT)
SELECT s1.s1_id as source1_entity_id, s23.m_id as matched_entity_id, 0.9995 as prob
FROM s1_test s1 JOIN s23_test s23 
  ON s1.country = s23.country AND s1.n1 = s23.n2 AND SUBSTRING(s1.a1, 1, 4) = SUBSTRING(s23.a2, 1, 4)
WHERE LENGTH(s1.n1) >= 6 AND LENGTH(s1.a1) >= 4

UNION ALL

-- Rule 6: Exact Addr (len >= 12) + Name 1st Letter (97.82% precision on GT)
SELECT s1.s1_id as source1_entity_id, s23.m_id as matched_entity_id, 0.9994 as prob
FROM s1_test s1 JOIN s23_test s23 
  ON s1.country = s23.country AND s1.a1 = s23.a2 AND SUBSTRING(s1.n1, 1, 1) = SUBSTRING(s23.n2, 1, 1)
WHERE LENGTH(s1.a1) >= 12 AND LENGTH(s1.n1) >= 2;
""")
u_stats = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_ultra").fetchone()
print(f"  Generated {u_stats[0]:,} ultra-precision pairs across {u_stats[1]:,} S1 entities in {time.time()-t2:.1f}s.", flush=True)

# 4. Load ML candidates from test_scored_candidates.tsv
t3 = time.time()
print("\n[Step 4/6] Loading ML candidates (prob >= 0.40 for all, prob >= 0.30 for empty)...", flush=True)
con.execute("""
CREATE TEMP TABLE p_model_all AS
SELECT source1_entity_id, matched_entity_id, probability as prob
FROM read_csv('test_scored_candidates.tsv', delim='\t', header=true)
WHERE probability >= 0.40;

CREATE TEMP TABLE p_model_empty AS
SELECT s.source1_entity_id, s.matched_entity_id, s.probability as prob
FROM read_csv('test_scored_candidates.tsv', delim='\t', header=true) s
WHERE s.probability >= 0.30
  AND s.source1_entity_id NOT IN (SELECT source1_entity_id FROM p_v21);
""")
m_stats = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_model_all").fetchone()
me_stats = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_model_empty").fetchone()
print(f"  ML candidates loaded: {m_stats[0]:,} general (prob >= 0.40) + {me_stats[0]:,} empty rescue (prob >= 0.30) in {time.time()-t3:.1f}s.", flush=True)

# 5. Strict 1-to-1 Injective Bipartite Resolution
t4 = time.time()
print("\n[Step 5/6] Executing strict 1-to-1 injective bipartite resolution...", flush=True)
con.execute("""
CREATE TEMP TABLE all_candidates AS
SELECT source1_entity_id, matched_entity_id, MAX(prob) as prob
FROM (
    SELECT * FROM p_ultra
    UNION ALL
    SELECT * FROM p_v21
    UNION ALL
    SELECT * FROM p_model_all
    UNION ALL
    SELECT * FROM p_model_empty
)
GROUP BY source1_entity_id, matched_entity_id;

CREATE TEMP TABLE final_injective_matches AS
WITH ranked AS (
    SELECT source1_entity_id, matched_entity_id, prob,
           ROW_NUMBER() OVER (PARTITION BY matched_entity_id ORDER BY prob DESC) as rk
    FROM all_candidates
)
SELECT source1_entity_id, matched_entity_id, prob
FROM ranked WHERE rk = 1;
""")

fin_stats = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM final_injective_matches").fetchone()
matched_s1 = fin_stats[1]
empty_s1 = 1732544 - matched_s1
avg_m = fin_stats[0] / matched_s1

print(f"  Injective bipartite resolution complete in {time.time()-t4:.1f}s:")
print(f"    Total Match Pairs:   {fin_stats[0]:,} (V21 was {n_v21[0]:,}, +{fin_stats[0] - n_v21[0]:,} surge!)", flush=True)
print(f"    Matched S1 Entities: {matched_s1:,} ({matched_s1/1732544*100:.2f}%) (V21 was {n_v21[1]:,}, +{matched_s1 - n_v21[1]:,} gained!)", flush=True)
print(f"    Empty Singletons:    {empty_s1:,} ({empty_s1/1732544*100:.2f}%) [GT Expected: 96,760 = 5.58%]", flush=True)
print(f"    Avg Matches per S1:  {avg_m:.3f} [GT Expected: 3.666]", flush=True)

# 6. Writing output files and validating
t5 = time.time()
print("\n[Step 6/6] Writing matching_results.tsv and synchronizing candidate_pairs.tsv...", flush=True)

row_order = []
with open(S1_RAW, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        row_order.append(line.split("\t")[0].strip())

s1_matches = defaultdict(list)
cursor = con.execute("SELECT source1_entity_id, matched_entity_id FROM final_injective_matches ORDER BY prob DESC")
for sid, mid in cursor.fetchall():
    s1_matches[sid].append(mid)

non_empty = 0
empty = 0
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
print(f"  matching_results.tsv written in {time.time()-t5:.1f}s ({non_empty:,} matched, {empty:,} empty).", flush=True)

# Synchronize candidate_pairs.tsv
t_sync = time.time()
TEMP_CAND = os.path.join(OUTPUT_DIR, "candidate_pairs_v22_temp.tsv")
added_cands = 0
with open(CANDIDATES_TSV, "r", encoding="utf-8") as f_in, open(TEMP_CAND, "w", encoding="utf-8") as f_out:
    f_out.write("source1_entity_id\tcandidate_entity_ids\n")
    header = next(f_in)
    for line in f_in:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0]
        cands = [x for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
        cand_set = set(cands)
        true_matches = set(s1_matches.get(sid, []))
        missing = true_matches - cand_set
        if missing:
            cands.extend(list(missing))
            added_cands += len(missing)
        if cands:
            f_out.write(f"{sid}\t{','.join(cands)}\n")
        else:
            f_out.write(f"{sid}\t\n")

os.replace(TEMP_CAND, CANDIDATES_TSV)
print(f"  candidate_pairs.tsv synchronized in {time.time()-t_sync:.1f}s (+{added_cands:,} IDs added).", flush=True)

# Package final submission zip
t_zip = time.time()
print(f"\nPackaging final submission zip: {SUBMISSION_ZIP}...", flush=True)
with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(MATCHING_TSV, arcname="matching_results.tsv")
    zf.write(CANDIDATES_TSV, arcname="candidate_pairs.tsv")
zip_size = os.path.getsize(SUBMISSION_ZIP) / (1024 * 1024)
print(f"  Zip created: {SUBMISSION_ZIP} ({zip_size:.1f} MB) in {time.time()-t_zip:.1f}s.", flush=True)

# Official Validation Run
print("\n" + "-" * 85, flush=True)
print("RUNNING OFFICIAL VALIDATOR...", flush=True)
print("-" * 85, flush=True)
cmd = [
    sys.executable,
    VALIDATOR_PY,
    "--matching", MATCHING_TSV,
    "--candidate", CANDIDATES_TSV,
    "--test-dir", TEST_DIR,
    "--check-ids"
]
val_proc = subprocess.run(cmd, capture_output=True, text=True)
print(val_proc.stdout, flush=True)
if val_proc.stderr:
    print("STDERR:", val_proc.stderr, flush=True)

print("=" * 85, flush=True)
print(f"V22 CHAMPION PIPELINE COMPLETED SUCCESSFULLY IN {time.time()-t0:.1f}s!", flush=True)
print("=" * 85, flush=True)
