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
SUBMISSION_ZIP = os.path.join(BASE, "team_antigravity_final_v27_submission.zip")
VALIDATOR_PY = r"D:\student_resource\student_resource\utils\validate_submission.py"
TEST_DIR = r"D:\student_resource\student_resource\dataset\test"

os.makedirs(r"D:\duckdb_temp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='8GB'")
con.execute("SET temp_directory='D:/duckdb_temp'")
con.execute("PRAGMA max_temp_directory_size='500GiB'")

print("=" * 90, flush=True)
print("V27 — HIGH CONFIDENCE EXPANSION PIPELINE", flush=True)
print("Strategy: V20 baseline + Ultra Rules + NEW scored pairs at prob>=0.95", flush=True)
print("Key insight: 738,894 already-ML-scored pairs NOT in V20 — adding them!", flush=True)
print("=" * 90, flush=True)

t0 = time.time()

# Step 1: Load V20 matches as the gold baseline (prob 0.9999 — protect all V20 decisions)
print("\n[Step 1] Loading V20 (0.769) baseline matches...", flush=True)
con.execute(f"""
CREATE TEMP TABLE p_v20 AS
SELECT source1_entity_id, UNNEST(STR_SPLIT(matched_entity_ids, ',')) AS matched_entity_id, 0.9999 AS prob
FROM read_csv('{V20_MATCH.replace(chr(92), "/")}', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';
""")
n_v20 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_v20").fetchone()
print(f"  V20 baseline: {n_v20[0]:,} pairs, {n_v20[1]:,} S1 ({time.time()-t0:.1f}s)", flush=True)

# Step 2: Load normalized test data for ultra-precision rules
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

# Step 3: Ultra-precision rules (certified >=99.37% GT precision)
t3 = time.time()
print("\n[Step 3] Certified ultra-precision rules (>=99.37% GT precision)...", flush=True)
con.execute("""
CREATE TEMP TABLE p_ultra AS
-- Rule 1: Exact Name + Exact Address (100.00%)
SELECT s1.s1_id AS source1_entity_id, s23.m_id AS matched_entity_id, 0.9995 AS prob
FROM s1_tbl s1 JOIN s23_tbl s23
  ON s1.country = s23.country AND s1.n1 = s23.n2 AND s1.a1 = s23.a2
WHERE LENGTH(s1.n1) >= 4 AND LENGTH(s1.a1) >= 4

UNION ALL
-- Rule 2: Exact Name + Addr Pfx 8 (99.37%)
SELECT s1.s1_id, s23.m_id, 0.9990 AS prob
FROM s1_tbl s1 JOIN s23_tbl s23
  ON s1.country = s23.country AND s1.n1 = s23.n2
     AND SUBSTRING(s1.a1, 1, 8) = SUBSTRING(s23.a2, 1, 8)
WHERE LENGTH(s1.n1) >= 4 AND LENGTH(s1.a1) >= 8

UNION ALL
-- Rule 3: Exact Addr + Name Pfx 8 (99.87%)
SELECT s1.s1_id, s23.m_id, 0.9992 AS prob
FROM s1_tbl s1 JOIN s23_tbl s23
  ON s1.country = s23.country AND s1.a1 = s23.a2
     AND SUBSTRING(s1.n1, 1, 8) = SUBSTRING(s23.n2, 1, 8)
WHERE LENGTH(s1.a1) >= 8 AND LENGTH(s1.n1) >= 8

UNION ALL
-- Rule 4: Exact Addr (len>=12) + Name Pfx 3 (99.48%)
SELECT s1.s1_id, s23.m_id, 0.9988 AS prob
FROM s1_tbl s1 JOIN s23_tbl s23
  ON s1.country = s23.country AND s1.a1 = s23.a2
     AND SUBSTRING(s1.n1, 1, 3) = SUBSTRING(s23.n2, 1, 3)
WHERE LENGTH(s1.a1) >= 12 AND LENGTH(s1.n1) >= 3;
""")
u_stats = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_ultra").fetchone()
print(f"  Ultra rules: {u_stats[0]:,} pairs, {u_stats[1]:,} S1 ({time.time()-t3:.1f}s).", flush=True)

# Step 4: Load HIGH-CONFIDENCE ML scored candidates NOT in V20 (prob >= 0.95)
# These are already scored by ML model — no feature extraction needed!
t4 = time.time()
print("\n[Step 4] Loading NEW high-confidence ML pairs (prob>=0.95, not in V20)...", flush=True)
con.execute(f"""
CREATE TEMP TABLE v20_set AS
SELECT source1_entity_id, UNNEST(STR_SPLIT(matched_entity_ids, ',')) AS matched_entity_id
FROM read_csv('{V20_MATCH.replace(chr(92), "/")}', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';

-- Load ALL scored candidates at high confidence
CREATE TEMP TABLE p_new_high AS
SELECT s.source1_entity_id, s.matched_entity_id, s.probability AS prob
FROM read_csv('test_scored_candidates.tsv', delim='\t', header=true) s
LEFT JOIN v20_set v ON s.source1_entity_id = v.source1_entity_id
                   AND s.matched_entity_id = v.matched_entity_id
WHERE s.probability >= 0.95
  AND v.source1_entity_id IS NULL;  -- Only NEW pairs not already in V20
""")
new_stats = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_new_high").fetchone()
print(f"  New high-conf pairs (prob>=0.95, not in V20): {new_stats[0]:,} pairs, {new_stats[1]:,} S1 ({time.time()-t4:.1f}s)", flush=True)

# Step 5: Merge all sources and apply injective resolution
t5 = time.time()
print("\n[Step 5] Merging all sources...", flush=True)
con.execute("""
CREATE TEMP TABLE all_candidates AS
SELECT source1_entity_id, matched_entity_id, MAX(prob) AS prob
FROM (
    SELECT source1_entity_id, matched_entity_id, prob FROM p_v20
    UNION ALL
    SELECT source1_entity_id, matched_entity_id, prob FROM p_ultra
    UNION ALL
    SELECT source1_entity_id, matched_entity_id, prob FROM p_new_high
)
GROUP BY source1_entity_id, matched_entity_id;
""")
all_stats = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM all_candidates").fetchone()
print(f"  Total merged: {all_stats[0]:,} pairs, {all_stats[1]:,} S1 ({time.time()-t5:.1f}s)", flush=True)

print("\n[Step 5b] Strict 1-to-1 injective bipartite resolution...", flush=True)
t5b = time.time()
con.execute("""
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
print(f"  Injective resolution done in {time.time()-t5b:.1f}s:", flush=True)
print(f"    Total Match Pairs:   {fin[0]:,}  (V20: {n_v20[0]:,},  +{fin[0]-n_v20[0]:,})", flush=True)
print(f"    Matched S1:          {matched_s1:,} ({matched_s1/1732544*100:.2f}%)  (V20: {n_v20[1]:,}, +{matched_s1-n_v20[1]:,})", flush=True)
print(f"    Empty/Singleton:     {empty_s1:,} ({empty_s1/1732544*100:.2f}%)", flush=True)
print(f"    Avg matches/S1:      {fin[0]/matched_s1:.3f}  (GT expected: ~3.666)", flush=True)

# Step 6: Write matching_results.tsv
t6 = time.time()
print("\n[Step 6] Writing matching_results.tsv...", flush=True)
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
print(f"  Written: {non_empty:,} matched, {empty:,} empty ({time.time()-t6:.1f}s)", flush=True)

# Step 7: Synchronize candidate_pairs.tsv (start from V20 candidates)
t7 = time.time()
print("\n[Step 7] Synchronizing candidate_pairs.tsv from V20 base...", flush=True)
all_cand_pairs = defaultdict(set)
with open(V20_CANDS, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0]
        cands = [x for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
        all_cand_pairs[sid].update(cands)

# Add ultra rule pairs to candidates
for sid, mid, prob in con.execute("SELECT source1_entity_id, matched_entity_id, prob FROM p_ultra").fetchall():
    all_cand_pairs[sid].add(mid)

# Add new high-conf pairs to candidates
for sid, mid, prob in con.execute("SELECT source1_entity_id, matched_entity_id, prob FROM p_new_high").fetchall():
    all_cand_pairs[sid].add(mid)

# Ensure all matches are in candidates
for sid, matches in s1_matches.items():
    for mid in matches:
        all_cand_pairs[sid].add(mid)

TEMP_CAND = os.path.join(OUTPUT_DIR, "candidate_pairs_v27_temp.tsv")
with open(TEMP_CAND, "w", encoding="utf-8") as f_out:
    f_out.write("source1_entity_id\tcandidate_entity_ids\n")
    for sid in row_order:
        cands = list(all_cand_pairs.get(sid, set()))
        if cands:
            f_out.write(f"{sid}\t{','.join(cands)}\n")
        else:
            f_out.write(f"{sid}\t\n")
os.replace(TEMP_CAND, CANDIDATES_TSV)
cand_size = os.path.getsize(CANDIDATES_TSV) / (1024 * 1024)
print(f"  candidate_pairs.tsv written ({cand_size:.1f} MB) in {time.time()-t7:.1f}s", flush=True)

# Step 8: Package zip
t8 = time.time()
print(f"\n[Step 8] Packaging {SUBMISSION_ZIP}...", flush=True)
with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(MATCHING_TSV, arcname="matching_results.tsv")
    zf.write(CANDIDATES_TSV, arcname="candidate_pairs.tsv")
zip_size = os.path.getsize(SUBMISSION_ZIP) / (1024 * 1024)
print(f"  Zip: {zip_size:.1f} MB in {time.time()-t8:.1f}s", flush=True)

# Step 9: Validate
print("\n" + "-" * 90, flush=True)
print("RUNNING OFFICIAL VALIDATOR...", flush=True)
cmd = [sys.executable, VALIDATOR_PY,
       "--matching", MATCHING_TSV, "--candidate", CANDIDATES_TSV,
       "--test-dir", TEST_DIR, "--check-ids"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout, flush=True)
if r.stderr: print("STDERR:", r.stderr, flush=True)

print("=" * 90, flush=True)
print(f"V27 COMPLETED in {time.time()-t0:.1f}s!", flush=True)
print(f"Key: Added {new_stats[0]:,} already-scored pairs at prob>=0.95 that V20's gate rejected!", flush=True)
print(f"Submission: {SUBMISSION_ZIP} ({zip_size:.1f} MB)", flush=True)
print("=" * 90, flush=True)
