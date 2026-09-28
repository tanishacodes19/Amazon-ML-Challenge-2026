import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, zipfile
import duckdb
from collections import defaultdict

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
OUTPUT_DIR = os.path.join(BASE, "output")
MATCHING_TSV = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATES_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
S1_RAW = r"D:\student_resource\student_resource\dataset\test\test_source1.tsv"
SUBMISSION_ZIP = os.path.join(BASE, "team_antigravity_final_v21_submission.zip")

os.makedirs(r"D:\duckdb_temp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")
con.execute("PRAGMA max_temp_directory_size='500GiB'")

print("=" * 80, flush=True)
print("AMAZON ML CHALLENGE 2026 — ULTIMATE V21 MASTER INJECTIVE SUBMISSION PIPELINE", flush=True)
print("Unlocking High-Purity Scored Candidates + Precision-Verified Empty S1 Rescues", flush=True)
print("=" * 80, flush=True)

t0 = time.time()

# 1. Load V20 baseline matches (prob = 0.999 to guarantee preservation of existing high-precision matches)
print("\n[Step 1/5] Loading V20 baseline matches...", flush=True)
con.execute(f"""
CREATE TEMP TABLE p_v20 AS
SELECT source1_entity_id, UNNEST(STR_SPLIT(matched_entity_ids, ',')) as matched_entity_id, 0.999 as prob
FROM read_csv('{MATCHING_TSV.replace(chr(92), '/')}', delim='\\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';
""")
n_v20 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_v20").fetchone()
print(f"  V20 baseline: {n_v20[0]:,} matches across {n_v20[1]:,} S1 entities in {time.time()-t0:.1f}s.", flush=True)

# 2. Load candidates from test_scored_candidates at prob >= 0.60 (94.1% precision benchmark)
t1 = time.time()
print("\n[Step 2/5] Loading high-confidence candidates (prob >= 0.60) from test_scored_candidates...", flush=True)
con.execute("""
CREATE TEMP TABLE p_scored AS
SELECT source1_entity_id, matched_entity_id, probability as prob
FROM read_csv('test_scored_candidates.tsv', delim='\\t', header=true)
WHERE probability >= 0.60;
""")
n_sc = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_scored").fetchone()
print(f"  Loaded {n_sc[0]:,} pairs covering {n_sc[1]:,} S1 entities in {time.time()-t1:.1f}s.", flush=True)

# 3. Load reference tables and identify remaining empty S1 entities
t2 = time.time()
print("\n[Step 3/5] Mining precision-verified rescues for remaining empty S1 entities...", flush=True)
con.execute("""
CREATE TEMP TABLE s1_all AS
SELECT entity_id as s1_id, country, name_normalized as n1, address_normalized as a1
FROM read_csv('normalized_data/test_source1_normalized.tsv', delim='\t', header=true);

CREATE TEMP TABLE s23_all AS
SELECT entity_id as m_id, country, name_normalized as n2, address_normalized as a2
FROM (
    SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/test_source2_normalized.tsv', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/test_source3_normalized.tsv', delim='\t', header=true)
);

CREATE TEMP TABLE covered_s1 AS
SELECT DISTINCT source1_entity_id FROM p_v20
UNION
SELECT DISTINCT source1_entity_id FROM p_scored;

CREATE TEMP TABLE empty_s1 AS
SELECT s.* FROM s1_all s
LEFT JOIN covered_s1 c ON s.s1_id = c.source1_entity_id
WHERE c.source1_entity_id IS NULL;
""")
n_empty = con.execute("SELECT COUNT(*) FROM empty_s1").fetchone()[0]
print(f"  Currently empty S1 entities needing rescue: {n_empty:,}.", flush=True)

# Generate High-Purity Rescues:
# Rule A: Exact Address (len >= 12) -> 95.12% Ground Truth precision
# Rule B: Name Prefix 6 + Addr Prefix 6 -> 88.76% Ground Truth precision
# Rule C: Exact Name (len >= 6) -> 85.80% Ground Truth precision
con.execute("""
CREATE TEMP TABLE p_rescues AS
-- Rule A: Exact Address (len >= 12)
SELECT e.s1_id as source1_entity_id, s.m_id as matched_entity_id, 0.95 as prob
FROM empty_s1 e JOIN s23_all s ON e.country = s.country AND e.a1 = s.a2
WHERE LENGTH(e.a1) >= 12
QUALIFY COUNT(*) OVER (PARTITION BY e.s1_id) <= 5

UNION ALL

-- Rule B: Name Prefix 6 + Addr Prefix 6
SELECT e.s1_id as source1_entity_id, s.m_id as matched_entity_id, 0.88 as prob
FROM empty_s1 e JOIN s23_all s ON e.country = s.country 
                               AND SUBSTRING(e.n1, 1, 6) = SUBSTRING(s.n2, 1, 6)
                               AND SUBSTRING(e.a1, 1, 6) = SUBSTRING(s.a2, 1, 6)
WHERE LENGTH(e.n1) >= 6 AND LENGTH(e.a1) >= 6
QUALIFY COUNT(*) OVER (PARTITION BY e.s1_id) <= 5

UNION ALL

-- Rule C: Exact Name (len >= 6)
SELECT e.s1_id as source1_entity_id, s.m_id as matched_entity_id, 0.85 as prob
FROM empty_s1 e JOIN s23_all s ON e.country = s.country AND e.n1 = s.n2
WHERE LENGTH(e.n1) >= 6
QUALIFY COUNT(*) OVER (PARTITION BY e.s1_id) <= 5;
""")
n_resc = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_rescues").fetchone()
print(f"  Generated {n_resc[0]:,} high-purity rescue pairs covering {n_resc[1]:,} previously empty S1 entities in {time.time()-t2:.1f}s!", flush=True)

# 4. Strict 1-to-1 Injective Bipartite Resolution
t3 = time.time()
print("\n[Step 4/5] Executing strict 1-to-1 injective bipartite resolution...", flush=True)
con.execute("""
CREATE TEMP TABLE all_candidates AS
SELECT source1_entity_id, matched_entity_id, MAX(prob) as prob
FROM (
    SELECT * FROM p_v20
    UNION ALL
    SELECT * FROM p_scored
    UNION ALL
    SELECT * FROM p_rescues
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

total_pairs, total_matched_s1 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM final_injective_matches").fetchone()
empty_count = 1732544 - total_matched_s1
print(f"  Injective resolution complete in {time.time()-t3:.1f}s:")
print(f"    Total Match Pairs:   {total_pairs:,} (was {n_v20[0]:,} in V20, +{total_pairs - n_v20[0]:,} surge!)", flush=True)
print(f"    Matched S1 Entities: {total_matched_s1:,} (was {n_v20[1]:,} in V20, +{total_matched_s1 - n_v20[1]:,} gained!)", flush=True)
print(f"    Empty Singletons:    {empty_count:,} (was 242,511 in V20, dropped by {242511 - empty_count:,})", flush=True)

# 5. Writing output files
t4 = time.time()
print("\n[Step 5/5] Writing matching_results.tsv and synchronizing candidate_pairs.tsv...", flush=True)

# Load reference row ordering
row_order = []
with open(S1_RAW, "r", encoding="utf-8") as f:
    header = next(f)
    for line in f:
        row_order.append(line.split("\t")[0].strip())

# Fetch all matches grouped by S1
s1_matches = defaultdict(list)
cursor = con.execute("SELECT source1_entity_id, matched_entity_id FROM final_injective_matches ORDER BY prob DESC")
for sid, mid in cursor.fetchall():
    s1_matches[sid].append(mid)

# Write matching_results.tsv
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
print(f"  matching_results.tsv written in {time.time()-t4:.1f}s ({non_empty:,} matched, {empty:,} empty).", flush=True)

# Synchronize candidate_pairs.tsv
t_sync = time.time()
TEMP_CAND = os.path.join(OUTPUT_DIR, "candidate_pairs_v21_temp.tsv")
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

print("\n" + "=" * 80, flush=True)
print(f"V21 ULTIMATE MASTER PIPELINE COMPLETED IN {time.time()-t0:.1f}s!", flush=True)
print("=" * 80, flush=True)
