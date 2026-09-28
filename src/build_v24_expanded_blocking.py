import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, zipfile, subprocess
import duckdb
from collections import defaultdict

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
OUTPUT_DIR = os.path.join(BASE, "output")
MATCHING_TSV = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATES_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
V20_TSV = os.path.join(BASE, "scratch", "v20_extracted", "matching_results.tsv")
S1_RAW = r"D:\student_resource\student_resource\dataset\test\test_source1.tsv"
SUBMISSION_ZIP = os.path.join(BASE, "team_antigravity_final_v24_submission.zip")
VALIDATOR_PY = r"D:\student_resource\student_resource\utils\validate_submission.py"
TEST_DIR = r"D:\student_resource\student_resource\dataset\test"

os.makedirs(r"D:\duckdb_temp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='6GB'")
con.execute("SET temp_directory='D:/duckdb_temp'")
con.execute("PRAGMA max_temp_directory_size='500GiB'")

print("=" * 90, flush=True)
print("AMAZON ML CHALLENGE 2026 — V24 EXPANDED BLOCKING SUBMISSION PIPELINE", flush=True)
print("Strategy: V20 baseline + Ultra Rules + ML + 5 New Fast Blocking Channels", flush=True)
print("=" * 90, flush=True)

t0 = time.time()

STOP_WORDS = "('the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'limited', 'private', 'services', 'solutions', 'enterprises', 'center', 'centre', 'group', 'india', 'us', 'usa', 'co', 'of')"
ADDR_STOP = "('street', 'avenue', 'road', 'floor', 'building', 'opposite', 'near', 'block', 'colony', 'nagar', 'sector', 'pennsylvania', 'california', 'texas', 'maharashtra', 'karnataka', 'delhi', 'haryana', 'tamil', 'nadu', 'kerala', 'bengal', 'pradesh', 'mumbai', 'bangalore', 'chennai', 'kolkata')"

# Step 1: Load V20 baseline
print("\n[Step 1] Loading V20 (0.769) baseline...", flush=True)
con.execute(f"""
CREATE TEMP TABLE p_v20 AS
SELECT source1_entity_id, UNNEST(STR_SPLIT(matched_entity_ids, ',')) as matched_entity_id, 0.9999 as prob
FROM read_csv('{V20_TSV.replace(chr(92), '/')}', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';
""")
n_v20 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_v20").fetchone()
print(f"  V20 baseline: {n_v20[0]:,} pairs, {n_v20[1]:,} S1 entities. ({time.time()-t0:.1f}s)", flush=True)

# Step 2: Load normalized test data
t2 = time.time()
print("\n[Step 2] Loading normalized test data...", flush=True)
con.execute("""
CREATE TEMP TABLE s1_tbl AS
SELECT entity_id AS source1_entity_id,
       name_normalized AS clean_name,
       address_normalized AS clean_addr,
       country,
       LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_number,
       COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS postal_code
FROM read_csv('normalized_data/test_source1_normalized.tsv', delim='\t', header=true);

CREATE TEMP TABLE s23_tbl AS
SELECT entity_id AS matched_entity_id,
       name_normalized AS clean_name,
       address_normalized AS clean_addr,
       country,
       LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_number,
       COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS postal_code
FROM (
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('normalized_data/test_source2_normalized.tsv', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('normalized_data/test_source3_normalized.tsv', delim='\t', header=true)
);
""")
print(f"  Tables loaded in {time.time()-t2:.1f}s.", flush=True)

# Step 3: Ultra-precision rules (same as V23, proven >=99.37%)
t3 = time.time()
print("\n[Step 3] Mining certified ultra rules (>=99.37% GT precision)...", flush=True)
con.execute("""
CREATE TEMP TABLE p_ultra AS
-- Rule 1: Exact Name + Exact Address (100.00% precision)
SELECT s1.source1_entity_id, s23.matched_entity_id, 0.9995 as prob
FROM s1_tbl s1 JOIN s23_tbl s23
  ON s1.country = s23.country AND s1.clean_name = s23.clean_name AND s1.clean_addr = s23.clean_addr
WHERE LENGTH(s1.clean_name) >= 4 AND LENGTH(s1.clean_addr) >= 4

UNION ALL

-- Rule 2: Exact Name + Addr Pfx 8 (99.37% precision)
SELECT s1.source1_entity_id, s23.matched_entity_id, 0.9990 as prob
FROM s1_tbl s1 JOIN s23_tbl s23
  ON s1.country = s23.country AND s1.clean_name = s23.clean_name
     AND SUBSTRING(s1.clean_addr, 1, 8) = SUBSTRING(s23.clean_addr, 1, 8)
WHERE LENGTH(s1.clean_name) >= 4 AND LENGTH(s1.clean_addr) >= 8

UNION ALL

-- Rule 3: Exact Addr + Name Pfx 8 (99.87% precision)
SELECT s1.source1_entity_id, s23.matched_entity_id, 0.9992 as prob
FROM s1_tbl s1 JOIN s23_tbl s23
  ON s1.country = s23.country AND s1.clean_addr = s23.clean_addr
     AND SUBSTRING(s1.clean_name, 1, 8) = SUBSTRING(s23.clean_name, 1, 8)
WHERE LENGTH(s1.clean_addr) >= 8 AND LENGTH(s1.clean_name) >= 8

UNION ALL

-- Rule 4: Exact Addr (len>=12) + Name Pfx 3 (99.48% precision)
SELECT s1.source1_entity_id, s23.matched_entity_id, 0.9988 as prob
FROM s1_tbl s1 JOIN s23_tbl s23
  ON s1.country = s23.country AND s1.clean_addr = s23.clean_addr
     AND SUBSTRING(s1.clean_name, 1, 3) = SUBSTRING(s23.clean_name, 1, 3)
WHERE LENGTH(s1.clean_addr) >= 12 AND LENGTH(s1.clean_name) >= 3;
""")
u_stats = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_ultra").fetchone()
print(f"  Ultra rules: {u_stats[0]:,} pairs, {u_stats[1]:,} S1 entities ({time.time()-t3:.1f}s).", flush=True)

# Step 4: ML model candidates at prob >= 0.60
t4 = time.time()
print("\n[Step 4] Loading ML candidates (prob >= 0.60)...", flush=True)
con.execute("""
CREATE TEMP TABLE p_model AS
SELECT source1_entity_id, matched_entity_id, probability as prob
FROM read_csv('test_scored_candidates.tsv', delim='\t', header=true)
WHERE probability >= 0.60;
""")
m_stats = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_model").fetchone()
print(f"  ML candidates: {m_stats[0]:,} pairs, {m_stats[1]:,} S1 entities ({time.time()-t4:.1f}s).", flush=True)

# Step 5: NEW FAST BLOCKING CHANNELS (to expand recall)
t5 = time.time()
print("\n[Step 5] Mining NEW blocking channels for expanded recall...", flush=True)

# First, identify which S1 entities currently have no matches
con.execute("""
CREATE TEMP TABLE already_matched AS
SELECT DISTINCT source1_entity_id FROM (
    SELECT source1_entity_id FROM p_v20
    UNION ALL
    SELECT source1_entity_id FROM p_ultra
    UNION ALL
    SELECT source1_entity_id FROM p_model
);

CREATE TEMP TABLE s1_unmatched AS
SELECT s.* FROM s1_tbl s
LEFT JOIN already_matched m ON s.source1_entity_id = m.source1_entity_id
WHERE m.source1_entity_id IS NULL;
""")
n_unm = con.execute("SELECT COUNT(*) FROM s1_unmatched").fetchone()[0]
print(f"  S1 entities with no matches yet: {n_unm:,}", flush=True)

# Channel A: Sorted 2 Distinctive Address Words (low fan-out, high precision)
print("  [Ch-A] Sorted 2 distinctive address words...", flush=True)
ta = time.time()
con.execute(f"""
CREATE TEMP TABLE cand_ch_a AS
WITH s1_top2 AS (
    SELECT source1_entity_id, country,
           CASE WHEN sorted_words[1] < sorted_words[2]
                THEN sorted_words[1] || '_' || sorted_words[2]
                ELSE sorted_words[2] || '_' || sorted_words[1] END as k
    FROM (
        SELECT source1_entity_id, country,
               LIST_SLICE(LIST_REVERSE_SORT(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')), 1, 2) as sorted_words
        FROM s1_tbl
    ) WHERE LEN(sorted_words) >= 2
),
s23_top2 AS (
    SELECT matched_entity_id, country,
           CASE WHEN sorted_words[1] < sorted_words[2]
                THEN sorted_words[1] || '_' || sorted_words[2]
                ELSE sorted_words[2] || '_' || sorted_words[1] END as k
    FROM (
        SELECT matched_entity_id, country,
               LIST_SLICE(LIST_REVERSE_SORT(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')), 1, 2) as sorted_words
        FROM s23_tbl
    ) WHERE LEN(sorted_words) >= 2
),
-- Only use keys that appear in <= 30 S1 entities per country (high precision)
vk AS (SELECT country, k FROM s1_top2 GROUP BY country, k HAVING COUNT(*) <= 30)
SELECT DISTINCT s1.source1_entity_id, s23.matched_entity_id, 0.75 as prob
FROM s1_top2 s1
JOIN vk ON s1.country = vk.country AND s1.k = vk.k
JOIN s23_top2 s23 ON s1.country = s23.country AND s1.k = s23.k
WHERE NOT EXISTS (SELECT 1 FROM already_matched am WHERE am.source1_entity_id = s1.source1_entity_id)
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 10;
""")
cnt_a = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_ch_a").fetchone()
print(f"    Ch-A: {cnt_a[0]:,} pairs, {cnt_a[1]:,} new S1 covered ({time.time()-ta:.1f}s)", flush=True)

# Channel B: House Number + Street Word (for unmatched S1)
print("  [Ch-B] House number + street word (unmatched S1)...", flush=True)
tb = time.time()
con.execute(f"""
CREATE TEMP TABLE cand_ch_b AS
WITH s1_hs AS (
    SELECT source1_entity_id, country, house_number || '_' || word as k
    FROM (
        SELECT source1_entity_id, country, house_number,
               UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as word
        FROM s1_unmatched
        WHERE LENGTH(house_number) >= 2
    )
),
s23_hs AS (
    SELECT matched_entity_id, country, house_number || '_' || word as k
    FROM (
        SELECT matched_entity_id, country, house_number,
               UNNEST(LIST_FILTER(STR_SPLIT(clean_addr, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')) as word
        FROM s23_tbl
        WHERE LENGTH(house_number) >= 2
    )
),
vk AS (SELECT country, k FROM s1_hs GROUP BY country, k HAVING COUNT(*) <= 30)
SELECT DISTINCT s1.source1_entity_id, s23.matched_entity_id, 0.72 as prob
FROM s1_hs s1
JOIN vk ON s1.country = vk.country AND s1.k = vk.k
JOIN s23_hs s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 10;
""")
cnt_b = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_ch_b").fetchone()
print(f"    Ch-B: {cnt_b[0]:,} pairs, {cnt_b[1]:,} new S1 covered ({time.time()-tb:.1f}s)", flush=True)

# Channel C: Postal Code + Name Prefix 4 (unmatched S1)
print("  [Ch-C] Postal code + name prefix 4 (unmatched S1)...", flush=True)
tc = time.time()
con.execute("""
CREATE TEMP TABLE cand_ch_c AS
WITH s1_pc AS (
    SELECT source1_entity_id, country, postal_code || '_' || SUBSTRING(clean_name, 1, 4) as k
    FROM s1_unmatched
    WHERE LENGTH(postal_code) >= 5 AND LENGTH(clean_name) >= 4
),
s23_pc AS (
    SELECT matched_entity_id, country, postal_code || '_' || SUBSTRING(clean_name, 1, 4) as k
    FROM s23_tbl
    WHERE LENGTH(postal_code) >= 5 AND LENGTH(clean_name) >= 4
),
vk AS (SELECT country, k FROM s1_pc GROUP BY country, k HAVING COUNT(*) <= 30)
SELECT DISTINCT s1.source1_entity_id, s23.matched_entity_id, 0.70 as prob
FROM s1_pc s1
JOIN vk ON s1.country = vk.country AND s1.k = vk.k
JOIN s23_pc s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 10;
""")
cnt_c = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_ch_c").fetchone()
print(f"    Ch-C: {cnt_c[0]:,} pairs, {cnt_c[1]:,} new S1 covered ({time.time()-tc:.1f}s)", flush=True)

# Channel D: Address prefix 12 chars (unmatched S1) — very precise
print("  [Ch-D] Address prefix 12 chars (unmatched S1)...", flush=True)
td = time.time()
con.execute("""
CREATE TEMP TABLE cand_ch_d AS
WITH s1_a12 AS (
    SELECT source1_entity_id, country, SUBSTRING(clean_addr, 1, 12) as k
    FROM s1_unmatched WHERE LENGTH(clean_addr) >= 12
),
s23_a12 AS (
    SELECT matched_entity_id, country, SUBSTRING(clean_addr, 1, 12) as k
    FROM s23_tbl WHERE LENGTH(clean_addr) >= 12
),
vk AS (SELECT country, k FROM s1_a12 GROUP BY country, k HAVING COUNT(*) <= 20)
SELECT DISTINCT s1.source1_entity_id, s23.matched_entity_id, 0.73 as prob
FROM s1_a12 s1
JOIN vk ON s1.country = vk.country AND s1.k = vk.k
JOIN s23_a12 s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 10;
""")
cnt_d = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_ch_d").fetchone()
print(f"    Ch-D: {cnt_d[0]:,} pairs, {cnt_d[1]:,} new S1 covered ({time.time()-td:.1f}s)", flush=True)

# Channel E: Distinctive brand word (len>=7) for unmatched S1, very low fan-out
print("  [Ch-E] Distinctive brand word (len>=7, unmatched S1)...", flush=True)
te = time.time()
con.execute(f"""
CREATE TEMP TABLE cand_ch_e AS
WITH s1_w7 AS (
    SELECT source1_entity_id, country,
           UNNEST(LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 7 AND x NOT IN {STOP_WORDS})) as w
    FROM s1_unmatched
),
s23_w7 AS (
    SELECT matched_entity_id, country,
           UNNEST(LIST_FILTER(STR_SPLIT(clean_name, ' '), x -> LENGTH(x) >= 7 AND x NOT IN {STOP_WORDS})) as w
    FROM s23_tbl
),
-- Only use words that appear in <= 20 S1 entities (very distinctive)
vk AS (SELECT country, w FROM s1_w7 GROUP BY country, w HAVING COUNT(*) <= 20)
SELECT DISTINCT s1.source1_entity_id, s23.matched_entity_id, 0.71 as prob
FROM s1_w7 s1
JOIN vk ON s1.country = vk.country AND s1.w = vk.w
JOIN s23_w7 s23 ON s1.country = s23.country AND s1.w = s23.w
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 10;
""")
cnt_e = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_ch_e").fetchone()
print(f"    Ch-E: {cnt_e[0]:,} pairs, {cnt_e[1]:,} new S1 covered ({time.time()-te:.1f}s)", flush=True)

print(f"\n  All new channels generated in {time.time()-t5:.1f}s total.", flush=True)

# Step 6: Merge ALL candidates and run ML scoring (if test_scored_candidates.tsv has scores for new pairs)
# Since ML model is pre-scored only on the original candidate set, new channel pairs get their channel prob
t6 = time.time()
print("\n[Step 6] Merging all candidate sources...", flush=True)
con.execute("""
CREATE TEMP TABLE all_candidates AS
SELECT source1_entity_id, matched_entity_id, MAX(prob) as prob
FROM (
    SELECT source1_entity_id, matched_entity_id, prob FROM p_v20
    UNION ALL
    SELECT source1_entity_id, matched_entity_id, prob FROM p_ultra
    UNION ALL
    SELECT source1_entity_id, matched_entity_id, prob FROM p_model
    UNION ALL
    SELECT source1_entity_id, matched_entity_id, prob FROM cand_ch_a
    UNION ALL
    SELECT source1_entity_id, matched_entity_id, prob FROM cand_ch_b
    UNION ALL
    SELECT source1_entity_id, matched_entity_id, prob FROM cand_ch_c
    UNION ALL
    SELECT source1_entity_id, matched_entity_id, prob FROM cand_ch_d
    UNION ALL
    SELECT source1_entity_id, matched_entity_id, prob FROM cand_ch_e
)
GROUP BY source1_entity_id, matched_entity_id;
""")
all_stats = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM all_candidates").fetchone()
print(f"  Total merged candidates: {all_stats[0]:,} pairs, {all_stats[1]:,} S1 entities ({time.time()-t6:.1f}s).", flush=True)

# Step 7: Strict 1-to-1 Injective Bipartite Resolution
t7 = time.time()
print("\n[Step 7] Strict 1-to-1 injective bipartite resolution...", flush=True)
con.execute("""
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
avg_m = fin_stats[0] / matched_s1 if matched_s1 > 0 else 0

print(f"  Injective resolution done in {time.time()-t7:.1f}s:", flush=True)
print(f"    Total Match Pairs:   {fin_stats[0]:,} (V20: {n_v20[0]:,}, delta: +{fin_stats[0] - n_v20[0]:,})", flush=True)
print(f"    Matched S1 Entities: {matched_s1:,} ({matched_s1/1732544*100:.2f}%) (V20: {n_v20[1]:,}, +{matched_s1 - n_v20[1]:,})", flush=True)
print(f"    Empty Singletons:    {empty_s1:,} ({empty_s1/1732544*100:.2f}%)", flush=True)
print(f"    Avg Matches per S1:  {avg_m:.3f}", flush=True)

# Step 8: Write output files
t8 = time.time()
print("\n[Step 8] Writing matching_results.tsv...", flush=True)
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
print(f"  matching_results.tsv written ({non_empty:,} matched, {empty:,} empty) in {time.time()-t8:.1f}s.", flush=True)

# Step 9: Synchronize candidate_pairs.tsv
t9 = time.time()
print("\n[Step 9] Synchronizing candidate_pairs.tsv...", flush=True)

# Build complete candidate set (matching + candidates)
# candidate_pairs must include all matching + additional from all channels
all_cand_pairs = defaultdict(set)

# First pass: load existing candidate_pairs from V20
V20_CANDS_TSV = os.path.join(BASE, "scratch", "v20_extracted", "candidate_pairs.tsv")
# Check if V20 candidate file exists
if os.path.exists(V20_CANDS_TSV):
    with open(V20_CANDS_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0]
            cands = [x for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
            all_cand_pairs[sid].update(cands)
    print(f"  Loaded V20 candidate pairs from {V20_CANDS_TSV}", flush=True)
else:
    # Fall back to current candidate_pairs.tsv
    with open(CANDIDATES_TSV, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0]
            cands = [x for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
            all_cand_pairs[sid].update(cands)
    print(f"  Loaded current candidate_pairs.tsv as base.", flush=True)

# Add all new channel candidates
cursor2 = con.execute("""
SELECT source1_entity_id, matched_entity_id FROM (
    SELECT source1_entity_id, matched_entity_id FROM cand_ch_a
    UNION ALL
    SELECT source1_entity_id, matched_entity_id FROM cand_ch_b
    UNION ALL
    SELECT source1_entity_id, matched_entity_id FROM cand_ch_c
    UNION ALL
    SELECT source1_entity_id, matched_entity_id FROM cand_ch_d
    UNION ALL
    SELECT source1_entity_id, matched_entity_id FROM cand_ch_e
    UNION ALL
    SELECT source1_entity_id, matched_entity_id FROM p_ultra
)
""")
added_cands = 0
for sid, mid in cursor2.fetchall():
    if mid not in all_cand_pairs[sid]:
        all_cand_pairs[sid].add(mid)
        added_cands += 1

# Ensure all matches are in candidates
for sid, matches in s1_matches.items():
    for mid in matches:
        if mid not in all_cand_pairs[sid]:
            all_cand_pairs[sid].add(mid)

# Write candidate_pairs.tsv
TEMP_CAND = os.path.join(OUTPUT_DIR, "candidate_pairs_v24_temp.tsv")
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
print(f"  candidate_pairs.tsv written ({cand_size:.1f} MB, +{added_cands:,} new IDs) in {time.time()-t9:.1f}s.", flush=True)

# Step 10: Package submission zip
t10 = time.time()
print(f"\n[Step 10] Packaging submission zip: {SUBMISSION_ZIP}...", flush=True)
with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(MATCHING_TSV, arcname="matching_results.tsv")
    zf.write(CANDIDATES_TSV, arcname="candidate_pairs.tsv")
zip_size = os.path.getsize(SUBMISSION_ZIP) / (1024 * 1024)
print(f"  Zip created: {zip_size:.1f} MB in {time.time()-t10:.1f}s.", flush=True)

# Step 11: Official validation
print("\n" + "-" * 90, flush=True)
print("RUNNING OFFICIAL VALIDATOR...", flush=True)
print("-" * 90, flush=True)
cmd = [
    sys.executable, VALIDATOR_PY,
    "--matching", MATCHING_TSV,
    "--candidate", CANDIDATES_TSV,
    "--test-dir", TEST_DIR,
    "--check-ids"
]
val_proc = subprocess.run(cmd, capture_output=True, text=True)
print(val_proc.stdout, flush=True)
if val_proc.stderr:
    print("STDERR:", val_proc.stderr, flush=True)

print("=" * 90, flush=True)
print(f"V24 EXPANDED BLOCKING PIPELINE COMPLETED IN {time.time()-t0:.1f}s!", flush=True)
print(f"Submission: {SUBMISSION_ZIP} ({zip_size:.1f} MB)", flush=True)
print("=" * 90, flush=True)
