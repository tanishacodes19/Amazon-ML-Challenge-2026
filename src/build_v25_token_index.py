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
V20_CANDS_TSV = os.path.join(BASE, "scratch", "v20_extracted", "candidate_pairs.tsv")
S1_RAW = r"D:\student_resource\student_resource\dataset\test\test_source1.tsv"
SUBMISSION_ZIP = os.path.join(BASE, "team_antigravity_final_v25_submission.zip")
VALIDATOR_PY = r"D:\student_resource\student_resource\utils\validate_submission.py"
TEST_DIR = r"D:\student_resource\student_resource\dataset\test"

os.makedirs(r"D:\duckdb_temp", exist_ok=True)
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='8GB'")
con.execute("SET temp_directory='D:/duckdb_temp'")
con.execute("PRAGMA max_temp_directory_size='500GiB'")

print("=" * 90, flush=True)
print("V25 — TOKEN INVERTED INDEX BLOCKING (Expanding Recall Beyond 57.85%)", flush=True)
print("Strategy: IDF-weighted distinctive token retrieval + ML gating", flush=True)
print("=" * 90, flush=True)

t0 = time.time()

STOP_WORDS = {'the','and','for','ltd','pvt','inc','corp','llc','company','limited',
              'private','services','solutions','enterprises','center','centre','group',
              'india','us','usa','co','of','a','an','in','at','to','by','on','is',
              'de','la','el','le','les','das','der','die','und','van','von','et',
              'au','du','en','sa','srl','bv','nv','ag','gmbh','sl','spa','sas'}

STOP_SQL = "('" + "','".join(sorted(STOP_WORDS)) + "')"

ADDR_STOP = "('street','avenue','road','floor','building','opposite','near','block'," \
            "'colony','nagar','sector','pennsylvania','california','texas','maharashtra'," \
            "'karnataka','delhi','haryana','tamil','nadu','kerala','bengal','pradesh'," \
            "'mumbai','bangalore','chennai','kolkata','hyderabad','pune','ahmedabad')"

# ---- Step 1: Load V20 baseline ----
print("\n[Step 1] Loading V20 (0.769) baseline...", flush=True)
con.execute(f"""
CREATE TEMP TABLE p_v20 AS
SELECT source1_entity_id,
       UNNEST(STR_SPLIT(matched_entity_ids, ',')) as matched_entity_id,
       0.9999 as prob
FROM read_csv('{V20_TSV.replace(chr(92), '/')}', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';
""")
n_v20 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_v20").fetchone()
print(f"  V20: {n_v20[0]:,} pairs, {n_v20[1]:,} S1 entities ({time.time()-t0:.1f}s)", flush=True)

# ---- Step 2: Load normalized test data ----
t2 = time.time()
print("\n[Step 2] Loading normalized test data...", flush=True)
con.execute("""
CREATE TEMP TABLE s1_tbl AS
SELECT entity_id AS s1_id,
       name_normalized AS n1,
       address_normalized AS a1,
       country,
       COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS postal_code,
       LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_num
FROM read_csv('normalized_data/test_source1_normalized.tsv', delim='\t', header=true);

CREATE TEMP TABLE s23_tbl AS
SELECT entity_id AS m_id,
       name_normalized AS n2,
       address_normalized AS a2,
       country,
       COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS postal_code,
       LTRIM(COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), ''), '0') AS house_num
FROM (
    SELECT entity_id, name_normalized, address_normalized, country
    FROM read_csv('normalized_data/test_source2_normalized.tsv', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, name_normalized, address_normalized, country
    FROM read_csv('normalized_data/test_source3_normalized.tsv', delim='\t', header=true)
);
""")
s1_cnt = con.execute("SELECT COUNT(*) FROM s1_tbl").fetchone()[0]
s23_cnt = con.execute("SELECT COUNT(*) FROM s23_tbl").fetchone()[0]
print(f"  S1: {s1_cnt:,}, S23: {s23_cnt:,} loaded in {time.time()-t2:.1f}s.", flush=True)

# ---- Step 3: Ultra-precision rules (V23 certified >= 99.37%) ----
t3 = time.time()
print("\n[Step 3] Mining certified ultra-precision rules (>=99.37%)...", flush=True)
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
print(f"  Ultra rules: {u_stats[0]:,} pairs, {u_stats[1]:,} S1 entities ({time.time()-t3:.1f}s).", flush=True)

# ---- Step 4: ML model candidates ----
t4 = time.time()
print("\n[Step 4] Loading ML candidates (prob >= 0.60)...", flush=True)
con.execute("""
CREATE TEMP TABLE p_model AS
SELECT source1_entity_id, matched_entity_id, probability AS prob
FROM read_csv('test_scored_candidates.tsv', delim='\t', header=true)
WHERE probability >= 0.60;
""")
m_stats = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM p_model").fetchone()
print(f"  ML candidates: {m_stats[0]:,} pairs, {m_stats[1]:,} S1 entities ({time.time()-t4:.1f}s).", flush=True)

# ---- Step 5: TOKEN INVERTED INDEX — THE KEY TO FIXING BLOCKING RECALL ----
t5 = time.time()
print("\n[Step 5] *** TOKEN INVERTED INDEX BLOCKING (fixing the 57.85% recall ceiling) ***", flush=True)

# 5a. Build token -> document frequency table (to compute IDF)
print("  [5a] Computing token document frequencies across S23...", flush=True)
con.execute(f"""
CREATE TEMP TABLE s23_name_tokens AS
SELECT m_id, country,
       UNNEST(LIST_FILTER(STR_SPLIT(n2, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_SQL})) AS tok
FROM s23_tbl;

CREATE TEMP TABLE s1_name_tokens AS
SELECT s1_id, country,
       UNNEST(LIST_FILTER(STR_SPLIT(n1, ' '), x -> LENGTH(x) >= 4 AND x NOT IN {STOP_SQL})) AS tok
FROM s1_tbl;
""")

# 5b. Compute document frequency for each token (how many S23 docs contain it)
print("  [5b] Computing token document frequencies...", flush=True)
con.execute("""
CREATE TEMP TABLE tok_df AS
SELECT country, tok, COUNT(DISTINCT m_id) AS df
FROM s23_name_tokens
GROUP BY country, tok;
""")

# 5c. The magic: identify S1 entities NOT yet in V20/ultra/model
con.execute("""
CREATE TEMP TABLE already_matched AS
SELECT DISTINCT source1_entity_id FROM (
    SELECT source1_entity_id FROM p_v20
    UNION ALL SELECT source1_entity_id FROM p_ultra
    UNION ALL SELECT source1_entity_id FROM p_model
);
CREATE TEMP TABLE s1_unmatched_ids AS
SELECT s1_id FROM s1_tbl
WHERE s1_id NOT IN (SELECT source1_entity_id FROM already_matched);
""")
n_unm = con.execute("SELECT COUNT(*) FROM s1_unmatched_ids").fetchone()[0]
print(f"  S1 entities with NO matches yet: {n_unm:,}", flush=True)

# 5d. Token inverted index: For each unmatched S1, find S23 that share distinctive tokens
# Key insight: use low-df tokens (appear in <= 200 docs) — these are distinctive
print("  [5c] Building token inverted index candidates (df <= 200)...", flush=True)
t5c = time.time()
con.execute(f"""
CREATE TEMP TABLE cand_token AS
WITH
-- Only use unmatched S1 tokens
s1_tok_filtered AS (
    SELECT s1t.s1_id, s1t.country, s1t.tok
    FROM s1_name_tokens s1t
    INNER JOIN s1_unmatched_ids u ON s1t.s1_id = u.s1_id
    -- Only use distinctive tokens (not common words, df <= 200 in S23)
    INNER JOIN tok_df td ON s1t.country = td.country AND s1t.tok = td.tok
    WHERE td.df <= 200 AND td.df >= 1
),
-- Join with S23 on matching distinctive token + same country
raw_pairs AS (
    SELECT s1f.s1_id AS source1_entity_id,
           s23t.m_id AS matched_entity_id,
           COUNT(*) AS shared_toks   -- number of shared distinctive tokens
    FROM s1_tok_filtered s1f
    JOIN s23_name_tokens s23t
      ON s1f.country = s23t.country AND s1f.tok = s23t.tok
    GROUP BY s1f.s1_id, s23t.m_id
),
-- Require at least 2 shared distinctive tokens for higher precision
filtered_pairs AS (
    SELECT source1_entity_id, matched_entity_id, shared_toks
    FROM raw_pairs
    WHERE shared_toks >= 2
)
SELECT source1_entity_id, matched_entity_id,
       -- Score based on number of shared tokens
       CASE WHEN shared_toks >= 4 THEN 0.82
            WHEN shared_toks >= 3 THEN 0.77
            ELSE 0.72
       END AS prob
FROM filtered_pairs
QUALIFY COUNT(*) OVER (PARTITION BY source1_entity_id) <= 15;
""")
cnt_tok = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_token").fetchone()
print(f"  Token index (2+ shared distinctive tokens): {cnt_tok[0]:,} pairs, {cnt_tok[1]:,} S1 covered ({time.time()-t5c:.1f}s)", flush=True)

# 5e. Single highly distinctive token (df <= 5) — nearly unique business names
print("  [5d] Ultra-distinctive single token candidates (df <= 5)...", flush=True)
t5d = time.time()
con.execute(f"""
CREATE TEMP TABLE cand_ultra_tok AS
WITH
s1_rare AS (
    SELECT s1t.s1_id, s1t.country, s1t.tok
    FROM s1_name_tokens s1t
    INNER JOIN s1_unmatched_ids u ON s1t.s1_id = u.s1_id
    INNER JOIN tok_df td ON s1t.country = td.country AND s1t.tok = td.tok
    WHERE td.df <= 5 AND LENGTH(s1t.tok) >= 6  -- rare AND long = very distinctive
),
raw_rare AS (
    SELECT s1r.s1_id AS source1_entity_id,
           s23t.m_id AS matched_entity_id,
           0.80 AS prob
    FROM s1_rare s1r
    JOIN s23_name_tokens s23t
      ON s1r.country = s23t.country AND s1r.tok = s23t.tok
)
SELECT source1_entity_id, matched_entity_id, MAX(prob) AS prob
FROM raw_rare
GROUP BY source1_entity_id, matched_entity_id
HAVING COUNT(*) >= 1  -- even 1 ultra-rare token is very convincing
QUALIFY COUNT(*) OVER (PARTITION BY source1_entity_id) <= 10;
""")
cnt_rare = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_ultra_tok").fetchone()
print(f"  Ultra-rare single token: {cnt_rare[0]:,} pairs, {cnt_rare[1]:,} S1 covered ({time.time()-t5d:.1f}s)", flush=True)

# 5f. Cross-field token matching: S1 name tokens matched to S23 address tokens (for alt-name entries)
print("  [5e] Cross-field: S1 name token in S23 address + S23 name prefix 3...", flush=True)
t5e = time.time()
con.execute(f"""
CREATE TEMP TABLE s23_addr_tokens AS
SELECT m_id, country,
       UNNEST(LIST_FILTER(STR_SPLIT(a2, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {STOP_SQL})) AS tok
FROM s23_tbl;

-- Postal code + name prefix 5 (for matched S1 too, as extra signal)
CREATE TEMP TABLE cand_postal_n5 AS
WITH s1_base AS (
    SELECT st.s1_id, st.country, st.postal_code, st.n1
    FROM s1_tbl st
    INNER JOIN s1_unmatched_ids u ON st.s1_id = u.s1_id
    WHERE LENGTH(st.postal_code) >= 5 AND LENGTH(st.n1) >= 5
),
s1_pn AS (
    SELECT s1_id, country, postal_code || '_' || SUBSTRING(n1, 1, 5) AS k
    FROM s1_base
),
s23_pn AS (
    SELECT m_id, country, postal_code || '_' || SUBSTRING(n2, 1, 5) AS k
    FROM s23_tbl WHERE LENGTH(postal_code) >= 5 AND LENGTH(n2) >= 5
),
vk AS (SELECT country, k FROM s1_pn GROUP BY country, k HAVING COUNT(*) <= 20)
SELECT DISTINCT s1.s1_id AS source1_entity_id, s23.m_id AS matched_entity_id, 0.73 AS prob
FROM s1_pn s1
JOIN vk ON s1.country = vk.country AND s1.k = vk.k
JOIN s23_pn s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.s1_id) <= 10;
""")
cnt_pn = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_postal_n5").fetchone()
print(f"  Postal+name-pfx-5: {cnt_pn[0]:,} pairs, {cnt_pn[1]:,} S1 covered ({time.time()-t5e:.1f}s)", flush=True)

# 5g. House number + name prefix 5 (highly precise combo)
print("  [5f] House number + name prefix 5...", flush=True)
t5f = time.time()
con.execute("""
CREATE TEMP TABLE cand_hn5 AS
WITH s1_hn_base AS (
    SELECT st.s1_id, st.country, st.house_num, st.n1
    FROM s1_tbl st
    INNER JOIN s1_unmatched_ids u ON st.s1_id = u.s1_id
    WHERE LENGTH(st.house_num) >= 1 AND LENGTH(st.n1) >= 5
),
s1_hn AS (
    SELECT s1_id, country, house_num || '_' || SUBSTRING(n1, 1, 5) AS k
    FROM s1_hn_base
),
s23_hn AS (
    SELECT m_id, country, house_num || '_' || SUBSTRING(n2, 1, 5) AS k
    FROM s23_tbl WHERE LENGTH(house_num) >= 1 AND LENGTH(n2) >= 5
),
vk AS (SELECT country, k FROM s1_hn GROUP BY country, k HAVING COUNT(*) <= 20)
SELECT DISTINCT s1.s1_id AS source1_entity_id, s23.m_id AS matched_entity_id, 0.76 AS prob
FROM s1_hn s1
JOIN vk ON s1.country = vk.country AND s1.k = vk.k
JOIN s23_hn s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.s1_id) <= 10;
""")
cnt_hn5 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_hn5").fetchone()
print(f"  House#+name-pfx-5: {cnt_hn5[0]:,} pairs, {cnt_hn5[1]:,} S1 covered ({time.time()-t5f:.1f}s)", flush=True)

# V24 channels (proven to work) for ALL S1 (not just unmatched)
print("  [5g] V24 fast channels (sorted addr words + house+street) for all S1...", flush=True)
t5g = time.time()
con.execute(f"""
CREATE TEMP TABLE cand_sorted_addr AS
WITH s1_top2 AS (
    SELECT s1_id AS source1_entity_id, country,
           CASE WHEN sw[1] < sw[2] THEN sw[1] || '_' || sw[2] ELSE sw[2] || '_' || sw[1] END AS k
    FROM (
        SELECT s1_id, country,
               LIST_SLICE(LIST_REVERSE_SORT(
                   LIST_FILTER(STR_SPLIT(a1, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')
               ), 1, 2) AS sw
        FROM s1_tbl
    ) WHERE LEN(sw) >= 2
),
s23_top2 AS (
    SELECT m_id AS matched_entity_id, country,
           CASE WHEN sw[1] < sw[2] THEN sw[1] || '_' || sw[2] ELSE sw[2] || '_' || sw[1] END AS k
    FROM (
        SELECT m_id, country,
               LIST_SLICE(LIST_REVERSE_SORT(
                   LIST_FILTER(STR_SPLIT(a2, ' '), x -> LENGTH(x) >= 5 AND x NOT IN {ADDR_STOP} AND x NOT SIMILAR TO '[0-9].*')
               ), 1, 2) AS sw
        FROM s23_tbl
    ) WHERE LEN(sw) >= 2
),
vk AS (SELECT country, k FROM s1_top2 GROUP BY country, k HAVING COUNT(*) <= 30)
SELECT DISTINCT s1.source1_entity_id, s23.matched_entity_id, 0.74 AS prob
FROM s1_top2 s1
JOIN vk ON s1.country = vk.country AND s1.k = vk.k
JOIN s23_top2 s23 ON s1.country = s23.country AND s1.k = s23.k
QUALIFY COUNT(*) OVER (PARTITION BY s1.source1_entity_id) <= 10;
""")
cnt_sa = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM cand_sorted_addr").fetchone()
print(f"  Sorted addr words (all S1): {cnt_sa[0]:,} pairs, {cnt_sa[1]:,} S1 covered ({time.time()-t5g:.1f}s)", flush=True)

print(f"\n  All new blocking channels generated in {time.time()-t5:.1f}s total.", flush=True)

# ---- Step 6: Merge ALL candidates ----
t6 = time.time()
print("\n[Step 6] Merging all candidate sources...", flush=True)
con.execute("""
CREATE TEMP TABLE all_candidates AS
SELECT source1_entity_id, matched_entity_id, MAX(prob) AS prob
FROM (
    SELECT source1_entity_id, matched_entity_id, prob FROM p_v20
    UNION ALL SELECT source1_entity_id, matched_entity_id, prob FROM p_ultra
    UNION ALL SELECT source1_entity_id, matched_entity_id, prob FROM p_model
    UNION ALL SELECT source1_entity_id, matched_entity_id, prob FROM cand_token
    UNION ALL SELECT source1_entity_id, matched_entity_id, prob FROM cand_ultra_tok
    UNION ALL SELECT source1_entity_id, matched_entity_id, prob FROM cand_postal_n5
    UNION ALL SELECT source1_entity_id, matched_entity_id, prob FROM cand_hn5
    UNION ALL SELECT source1_entity_id, matched_entity_id, prob FROM cand_sorted_addr
)
GROUP BY source1_entity_id, matched_entity_id;
""")
all_stats = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM all_candidates").fetchone()
print(f"  Total merged: {all_stats[0]:,} pairs, {all_stats[1]:,} S1 entities ({time.time()-t6:.1f}s).", flush=True)

# ---- Step 7: Injective bipartite resolution ----
t7 = time.time()
print("\n[Step 7] Strict 1-to-1 injective bipartite resolution...", flush=True)
con.execute("""
CREATE TEMP TABLE final_injective AS
WITH ranked AS (
    SELECT source1_entity_id, matched_entity_id, prob,
           ROW_NUMBER() OVER (PARTITION BY matched_entity_id ORDER BY prob DESC) AS rk
    FROM all_candidates
)
SELECT source1_entity_id, matched_entity_id, prob FROM ranked WHERE rk = 1;
""")
fin = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM final_injective").fetchone()
matched_s1 = fin[1]
empty_s1 = 1732544 - matched_s1

print(f"  Done in {time.time()-t7:.1f}s:", flush=True)
print(f"    Match Pairs:     {fin[0]:,}  (V20: {n_v20[0]:,},  +{fin[0]-n_v20[0]:,})", flush=True)
print(f"    Matched S1:      {matched_s1:,} ({matched_s1/1732544*100:.2f}%)  (V20: {n_v20[1]:,}, +{matched_s1-n_v20[1]:,})", flush=True)
print(f"    Empty/Singleton: {empty_s1:,} ({empty_s1/1732544*100:.2f}%)", flush=True)
print(f"    Avg matches/S1:  {fin[0]/matched_s1:.3f}  (GT expected: ~3.666)", flush=True)

# ---- Step 8: Write output ----
t8 = time.time()
print("\n[Step 8] Writing matching_results.tsv...", flush=True)
row_order = []
with open(S1_RAW, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        row_order.append(line.split("\t")[0].strip())

s1_matches = defaultdict(list)
cursor = con.execute("SELECT source1_entity_id, matched_entity_id FROM final_injective ORDER BY prob DESC")
for sid, mid in cursor.fetchall():
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
print(f"  Written: {non_empty:,} matched, {empty:,} empty ({time.time()-t8:.1f}s).", flush=True)

# ---- Step 9: Sync candidate_pairs.tsv ----
t9 = time.time()
print("\n[Step 9] Synchronizing candidate_pairs.tsv...", flush=True)
all_cand_pairs = defaultdict(set)

# Load V20 candidate pairs as base
with open(V20_CANDS_TSV, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0]
        cands = [x for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
        all_cand_pairs[sid].update(cands)

# Add all new channel candidates
cursor2 = con.execute("""
SELECT source1_entity_id, matched_entity_id FROM (
    SELECT source1_entity_id, matched_entity_id FROM p_ultra
    UNION ALL SELECT source1_entity_id, matched_entity_id FROM cand_token
    UNION ALL SELECT source1_entity_id, matched_entity_id FROM cand_ultra_tok
    UNION ALL SELECT source1_entity_id, matched_entity_id FROM cand_postal_n5
    UNION ALL SELECT source1_entity_id, matched_entity_id FROM cand_hn5
    UNION ALL SELECT source1_entity_id, matched_entity_id FROM cand_sorted_addr
)
""")
for sid, mid in cursor2.fetchall():
    all_cand_pairs[sid].add(mid)

# Ensure all matches are in candidates
for sid, matches in s1_matches.items():
    for mid in matches:
        all_cand_pairs[sid].add(mid)

TEMP_CAND = os.path.join(OUTPUT_DIR, "candidate_pairs_v25_temp.tsv")
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
print(f"  candidate_pairs.tsv written ({cand_size:.1f} MB) in {time.time()-t9:.1f}s.", flush=True)

# ---- Step 10: Package zip ----
t10 = time.time()
print(f"\n[Step 10] Packaging {SUBMISSION_ZIP}...", flush=True)
with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(MATCHING_TSV, arcname="matching_results.tsv")
    zf.write(CANDIDATES_TSV, arcname="candidate_pairs.tsv")
zip_size = os.path.getsize(SUBMISSION_ZIP) / (1024 * 1024)
print(f"  Zip: {zip_size:.1f} MB in {time.time()-t10:.1f}s.", flush=True)

# ---- Step 11: Official validation ----
print("\n" + "-" * 90, flush=True)
print("RUNNING OFFICIAL VALIDATOR...", flush=True)
cmd = [sys.executable, VALIDATOR_PY,
       "--matching", MATCHING_TSV, "--candidate", CANDIDATES_TSV,
       "--test-dir", TEST_DIR, "--check-ids"]
val_proc = subprocess.run(cmd, capture_output=True, text=True)
print(val_proc.stdout, flush=True)
if val_proc.stderr:
    print("STDERR:", val_proc.stderr, flush=True)

print("=" * 90, flush=True)
print(f"V25 TOKEN INVERTED INDEX PIPELINE COMPLETE in {time.time()-t0:.1f}s!", flush=True)
print(f"Submission: {SUBMISSION_ZIP} ({zip_size:.1f} MB)", flush=True)
print("=" * 90, flush=True)
