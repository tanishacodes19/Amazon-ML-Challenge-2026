import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, time, duckdb
import polars as pl
from rapidfuzz import fuzz

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
OUTPUT_DIR = os.path.join(BASE, "output")
MATCHING_FILE = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUT_CLEAN = os.path.join(OUTPUT_DIR, "matching_results_v16_injective.tsv")

print("=" * 70)
print("POST-PROCESSING: S23 INJECTIVE CONSTRAINT (RESOLVING MULTI-CLAIMS)")
print("=" * 70)

t0 = time.time()
print("1. Loading matching_results.tsv...")
df = pl.read_csv(MATCHING_FILE, separator="\t", has_header=True)
print(f"   Loaded {len(df):,} S1 rows in {time.time()-t0:.1f}s")

# Unnest to pairs
print("2. Unnesting match pairs...")
t1 = time.time()
non_empty = df.filter(pl.col("matched_entity_ids") != "")
s1_list = non_empty["source1_entity_id"].to_list()
m_str_list = non_empty["matched_entity_ids"].to_list()

all_pairs = []
for sid, m_str in zip(s1_list, m_str_list):
    for mid in m_str.split(","):
        if mid:
            all_pairs.append((sid, mid))

print(f"   Total match pairs: {len(all_pairs):,} in {time.time()-t1:.1f}s")

# Count occurrences of each matched_entity_id
from collections import Counter
m_counts = Counter(mid for _, mid in all_pairs)
multi_m = {mid for mid, c in m_counts.items() if c > 1}
single_pairs = [(sid, mid) for sid, mid in all_pairs if mid not in multi_m]
conflict_pairs = [(sid, mid) for sid, mid in all_pairs if mid in multi_m]

print(f"   Unique S23 entities: {len(m_counts):,}")
print(f"   S23 entities with exactly 1 claim: {len(m_counts) - len(multi_m):,} (Pairs: {len(single_pairs):,})")
print(f"   S23 entities with >1 claims: {len(multi_m):,} (Conflicting Pairs: {len(conflict_pairs):,})")

# Load normalized data for conflict resolution
print("\n3. Loading entity text for resolving conflicting pairs...")
t2 = time.time()
NORM_DIR = os.path.join(BASE, "normalized_data")

conflict_s1 = {sid for sid, _ in conflict_pairs}
conflict_s23 = multi_m

con = duckdb.connect()
con.execute(f"""
CREATE TEMP TABLE s1_txt AS
SELECT entity_id, COALESCE(name_normalized, '') as name, COALESCE(address_normalized, '') as addr
FROM read_csv('{os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")}', delim='\t', header=true)
WHERE entity_id IN (SELECT unnest($1));
""", [list(conflict_s1)])

con.execute(f"""
CREATE TEMP TABLE s23_txt AS
SELECT entity_id, COALESCE(name_normalized, '') as name, COALESCE(address_normalized, '') as addr
FROM (
    SELECT entity_id, name_normalized, address_normalized FROM read_csv('{os.path.join(NORM_DIR, "test_source2_normalized.tsv").replace("\\", "/")}', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, name_normalized, address_normalized FROM read_csv('{os.path.join(NORM_DIR, "test_source3_normalized.tsv").replace("\\", "/")}', delim='\t', header=true)
) WHERE entity_id IN (SELECT unnest($1));
""", [list(conflict_s23)])

s1_dict = {r[0]: (r[1], r[2]) for r in con.execute("SELECT entity_id, name, addr FROM s1_txt").fetchall()}
s23_dict = {r[0]: (r[1], r[2]) for r in con.execute("SELECT entity_id, name, addr FROM s23_txt").fetchall()}
print(f"   Loaded {len(s1_dict):,} S1 texts and {len(s23_dict):,} S23 texts in {time.time()-t2:.1f}s")

# Resolve conflicts by highest text similarity
print("4. Scoring and resolving conflicts by maximal similarity...")
t3 = time.time()
best_for_m = {} # mid -> (best_score, sid)

for sid, mid in conflict_pairs:
    s1_t = s1_dict.get(sid, ("", ""))
    m_t = s23_dict.get(mid, ("", ""))
    nr = fuzz.ratio(s1_t[0], m_t[0]) / 100.0 if s1_t[0] and m_t[0] else 0.0
    ar = fuzz.ratio(s1_t[1], m_t[1]) / 100.0 if s1_t[1] and m_t[1] else 0.0
    sim = 0.50 * nr + 0.50 * ar
    
    if mid not in best_for_m or sim > best_for_m[mid][0]:
        best_for_m[mid] = (sim, sid)

resolved_pairs = [(sid, mid) for mid, (sim, sid) in best_for_m.items()]
print(f"   Resolved {len(multi_m):,} conflicts into {len(resolved_pairs):,} optimal pairs in {time.time()-t3:.1f}s")

# Combine single_pairs + resolved_pairs
final_pairs = single_pairs + resolved_pairs
print(f"\n5. Final strictly injective matches: {len(final_pairs):,} (eliminated {len(conflict_pairs) - len(resolved_pairs):,} false duplicate claims!)")

# Group by S1
from collections import defaultdict
s1_to_matches = defaultdict(list)
for sid, mid in final_pairs:
    s1_to_matches[sid].append(mid)

print("6. Formatting and saving to output/matching_results_v16_injective.tsv...")
t4 = time.time()
S1_RAW = r"D:\student_resource\student_resource\dataset\test\test_source1.tsv".replace("\\", "/")
s1_raw_df = pl.read_csv(S1_RAW, separator="\t", has_header=True)
all_s1_ids = s1_raw_df["entity_id"].to_list()

clean_rows = []
for sid in all_s1_ids:
    m_list = s1_to_matches.get(sid, [])
    clean_rows.append({
        "source1_entity_id": sid,
        "matched_entity_ids": ",".join(m_list)
    })

clean_df = pl.DataFrame(clean_rows)
clean_df.write_csv(OUT_CLEAN, separator="\t")
print(f"   Exported {len(clean_df):,} rows in {time.time()-t4:.1f}s!")
print(f"\nTotal process completed in {time.time()-t0:.1f}s.")
