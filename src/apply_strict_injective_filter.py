import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, time, duckdb
from collections import Counter, defaultdict
from rapidfuzz import fuzz

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
OUTPUT_DIR = os.path.join(BASE, "output")
MATCHING_FILE = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUT_CLEAN = os.path.join(OUTPUT_DIR, "matching_results_injective.tsv")

print("=" * 70)
print("STRICT INJECTIVE FILTERING FOR MATCHING_RESULTS.TSV")
print("=" * 70)

t0 = time.time()
print("1. Scanning matching_results.tsv for duplicate S23 claims...")

all_pairs = []
row_order = []

with open(MATCHING_FILE, "r", encoding="utf-8") as f:
    header = next(f) # 'source1_entity_id\tmatched_entity_ids\n'
    for line in f:
        parts = line.rstrip("\n").split("\t")
        sid = parts[0]
        row_order.append(sid)
        if len(parts) > 1 and parts[1].strip():
            for mid in parts[1].strip().split(","):
                if mid:
                    all_pairs.append((sid, mid))

print(f"   Scanned {len(row_order):,} S1 rows, {len(all_pairs):,} total match pairs in {time.time()-t0:.1f}s")

# Identify multi-claimed S23
m_counts = Counter(mid for _, mid in all_pairs)
multi_m = {mid for mid, count in m_counts.items() if count > 1}
print(f"   Unique S23 entities: {len(m_counts):,}")
print(f"   S23 with >1 claims: {len(multi_m):,} (causing {sum(v-1 for v in m_counts.values() if v > 1):,} duplicate claims)")

# For multi-claimed S23, load normalized text to pick the best S1
print("\n2. Loading text for multi-claimed entities...")
t1 = time.time()
conflict_pairs = [(sid, mid) for sid, mid in all_pairs if mid in multi_m]
conflict_s1 = list({sid for sid, _ in conflict_pairs})
conflict_s23 = list(multi_m)

NORM_DIR = os.path.join(BASE, "normalized_data")
con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

con.execute(f"""
CREATE TEMP TABLE s1_txt AS
SELECT entity_id, COALESCE(name_normalized, '') as name, COALESCE(address_normalized, '') as addr
FROM read_csv('{os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")}', delim='\t', header=true)
WHERE entity_id IN (SELECT unnest($1));
""", [conflict_s1])

con.execute(f"""
CREATE TEMP TABLE s23_txt AS
SELECT entity_id, COALESCE(name_normalized, '') as name, COALESCE(address_normalized, '') as addr
FROM (
    SELECT entity_id, name_normalized, address_normalized FROM read_csv('{os.path.join(NORM_DIR, "test_source2_normalized.tsv").replace("\\", "/")}', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, name_normalized, address_normalized FROM read_csv('{os.path.join(NORM_DIR, "test_source3_normalized.tsv").replace("\\", "/")}', delim='\t', header=true)
) WHERE entity_id IN (SELECT unnest($1));
""", [conflict_s23])

s1_dict = {r[0]: (r[1], r[2]) for r in con.execute("SELECT entity_id, name, addr FROM s1_txt").fetchall()}
s23_dict = {r[0]: (r[1], r[2]) for r in con.execute("SELECT entity_id, name, addr FROM s23_txt").fetchall()}
print(f"   Loaded text in {time.time()-t1:.1f}s")

print("3. Disambiguating multi-claimed S23 entities...")
t2 = time.time()
best_s1_for_s23 = {} # mid -> (score, sid)
for sid, mid in conflict_pairs:
    s1_t = s1_dict.get(sid, ("", ""))
    m_t = s23_dict.get(mid, ("", ""))
    nr = fuzz.ratio(s1_t[0], m_t[0]) / 100.0 if s1_t[0] and m_t[0] else 0.0
    ar = fuzz.ratio(s1_t[1], m_t[1]) / 100.0 if s1_t[1] and m_t[1] else 0.0
    sim = 0.50 * nr + 0.50 * ar
    if mid not in best_s1_for_s23 or sim > best_s1_for_s23[mid][0]:
        best_s1_for_s23[mid] = (sim, sid)

# Set of winning (sid, mid) pairs for conflicts
winning_conflict_pairs = {(sid, mid) for mid, (score, sid) in best_s1_for_s23.items()}
print(f"   Disambiguated in {time.time()-t2:.1f}s")

# Build valid pairs lookup
# A pair (sid, mid) is accepted if: mid not in multi_m OR (sid, mid) in winning_conflict_pairs
s1_accepted_m = defaultdict(list)
for sid, mid in all_pairs:
    if mid not in multi_m:
        s1_accepted_m[sid].append(mid)
    elif (sid, mid) in winning_conflict_pairs:
        s1_accepted_m[sid].append(mid)

total_injective_matches = sum(len(l) for l in s1_accepted_m.values())
print(f"\nTotal Injective Matches: {total_injective_matches:,} (Eliminated {len(all_pairs) - total_injective_matches:,} illegal duplicate claims)")

print("4. Writing clean matching_results_injective.tsv in exact original order...")
t3 = time.time()
empty_count = 0
non_empty_count = 0

with open(OUT_CLEAN, "w", encoding="utf-8") as f_out:
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in row_order:
        matches = s1_accepted_m.get(sid, [])
        if matches:
            f_out.write(f"{sid}\t{','.join(matches)}\n")
            non_empty_count += 1
        else:
            f_out.write(f"{sid}\t\n")
            empty_count += 1

print(f"   Export complete in {time.time()-t3:.1f}s!")
print(f"   Rows: {empty_count + non_empty_count:,} ({empty_count:,} empty, {non_empty_count:,} non-empty)")
print(f"Total time: {time.time()-t0:.1f}s")
