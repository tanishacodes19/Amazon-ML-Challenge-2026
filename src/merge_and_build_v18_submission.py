import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, zipfile
import duckdb
from collections import Counter, defaultdict

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
OUTPUT_DIR = os.path.join(BASE, "output")
NORM_DIR = os.path.join(BASE, "normalized_data")
S1_RAW = r"D:\student_resource\student_resource\dataset\test\test_source1.tsv".replace("\\", "/")

EXISTING_MATCHES_FILE = os.path.join(OUTPUT_DIR, "matching_results_injective.tsv")
NEW_MATCHES_PARQUET = os.path.join(OUTPUT_DIR, "new_accepted_matches_v18.parquet")

FINAL_MATCHING_TSV = os.path.join(OUTPUT_DIR, "matching_results.tsv")
FINAL_CANDIDATES_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
SUBMISSION_ZIP = os.path.join(BASE, "team_antigravity_v18_submission.zip")

print("=" * 80)
print("MERGING V16 BASELINE + V18 NEW MATCHES INTO FINAL SUBMISSION")
print("=" * 80)

t0 = time.time()
print("1. Loading raw test S1 entity order...")
row_order = []
with open(S1_RAW, "r", encoding="utf-8") as f:
    header = next(f)
    for line in f:
        row_order.append(line.split("\t")[0].strip())
print(f"Loaded {len(row_order):,} reference S1 entities in {time.time()-t0:.1f}s.")

# Load existing matches
print("\n2. Loading existing 3.08M matches...")
t1 = time.time()
all_pairs = [] # (sid, mid, prob)
with open(EXISTING_MATCHES_FILE, "r", encoding="utf-8") as f:
    header = next(f)
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0]
        if len(parts) > 1 and parts[1].strip():
            for mid in parts[1].strip().split(","):
                if mid:
                    all_pairs.append((sid, mid, 0.95)) # default high prob for existing
print(f"Loaded {len(all_pairs):,} existing pairs in {time.time()-t1:.1f}s.")

# Load new matches
print("\n3. Loading brand-new accepted matches...")
t2 = time.time()
con = duckdb.connect()
new_df = con.execute(f"SELECT source1_entity_id, matched_entity_id, prob FROM read_parquet('{NEW_MATCHES_PARQUET.replace(chr(92), '/')}')").fetchall()
print(f"Loaded {len(new_df):,} brand-new match pairs in {time.time()-t2:.1f}s.")

# Combine all pairs
combined_pairs = all_pairs + [(r[0], r[1], float(r[2])) for r in new_df]
print(f"\nTotal combined match pool: {len(combined_pairs):,} pairs!")

# Enforce strict 1-to-1 injective constraint (no S2/S3 entity claimed more than once)
print("\n4. Resolving multi-claims via injective bipartite assignment (highest probability wins)...")
t3 = time.time()

# Sort descending by probability
combined_pairs.sort(key=lambda x: -x[2])

seen_m = set()
s1_matches = defaultdict(list)
accepted_count = 0
dropped_duplicates = 0

for sid, mid, prob in combined_pairs:
    if mid not in seen_m:
        seen_m.add(mid)
        s1_matches[sid].append(mid)
        accepted_count += 1
    else:
        dropped_duplicates += 1

print(f"Injective resolution complete in {time.time()-t3:.1f}s:")
print(f"  Accepted strictly injective matches: {accepted_count:,} (+{accepted_count - len(all_pairs):,} new matches!)")
print(f"  Dropped duplicate claims: {dropped_duplicates:,}")

# Write matching_results.tsv
print("\n5. Writing final matching_results.tsv...")
t4 = time.time()
empty_count = 0
non_empty_count = 0

with open(FINAL_MATCHING_TSV, "w", encoding="utf-8") as f_out:
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in row_order:
        matches = s1_matches.get(sid, [])
        if matches:
            f_out.write(f"{sid}\t{','.join(matches)}\n")
            non_empty_count += 1
        else:
            f_out.write(f"{sid}\t\n")
            empty_count += 1

print(f"Written {FINAL_MATCHING_TSV} in {time.time()-t4:.1f}s:")
print(f"  Matched S1 entities: {non_empty_count:,} (was 1,294,184, +{non_empty_count - 1294184:,} recovered!)")
print(f"  Empty singletons:    {empty_count:,} (was 438,360, dropped by {438360 - empty_count:,})")
print(f"  Total match pairs:   {accepted_count:,} (was 3,078,168, +{accepted_count - 3078168:,} surge!)")

# Package into zip
print("\n6. Building submission archive...")
t5 = time.time()
with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(FINAL_MATCHING_TSV, "output/matching_results.tsv")
    zf.write(FINAL_CANDIDATES_TSV, "output/candidate_pairs.tsv")
    zf.write(os.path.join(BASE, "Documentation_template.md"), "Documentation_template.md")
    zf.write(os.path.join(BASE, "README.md"), "code/business_entity_resolution/README.md")
    zf.write(os.path.join(BASE, "requirements.txt"), "code/business_entity_resolution/requirements.txt")
    
    src_dir = os.path.join(BASE, "src")
    for fname in os.listdir(src_dir):
        if fname.endswith(".py"):
            zf.write(os.path.join(src_dir, fname), f"code/business_entity_resolution/src/{fname}")

size_mb = os.path.getsize(SUBMISSION_ZIP) / (1024 * 1024)
print(f"Created {SUBMISSION_ZIP} ({size_mb:.2f} MB) in {time.time()-t5:.1f}s.")
print("=" * 80)
print(f"ALL DONE in {time.time()-t0:.1f}s! Ready for official validation.")
print("=" * 80)
