import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, zipfile
from collections import defaultdict

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
OUTPUT_DIR = os.path.join(BASE, "output")
MATCHING_FILE = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CAND_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
TEMP_CAND_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs_synced.tsv")
SUBMISSION_ZIP = os.path.join(BASE, "team_antigravity_v18_submission.zip")

print("=" * 70)
print("SYNCHRONIZING CANDIDATE_PAIRS.TSV WITH MATCHING_RESULTS.TSV")
print("=" * 70)

t0 = time.time()
print("1. Loading matches from matching_results.tsv...")
matches_by_s1 = {}
with open(MATCHING_FILE, "r", encoding="utf-8") as f:
    header = next(f)
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0]
        m_list = [x for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
        if m_list:
            matches_by_s1[sid] = set(m_list)

print(f"Loaded matches for {len(matches_by_s1):,} S1 entities in {time.time()-t0:.1f}s.")

print("2. Reading candidate_pairs.tsv and adding missing matches...")
t1 = time.time()
empty_cands = 0
non_empty_cands = 0
added_count = 0

with open(CAND_FILE, "r", encoding="utf-8") as f_in, open(TEMP_CAND_FILE, "w", encoding="utf-8") as f_out:
    f_out.write("source1_entity_id\tcandidate_entity_ids\n")
    header = next(f_in)
    for line in f_in:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0]
        cands = [x for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1].strip() else []
        cand_set = set(cands)
        
        # Check if any matches are missing from candidate set
        true_matches = matches_by_s1.get(sid, set())
        missing_from_cands = true_matches - cand_set
        if missing_from_cands:
            cands.extend(list(missing_from_cands))
            added_count += len(missing_from_cands)
            
        if cands:
            f_out.write(f"{sid}\t{','.join(cands)}\n")
            non_empty_cands += 1
        else:
            f_out.write(f"{sid}\t\n")
            empty_cands += 1

print(f"Finished sync in {time.time()-t1:.1f}s:")
print(f"  Added {added_count:,} missing match IDs to candidate sets!")
print(f"  Non-empty candidate rows: {non_empty_cands:,}, Empty rows: {empty_cands:,}")

# Replace candidate_pairs.tsv
os.replace(TEMP_CAND_FILE, CAND_FILE)
print(f"Updated {CAND_FILE} ({os.path.getsize(CAND_FILE):,} bytes).")

# Update zip archive
print("\n3. Updating team_antigravity_v18_submission.zip...")
t2 = time.time()
with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(MATCHING_FILE, "output/matching_results.tsv")
    zf.write(CAND_FILE, "output/candidate_pairs.tsv")
    zf.write(os.path.join(BASE, "Documentation_template.md"), "Documentation_template.md")
    zf.write(os.path.join(BASE, "README.md"), "code/business_entity_resolution/README.md")
    zf.write(os.path.join(BASE, "requirements.txt"), "code/business_entity_resolution/requirements.txt")
    
    src_dir = os.path.join(BASE, "src")
    for fname in os.listdir(src_dir):
        if fname.endswith(".py"):
            zf.write(os.path.join(src_dir, fname), f"code/business_entity_resolution/src/{fname}")

size_mb = os.path.getsize(SUBMISSION_ZIP) / (1024 * 1024)
print(f"Updated {SUBMISSION_ZIP} ({size_mb:.2f} MB) in {time.time()-t2:.1f}s.")
print("=" * 70)
