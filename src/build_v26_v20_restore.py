import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os, zipfile, shutil, subprocess

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
OUTPUT_DIR = os.path.join(BASE, "output")
V20_MATCH = os.path.join(BASE, "scratch", "v20_extracted", "matching_results.tsv")
V20_CANDS = os.path.join(BASE, "scratch", "v20_extracted", "candidate_pairs.tsv")
MATCHING_TSV = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATES_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
SUBMISSION_ZIP = os.path.join(BASE, "team_antigravity_final_v26_v20restore.zip")
VALIDATOR_PY = r"D:\student_resource\student_resource\utils\validate_submission.py"
TEST_DIR = r"D:\student_resource\student_resource\dataset\test"

print("=" * 70, flush=True)
print("V26 — PURE V20 RESTORE (Guaranteed 0.769 baseline)", flush=True)
print("=" * 70, flush=True)

# Copy V20 files back to output/
print("Restoring V20 matching_results.tsv and candidate_pairs.tsv...", flush=True)
shutil.copy2(V20_MATCH, MATCHING_TSV)
shutil.copy2(V20_CANDS, CANDIDATES_TSV)
print(f"  matching_results.tsv: {os.path.getsize(MATCHING_TSV)/(1024*1024):.1f} MB", flush=True)
print(f"  candidate_pairs.tsv:  {os.path.getsize(CANDIDATES_TSV)/(1024*1024):.1f} MB", flush=True)

# Validate
print("\nRunning validator...", flush=True)
cmd = [sys.executable, VALIDATOR_PY,
       "--matching", MATCHING_TSV, "--candidate", CANDIDATES_TSV,
       "--test-dir", TEST_DIR, "--check-ids"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout, flush=True)
if r.stderr: print("STDERR:", r.stderr, flush=True)

# Package zip
print("Packaging zip...", flush=True)
with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(MATCHING_TSV, arcname="matching_results.tsv")
    zf.write(CANDIDATES_TSV, arcname="candidate_pairs.tsv")
print(f"  {SUBMISSION_ZIP}  ({os.path.getsize(SUBMISSION_ZIP)/(1024*1024):.1f} MB)", flush=True)
print("\nV26 = Pure V20 restore. Submit this for a guaranteed 0.769!", flush=True)
