import os
import sys
import zipfile
import time

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
ZIP_NAME = os.path.join(BASE, "team_antigravity_v16_submission.zip")

print("=" * 70)
print("BUILDING FINAL SUBMISSION ZIP PACKAGE")
print("=" * 70)

t0 = time.time()
with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
    # 1. output/ folder
    print("Adding output/matching_results.tsv...")
    zf.write(os.path.join(BASE, "output", "matching_results.tsv"), "output/matching_results.tsv")
    print("Adding output/candidate_pairs.tsv...")
    zf.write(os.path.join(BASE, "output", "candidate_pairs.tsv"), "output/candidate_pairs.tsv")
    
    # 2. Documentation_template.md
    print("Adding Documentation_template.md...")
    zf.write(os.path.join(BASE, "Documentation_template.md"), "Documentation_template.md")
    
    # 3. code/business_entity_resolution/
    print("Adding code/business_entity_resolution/README.md & requirements.txt...")
    zf.write(os.path.join(BASE, "README.md"), "code/business_entity_resolution/README.md")
    zf.write(os.path.join(BASE, "requirements.txt"), "code/business_entity_resolution/requirements.txt")
    
    # Add key src files
    src_dir = os.path.join(BASE, "src")
    for f in os.listdir(src_dir):
        if f.endswith(".py"):
            print(f"  Adding src/{f}...")
            zf.write(os.path.join(src_dir, f), f"code/business_entity_resolution/src/{f}")

size_mb = os.path.getsize(ZIP_NAME) / (1024 * 1024)
print("=" * 70)
print(f"Submission zip created: {ZIP_NAME}")
print(f"Total size: {size_mb:.2f} MB in {time.time()-t0:.1f}s")
print("=" * 70)
