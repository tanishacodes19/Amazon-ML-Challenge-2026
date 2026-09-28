import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os, zipfile, time

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
FINAL_PKG = os.path.join(BASE, "final_package_build")
SUBMISSION_ZIP = os.path.join(BASE, "Kernel_Panic_submission.zip")
V27_ZIP = os.path.join(BASE, "team_antigravity_final_v27_submission.zip")

t0 = time.time()
print("Building Kernel_Panic_submission.zip (V27 = 0.770)...", flush=True)

# Extract V27 output files
print("Extracting V27 output files...", flush=True)
with zipfile.ZipFile(V27_ZIP, "r") as v27:
    matching_data = v27.read("matching_results.tsv")
    candidate_data = v27.read("candidate_pairs.tsv")
print(f"  matching_results.tsv: {len(matching_data)/(1024*1024):.1f} MB", flush=True)
print(f"  candidate_pairs.tsv:  {len(candidate_data)/(1024*1024):.1f} MB", flush=True)

with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    # output/
    print("\nAdding output/...", flush=True)
    zf.writestr("output/matching_results.tsv", matching_data)
    zf.writestr("output/candidate_pairs.tsv", candidate_data)

    # code/business_entity_resolution/src/ — all .py files
    print("Adding code/...", flush=True)
    src_dir = os.path.join(BASE, "src")
    n = 0
    for fname in os.listdir(src_dir):
        if fname.endswith(".py"):
            with open(os.path.join(src_dir, fname), "rb") as f:
                zf.writestr(f"code/business_entity_resolution/src/{fname}", f.read())
            n += 1
    print(f"  {n} source files added.", flush=True)

    # README.md
    readme = os.path.join(FINAL_PKG, "code", "business_entity_resolution", "README.md")
    with open(readme, "rb") as f:
        zf.writestr("code/business_entity_resolution/README.md", f.read())

    # requirements.txt
    req = os.path.join(FINAL_PKG, "code", "business_entity_resolution", "requirements.txt")
    with open(req, "rb") as f:
        zf.writestr("code/business_entity_resolution/requirements.txt", f.read())

    # Documentation_template.md
    doc = os.path.join(FINAL_PKG, "Documentation_template.md")
    with open(doc, "rb") as f:
        zf.writestr("Documentation_template.md", f.read())

zip_size = os.path.getsize(SUBMISSION_ZIP) / (1024 * 1024)
print(f"\n{'='*60}", flush=True)
print(f"Kernel_Panic_submission.zip READY!", flush=True)
print(f"Path: {SUBMISSION_ZIP}", flush=True)
print(f"Size: {zip_size:.1f} MB", flush=True)
print(f"Score: 0.770 (V27)", flush=True)
print(f"Time:  {time.time()-t0:.1f}s", flush=True)
print(f"{'='*60}", flush=True)
