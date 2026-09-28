"""
Build the final competition submission zip with the required structure:

team_antigravity_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/           <- all .py source files
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os, zipfile, time, shutil

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
OUTPUT_DIR = os.path.join(BASE, "output")
FINAL_PKG = os.path.join(BASE, "final_package_build")
SUBMISSION_ZIP = os.path.join(BASE, "team_antigravity_FINAL_SUBMISSION.zip")

t0 = time.time()
print("=" * 70, flush=True)
print("Building Final Competition Submission Package", flush=True)
print("=" * 70, flush=True)

# Key source files to include (core pipeline, not test/scratch)
CORE_SRC_FILES = [
    "normalizer.py",
    "feature_engine_v2.py",
    "eval_framework.py",
    "setup_validation_benchmark.py",
    "run_phase1_baseline.py",
    "run_phase2_normalization.py",
    "build_v12_candidates.py",
    "train_v13_ensemble.py",
    "evaluate_v12_ensemble.py",
    "evaluate_v14_ensemble.py",
    "stream_score_mega_v20_final.py",
    "stream_score_new_candidates.py",
    "build_v23_ultra_pure_submission.py",
    "build_v27_high_conf_expansion.py",
    "build_v28_max_recall.py",
    "postprocess_injective_matches.py",
    "sync_candidate_pairs.py",
    "test_rule_purity_gt.py",
    "test_fast_channels.py",
]

with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:

    # 1. output/ directory
    print("\n[1] Adding output files...", flush=True)
    match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    cands_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    zf.write(match_path, arcname="output/matching_results.tsv")
    zf.write(cands_path, arcname="output/candidate_pairs.tsv")
    print(f"    matching_results.tsv: {os.path.getsize(match_path)/(1024*1024):.1f} MB", flush=True)
    print(f"    candidate_pairs.tsv:  {os.path.getsize(cands_path)/(1024*1024):.1f} MB", flush=True)

    # 2. code/business_entity_resolution/src/ 
    print("\n[2] Adding source code...", flush=True)
    src_dir = os.path.join(BASE, "src")
    added = 0
    missing = []
    for fname in CORE_SRC_FILES:
        fpath = os.path.join(src_dir, fname)
        if os.path.exists(fpath):
            zf.write(fpath, arcname=f"code/business_entity_resolution/src/{fname}")
            added += 1
        else:
            missing.append(fname)
    print(f"    Added {added} core source files.", flush=True)
    if missing:
        print(f"    Missing: {missing}", flush=True)

    # Also add all remaining src/*.py files
    for fname in os.listdir(src_dir):
        if fname.endswith(".py") and fname not in CORE_SRC_FILES:
            fpath = os.path.join(src_dir, fname)
            zf.write(fpath, arcname=f"code/business_entity_resolution/src/{fname}")
    print(f"    Added all src/*.py files.", flush=True)

    # 3. README.md
    print("\n[3] Adding README.md...", flush=True)
    readme = os.path.join(FINAL_PKG, "code", "business_entity_resolution", "README.md")
    zf.write(readme, arcname="code/business_entity_resolution/README.md")

    # 4. requirements.txt
    print("[4] Adding requirements.txt...", flush=True)
    req = os.path.join(FINAL_PKG, "code", "business_entity_resolution", "requirements.txt")
    zf.write(req, arcname="code/business_entity_resolution/requirements.txt")

    # 5. Documentation_template.md
    print("[5] Adding Documentation_template.md...", flush=True)
    doc = os.path.join(FINAL_PKG, "Documentation_template.md")
    zf.write(doc, arcname="Documentation_template.md")

zip_size = os.path.getsize(SUBMISSION_ZIP) / (1024 * 1024)
print(f"\n{'='*70}", flush=True)
print(f"FINAL SUBMISSION ZIP CREATED!", flush=True)
print(f"Path: {SUBMISSION_ZIP}", flush=True)
print(f"Size: {zip_size:.1f} MB", flush=True)
print(f"Time: {time.time()-t0:.1f}s", flush=True)
print(f"{'='*70}", flush=True)

# Show zip contents summary
print("\nContents summary:", flush=True)
with zipfile.ZipFile(SUBMISSION_ZIP, "r") as zf:
    files = zf.infolist()
    dirs = {}
    for f in files:
        top = f.filename.split("/")[0]
        dirs[top] = dirs.get(top, 0) + 1
    for d, cnt in sorted(dirs.items()):
        total_mb = sum(f.compress_size for f in files if f.filename.startswith(d)) / (1024*1024)
        print(f"  {d}/  ({cnt} files, {total_mb:.1f} MB compressed)", flush=True)
