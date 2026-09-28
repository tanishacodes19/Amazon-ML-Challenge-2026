import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, polars as pl
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v14_cands.parquet"))
s1 = pl.read_parquet(os.path.join(VAL_DIR, "val_s1.parquet"))
s23 = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet"))

cand_set = set(zip(cands["source1_entity_id"], cands["matched_entity_id"]))
gt_set = set(zip(gt["source1_entity_id"], gt["matched_entity_id"]))

missed = gt_set - cand_set
print(f"Total Missed by V14 Blocking: {len(missed):,} / {len(gt_set):,} ({len(missed)/len(gt_set)*100:.2f}%)")

s1_dict = {r["source1_entity_id"]: r for r in s1.to_dicts()}
s23_dict = {r["matched_entity_id"]: r for r in s23.to_dicts()}

print("\nSample 25 Missed Pairs:")
for i, (sid, mid) in enumerate(list(missed)[:25]):
    s1_r = s1_dict.get(sid, {})
    s23_r = s23_dict.get(mid, {})
    name1 = s1_r.get("business_name")
    addr1 = s1_r.get("business_address")
    name2 = s23_r.get("business_name")
    addr2 = s23_r.get("business_address")
    print(f"\n{i+1}. [S1 ]: {name1}  ||  {addr1}")
    print(f"   [S23]: {name2}  ||  {addr2}")
