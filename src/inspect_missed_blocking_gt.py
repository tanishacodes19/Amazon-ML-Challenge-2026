import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import os, numpy as np, polars as pl

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
pairs_df = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_features52_df.parquet"))
s1 = pl.read_parquet(os.path.join(VAL_DIR, "val_s1_v13_clean.parquet"))
s23 = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet"))

v16_pairs = set(zip(pairs_df["source1_entity_id"], pairs_df["matched_entity_id"]))
gt_all = list(zip(gt["source1_entity_id"], gt["matched_entity_id"]))

missed_blocking = [pair for pair in gt_all if pair not in v16_pairs]
print(f"Total GT: {len(gt_all):,}")
print(f"V16 Retrieved GT: {len(gt_all) - len(missed_blocking):,} ({(len(gt_all) - len(missed_blocking))/len(gt_all):.2%})")
print(f"Missed by ALL 16 blocking channels: {len(missed_blocking):,} ({len(missed_blocking)/len(gt_all):.2%})")

s1_dict = {r["source1_entity_id"]: r for r in s1.to_dicts()}
s23_dict = {r["matched_entity_id"]: r for r in s23.to_dicts()}

print("\nSample 25 GT Pairs Missed by ALL 16 Blocking Channels:")
for sid, mid in missed_blocking[:25]:
    r1 = s1_dict.get(sid, {})
    r2 = s23_dict.get(mid, {})
    print(f"\nS1 : {r1.get('business_name')}  ||  {r1.get('business_address')} ({r1.get('country')})")
    print(f"S23: {r2.get('business_name')}  ||  {r2.get('business_address')} ({r2.get('country')})")
