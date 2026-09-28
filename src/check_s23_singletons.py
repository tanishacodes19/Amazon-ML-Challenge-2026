import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import polars as pl

gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
s23 = pl.read_parquet('validation_benchmark/val_s23_v13_clean.parquet')
s1 = pl.read_parquet('validation_benchmark/val_s1_v13_clean.parquet')

print(f"Total S1 entities: {len(s1):,}")
print(f"Total S23 entities in validation: {len(s23):,}")
print(f"Total GT pairs: {len(gt):,}")

gt_s23 = set(gt['matched_entity_id'])
print(f"Unique S23 entities in GT: {len(gt_s23):,}")
print(f"S23 entities NOT in GT (singletons): {len(s23) - len(gt_s23):,}")

# Among singletons in S23, how many have missing address vs valid address?
s23_dict = {r["matched_entity_id"]: r for r in s23.to_dicts()}
singletons = [mid for mid in s23['matched_entity_id'] if mid not in gt_s23]
s_missing_addr = 0
s_valid_addr = 0
for mid in singletons:
    addr = s23_dict[mid].get('business_address')
    if addr is None or str(addr).strip().lower() in ['none', 'nan', '']:
        s_missing_addr += 1
    else:
        s_valid_addr += 1

print(f"Singletons in S23: {len(singletons):,}")
print(f"  Singletons with valid address: {s_valid_addr:,}")
print(f"  Singletons with MISSING address: {s_missing_addr:,}")
