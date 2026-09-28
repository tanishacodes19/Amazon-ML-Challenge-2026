import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import polars as pl

gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
s23 = pl.read_parquet('validation_benchmark/val_s23_v13_clean.parquet')

s23_dict = {r["matched_entity_id"]: r for r in s23.to_dicts()}

# Count how many GT pairs have missing S23 address
gt_m_ids = gt['matched_entity_id'].to_list()
missing_addr_count = 0
valid_addr_count = 0

for mid in gt_m_ids:
    r = s23_dict.get(mid, {})
    addr = r.get('business_address')
    c_addr = r.get('clean_addr')
    if addr is None or str(addr).strip().lower() in ['none', 'nan', ''] or c_addr is None or str(c_addr).strip() in ['nan', '']:
        missing_addr_count += 1
    else:
        valid_addr_count += 1

print(f"Total Ground Truth pairs: {len(gt):,}")
print(f"Ground Truth pairs with VALID S23 address: {valid_addr_count:,} ({valid_addr_count/len(gt)*100:.2f}%)")
print(f"Ground Truth pairs with MISSING S23 address: {missing_addr_count:,} ({missing_addr_count/len(gt)*100:.2f}%)")

# Let's inspect a few of those missing S23 address GT pairs if any!
if missing_addr_count > 0:
    s1 = pl.read_parquet('validation_benchmark/val_s1_v13_clean.parquet')
    s1_dict = {r["source1_entity_id"]: r for r in s1.to_dicts()}
    print("\nSample 10 GT pairs where S23 address is missing:")
    count = 0
    for sid, mid in zip(gt['source1_entity_id'], gt['matched_entity_id']):
        r2 = s23_dict.get(mid, {})
        addr = r2.get('business_address')
        if addr is None or str(addr).strip().lower() in ['none', 'nan', '']:
            r1 = s1_dict.get(sid, {})
            print(f"\nGT Pair {count+1}:")
            print(f"  S1 : [{sid}] {r1.get('business_name')} || {r1.get('business_address')}")
            print(f"  S23: [{mid}] {r2.get('business_name')} || {r2.get('business_address')}")
            count += 1
            if count >= 10: break
