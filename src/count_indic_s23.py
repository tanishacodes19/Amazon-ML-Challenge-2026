import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import polars as pl

s23_val = pl.read_parquet('validation_benchmark/val_s23_v13_clean.parquet')
print(f"Total S23 entities in validation: {len(s23_val):,}")

# Check non-ASCII characters in original business_name
def has_non_ascii(s):
    if not s: return False
    return any(ord(c) > 127 for c in str(s))

non_ascii_mask = [has_non_ascii(n) for n in s23_val['business_name']]
non_ascii_count = sum(non_ascii_mask)
print(f"S23 entities with non-ASCII in business_name: {non_ascii_count:,} ({non_ascii_count/len(s23_val)*100:.2f}%)")

# How many of these are in Ground Truth?
gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
gt_m_ids = set(gt['matched_entity_id'])
non_ascii_in_gt = sum(1 for mid, flag in zip(s23_val['matched_entity_id'], non_ascii_mask) if flag and mid in gt_m_ids)
print(f"Non-ASCII S23 entities in Ground Truth: {non_ascii_in_gt:,}")
