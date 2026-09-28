import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import polars as pl
gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
s1 = pl.read_parquet('validation_benchmark/val_s1_v13_clean.parquet')
s23 = pl.read_parquet('validation_benchmark/val_s23_v13_clean.parquet')

# Check S1-140830208
print("S1-140830208 in GT:")
print(gt.filter(pl.col('source1_entity_id') == 'S1-140830208'))

# Check S2-232964779
print("S2-232964779 in GT:")
print(gt.filter(pl.col('matched_entity_id') == 'S2-232964779'))
