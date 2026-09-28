import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import polars as pl
s23 = pl.read_parquet('validation_benchmark/val_s23_v13_clean.parquet')
print(s23.filter(pl.col('matched_entity_id') == 'S2-635830112'))
