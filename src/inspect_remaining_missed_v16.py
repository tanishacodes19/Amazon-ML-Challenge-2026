import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, duckdb, polars as pl
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v16_cands.parquet"))
s1 = pl.read_parquet(os.path.join(VAL_DIR, "val_s1_v13_clean.parquet"))
s23 = pl.read_parquet(os.path.join(VAL_DIR, "val_s23_v13_clean.parquet"))

con = duckdb.connect()
con.register("gt", gt.to_pandas())
con.register("cands", cands.to_pandas())
con.register("s1", s1.to_pandas())
con.register("s23", s23.to_pandas())

missed = con.execute("""
SELECT g.source1_entity_id, g.matched_entity_id,
       s1.business_name as n1, s1.business_address as a1,
       s23.business_name as n2, s23.business_address as a2
FROM gt g
LEFT JOIN cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
JOIN s1 ON g.source1_entity_id = s1.source1_entity_id
JOIN s23 ON g.matched_entity_id = s23.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""").fetchdf()

print(f"Remaining Missed GT: {len(missed):,} / {len(gt):,} (Recall: {(len(gt)-len(missed))/len(gt)*100:.2f}%)")
print("\nTop 15 Remaining Missed Samples:")
for i, r in missed.head(15).iterrows():
    print(f"{i+1}. S1 : {r['n1']}  ||  {r['a1']}")
    print(f"   S23: {r['n2']}  ||  {r['a2']}\n")
