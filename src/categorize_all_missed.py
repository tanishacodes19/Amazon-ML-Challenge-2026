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
       s1.clean_name as n1, s1.clean_addr as a1,
       s23.clean_name as n2, s23.clean_addr as a2,
       s1.country
FROM gt g
LEFT JOIN cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
JOIN s1 ON g.source1_entity_id = s1.source1_entity_id
JOIN s23 ON g.matched_entity_id = s23.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""").fetchdf()

total_missed = len(missed)
print(f"Total Missed True Matches: {total_missed:,}")

# Categorize:
# 1. Missing address in S23
null_addr = missed["a2"].isna() | (missed["a2"].str.strip() == "") | (missed["a2"].str.lower() == "none") | (missed["a2"].str.lower() == "nan")
c_null_addr = sum(null_addr)

# 2. Shared phone number in raw address
# 3. Exact first word of name match
first_word_s1 = missed["n1"].str.split(" ").str[0]
first_word_s23 = missed["n2"].str.split(" ").str[0]
c_first_word = sum((first_word_s1 == first_word_s23) & (~null_addr))

# 4. Any shared 5+ char word in name
shared_word_name = []
for n1, n2 in zip(missed["n1"], missed["n2"]):
    w1 = {w for w in str(n1).split() if len(w) >= 5}
    w2 = {w for w in str(n2).split() if len(w) >= 5}
    shared_word_name.append(len(w1 & w2) > 0)
c_shared_name_word = sum(shared_word_name)

# 5. Any shared 5+ char word in address
shared_word_addr = []
for a1, a2 in zip(missed["a1"], missed["a2"]):
    w1 = {w for w in str(a1).split() if len(w) >= 5}
    w2 = {w for w in str(a2).split() if len(w) >= 5}
    shared_word_addr.append(len(w1 & w2) > 0)
c_shared_addr_word = sum(shared_word_addr)

print("Breakdown of remaining 2,791 missed matches:")
print(f"1. S23 has NULL / Missing Address:          {c_null_addr:,} ({c_null_addr/total_missed*100:.1f}%)")
print(f"2. Exact First Word of Name Match:          {c_first_word:,} ({c_first_word/total_missed*100:.1f}%)")
print(f"3. Shares at least ONE 5+ char Name Word:   {c_shared_name_word:,} ({c_shared_name_word/total_missed*100:.1f}%)")
print(f"4. Shares at least ONE 5+ char Addr Word:   {c_shared_addr_word:,} ({c_shared_addr_word/total_missed*100:.1f}%)")

# Both shared word in name OR address
either = [sn or sa for sn, sa in zip(shared_word_name, shared_word_addr)]
print(f"5. Shares 5+ char word in Name OR Address:  {sum(either):,} ({sum(either)/total_missed*100:.1f}%)")
