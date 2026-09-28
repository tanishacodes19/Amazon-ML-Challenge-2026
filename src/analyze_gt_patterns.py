import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, duckdb

print("Analyzing Ground Truth relationships in Training Set...")
BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
GT_FILE = os.path.join(BASE, "ground_truth_pairs.tsv").replace("\\", "/")
S1_NORM = os.path.join(BASE, "normalized_data", "train_source1_normalized.tsv").replace("\\", "/")
S2_NORM = os.path.join(BASE, "normalized_data", "train_source2_normalized.tsv").replace("\\", "/")
S3_NORM = os.path.join(BASE, "normalized_data", "train_source3_normalized.tsv").replace("\\", "/")

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")

# Sample 100,000 GT pairs
print("1. Sampling 100k GT pairs...")
con.execute(f"""
CREATE TEMP TABLE gt_sample AS
SELECT source1_entity_id, matched_entity_id
FROM read_csv('{GT_FILE}', delim='\t', header=true)
USING SAMPLE reservoir (100000 ROWS) REPEATABLE (42);

CREATE TEMP TABLE s1_sub AS
SELECT entity_id, name_normalized as n1, address_normalized as a1, country as c1
FROM read_csv('{S1_NORM}', delim='\t', header=true)
WHERE entity_id IN (SELECT source1_entity_id FROM gt_sample);

CREATE TEMP TABLE s23_sub AS
SELECT entity_id, name_normalized as n2, address_normalized as a2, country as c2
FROM (
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('{S2_NORM}', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('{S3_NORM}', delim='\t', header=true)
)
WHERE entity_id IN (SELECT matched_entity_id FROM gt_sample);
""")

con.execute("""
CREATE TEMP TABLE analysis AS
SELECT g.source1_entity_id, g.matched_entity_id,
       s1.n1, s1.a1, s1.c1,
       s2.n2, s2.a2, s2.c2
FROM gt_sample g
JOIN s1_sub s1 ON g.source1_entity_id = s1.entity_id
JOIN s23_sub s2 ON g.matched_entity_id = s2.entity_id;
""")

print("\n2. Checking blocking channel coverage on 100k GT pairs:")
# Check different matching mechanisms:
con.execute("""
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN c1 = c2 THEN 1 ELSE 0 END) as same_country,
    SUM(CASE WHEN SUBSTRING(n1, 1, 3) = SUBSTRING(n2, 1, 3) THEN 1 ELSE 0 END) as name_prefix3,
    SUM(CASE WHEN SUBSTRING(n1, 1, 4) = SUBSTRING(n2, 1, 4) THEN 1 ELSE 0 END) as name_prefix4,
    SUM(CASE WHEN SUBSTRING(a1, 1, 6) = SUBSTRING(a2, 1, 6) THEN 1 ELSE 0 END) as addr_prefix6,
    SUM(CASE WHEN SUBSTRING(a1, 1, 10) = SUBSTRING(a2, 1, 10) THEN 1 ELSE 0 END) as addr_prefix10,
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(n1, ' '), STR_SPLIT(n2, ' '))) >= 1 THEN 1 ELSE 0 END) as share_1_name_word,
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(n1, ' '), STR_SPLIT(n2, ' '))) >= 2 THEN 1 ELSE 0 END) as share_2_name_words,
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(a1, ' '), STR_SPLIT(a2, ' '))) >= 1 THEN 1 ELSE 0 END) as share_1_addr_word,
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(a1, ' '), STR_SPLIT(a2, ' '))) >= 2 THEN 1 ELSE 0 END) as share_2_addr_words,
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(a1, ' '), STR_SPLIT(a2, ' '))) >= 3 THEN 1 ELSE 0 END) as share_3_addr_words
FROM analysis;
""")
res = con.execute("SELECT * FROM analysis LIMIT 0").description
row = con.execute("""
SELECT 
    COUNT(*),
    SUM(CASE WHEN c1 = c2 THEN 1 ELSE 0 END),
    SUM(CASE WHEN SUBSTRING(n1, 1, 3) = SUBSTRING(n2, 1, 3) THEN 1 ELSE 0 END),
    SUM(CASE WHEN SUBSTRING(n1, 1, 4) = SUBSTRING(n2, 1, 4) THEN 1 ELSE 0 END),
    SUM(CASE WHEN SUBSTRING(a1, 1, 6) = SUBSTRING(a2, 1, 6) THEN 1 ELSE 0 END),
    SUM(CASE WHEN SUBSTRING(a1, 1, 10) = SUBSTRING(a2, 1, 10) THEN 1 ELSE 0 END),
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(n1, ' '), STR_SPLIT(n2, ' '))) >= 1 THEN 1 ELSE 0 END),
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(n1, ' '), STR_SPLIT(n2, ' '))) >= 2 THEN 1 ELSE 0 END),
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(a1, ' '), STR_SPLIT(a2, ' '))) >= 1 THEN 1 ELSE 0 END),
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(a1, ' '), STR_SPLIT(a2, ' '))) >= 2 THEN 1 ELSE 0 END),
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(a1, ' '), STR_SPLIT(a2, ' '))) >= 3 THEN 1 ELSE 0 END)
FROM analysis;
""").fetchall()[0]

labels = [
    "Total Sampled", "Same Country", "Name Prefix 3", "Name Prefix 4", 
    "Addr Prefix 6", "Addr Prefix 10", "Share >= 1 Name Word", "Share >= 2 Name Words",
    "Share >= 1 Addr Word", "Share >= 2 Addr Words", "Share >= 3 Addr Words"
]
for lbl, val in zip(labels, row):
    pct = val / row[0] * 100
    print(f"  {lbl:<25}: {val:6,d} ({pct:5.2f}%)")
