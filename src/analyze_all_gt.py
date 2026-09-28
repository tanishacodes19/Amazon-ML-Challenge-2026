import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, time

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")

print("Sampling 200,000 GT pairs from training set to analyze true match distribution...", flush=True)
t0 = time.time()
con.execute("""
CREATE TEMP TABLE gt_sample AS
SELECT source1_entity_id, matched_entity_id
FROM read_csv('ground_truth_pairs.tsv', delim='\t', header=true)
USING SAMPLE reservoir (200000 ROWS) REPEATABLE (42);

CREATE TEMP TABLE s1_sub AS
SELECT entity_id, name_normalized as n1, address_normalized as a1, country as c1
FROM read_csv('normalized_data/train_source1_normalized.tsv', delim='\t', header=true)
WHERE entity_id IN (SELECT source1_entity_id FROM gt_sample);

CREATE TEMP TABLE s23_sub AS
SELECT entity_id, name_normalized as n2, address_normalized as a2, country as c2
FROM (
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('normalized_data/train_source2_normalized.tsv', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('normalized_data/train_source3_normalized.tsv', delim='\t', header=true)
)
WHERE entity_id IN (SELECT matched_entity_id FROM gt_sample);

CREATE TEMP TABLE sample_joined AS
SELECT g.source1_entity_id, g.matched_entity_id, s1.n1, s1.a1, s1.c1, s2.n2, s2.a2, s2.c2
FROM gt_sample g
JOIN s1_sub s1 ON g.source1_entity_id = s1.entity_id
JOIN s23_sub s2 ON g.matched_entity_id = s2.entity_id;
""")
print(f"Sample joined in {time.time()-t0:.1f}s.", flush=True)

stats = con.execute("""
SELECT 
    COUNT(*),
    SUM(CASE WHEN n1 = n2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    SUM(CASE WHEN a1 = a2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    SUM(CASE WHEN n1 LIKE '%' || n2 || '%' OR n2 LIKE '%' || n1 || '%' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    SUM(CASE WHEN a1 LIKE '%' || a2 || '%' OR a2 LIKE '%' || a1 || '%' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    SUM(CASE WHEN SUBSTRING(n1, 1, 4) = SUBSTRING(n2, 1, 4) THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    SUM(CASE WHEN SUBSTRING(a1, 1, 8) = SUBSTRING(a2, 1, 8) THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(n1, ' '), STR_SPLIT(n2, ' '))) >= 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(a1, ' '), STR_SPLIT(a2, ' '))) >= 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(n1, ' '), STR_SPLIT(n2, ' '))) >= 1 AND LEN(LIST_INTERSECT(STR_SPLIT(a1, ' '), STR_SPLIT(a2, ' '))) >= 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    SUM(CASE WHEN LEN(LIST_INTERSECT(STR_SPLIT(n1, ' '), STR_SPLIT(n2, ' '))) >= 1 OR LEN(LIST_INTERSECT(STR_SPLIT(a1, ' '), STR_SPLIT(a2, ' '))) >= 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
FROM sample_joined;
""").fetchall()[0]

labels = [
    "Total Sampled", "Exact Name", "Exact Address", "Name Contains", "Address Contains",
    "Name Prefix 4", "Address Prefix 8", "Share >= 1 Name Word", "Share >= 1 Addr Word",
    "Share (>=1 Name Word AND >=1 Addr Word)", "Share (>=1 Name Word OR >=1 Addr Word)"
]

for l, s in zip(labels, stats):
    if l == "Total Sampled":
        print(f"  {l:<45}: {int(s):,}")
    else:
        print(f"  {l:<45}: {s:6.2f}%")
