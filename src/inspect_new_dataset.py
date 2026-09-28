import duckdb
con = duckdb.connect()

# Ground truth format
print("=== GROUND TRUTH FORMAT ===")
gt = con.execute("SELECT * FROM read_csv('D:/student_resource/student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true) LIMIT 10").fetchall()
for r in gt:
    print(r)

# Column names
cols = con.execute("SELECT column_name FROM (DESCRIBE SELECT * FROM read_csv('D:/student_resource/student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true))").fetchall()
print("\nGT Columns:", [c[0] for c in cols])

# GT stats
stats = con.execute("""
SELECT 
    COUNT(*) as n_rows,
    COUNT(DISTINCT source1_entity_id) as n_s1,
    SUM(CASE WHEN matched_entity_ids = '' OR matched_entity_ids IS NULL THEN 1 ELSE 0 END) as singletons
FROM read_csv('D:/student_resource/student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true)
""").fetchall()
print("\nGT Stats:", stats)

# Source counts
for src, path in [
    ("train_s1", "D:/student_resource/student_resource/dataset/train/train_source1.tsv"),
    ("train_s2", "D:/student_resource/student_resource/dataset/train/train_source2.tsv"),
    ("train_s3", "D:/student_resource/student_resource/dataset/train/train_source3.tsv"),
    ("test_s1", "D:/student_resource/student_resource/dataset/test/test_source1.tsv"),
    ("test_s2", "D:/student_resource/student_resource/dataset/test/test_source2.tsv"),
    ("test_s3", "D:/student_resource/student_resource/dataset/test/test_source3.tsv"),
]:
    n = con.execute(f"SELECT COUNT(*) FROM read_csv('{path}', delim='\t', header=true)").fetchone()[0]
    print(f"{src}: {n:,} rows")

# Country distribution
print("\n=== TRAIN COUNTRIES ===")
for src, path in [
    ("train_s1", "D:/student_resource/student_resource/dataset/train/train_source1.tsv"),
    ("train_s2", "D:/student_resource/student_resource/dataset/train/train_source2.tsv"),
]:
    res = con.execute(f"SELECT country, COUNT(*) as cnt FROM read_csv('{path}', delim='\t', header=true) GROUP BY country ORDER BY cnt DESC").fetchall()
    print(f"{src}: {res}")

print("\n=== TEST COUNTRIES ===")
for src, path in [
    ("test_s1", "D:/student_resource/student_resource/dataset/test/test_source1.tsv"),
    ("test_s2", "D:/student_resource/student_resource/dataset/test/test_source2.tsv"),
]:
    res = con.execute(f"SELECT country, COUNT(*) as cnt FROM read_csv('{path}', delim='\t', header=true) GROUP BY country ORDER BY cnt DESC").fetchall()
    print(f"{src}: {res}")

# Check our existing normalized data vs the new dataset
print("\n=== COMPARING DATASETS ===")
# Check if D:\student_resource data is the same as C:\Users\Admin\Documents\Amazon-ML-Dataset
old_s1 = con.execute("SELECT COUNT(*) FROM read_csv('C:/Users/Admin/Documents/Amazon-ML-Dataset/student_resource/dataset/train/train_source1.tsv', delim='\t', header=true)").fetchone()[0]
new_s1 = con.execute("SELECT COUNT(*) FROM read_csv('D:/student_resource/student_resource/dataset/train/train_source1.tsv', delim='\t', header=true)").fetchone()[0]
print(f"Old train_s1: {old_s1:,}, New train_s1: {new_s1:,}, Same: {old_s1 == new_s1}")

old_test = con.execute("SELECT COUNT(*) FROM read_csv('C:/Users/Admin/Documents/Amazon-ML-Dataset/student_resource/dataset/test/test_source1.tsv', delim='\t', header=true)").fetchone()[0]
new_test = con.execute("SELECT COUNT(*) FROM read_csv('D:/student_resource/student_resource/dataset/test/test_source1.tsv', delim='\t', header=true)").fetchone()[0]
print(f"Old test_s1: {old_test:,}, New test_s1: {new_test:,}, Same: {old_test == new_test}")

# Check ground truth format - the NEW one has a DIFFERENT format (aggregated)
print("\n=== NEW GT FORMAT (first 5 rows) ===")
gt_sample = con.execute("SELECT * FROM read_csv('D:/student_resource/student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true) LIMIT 5").fetchall()
for r in gt_sample:
    print(r)

# Check OUR ground truth format
print("\n=== OUR GT FORMAT (first 5 rows) ===")
our_gt = con.execute("SELECT * FROM read_csv('C:/Users/Admin/Documents/Amazon-ML-Challenge-2026/ground_truth_pairs.tsv', delim='\t', header=true) LIMIT 5").fetchall()
cols2 = con.execute("SELECT column_name FROM (DESCRIBE SELECT * FROM read_csv('C:/Users/Admin/Documents/Amazon-ML-Challenge-2026/ground_truth_pairs.tsv', delim='\t', header=true))").fetchall()
print("Columns:", [c[0] for c in cols2])
for r in our_gt:
    print(r)
