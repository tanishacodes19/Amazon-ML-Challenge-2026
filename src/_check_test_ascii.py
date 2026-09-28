import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
import duckdb

con = duckdb.connect()
s1_path = "D:/student_resource/student_resource/dataset/test/test_source1.tsv"

total = con.execute(f"SELECT COUNT(*) FROM read_csv('{s1_path}', delim='\\t', header=true)").fetchone()[0]
non_ascii = con.execute(f"SELECT COUNT(*) FROM read_csv('{s1_path}', delim='\\t', header=true) WHERE REGEXP_MATCHES(business_name, '[^\\x00-\\x7F]')").fetchone()[0]
non_ascii_addr = con.execute(f"SELECT COUNT(*) FROM read_csv('{s1_path}', delim='\\t', header=true) WHERE REGEXP_MATCHES(business_address, '[^\\x00-\\x7F]')").fetchone()[0]

print(f"Total Test S1: {total:,}")
print(f"Non-ASCII names: {non_ascii:,} ({non_ascii/total*100:.2f}%)")
print(f"Non-ASCII addrs: {non_ascii_addr:,} ({non_ascii_addr/total*100:.2f}%)")
