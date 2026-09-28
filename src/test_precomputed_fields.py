import sys
import duckdb
import time

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

S1_NORM = 'C:/Users/Admin/Documents/Amazon-ML-Challenge-2026/normalized_data/test_source1_normalized.tsv'

t0 = time.time()
print("Creating s1_tbl with pre-parsed house and postal in DuckDB...")
con.execute(f"""
CREATE TEMP TABLE s1_tbl AS
SELECT 
    entity_id,
    COALESCE(name_normalized, '') AS name_norm,
    COALESCE(address_normalized, '') AS addr_norm,
    COALESCE(country, '') AS country,
    COALESCE(REGEXP_EXTRACT(address_normalized, '(?:^|#|\\bno\\.?|\\bplot\\.?|\\bflat\\.?|\\bunit\\s*)\\s*(\\d+[a-z]?)', 1), '') AS house_number,
    COALESCE(REGEXP_EXTRACT(address_normalized, '\\b(\\d{5,6})\\b', 1), '') AS postal_code
FROM read_csv('{S1_NORM}', delim='\\t', header=true);
""")
print(f"Loaded and parsed 1.73M S1 records in {time.time()-t0:.2f}s!")

sample = con.execute("SELECT * FROM s1_tbl LIMIT 5").df()
print(sample)
