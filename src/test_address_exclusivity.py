import sys
sys.stdout.reconfigure(encoding='utf-8')
import duckdb

con = duckdb.connect()
con.execute("""
CREATE TEMP TABLE all_ent AS 
SELECT 'S1' as src, entity_id, business_name, business_address 
FROM read_csv('normalized_data/train_source1_normalized.tsv', delim='\t', header=true) 
WHERE address_normalized LIKE '%85%wayne%ticonderoga%' OR address_normalized LIKE '%85%wanye%ticonderoga%' 
UNION ALL 
SELECT 'S2' as src, entity_id, business_name, business_address 
FROM read_csv('normalized_data/train_source2_normalized.tsv', delim='\t', header=true) 
WHERE address_normalized LIKE '%85%wayne%ticonderoga%' OR address_normalized LIKE '%85%wanye%ticonderoga%' 
UNION ALL 
SELECT 'S3' as src, entity_id, business_name, business_address 
FROM read_csv('normalized_data/train_source3_normalized.tsv', delim='\t', header=true) 
WHERE address_normalized LIKE '%85%wayne%ticonderoga%' OR address_normalized LIKE '%85%wanye%ticonderoga%';
""")
rows = con.execute("SELECT * FROM all_ent").fetchall()
print(f"Total entities at '85 Wayne Avenue, Ticonderoga': {len(rows)}")
for r in rows:
    print(f"  [{r[0]}] {r[1]}: {r[2]!r} || {r[3]!r}")
