import sys
sys.stdout.reconfigure(encoding='utf-8')
import duckdb

con = duckdb.connect()
con.execute("PRAGMA memory_limit='3500MB'")
S1 = 'normalized_data/train_source1_normalized.tsv'
S2 = 'normalized_data/train_source2_normalized.tsv'
GT = 'ground_truth_pairs.tsv'

con.execute(f"""
CREATE TEMP TABLE s1 AS 
SELECT entity_id, business_name, address_normalized, country 
FROM read_csv('{S1}', delim='\t', header=true) 
WHERE address_normalized IS NOT NULL AND LENGTH(address_normalized) >= 15;

CREATE TEMP TABLE s2 AS 
SELECT entity_id, business_name, address_normalized, country 
FROM read_csv('{S2}', delim='\t', header=true) 
WHERE address_normalized IS NOT NULL AND LENGTH(address_normalized) >= 15;

CREATE TEMP TABLE gt AS 
SELECT source1_entity_id, matched_entity_id 
FROM read_csv('{GT}', delim='\t', header=true);

CREATE TEMP TABLE collocated_negs AS 
SELECT s1.entity_id as s1_id, s2.entity_id as s2_id, 
       s1.business_name as n1, s2.business_name as n2, 
       s1.address_normalized as a 
FROM s1 
JOIN s2 ON s1.country = s2.country AND s1.address_normalized = s2.address_normalized 
LEFT JOIN gt ON s1.entity_id = gt.source1_entity_id AND s2.entity_id = gt.matched_entity_id 
WHERE gt.matched_entity_id IS NULL;
""")

cnt = con.execute("SELECT COUNT(*) FROM collocated_negs").fetchone()[0]
print(f"Total Co-located Hard Negatives in Training Data: {cnt:,}")

print("\nSample 10 Co-located Hard Negatives (Exact same address, completely different business):")
for r in con.execute("SELECT n1, n2, a FROM collocated_negs USING SAMPLE 10").fetchall():
    print(f"  S1:  {r[0]}")
    print(f"  S2:  {r[1]}")
    print(f"  ADR: {r[2]}")
    print()
