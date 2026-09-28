import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import duckdb
import re
import pandas as pd

con = duckdb.connect()
con.execute("""
CREATE TEMP TABLE gt AS SELECT * FROM read_parquet('validation_benchmark/val_gt.parquet');
CREATE TEMP TABLE cands AS SELECT source1_entity_id, matched_entity_id FROM read_parquet('validation_benchmark/val_v12_cands.parquet');
CREATE TEMP TABLE s1 AS SELECT * FROM read_parquet('validation_benchmark/val_s1.parquet');
CREATE TEMP TABLE s23 AS SELECT * FROM read_parquet('validation_benchmark/val_s23.parquet');

CREATE TEMP TABLE missed_v12 AS
SELECT g.source1_entity_id, g.matched_entity_id,
       s1.business_name as s1_name, s1.business_address as s1_addr, s1.country as s1_country,
       s23.business_name as m_name, s23.business_address as m_addr, s23.country as m_country
FROM gt g
LEFT JOIN cands c ON g.source1_entity_id = c.source1_entity_id AND g.matched_entity_id = c.matched_entity_id
JOIN s1 ON g.source1_entity_id = s1.source1_entity_id
JOIN s23 ON g.matched_entity_id = s23.matched_entity_id
WHERE c.matched_entity_id IS NULL;
""")

df = con.execute('SELECT source1_entity_id, matched_entity_id, s1_name, m_name, s1_addr, m_addr, s1_country FROM missed_v12').df()

def clean_brand(name):
    name = str(name).lower()
    name = re.sub(r'^(?:pvt|ltd|inc|corp|llc|co|company|private|limited|dr|shri|sri)\s+', '', name)
    name = re.sub(r'\s+(?:pvt|ltd|inc|corp|llc|co|company|private|limited)$', '', name)
    words = [w for w in re.split(r'[^a-zA-Z0-9]+', name) if len(w) >= 3 and w not in {'the', 'and', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc', 'company', 'private'}]
    return '_'.join(words[:2]) if len(words) >= 2 else (words[0] if words else '')

rem = []
for _, r in df.iterrows():
    b1 = clean_brand(r['s1_name'])
    b2 = clean_brand(r['m_name'])
    brand_hit = (b1 and b2 and b1 == b2)
    
    h1 = re.search(r'\b0*(\d+)\b', str(r['s1_addr']))
    h2 = re.search(r'\b0*(\d+)\b', str(r['m_addr']))
    house_hit = bool(h1 and h2 and h1.group(1) == h2.group(1) and len(h1.group(1)) >= 2)
    
    if not (brand_hit or house_hit):
        rem.append(r)

rem_df = pd.DataFrame(rem)
print(f"Remaining stubborn missed pairs: {len(rem_df):,}")
print(rem_df['s1_country'].value_counts())
print('\nSample 10 stubborn missed pairs:')
for idx, r in rem_df.head(10).iterrows():
    print(f"S1: [{r['s1_name']}] | [{r['s1_addr']}]")
    print(f"M:  [{r['m_name']}] | [{r['m_addr']}]")
    print('-' * 70)
