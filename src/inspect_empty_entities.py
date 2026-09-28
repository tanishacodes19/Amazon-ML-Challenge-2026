import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, time

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")

print("Loading test S2 and S3 once into memory...", flush=True)
t0 = time.time()
con.execute("""
CREATE TEMP TABLE s23 AS
SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('normalized_data/test_source2_normalized.tsv', delim='\t', header=true)
UNION ALL
SELECT entity_id, name_normalized, address_normalized, country FROM read_csv('normalized_data/test_source3_normalized.tsv', delim='\t', header=true);
""")
print(f"Loaded in {time.time()-t0:.1f}s.", flush=True)

empty_s1 = [
    ("S1-106407869", "vision partners", "newton rd", "US"),
    ("S1-156285671", "team ecole", "franklin roosevelt", "France"),
    ("S1-689823050", "red perfect", "mirzapur", "India"),
    ("S1-280204013", "om constructions", "karauli", "India"),
    ("S1-378191895", "east marketing", "exchange place", "India"),
    ("S1-750279127", "sra export", "lajpat nagar", "India"),
    ("S1-550154910", "tech management", "chanchalguda", "India"),
    ("S1-897975128", "prime aditya agro", "brahmanapalli", "India"),
    ("S1-444801830", "site win", "rasta peth", "India"),
    ("S1-466641145", "elephant centre", "lachassaigne", "France")
]

for sid, n_sub, a_sub, ctry in empty_s1:
    print(f"\n--- Checking {sid} ({n_sub} | {a_sub} | {ctry}) ---", flush=True)
    q = f"""
    SELECT entity_id, name_normalized, address_normalized
    FROM s23
    WHERE country = '{ctry}' AND (
        address_normalized LIKE '%{a_sub}%' 
        OR name_normalized LIKE '%{n_sub}%'
    )
    LIMIT 5;
    """
    rows = con.execute(q).fetchall()
    if rows:
        for r in rows:
            print(f"   MATCH FOUND: {r[0]} | {r[1]} | {r[2]}", flush=True)
    else:
        print("   NO MATCH FOUND.", flush=True)
