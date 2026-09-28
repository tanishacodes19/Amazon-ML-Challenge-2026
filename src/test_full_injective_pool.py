import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, time

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")

print("Testing full merge of test_scored_candidates + V20 matches...", flush=True)
t0 = time.time()

# 1. Load V20 accepted matches (4,362,169 pairs)
con.execute("""
CREATE TEMP TABLE v20_pairs AS
SELECT source1_entity_id, UNNEST(STR_SPLIT(matched_entity_ids, ',')) as matched_entity_id, 0.999 as prob
FROM read_csv('output/matching_results.tsv', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';
""")
n_v20 = con.execute("SELECT COUNT(*) FROM v20_pairs").fetchone()[0]
print(f"Loaded {n_v20:,} V20 match pairs.", flush=True)

# 2. Check candidate pairs in test_scored_candidates at various probability thresholds
for min_p in [0.80, 0.70, 0.60, 0.50]:
    print(f"\n--- Testing min_prob = {min_p:.2f} ---", flush=True)
    t_p = time.time()
    res = con.execute(f"""
    WITH combined AS (
        SELECT source1_entity_id, matched_entity_id, prob FROM v20_pairs
        UNION ALL
        SELECT source1_entity_id, matched_entity_id, probability as prob
        FROM read_csv('test_scored_candidates.tsv', delim='\\t', header=true)
        WHERE probability >= {min_p}
    ),
    deduped AS (
        SELECT source1_entity_id, matched_entity_id, MAX(prob) as prob
        FROM combined
        GROUP BY source1_entity_id, matched_entity_id
    ),
    ranked AS (
        SELECT source1_entity_id, matched_entity_id, prob,
               ROW_NUMBER() OVER (PARTITION BY matched_entity_id ORDER BY prob DESC) as rk
        FROM deduped
    )
    SELECT COUNT(*), COUNT(DISTINCT source1_entity_id), COUNT(DISTINCT matched_entity_id)
    FROM ranked WHERE rk = 1;
    """).fetchall()[0]
    print(f"Injective Result: {res[0]:,} pairs | {res[1]:,} S1 entities ({res[1]/1732544*100:.2f}%) | {res[2]:,} matched S23 entities in {time.time()-t_p:.1f}s", flush=True)
