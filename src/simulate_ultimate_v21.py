import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, time

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3500MB'")
con.execute("SET temp_directory='D:/duckdb_temp'")

print("Testing min_prob sweep (0.70, 0.60, 0.55, 0.50) on V21 simulation...", flush=True)

# 1. Load V20 baseline matches (prob = 0.999)
con.execute("""
CREATE TEMP TABLE p_v20 AS
SELECT source1_entity_id, UNNEST(STR_SPLIT(matched_entity_ids, ',')) as matched_entity_id, 0.999 as prob
FROM read_csv('output/matching_results.tsv', delim='\t', header=true)
WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '';
""")

con.execute("""
CREATE TEMP TABLE s1_all AS
SELECT entity_id as s1_id, country, name_normalized as n1, address_normalized as a1
FROM read_csv('normalized_data/test_source1_normalized.tsv', delim='\t', header=true);

CREATE TEMP TABLE s23_all AS
SELECT entity_id as m_id, country, name_normalized as n2, address_normalized as a2
FROM (
    SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/test_source2_normalized.tsv', delim='\t', header=true)
    UNION ALL
    SELECT entity_id, country, name_normalized, address_normalized FROM read_csv('normalized_data/test_source3_normalized.tsv', delim='\t', header=true)
);
""")

for min_p in [0.70, 0.60, 0.55, 0.50]:
    t_start = time.time()
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE p_scored AS
    SELECT source1_entity_id, matched_entity_id, probability as prob
    FROM read_csv('test_scored_candidates.tsv', delim='\\t', header=true)
    WHERE probability >= {min_p};

    CREATE OR REPLACE TEMP TABLE covered_s1 AS
    SELECT DISTINCT source1_entity_id FROM p_v20
    UNION
    SELECT DISTINCT source1_entity_id FROM p_scored;

    CREATE OR REPLACE TEMP TABLE empty_s1 AS
    SELECT s.* FROM s1_all s
    LEFT JOIN covered_s1 c ON s.s1_id = c.source1_entity_id
    WHERE c.source1_entity_id IS NULL;

    CREATE OR REPLACE TEMP TABLE p_rescues AS
    SELECT e.s1_id as source1_entity_id, s.m_id as matched_entity_id, 0.95 as prob
    FROM empty_s1 e JOIN s23_all s ON e.country = s.country AND e.a1 = s.a2
    WHERE LENGTH(e.a1) >= 12
    QUALIFY COUNT(*) OVER (PARTITION BY e.s1_id) <= 5

    UNION ALL

    SELECT e.s1_id as source1_entity_id, s.m_id as matched_entity_id, 0.88 as prob
    FROM empty_s1 e JOIN s23_all s ON e.country = s.country 
                                   AND SUBSTRING(e.n1, 1, 6) = SUBSTRING(s.n2, 1, 6)
                                   AND SUBSTRING(e.a1, 1, 6) = SUBSTRING(s.a2, 1, 6)
    WHERE LENGTH(e.n1) >= 6 AND LENGTH(e.a1) >= 6
    QUALIFY COUNT(*) OVER (PARTITION BY e.s1_id) <= 5

    UNION ALL

    SELECT e.s1_id as source1_entity_id, s.m_id as matched_entity_id, 0.85 as prob
    FROM empty_s1 e JOIN s23_all s ON e.country = s.country AND e.n1 = s.n2
    WHERE LENGTH(e.n1) >= 6
    QUALIFY COUNT(*) OVER (PARTITION BY e.s1_id) <= 5;

    CREATE OR REPLACE TEMP TABLE all_candidates AS
    SELECT source1_entity_id, matched_entity_id, MAX(prob) as prob
    FROM (
        SELECT * FROM p_v20
        UNION ALL
        SELECT * FROM p_scored
        UNION ALL
        SELECT * FROM p_rescues
    )
    GROUP BY source1_entity_id, matched_entity_id;

    CREATE OR REPLACE TEMP TABLE final_injective_matches AS
    WITH ranked AS (
        SELECT source1_entity_id, matched_entity_id, prob,
               ROW_NUMBER() OVER (PARTITION BY matched_entity_id ORDER BY prob DESC) as rk
        FROM all_candidates
    )
    SELECT source1_entity_id, matched_entity_id, prob
    FROM ranked WHERE rk = 1;
    """)
    pairs, matched_s1 = con.execute("SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM final_injective_matches").fetchone()
    print(f"min_p = {min_p:.2f} -> Injective Pairs: {pairs:,} | Matched S1: {matched_s1:,} ({matched_s1/1732544*100:.2f}%) | Empty: {1732544-matched_s1:,} in {time.time()-t_start:.1f}s", flush=True)
