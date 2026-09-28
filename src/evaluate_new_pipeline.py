import os
import time
import csv
import numpy as np
import polars as pl
import xgboost as xgb
from rapidfuzz import fuzz
from normalizer import normalize_business_name, normalize_address, extract_structured_fields
from eval_framework import load_benchmark, compute_blocking_metrics, compute_macro_f05, V4_FEATURES

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")
LOG_PATH = os.path.join(BASE, "experiments_log.csv")

print("=" * 70)
print("EVALUATING IMPROVED PIPELINE: 72.94% RECALL CANDIDATES + BALANCED MODELS")
print("=" * 70)

t0 = time.time()
s1, gt, baseline_cands, s23 = load_benchmark()
v6_cands = pl.read_parquet(os.path.join(VAL_DIR, "val_v6_cands.parquet"))

blocking_res = compute_blocking_metrics(v6_cands, gt, n_s1=len(s1))
print(f"\nV6 Candidates:     {blocking_res['candidate_count']:,} ({blocking_res['avg_candidates_per_s1']:.2f} per S1)")
print(f"Blocking Recall:   {blocking_res['blocking_recall']*100:.2f}% (Recovered: {blocking_res['recovered_matches']:,} / {blocking_res['ground_truth_count']:,})")

# Check if V6 features already cached, otherwise compute
v6_feat_cache = os.path.join(VAL_DIR, "val_v6_features_X.npy")

# Helper functions
def token_set(s):
    if not s: return set()
    return set(s.split())

def numeric_tokens(s):
    if not s: return set()
    return {x for x in s.split() if any(ch.isdigit() for ch in x)}

def first_token(s):
    if not s: return ""
    return s.split()[0]

def last_token(s):
    if not s: return ""
    return s.split()[-1]

def house_number(s):
    if not s: return ""
    for token in s.split():
        if token and token[0].isdigit():
            return token
    return ""

def jaccard(a, b):
    A = token_set(a)
    B = token_set(b)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def numeric_overlap(a, b):
    A = numeric_tokens(a)
    B = numeric_tokens(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

pairs = v6_cands.join(
    s1.select([
        pl.col("source1_entity_id"),
        pl.col("business_name").alias("s1_name_raw"),
        pl.col("business_address").alias("s1_addr_raw"),
        pl.col("country").alias("s1_country"),
    ]),
    on="source1_entity_id",
    how="left"
).join(
    s23.select([
        pl.col("matched_entity_id"),
        pl.col("business_name").alias("m_name_raw"),
        pl.col("business_address").alias("m_addr_raw"),
        pl.col("country").alias("m_country"),
    ]),
    on="matched_entity_id",
    how="left"
)

if os.path.exists(v6_feat_cache):
    print("\nLoading cached V6 feature matrix...")
    X_v6 = np.load(v6_feat_cache)
else:
    print("\nComputing robustly normalized features on V6 candidates...")
    unique_names = set(pairs["s1_name_raw"].unique().drop_nulls()).union(set(pairs["m_name_raw"].unique().drop_nulls()))
    unique_addrs = set(pairs["s1_addr_raw"].unique().drop_nulls()).union(set(pairs["m_addr_raw"].unique().drop_nulls()))
    norm_name_map = {n: normalize_business_name(n) for n in unique_names if n}
    norm_addr_map = {a: normalize_address(a) for a in unique_addrs if a}
    
    rows = []
    for r in pairs.iter_rows(named=True):
        n1 = norm_name_map.get(r["s1_name_raw"], "")
        n2 = norm_name_map.get(r["m_name_raw"], "")
        a1 = norm_addr_map.get(r["s1_addr_raw"], "")
        a2 = norm_addr_map.get(r["m_addr_raw"], "")
        c1 = r["s1_country"] or ""
        c2 = r["m_country"] or ""
        
        nt1 = token_set(n1)
        nt2 = token_set(n2)
        h1 = house_number(a1)
        h2 = house_number(a2)
        
        name_len1 = len(n1)
        name_len2 = len(n2)
        addr_len1 = len(a1)
        addr_len2 = len(a2)
        
        name_ratio = (fuzz.ratio(n1, n2) / 100.0) if (n1 and n2) else 0.0
        addr_ratio = (fuzz.ratio(a1, a2) / 100.0) if (a1 and a2) else 0.0
        name_partial = (fuzz.partial_ratio(n1, n2) / 100.0) if (n1 and n2) else 0.0
        addr_partial = (fuzz.partial_ratio(a1, a2) / 100.0) if (a1 and a2) else 0.0
        name_sort = (fuzz.token_sort_ratio(n1, n2) / 100.0) if (n1 and n2) else 0.0
        name_set = (fuzz.token_set_ratio(n1, n2) / 100.0) if (n1 and n2) else 0.0
        addr_sort = (fuzz.token_sort_ratio(a1, a2) / 100.0) if (a1 and a2) else 0.0
        addr_set = (fuzz.token_set_ratio(a1, a2) / 100.0) if (a1 and a2) else 0.0
        name_j = jaccard(n1, n2)
        addr_j = jaccard(a1, a2)
        num_j = numeric_overlap(a1, a2)
        
        rows.append({
            "name_exact": int(n1 == n2 and n1 != ""),
            "name_contains": int(n1 != "" and n2 != "" and (n1 in n2 or n2 in n1)),
            "name_prefix3": int(n1[:3] == n2[:3] and len(n1) >= 3 and len(n2) >= 3),
            "name_prefix5": int(n1[:5] == n2[:5] and len(n1) >= 5 and len(n2) >= 5),
            "name_suffix4": int(n1[-4:] == n2[-4:] and len(n1) >= 4 and len(n2) >= 4),
            "name_first_token_match": int(first_token(n1) == first_token(n2) and first_token(n1) != ""),
            "name_last_token_match": int(last_token(n1) == last_token(n2) and last_token(n1) != ""),
            "name_shared_token_count": len(nt1 & nt2),
            "name_token_jaccard": name_j,
            "name_ratio": name_ratio,
            "name_partial_ratio": name_partial,
            "name_token_sort": name_sort,
            "name_token_set": name_set,
            "name_len_s1": name_len1,
            "name_len_match": name_len2,
            "name_len_abs_diff": abs(name_len1 - name_len2),
            "name_len_relative_diff": abs(name_len1 - name_len2) / max(name_len1, 1),
            "address_exact": int(a1 == a2 and a1 != ""),
            "address_contains": int(a1 != "" and a2 != "" and (a1 in a2 or a2 in a1)),
            "address_prefix5": int(a1[:5] == a2[:5] and len(a1) >= 5 and len(a2) >= 5),
            "address_prefix8": int(a1[:8] == a2[:8] and len(a1) >= 8 and len(a2) >= 8),
            "address_suffix8": int(a1[-8:] == a2[-8:] and len(a1) >= 8 and len(a2) >= 8),
            "house_match": int(h1 != "" and h1 == h2),
            "address_ratio": addr_ratio,
            "address_partial_ratio": addr_partial,
            "address_token_sort": addr_sort,
            "address_token_set": addr_set,
            "address_token_jaccard": addr_j,
            "address_numeric_overlap": num_j,
            "address_len_s1": addr_len1,
            "address_len_match": addr_len2,
            "address_len_abs_diff": abs(addr_len1 - addr_len2),
            "address_len_relative_diff": abs(addr_len1 - addr_len2) / max(addr_len1, 1),
            "country_match": int(c1 != "" and c1 == c2),
            "country_missing": int(c1 == "" or c2 == ""),
            "source2": int(r["matched_entity_id"].startswith("S2-")),
            "source3": int(r["matched_entity_id"].startswith("S3-")),
            "name_address_ratio_mean": (name_ratio + addr_ratio) / 2.0,
            "name_address_ratio_product": name_ratio * addr_ratio,
            "strong_name_address": int(name_ratio >= 0.90 and addr_ratio >= 0.70),
            "strong_name_house": int(name_ratio >= 0.90 and h1 != "" and h1 == h2),
        })
    feat_df = pl.DataFrame(rows)
    X_v6 = feat_df.select(V4_FEATURES).fill_null(0).to_numpy().astype(np.float32)
    np.save(v6_feat_cache, X_v6)

def evaluate_model(model_path, model_label):
    print(f"\n--- Testing {model_label} ---")
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    probs = clf.predict_proba(X_v6)[:, 1]
    
    best_score = -1
    best_res = None
    best_t = None
    
    for t in [0.50, 0.60, 0.70, 0.76, 0.80, 0.85, 0.90, 0.92, 0.95]:
        pred_mask = probs >= t
        pred_pairs = pairs.select(["source1_entity_id", "matched_entity_id"]).filter(pred_mask)
        res = compute_macro_f05(pred_pairs, gt, s1)
        if res['macro_f05'] > best_score:
            best_score = res['macro_f05']
            best_res = res
            best_t = t
        if t in [0.70, 0.80, 0.85, 0.90, 0.95]:
            print(f"  t={t:.2f} | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['precision']*100:.2f}% | Rec: {res['recall']*100:.2f}% | Preds: {res['predicted_matches']:,}")
            
    print(f"  >> BEST: Macro F0.5 = {best_res['macro_f05']:.4f} @ t={best_t:.2f} (Prec: {best_res['precision']*100:.2f}%, Rec: {best_res['recall']*100:.2f}%)")
    return best_t, best_res

# Evaluate Model B: 50% Random + 50% Hard Negatives
m_b_path = os.path.join(BASE, "model", "xgboost_balanced_b50.json")
if os.path.exists(m_b_path):
    t_b, res_b = evaluate_model(m_b_path, "Model B (50% Hard + 50% Random Negatives)")
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["EXP-2", f"V6 Candidates (72.94% recall) + Model B (50/50) @ t={t_b:.2f}", "72.94%", len(v6_cands), f"{res_b['precision']*100:.2f}%", f"{res_b['recall']*100:.2f}%", f"{res_b['macro_f05']:.4f}", f"{time.time()-t0:.1f}", "Breakthrough in blocking recall + balanced negative training"])

# Evaluate Model C: 75% Hard + 25% Random Negatives
m_c_path = os.path.join(BASE, "model", "xgboost_balanced_c75.json")
if os.path.exists(m_c_path):
    t_c, res_c = evaluate_model(m_c_path, "Model C (75% Hard + 25% Random Negatives)")
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["EXP-3", f"V6 Candidates (72.94% recall) + Model C (75/25) @ t={t_c:.2f}", "72.94%", len(v6_cands), f"{res_c['precision']*100:.2f}%", f"{res_c['recall']*100:.2f}%", f"{res_c['macro_f05']:.4f}", f"{time.time()-t0:.1f}", "Model C with 75% hard / 25% random negatives"])

print("\nEvaluation complete.")
