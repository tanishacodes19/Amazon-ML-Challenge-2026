import re
import numpy as np
import polars as pl
from rapidfuzz import fuzz

V5_FEATURES_EXPANDED = [
    # --- NAME SIMILARITY (19 features) ---
    "name_exact",
    "name_contains",
    "name_prefix3",
    "name_prefix5",
    "name_suffix4",
    "name_first_token_match",
    "name_last_token_match",
    "name_shared_token_count",
    "name_token_jaccard",
    "name_token_dice",
    "name_char3_jaccard",
    "name_char4_jaccard",
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort",
    "name_token_set",
    "name_acronym_match",
    "name_len_diff",
    "name_len_rel_diff",

    # --- ADDRESS SIMILARITY (18 features) ---
    "address_exact",
    "address_contains",
    "address_prefix5",
    "address_prefix8",
    "address_suffix8",
    "address_char3_jaccard",
    "address_char4_jaccard",
    "house_match",
    "postal_exact",
    "postal_prefix3",
    "address_ratio",
    "address_partial_ratio",
    "address_token_sort",
    "address_token_set",
    "address_token_jaccard",
    "address_numeric_overlap",
    "address_len_diff",
    "address_len_rel_diff",

    # --- COUNTRY & METADATA (4 features) ---
    "country_match",
    "country_missing",
    "source2",
    "source3",

    # --- CROSS / INTERACTION / GATING FEATURES (11 features) ---
    "name_address_ratio_mean",
    "name_address_ratio_product",
    "name_address_harmonic",
    "name_postal_cross",
    "name_house_cross",
    "strong_name_address",
    "strong_name_house",
    "strong_name_postal",
    "name_gate_40",
    "name_gate_60",
    "addr_gate_50"
]

def get_char_ngrams(s: str, n: int) -> set:
    if not s or len(s) < n:
        return set()
    s_squashed = "".join(s.split())
    return {s_squashed[i:i+n] for i in range(len(s_squashed) - n + 1)}

def jaccard_sets(A: set, B: set) -> float:
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def dice_sets(A: set, B: set) -> float:
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return (2.0 * len(A & B)) / (len(A) + len(B))

def is_acronym(s1: str, s2: str) -> int:
    t1 = s1.split()
    t2 = s2.split()
    if len(t1) >= 2 and len(s2) == len(t1):
        acr = "".join(w[0] for w in t1 if w)
        if acr == s2:
            return 1
    if len(t2) >= 2 and len(s1) == len(t2):
        acr = "".join(w[0] for w in t2 if w)
        if acr == s1:
            return 1
    return 0

def extract_features_df(pairs_df):
    """
    Input df expected columns:
      s1_name_norm, matched_name_norm
      s1_addr_norm, matched_addr_norm
      s1_house, matched_house
      s1_postal, matched_postal
      s1_country, matched_country
      matched_entity_id
    """
    rows = []
    
    for r in pairs_df.iter_rows(named=True):
        n1 = r.get("s1_name_norm") or ""
        n2 = r.get("matched_name_norm") or ""
        a1 = r.get("s1_addr_norm") or ""
        a2 = r.get("matched_addr_norm") or ""
        
        h1 = r.get("s1_house") or ""
        h2 = r.get("matched_house") or ""
        p1 = r.get("s1_postal") or ""
        p2 = r.get("matched_postal") or ""
        c1 = r.get("s1_country") or ""
        c2 = r.get("matched_country") or ""
        mid = r.get("matched_entity_id") or ""
        
        nt1 = set(n1.split())
        nt2 = set(n2) if isinstance(n2, set) else set(n2.split())
        at1 = set(a1.split())
        at2 = set(a2) if isinstance(a2, set) else set(a2.split())
        
        # Numeric tokens
        num1 = {x for x in at1 if any(c.isdigit() for c in x)}
        num2 = {x for x in at2 if any(c.isdigit() for c in x)}
        
        # Character n-grams
        n_g3_1 = get_char_ngrams(n1, 3)
        n_g3_2 = get_char_ngrams(n2, 3)
        n_g4_1 = get_char_ngrams(n1, 4)
        n_g4_2 = get_char_ngrams(n2, 4)
        
        a_g3_1 = get_char_ngrams(a1, 3)
        a_g3_2 = get_char_ngrams(a2, 3)
        a_g4_1 = get_char_ngrams(a1, 4)
        a_g4_2 = get_char_ngrams(a2, 4)
        
        # Similarities
        name_ratio = (fuzz.ratio(n1, n2) / 100.0) if (n1 and n2) else 0.0
        addr_ratio = (fuzz.ratio(a1, a2) / 100.0) if (a1 and a2) else 0.0
        name_partial = (fuzz.partial_ratio(n1, n2) / 100.0) if (n1 and n2) else 0.0
        addr_partial = (fuzz.partial_ratio(a1, a2) / 100.0) if (a1 and a2) else 0.0
        name_sort = (fuzz.token_sort_ratio(n1, n2) / 100.0) if (n1 and n2) else 0.0
        name_set = (fuzz.token_set_ratio(n1, n2) / 100.0) if (n1 and n2) else 0.0
        addr_sort = (fuzz.token_sort_ratio(a1, a2) / 100.0) if (a1 and a2) else 0.0
        addr_set = (fuzz.token_set_ratio(a1, a2) / 100.0) if (a1 and a2) else 0.0
        
        tok_j = jaccard_sets(nt1, nt2)
        tok_d = dice_sets(nt1, nt2)
        addr_tok_j = jaccard_sets(at1, at2)
        num_j = jaccard_sets(num1, num2)
        
        h_match = int(bool(h1) and bool(h2) and h1 == h2)
        p_match = int(bool(p1) and bool(p2) and p1 == p2)
        p_pref = int(bool(p1) and bool(p2) and p1[:3] == p2[:3] and len(p1) >= 3 and len(p2) >= 3)
        
        f1 = n1.split()[0] if n1 else ""
        f2 = n2.split()[0] if n2 else ""
        l1 = n1.split()[-1] if n1 else ""
        l2 = n2.split()[-1] if n2 else ""
        
        acr_match = is_acronym(n1, n2)
        
        harmonic_mean = (2.0 * name_ratio * addr_ratio) / (name_ratio + addr_ratio + 1e-6)
        
        rows.append({
            # NAME
            "name_exact": int(n1 == n2 and n1 != ""),
            "name_contains": int(n1 != "" and n2 != "" and (n1 in n2 or n2 in n1)),
            "name_prefix3": int(n1[:3] == n2[:3] and len(n1) >= 3 and len(n2) >= 3),
            "name_prefix5": int(n1[:5] == n2[:5] and len(n1) >= 5 and len(n2) >= 5),
            "name_suffix4": int(n1[-4:] == n2[-4:] and len(n1) >= 4 and len(n2) >= 4),
            "name_first_token_match": int(f1 == f2 and f1 != ""),
            "name_last_token_match": int(l1 == l2 and l1 != ""),
            "name_shared_token_count": len(nt1 & nt2),
            "name_token_jaccard": tok_j,
            "name_token_dice": tok_d,
            "name_char3_jaccard": jaccard_sets(n_g3_1, n_g3_2),
            "name_char4_jaccard": jaccard_sets(n_g4_1, n_g4_2),
            "name_ratio": name_ratio,
            "name_partial_ratio": name_partial,
            "name_token_sort": name_sort,
            "name_token_set": name_set,
            "name_acronym_match": acr_match,
            "name_len_diff": abs(len(n1) - len(n2)),
            "name_len_rel_diff": abs(len(n1) - len(n2)) / max(len(n1), 1),

            # ADDRESS
            "address_exact": int(a1 == a2 and a1 != ""),
            "address_contains": int(a1 != "" and a2 != "" and (a1 in a2 or a2 in a1)),
            "address_prefix5": int(a1[:5] == a2[:5] and len(a1) >= 5 and len(a2) >= 5),
            "address_prefix8": int(a1[:8] == a2[:8] and len(a1) >= 8 and len(a2) >= 8),
            "address_suffix8": int(a1[-8:] == a2[-8:] and len(a1) >= 8 and len(a2) >= 8),
            "address_char3_jaccard": jaccard_sets(a_g3_1, a_g3_2),
            "address_char4_jaccard": jaccard_sets(a_g4_1, a_g4_2),
            "house_match": h_match,
            "postal_exact": p_match,
            "postal_prefix3": p_pref,
            "address_ratio": addr_ratio,
            "address_partial_ratio": addr_partial,
            "address_token_sort": addr_sort,
            "address_token_set": addr_set,
            "address_token_jaccard": addr_tok_j,
            "address_numeric_overlap": num_j,
            "address_len_diff": abs(len(a1) - len(a2)),
            "address_len_rel_diff": abs(len(a1) - len(a2)) / max(len(a1), 1),

            # COUNTRY / SOURCE
            "country_match": int(c1 != "" and c1 == c2),
            "country_missing": int(c1 == "" or c2 == ""),
            "source2": int(mid.startswith("S2-")),
            "source3": int(mid.startswith("S3-")),

            # CROSS
            "name_address_ratio_mean": (name_ratio + addr_ratio) / 2.0,
            "name_address_ratio_product": name_ratio * addr_ratio,
            "name_address_harmonic": harmonic_mean,
            "name_postal_cross": name_ratio * p_match,
            "name_house_cross": name_ratio * h_match,
            "strong_name_address": int(name_ratio >= 0.90 and addr_ratio >= 0.70),
            "strong_name_house": int(name_ratio >= 0.90 and h_match == 1),
            "strong_name_postal": int(name_ratio >= 0.90 and p_match == 1),
            "name_gate_40": int(name_ratio >= 0.40 or tok_j >= 0.35 or acr_match == 1),
            "name_gate_60": int(name_ratio >= 0.60 or tok_j >= 0.50),
            "addr_gate_50": int(addr_ratio >= 0.50 or h_match == 1 or p_match == 1)
        })
        
    return pl.DataFrame(rows)
