import os
import duckdb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from rapidfuzz import fuzz

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
VAL_DIR = os.path.join(BASE, "validation_benchmark")

# 41 Features currently in V4 / V5.5
V4_FEATURES = [
    "name_exact", "name_contains", "name_prefix3", "name_prefix5", "name_suffix4",
    "name_first_token_match", "name_last_token_match", "name_shared_token_count", "name_token_jaccard",
    "name_ratio", "name_partial_ratio", "name_token_sort", "name_token_set",
    "name_len_s1", "name_len_match", "name_len_abs_diff", "name_len_relative_diff",
    "address_exact", "address_contains", "address_prefix5", "address_prefix8", "address_suffix8",
    "house_match", "address_ratio", "address_partial_ratio", "address_token_sort", "address_token_set",
    "address_token_jaccard", "address_numeric_overlap", "address_len_s1", "address_len_match",
    "address_len_abs_diff", "address_len_relative_diff", "country_match", "country_missing",
    "source2", "source3", "name_address_ratio_mean", "name_address_ratio_product",
    "strong_name_address", "strong_name_house"
]

def load_benchmark():
    s1 = pl.read_parquet(os.path.join(VAL_DIR, "val_s1.parquet"))
    gt = pl.read_parquet(os.path.join(VAL_DIR, "val_gt.parquet"))
    cands = pl.read_parquet(os.path.join(VAL_DIR, "val_cands.parquet"))
    s23 = pl.read_parquet(os.path.join(VAL_DIR, "val_s23.parquet"))
    return s1, gt, cands, s23

def compute_blocking_metrics(cands_df, gt_df, n_s1=25000):
    total_candidates = len(cands_df)
    total_gt = len(gt_df)
    
    # Inner join to find recovered pairs
    recovered = len(gt_df.join(
        cands_df.unique(subset=["source1_entity_id", "matched_entity_id"]),
        on=["source1_entity_id", "matched_entity_id"],
        how="inner"
    ))
    
    recall = recovered / total_gt if total_gt > 0 else 0.0
    avg_cands = total_candidates / n_s1
    
    return {
        "candidate_count": total_candidates,
        "avg_candidates_per_s1": avg_cands,
        "ground_truth_count": total_gt,
        "recovered_matches": recovered,
        "blocking_recall": recall
    }

def compute_macro_f05(pred_pairs_df, gt_df, s1_df):
    """
    Computes exact Macro F0.5 across all S1 entities, including singletons.
    """
    all_s1 = set(s1_df["source1_entity_id"].to_list())
    n_s1 = len(all_s1)
    
    gt_map = {}
    for row in gt_df.iter_rows():
        gt_map.setdefault(row[0], set()).add(row[1])
        
    pred_map = {}
    if len(pred_pairs_df) > 0:
        for row in pred_pairs_df.iter_rows():
            pred_map.setdefault(row[0], set()).add(row[1])
            
    scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    singletons = 0
    singleton_fp = 0
    
    for s1_id in all_s1:
        true_set = gt_map.get(s1_id, set())
        pred_set = pred_map.get(s1_id, set())
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if len(true_set) == 0:
            singletons += 1
            if len(pred_set) > 0:
                singleton_fp += 1
                f05 = 0.0
            else:
                f05 = 1.0
        else:
            if tp == 0:
                f05 = 0.0
            else:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f05 = (1.25 * p * r) / (0.25 * p + r)
                
        scores.append(f05)
        
    macro_f05 = float(np.mean(scores))
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    singleton_fp_rate = singleton_fp / singletons if singletons > 0 else 0.0
    
    return {
        "macro_f05": macro_f05,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "predicted_matches": total_tp + total_fp,
        "singletons": singletons,
        "singleton_fp": singleton_fp,
        "singleton_fp_rate": singleton_fp_rate
    }
