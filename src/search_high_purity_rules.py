import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, polars as pl
from sklearn.tree import DecisionTreeClassifier, export_text
from feature_engine_v2 import V5_FEATURES_EXPANDED

print("1. Loading validation benchmark data...")
gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')

import xgboost as xgb, lightgbm as lgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('model/xgboost_v13_52features.json')
lgb_model = lgb.Booster(model_file='model/lightgbm_v13_52features.txt')
p_ens = 0.50 * xgb_model.predict_proba(X_val)[:, 1] + 0.50 * lgb_model.predict(X_val)

s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)

# Baseline V16 mask
feat_dict = {f: X_val[:, idx] for idx, f in enumerate(V5_FEATURES_EXPANDED)}
name_r = feat_dict["name_ratio"]
addr_r = feat_dict["address_ratio"]
house_m = feat_dict["house_match"]
name_c = feat_dict["name_contains"]
is_acronym = feat_dict["name_acronym_match"]

base_gate = (p_ens >= 0.972) & ((name_r >= 0.20) | (addr_r >= 0.40) | (house_m == 1))
rescue1 = (p_ens >= 0.90) & (((name_r >= 0.60) & (addr_r >= 0.60)) | ((name_r >= 0.90) & (house_m == 1)) | (addr_r >= 0.95))
co_location = (name_r >= 0.40)
r_contain = (p_ens >= 0.95) & (name_c == 1) & (addr_r >= 0.65)
r_acronym = (p_ens >= 0.90) & (is_acronym == 1) & (addr_r >= 0.40)
v16_mask = ((base_gate | rescue1) & co_location) | r_contain | r_acronym

# Unaccepted pool
unacc = ~v16_mask
X_unacc = X_val[unacc]
y_unacc = is_gt[unacc].astype(int)

print(f"Unaccepted candidate pairs: {len(X_unacc):,} (Positives: {y_unacc.sum():,}, Negatives: {(y_unacc==0).sum():,})")
print(f"Baseline purity of unaccepted pool: {y_unacc.sum() / len(X_unacc) * 100:.2f}%")

# Train shallow DecisionTreeClassifier to discover any high-purity leaf
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, class_weight='balanced', random_state=42)
dt.fit(X_unacc, y_unacc)

# Get predictions and leaf node assignments
leaf_ids = dt.apply(X_unacc)
unique_leaves = np.unique(leaf_ids)

print("\n--- Decision Tree Leaves Purity Analysis ---")
found_high_purity = False
for leaf in unique_leaves:
    mask = leaf_ids == leaf
    n_total = mask.sum()
    n_pos = y_unacc[mask].sum()
    n_neg = n_total - n_pos
    purity = n_pos / n_total if n_total > 0 else 0
    if purity >= 0.70 and n_pos >= 50:
        found_high_purity = True
        print(f"Leaf {leaf}: TP={n_pos:,}, FP={n_neg:,}, Purity={purity*100:.2f}%, Total={n_total:,}")

if not found_high_purity:
    print("No leaf found with purity >= 70% and TP >= 50.")
    print("Highest purity leaves:")
    leaves_by_purity = []
    for leaf in unique_leaves:
        mask = leaf_ids == leaf
        n_total = mask.sum()
        n_pos = y_unacc[mask].sum()
        purity = n_pos / n_total if n_total > 0 else 0
        leaves_by_purity.append((purity, n_pos, n_total - n_pos, leaf))
    leaves_by_purity.sort(reverse=True)
    for purity, n_pos, n_neg, leaf in leaves_by_purity[:5]:
        print(f"  Leaf {leaf}: TP={n_pos:,}, FP={n_neg:,}, Purity={purity*100:.2f}%, Total={n_pos+n_neg:,}")

print("\nDecision Tree Rules:")
print(export_text(dt, feature_names=V5_FEATURES_EXPANDED))
