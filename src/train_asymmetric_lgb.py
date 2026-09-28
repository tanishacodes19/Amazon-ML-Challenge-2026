import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time, os, numpy as np, polars as pl
import lightgbm as lgb
from feature_engine_v2 import V5_FEATURES_EXPANDED

print("1. Loading V17 training data (319k pairs)...")
t0 = time.time()
X_train = np.load('model/train_v17_X.npy')
y_train = np.load('model/train_v17_y.npy')
print(f"Loaded X_train {X_train.shape} in {time.time()-t0:.1f}s")

# Train LightGBM with scale_pos_weight = 0.25 (4x heavy penalty on FP)
print("\n2. Training Asymmetric Precision-Heavy LightGBM (scale_pos_weight=0.25)...")
t_train = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    num_leaves=127,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.80,
    colsample_bytree=0.80,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=0.25, # 4x penalty on false positive!
    random_state=42,
    n_jobs=4,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_model.booster_.save_model('model/lightgbm_asym_52features.txt')
print(f"Training completed and saved to model/lightgbm_asym_52features.txt in {time.time()-t_train:.1f}s")

print("\n3. Loading validation benchmark data (623k pairs)...")
t_val = time.time()
X_val = np.load('validation_benchmark/val_v16_features52_X.npy')
pairs_df = pl.read_parquet('validation_benchmark/val_v16_features52_df.parquet')
gt = pl.read_parquet('validation_benchmark/val_gt.parquet')
s1_ids = pairs_df['source1_entity_id'].to_numpy()
m_ids = pairs_df['matched_entity_id'].to_numpy()
gt_set = set(zip(gt['source1_entity_id'], gt['matched_entity_id']))
is_gt = np.array([pair in gt_set for pair in zip(s1_ids, m_ids)], dtype=bool)
total_gt = len(gt_set)

print("\n4. Predicting on validation set...")
p_val = lgb_model.predict_proba(X_val)[:, 1]

def eval_injective(p_scores, thresh):
    mask = p_scores >= thresh
    cands = np.where(mask)[0]
    s_ord = cands[np.argsort(-p_scores[cands])]
    s23_seen = set()
    accepted = []
    for i in s_ord:
        mid = m_ids[i]
        if mid not in s23_seen:
            s23_seen.add(mid)
            accepted.append(i)
            
    m_eval = np.zeros(len(p_scores), dtype=bool)
    m_eval[accepted] = True
    tp = (m_eval & is_gt).sum()
    fp = (m_eval & ~is_gt).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / total_gt
    f05 = 1.25 * p * r / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
    return tp, fp, p, r, f05

print("\n--- Sweeping Thresholds for Asymmetric Model ---")
for t in [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30]:
    tp, fp, p, r, f05 = eval_injective(p_val, t)
    print(f"t={t:.2f} | TP={tp:6,d} | FP={fp:5,d} | P={p*100:6.2f}% | R={r*100:6.2f}% | F0.5={f05:.6f}")
