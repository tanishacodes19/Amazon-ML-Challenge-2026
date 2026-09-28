import os
import time
import duckdb
import numpy as np
import polars as pl
import xgboost as xgb
from eval_framework import V4_FEATURES

BASE = r"C:\Users\Admin\Documents\Amazon-ML-Challenge-2026"
FEATURE_DIR = os.path.join(BASE, "training_data", "features_v4")
FEATURE_GLOB = os.path.join(FEATURE_DIR, "features_*.parquet").replace("\\", "/")
MODEL_DIR = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 4: TRAINING BALANCED-NEGATIVE XGBOOST MODELS")
print("=" * 70)

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='3500MB'")

feature_cols = ", ".join(V4_FEATURES)

print("Extracting 1,000,000 positive training pairs...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE positives AS
SELECT {feature_cols}, label, source1_entity_id
FROM read_parquet('{FEATURE_GLOB}')
WHERE label = 1
USING SAMPLE reservoir (1000000 ROWS) REPEATABLE (42)
""")
pos_count = con.execute("SELECT COUNT(*) FROM positives").fetchone()[0]
print(f"Positives: {pos_count:,}")

print("Extracting hard-negative pool...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE hard_neg_pool AS
SELECT {feature_cols}, label, source1_entity_id
FROM read_parquet('{FEATURE_GLOB}')
WHERE label = 0
  AND (
      name_contains = 1
      OR name_first_token_match = 1
      OR name_token_jaccard >= 0.30
      OR name_ratio >= 0.60
      OR name_partial_ratio >= 0.70
      OR address_contains = 1
      OR address_ratio >= 0.50
      OR house_match = 1
      OR strong_name_address = 1
      OR strong_name_house = 1
  )
""")
hard_pool_count = con.execute("SELECT COUNT(*) FROM hard_neg_pool").fetchone()[0]
print(f"Hard negative pool: {hard_pool_count:,}")

print("Extracting random-negative pool (excluding hard lookalikes)...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE random_neg_pool AS
SELECT {feature_cols}, label, source1_entity_id
FROM read_parquet('{FEATURE_GLOB}')
WHERE label = 0
  AND name_ratio < 0.60
  AND address_ratio < 0.50
  AND name_first_token_match = 0
  AND house_match = 0
""")
random_pool_count = con.execute("SELECT COUNT(*) FROM random_neg_pool").fetchone()[0]
print(f"Random negative pool: {random_pool_count:,}")

def train_and_save_model(model_name, n_hard, n_random):
    print(f"\nBuilding dataset for {model_name}: {n_hard:,} hard + {n_random:,} random negatives...")
    
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE combined_data AS
    SELECT * FROM positives
    UNION ALL
    SELECT * FROM hard_neg_pool USING SAMPLE reservoir ({n_hard} ROWS) REPEATABLE (43)
    UNION ALL
    SELECT * FROM random_neg_pool USING SAMPLE reservoir ({n_random} ROWS) REPEATABLE (44)
    """)
    
    # Entity-level train/validation split
    con.execute("""
    CREATE OR REPLACE TEMP TABLE train_set AS
    SELECT * FROM combined_data
    WHERE ABS(HASH(source1_entity_id)) % 4 != 0
    """)
    
    con.execute("""
    CREATE OR REPLACE TEMP TABLE val_set AS
    SELECT * FROM combined_data
    WHERE ABS(HASH(source1_entity_id)) % 4 = 0
    """)
    
    n_train = con.execute("SELECT COUNT(*) FROM train_set").fetchone()[0]
    n_val = con.execute("SELECT COUNT(*) FROM val_set").fetchone()[0]
    print(f"Train rows: {n_train:,} | Valid rows: {n_val:,}")
    
    train_df = con.execute(f"SELECT {feature_cols}, label FROM train_set").pl()
    val_df = con.execute(f"SELECT {feature_cols}, label FROM val_set").pl()
    
    X_train = train_df.select(V4_FEATURES).fill_null(0).to_numpy().astype(np.float32)
    y_train = train_df["label"].to_numpy().astype(np.int8)
    
    X_val = val_df.select(V4_FEATURES).fill_null(0).to_numpy().astype(np.float32)
    y_val = val_df["label"].to_numpy().astype(np.int8)
    
    print(f"Training XGBoost for {model_name}...")
    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=900,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.90,
        min_child_weight=3,
        gamma=0,
        reg_alpha=0.10,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_bin=256,
        random_state=42,
        n_jobs=2
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=150
    )
    
    model_path = os.path.join(MODEL_DIR, f"{model_name}.json")
    clf.save_model(model_path)
    print(f"Model saved to: {model_path} (Training time: {time.time()-t0:.1f}s)")
    return model_path

# Train Model B: 50% hard + 50% random negatives (500k each)
train_and_save_model("xgboost_balanced_b50", n_hard=500_000, n_random=500_000)

# Train Model C: 75% hard + 25% random negatives (750k hard + 250k random)
train_and_save_model("xgboost_balanced_c75", n_hard=750_000, n_random=250_000)

print("\nAll balanced models trained successfully.")
