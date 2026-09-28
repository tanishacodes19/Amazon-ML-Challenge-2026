import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# F0.5 SCORE
# ============================================================

def f0_5_score(y_true, y_pred):
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    beta = 0.5

    denominator = (beta ** 2 * precision) + recall

    if denominator == 0:
        return 0.0

    return (
        (1 + beta ** 2)
        * precision
        * recall
        / denominator
    )


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_thresholds(model, X_valid, y_valid):

    probabilities = model.predict_proba(X_valid)[:, 1]

    best_threshold = 0.5
    best_score = -1

    print("\nTHRESHOLD EVALUATION")
    print("=" * 70)

    for threshold in np.arange(0.30, 0.91, 0.05):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_valid,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_valid,
            predictions,
            zero_division=0
        )

        f05 = f0_5_score(
            y_valid,
            predictions
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F0.5: {f05:.4f}"
        )

        if f05 > best_score:
            best_score = f05
            best_threshold = threshold

    print("\nBEST THRESHOLD")
    print("=" * 70)
    print(f"Threshold : {best_threshold:.2f}")
    print(f"F0.5      : {best_score:.4f}")

    return best_threshold, best_score


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("XGBOOST MODEL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # TEMPORARY TEST DATA
    # ONLY to verify that the ML pipeline works.
    # We will NOT use this for the actual challenge.
    # --------------------------------------------------------

    rng = np.random.default_rng(42)

    X = rng.random((2000, 10))

    # Create a simple relationship between features and label
    score = (
        2.5 * X[:, 0]
        + 2.0 * X[:, 1]
        + 1.5 * X[:, 2]
        + 0.5 * X[:, 3]
    )

    y = (score > 3.0).astype(int)

    print(f"Samples         : {len(X)}")
    print(f"Features        : {X.shape[1]}")
    print(f"Positive matches: {y.sum()}")
    print(f"Negative pairs  : {(y == 0).sum()}")

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nDATA SPLIT")
    print("=" * 70)
    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_valid)}")

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=4
    )

    print("\nTRAINING XGBOOST...")
    print("=" * 70)

    model.fit(
        X_train,
        y_train
    )

    print("Training completed!")

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_valid,
        predictions,
        zero_division=0
    )

    f05 = f0_5_score(
        y_valid,
        predictions
    )

    print("\nMODEL PERFORMANCE @ 0.50")
    print("=" * 70)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"F0.5 Score: {f05:.4f}")

    # --------------------------------------------------------
    # FIND BEST THRESHOLD
    # --------------------------------------------------------

    best_threshold, best_f05 = evaluate_thresholds(
        model,
        X_valid,
        y_valid
    )

    print("\nMODEL TEST COMPLETE")
    print("=" * 70)
    print("XGBoost         : OK")
    print("F0.5 evaluation : OK")
    print("Threshold tune  : OK")