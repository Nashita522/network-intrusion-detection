import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "data/processed/results/xgboost_binary/xgboost_binary.json"
FEATURE_PATH = "data/processed/leakage_safe/model_ready/selected_features.csv"
DATA_DIR = "data/processed/unseen_day"

OUTPUT_DIR = "data/processed/results/unseen_day_xgboost"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# IMPORTANT:
# XGBoost model saved as JSON can be loaded directly.
# ============================================================

from xgboost import XGBClassifier

print("=" * 80)
print("UNSEEN-DAY XGBOOST ROBUSTNESS EVALUATION")
print("=" * 80)

# ------------------------------------------------------------
# Load selected features
# ------------------------------------------------------------

print("\nLoading selected features...")

feature_df = pd.read_csv(FEATURE_PATH)

if "Feature" in feature_df.columns:
    selected_features = feature_df["Feature"].tolist()
else:
    selected_features = feature_df.iloc[:, 0].tolist()

print(f"Selected features: {len(selected_features)}")

# ------------------------------------------------------------
# Load trained XGBoost model
# ------------------------------------------------------------

print("\nLoading trained XGBoost model...")

model = XGBClassifier()
model.load_model(MODEL_PATH)

print("XGBoost model loaded successfully.")

# ------------------------------------------------------------
# Friday scenarios
# ------------------------------------------------------------

test_scenarios = [
    "Friday_DDoS",
    "Friday_PortScan",
    "Friday_Morning",
]

all_results = []
all_predictions = []

# ============================================================
# Evaluate each unseen Friday scenario
# ============================================================

for scenario in test_scenarios:

    filename = f"{scenario}_unseen_dataset.csv"
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        print(f"\nWARNING: Missing {path}")
        continue

    print("\n" + "=" * 80)
    print(f"EVALUATING: {scenario}")
    print("=" * 80)

    df = pd.read_csv(path, low_memory=False)

    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # Clean feature values
    # --------------------------------------------------------

    df = df.replace([np.inf, -np.inf], np.nan)

    # Train-only selected features
    missing_features = [
        f for f in selected_features
        if f not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing selected features in {scenario}: "
            f"{missing_features}"
        )

    X = df[selected_features].copy()

    # --------------------------------------------------------
    # IMPORTANT:
    # Use medians from training data only.
    # These are calculated from the original training set.
    # --------------------------------------------------------

    train_X_path = (
        "data/processed/leakage_safe/model_ready/"
        "X_train_binary_selected.csv"
    )

    print("Loading training data for train-only imputation...")

    train_sample = pd.read_csv(
        train_X_path,
        nrows=100000,
        low_memory=False
    )

    train_medians = train_sample.median(numeric_only=True)

    X = X.fillna(train_medians)
    X = X.fillna(0)

    # --------------------------------------------------------
    # Binary ground truth
    # --------------------------------------------------------

    y = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .ne("BENIGN")
        .astype(int)
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("Running prediction...")

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(
        y, predictions, zero_division=0
    )
    recall = recall_score(
        y, predictions, zero_division=0
    )
    f1 = f1_score(
        y, predictions, zero_division=0
    )

    if len(np.unique(y)) == 2:
        roc_auc = roc_auc_score(y, probabilities)
    else:
        roc_auc = None

    cm = confusion_matrix(y, predictions)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        tn = fp = fn = tp = None
        fpr = None

    print("\nResults:")
    print(f"Accuracy : {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1       : {f1:.6f}")
    print(f"ROC-AUC  : {roc_auc}")
    print(f"FPR      : {fpr}")

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save scenario metrics
    # --------------------------------------------------------

    metrics = {
        "scenario": scenario,
        "rows": int(len(df)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "fpr": float(fpr) if fpr is not None else None,
        "true_negative": int(tn) if tn is not None else None,
        "false_positive": int(fp) if fp is not None else None,
        "false_negative": int(fn) if fn is not None else None,
        "true_positive": int(tp) if tp is not None else None,
    }

    metrics_path = os.path.join(
        OUTPUT_DIR,
        f"{scenario}_metrics.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_BENIGN", "Actual_ATTACK"],
        columns=["Predicted_BENIGN", "Predicted_ATTACK"],
    )

    cm_path = os.path.join(
        OUTPUT_DIR,
        f"{scenario}_confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "Actual": y,
        "Predicted": predictions,
        "Attack_Probability": probabilities,
    })

    prediction_path = os.path.join(
        OUTPUT_DIR,
        f"{scenario}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    all_results.append(metrics)

    all_predictions.append(
        prediction_df.assign(Scenario=scenario)
    )

# ============================================================
# Overall Friday evaluation
# ============================================================

if not all_predictions:
    raise RuntimeError("No unseen-day scenarios were evaluated.")

print("\n" + "=" * 80)
print("OVERALL FRIDAY UNSEEN-DAY EVALUATION")
print("=" * 80)

combined_predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

y_all = combined_predictions["Actual"]
pred_all = combined_predictions["Predicted"]
prob_all = combined_predictions["Attack_Probability"]

accuracy = accuracy_score(y_all, pred_all)
precision = precision_score(
    y_all, pred_all, zero_division=0
)
recall = recall_score(
    y_all, pred_all, zero_division=0
)
f1 = f1_score(
    y_all, pred_all, zero_division=0
)
roc_auc = roc_auc_score(
    y_all, prob_all
)

cm = confusion_matrix(
    y_all,
    pred_all
)

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)

overall_metrics = {
    "evaluation": "Friday_Unseen_Day",
    "rows": int(len(combined_predictions)),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "fpr": float(fpr),
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),
}

print(f"\nAccuracy : {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall   : {recall:.6f}")
print(f"F1       : {f1:.6f}")
print(f"ROC-AUC  : {roc_auc:.6f}")
print(f"FPR      : {fpr:.6f}")

print("\nConfusion Matrix:")
print(cm)

# ------------------------------------------------------------
# Save overall results
# ------------------------------------------------------------

with open(
    os.path.join(
        OUTPUT_DIR,
        "overall_metrics.json"
    ),
    "w"
) as f:
    json.dump(
        overall_metrics,
        f,
        indent=4
    )

pd.DataFrame(all_results).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "scenario_comparison.csv"
    ),
    index=False
)

pd.DataFrame(cm).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "overall_confusion_matrix.csv"
    ),
    index=False
)

combined_predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "overall_predictions.csv"
    ),
    index=False
)

print("\n" + "=" * 80)
print("UNSEEN-DAY EVALUATION COMPLETE")
print("=" * 80)

print(f"Output directory: {OUTPUT_DIR}")
