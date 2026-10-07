import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


BASE = Path("data/processed/results/xgboost_multiclass")

pred_path = BASE / "predictions.csv"
metrics_path = BASE / "metrics.json"

pred = pd.read_csv(pred_path)

print("Prediction columns:", list(pred.columns))

actual_col = next(
    c for c in pred.columns
    if c.lower() in ["actual", "y_true", "true_label"]
)

predicted_col = next(
    c for c in pred.columns
    if c.lower() in ["predicted", "y_pred", "prediction"]
)

y_true = pred[actual_col]
y_pred = pred[predicted_col]

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

with open(metrics_path, "r") as f:
    metrics = json.load(f)

metrics["macro_precision"] = float(macro_precision)
metrics["macro_recall"] = float(macro_recall)
metrics["macro_f1"] = float(macro_f1)

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("\nXGBoost Multiclass Macro Metrics")
print("=" * 50)
print(f"Macro Precision : {macro_precision:.6f}")
print(f"Macro Recall    : {macro_recall:.6f}")
print(f"Macro F1        : {macro_f1:.6f}")

print("\nmetrics.json updated successfully.")
