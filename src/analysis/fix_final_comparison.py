import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve, auc


BASE = Path("data/processed/results")
OUT = BASE / "final_comparison"
OUT.mkdir(exist_ok=True)


MODELS = {
    "XGBoost": {
        "binary": "xgboost_binary",
        "multiclass": "xgboost_multiclass",
    },
    "Random Forest": {
        "binary": "random_forest_binary",
        "multiclass": "random_forest_multiclass",
    },
    "Neural Network": {
        "binary": "neural_network_binary",
        "multiclass": "neural_network_multiclass",
    },
}


def load(folder):
    with open(BASE / folder / "metrics.json") as f:
        return json.load(f)


# ============================================================
# BINARY FINAL TABLE
# ============================================================

rows = []

for model, folders in MODELS.items():

    m = load(folders["binary"])

    rows.append({
        "Model": model,
        "Accuracy": m["accuracy"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1": m.get("f1_score", m.get("f1")),
        "ROC-AUC": m["roc_auc"],
        "FPR": m["false_positive_rate"],
        "Training Time (sec)": m["training_time_seconds"],
        "Prediction Time (sec)": m.get(
            "prediction_time_seconds",
            np.nan
        ),
    })


binary = pd.DataFrame(rows)

binary.to_csv(
    OUT / "binary_model_comparison.csv",
    index=False
)


# ============================================================
# MULTICLASS FINAL TABLE
# ============================================================

rows = []

for model, folders in MODELS.items():

    m = load(folders["multiclass"])

    rows.append({
        "Model": model,
        "Accuracy": m["accuracy"],
        "Weighted Precision": m["weighted_precision"],
        "Weighted Recall": m["weighted_recall"],
        "Weighted F1": m["weighted_f1"],
        "Macro Precision": m.get("macro_precision", np.nan),
        "Macro Recall": m.get("macro_recall", np.nan),
        "Macro F1": m.get("macro_f1", np.nan),
        "Training Time (sec)": m["training_time_seconds"],
        "Prediction Time (sec)": m.get(
            "prediction_time_seconds",
            np.nan
        ),
    })


multiclass = pd.DataFrame(rows)

multiclass.to_csv(
    OUT / "multiclass_model_comparison.csv",
    index=False
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 90)
print("FINAL BINARY MODEL COMPARISON")
print("=" * 90)

print(binary.to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}"
))


print("\n" + "=" * 90)
print("FINAL MULTICLASS MODEL COMPARISON")
print("=" * 90)

print(multiclass.to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}"
))


# ============================================================
# ROC + PR CURVES
# ============================================================

plt.figure(figsize=(9, 7))

for model, folders in MODELS.items():

    path = BASE / folders["binary"] / "predictions.csv"

    pred = pd.read_csv(path)

    # Case-insensitive column matching
    cols = {c.lower(): c for c in pred.columns}

    actual_col = cols["actual"]
    prob_col = cols["attack_probability"]

    y_true = pred[actual_col]

    if y_true.dtype == object:

        y_true = (
            y_true.astype(str)
            .str.upper()
            .map({
                "BENIGN": 0,
                "ATTACK": 1,
                "FALSE": 0,
                "TRUE": 1
            })
        )

    y_true = y_true.astype(int)

    y_prob = pred[prob_col].astype(float)

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    score = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{model} (AUC={score:.4f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Binary Classification ROC Curves")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUT / "binary_roc_curves.png",
    dpi=300
)

plt.close()


# ============================================================
# PRECISION-RECALL
# ============================================================

plt.figure(figsize=(9, 7))

for model, folders in MODELS.items():

    path = BASE / folders["binary"] / "predictions.csv"

    pred = pd.read_csv(path)

    cols = {c.lower(): c for c in pred.columns}

    actual_col = cols["actual"]
    prob_col = cols["attack_probability"]

    y_true = pred[actual_col]

    if y_true.dtype == object:

        y_true = (
            y_true.astype(str)
            .str.upper()
            .map({
                "BENIGN": 0,
                "ATTACK": 1,
                "FALSE": 0,
                "TRUE": 1
            })
        )

    y_true = y_true.astype(int)

    y_prob = pred[prob_col].astype(float)

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_prob
    )

    plt.plot(
        recall,
        precision,
        label=model
    )


plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(
    "Binary Classification Precision-Recall Curves"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUT / "binary_precision_recall_curves.png",
    dpi=300
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 90)
print("FINAL REPORT FIXED SUCCESSFULLY")
print("=" * 90)

print("\nSaved:")
print(OUT / "binary_model_comparison.csv")
print(OUT / "multiclass_model_comparison.csv")
print(OUT / "binary_roc_curves.png")
print(OUT / "binary_precision_recall_curves.png")

print("\nDONE.")
