import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc
)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("data/processed/results")
OUTPUT_DIR = BASE_DIR / "final_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "XGBoost": {
        "binary": BASE_DIR / "xgboost_binary",
        "multiclass": BASE_DIR / "xgboost_multiclass",
    },
    "Random Forest": {
        "binary": BASE_DIR / "random_forest_binary",
        "multiclass": BASE_DIR / "random_forest_multiclass",
    },
    "Neural Network": {
        "binary": BASE_DIR / "neural_network_binary",
        "multiclass": BASE_DIR / "neural_network_multiclass",
    },
}


# ============================================================
# LOAD METRICS
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


binary_rows = []
multiclass_rows = []

for model, paths in MODELS.items():

    binary_metrics = load_json(paths["binary"] / "metrics.json")
    multiclass_metrics = load_json(paths["multiclass"] / "metrics.json")

    binary_rows.append({
        "Model": model,
        "Accuracy": binary_metrics.get("accuracy"),
        "Precision": binary_metrics.get("precision"),
        "Recall": binary_metrics.get("recall"),
        "F1": binary_metrics.get("f1"),
        "ROC-AUC": binary_metrics.get("roc_auc"),
        "FPR": binary_metrics.get("fpr"),
        "Training Time (sec)": binary_metrics.get("training_time_seconds"),
        "Prediction Time (sec)": binary_metrics.get("prediction_time_seconds"),
    })

    multiclass_rows.append({
        "Model": model,
        "Accuracy": multiclass_metrics.get("accuracy"),
        "Weighted Precision": multiclass_metrics.get("weighted_precision"),
        "Weighted Recall": multiclass_metrics.get("weighted_recall"),
        "Weighted F1": multiclass_metrics.get("weighted_f1"),
        "Macro Precision": multiclass_metrics.get("macro_precision"),
        "Macro Recall": multiclass_metrics.get("macro_recall"),
        "Macro F1": multiclass_metrics.get("macro_f1"),
        "Training Time (sec)": multiclass_metrics.get("training_time_seconds"),
        "Prediction Time (sec)": multiclass_metrics.get("prediction_time_seconds"),
    })


binary_df = pd.DataFrame(binary_rows)
multiclass_df = pd.DataFrame(multiclass_rows)

# Save tables
binary_df.to_csv(
    OUTPUT_DIR / "binary_model_comparison.csv",
    index=False
)

multiclass_df.to_csv(
    OUTPUT_DIR / "multiclass_model_comparison.csv",
    index=False
)


# ============================================================
# PRINT FINAL TABLES
# ============================================================

print("\n" + "=" * 80)
print("FINAL BINARY MODEL COMPARISON")
print("=" * 80)

print(binary_df.to_string(index=False))

print("\n" + "=" * 80)
print("FINAL MULTICLASS MODEL COMPARISON")
print("=" * 80)

print(multiclass_df.to_string(index=False))


# ============================================================
# HELPER: SAVE BAR CHART
# ============================================================

def save_bar_chart(df, metric, filename, title, ylabel="Score"):

    plt.figure(figsize=(9, 6))

    values = df[metric].astype(float)

    plt.bar(df["Model"], values)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Model")

    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)

    for i, value in enumerate(values):
        plt.text(
            i,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


# ============================================================
# BINARY PERFORMANCE CHARTS
# ============================================================

for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]:

    save_bar_chart(
        binary_df,
        metric,
        f"binary_{metric.lower().replace('-', '_')}.png",
        f"Binary Classification - {metric} Comparison"
    )


save_bar_chart(
    binary_df,
    "FPR",
    "binary_fpr_comparison.png",
    "Binary Classification - False Positive Rate",
    ylabel="False Positive Rate"
)


# ============================================================
# MULTICLASS PERFORMANCE
# ============================================================

for metric in [
    "Accuracy",
    "Weighted F1",
    "Macro F1"
]:

    save_bar_chart(
        multiclass_df,
        metric,
        f"multiclass_{metric.lower().replace(' ', '_')}.png",
        f"Multiclass Classification - {metric} Comparison"
    )


# ============================================================
# COMBINED MODEL PERFORMANCE
# ============================================================

plt.figure(figsize=(10, 6))

x = np.arange(len(binary_df))
width = 0.25

plt.bar(
    x - width,
    binary_df["Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x,
    binary_df["Precision"],
    width,
    label="Precision"
)

plt.bar(
    x + width,
    binary_df["F1"],
    width,
    label="F1"
)

plt.xticks(x, binary_df["Model"])
plt.ylabel("Score")
plt.title("Binary Classification Performance Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "binary_combined_performance.png",
    dpi=300
)

plt.close()


# ============================================================
# TRAINING TIME COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    binary_df["Model"],
    binary_df["Training Time (sec)"]
)

plt.title("Binary Model Training Time Comparison")
plt.ylabel("Training Time (seconds)")
plt.xlabel("Model")

plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "training_time_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# CONFUSION MATRICES
# ============================================================

def plot_confusion_matrix(model, task):

    path = MODELS[model][task] / "confusion_matrix.csv"

    if not path.exists():
        print(f"Missing: {path}")
        return

    cm = pd.read_csv(path, index_col=0)

    matrix = cm.values

    plt.figure(figsize=(10, 8))

    plt.imshow(matrix, aspect="auto")

    plt.title(f"{model} - {task.title()} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.colorbar()

    if len(cm.columns) <= 20:

        plt.xticks(
            range(len(cm.columns)),
            cm.columns,
            rotation=90,
            fontsize=7
        )

        plt.yticks(
            range(len(cm.index)),
            cm.index,
            fontsize=7
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{model.lower().replace(' ', '_')}_{task}_confusion_matrix.png",
        dpi=300
    )

    plt.close()


for model in MODELS:

    plot_confusion_matrix(model, "binary")
    plot_confusion_matrix(model, "multiclass")


# ============================================================
# BINARY ROC CURVES
# ============================================================

plt.figure(figsize=(9, 7))

for model, paths in MODELS.items():

    prediction_file = paths["binary"] / "predictions.csv"

    if not prediction_file.exists():
        print(f"Missing: {prediction_file}")
        continue

    pred = pd.read_csv(prediction_file)

    print(
        f"\n{model} binary prediction columns:",
        list(pred.columns)
    )

    # Automatically identify actual and probability columns
    actual_candidates = [
        "y_true",
        "true_label",
        "actual",
        "label",
        "BinaryLabel"
    ]

    probability_candidates = [
        "probability",
        "prob_attack",
        "attack_probability",
        "y_prob",
        "prob_1"
    ]

    actual_col = next(
        (c for c in actual_candidates if c in pred.columns),
        None
    )

    prob_col = next(
        (c for c in probability_candidates if c in pred.columns),
        None
    )

    if actual_col is None or prob_col is None:
        print(f"Could not identify columns for {model}")
        continue

    y_true = pred[actual_col]

    # Convert labels if necessary
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

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{model} (AUC={roc_auc:.4f})"
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
    OUTPUT_DIR / "binary_roc_curves.png",
    dpi=300
)

plt.close()


# ============================================================
# BINARY PRECISION-RECALL CURVES
# ============================================================

plt.figure(figsize=(9, 7))

for model, paths in MODELS.items():

    prediction_file = paths["binary"] / "predictions.csv"

    if not prediction_file.exists():
        continue

    pred = pd.read_csv(prediction_file)

    actual_candidates = [
        "y_true",
        "true_label",
        "actual",
        "label",
        "BinaryLabel"
    ]

    probability_candidates = [
        "probability",
        "prob_attack",
        "attack_probability",
        "y_prob",
        "prob_1"
    ]

    actual_col = next(
        (c for c in actual_candidates if c in pred.columns),
        None
    )

    prob_col = next(
        (c for c in probability_candidates if c in pred.columns),
        None
    )

    if actual_col is None or prob_col is None:
        continue

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
plt.title("Binary Classification Precision-Recall Curves")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "binary_precision_recall_curves.png",
    dpi=300
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

for model in ["XGBoost", "Random Forest"]:

    path = MODELS[model]["binary"] / "feature_importance.csv"

    if not path.exists():
        continue

    fi = pd.read_csv(path)

    # Detect feature and importance columns
    feature_col = None
    importance_col = None

    for col in fi.columns:

        lower = col.lower()

        if "feature" in lower or "name" in lower:
            feature_col = col

        if "importance" in lower:
            importance_col = col

    if feature_col is None:
        feature_col = fi.columns[0]

    if importance_col is None:
        importance_col = fi.columns[1]

    fi = fi.sort_values(
        importance_col,
        ascending=False
    ).head(20)

    plt.figure(figsize=(10, 8))

    plt.barh(
        fi[feature_col][::-1],
        fi[importance_col][::-1]
    )

    plt.title(
        f"{model} - Top 20 Feature Importance"
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{model.lower().replace(' ', '_')}_top20_feature_importance.png",
        dpi=300
    )

    plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL COMPARISON COMPLETED")
print("=" * 80)

print(f"\nOutput directory:")
print(OUTPUT_DIR.resolve())

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file.name)

print("\nDone.")
