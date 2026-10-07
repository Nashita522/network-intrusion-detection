import os
import json
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = "data/processed/results/unseen_day_xgboost"
OUTPUT_DIR = "data/processed/results/robustness_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("UNSEEN-DAY ROBUSTNESS ANALYSIS")
print("=" * 80)

# ============================================================
# Load scenario results
# ============================================================

scenario_file = os.path.join(
    INPUT_DIR,
    "scenario_comparison.csv"
)

df = pd.read_csv(scenario_file)

print("\nScenario results:")
print(df.to_string(index=False))

# ============================================================
# Load overall results
# ============================================================

overall_file = os.path.join(
    INPUT_DIR,
    "overall_metrics.json"
)

with open(overall_file, "r") as f:
    overall = json.load(f)

# ============================================================
# Save clean publication table
# ============================================================

table = df[
    [
        "scenario",
        "rows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "fpr",
    ]
].copy()

table["accuracy_percent"] = table["accuracy"] * 100
table["precision_percent"] = table["precision"] * 100
table["recall_percent"] = table["recall"] * 100
table["f1_percent"] = table["f1"] * 100
table["roc_auc_percent"] = table["roc_auc"] * 100
table["fpr_percent"] = table["fpr"] * 100

table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "unseen_day_robustness_results.csv"
    ),
    index=False
)

# ============================================================
# Overall Friday result
# ============================================================

overall_table = pd.DataFrame(
    [{
        "Evaluation": "Overall Friday Unseen-Day",
        "Rows": overall["rows"],
        "Accuracy": overall["accuracy"],
        "Precision": overall["precision"],
        "Recall": overall["recall"],
        "F1": overall["f1"],
        "ROC-AUC": overall["roc_auc"],
        "FPR": overall["fpr"],
    }]
)

overall_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "overall_unseen_day_result.csv"
    ),
    index=False
)

# ============================================================
# Graph helper
# ============================================================

def create_metric_plot(
    metric,
    title,
    ylabel,
    filename,
    percentage=False
):
    plt.figure(figsize=(10, 6))

    values = df[metric]

    if percentage:
        values = values * 100

    bars = plt.bar(
        df["scenario"],
        values
    )

    plt.title(title)
    plt.xlabel("Unseen Scenario")
    plt.ylabel(ylabel)

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Scenario-wise graphs
# ============================================================

create_metric_plot(
    "accuracy",
    "XGBoost Accuracy on Unseen Scenarios",
    "Accuracy",
    "unseen_accuracy.png",
    percentage=True
)

create_metric_plot(
    "precision",
    "XGBoost Precision on Unseen Scenarios",
    "Precision",
    "unseen_precision.png",
    percentage=True
)

create_metric_plot(
    "recall",
    "XGBoost Recall on Unseen Scenarios",
    "Recall",
    "unseen_recall.png",
    percentage=True
)

create_metric_plot(
    "f1",
    "XGBoost F1 Score on Unseen Scenarios",
    "F1 Score",
    "unseen_f1.png",
    percentage=True
)

create_metric_plot(
    "roc_auc",
    "XGBoost ROC-AUC on Unseen Scenarios",
    "ROC-AUC",
    "unseen_roc_auc.png",
    percentage=True
)

create_metric_plot(
    "fpr",
    "XGBoost False Positive Rate on Unseen Scenarios",
    "False Positive Rate (%)",
    "unseen_fpr.png",
    percentage=True
)

# ============================================================
# Combined performance graph
# ============================================================

plt.figure(figsize=(11, 6))

x = range(len(df))

plt.plot(
    x,
    df["accuracy"] * 100,
    marker="o",
    label="Accuracy"
)

plt.plot(
    x,
    df["precision"] * 100,
    marker="o",
    label="Precision"
)

plt.plot(
    x,
    df["recall"] * 100,
    marker="o",
    label="Recall"
)

plt.plot(
    x,
    df["f1"] * 100,
    marker="o",
    label="F1"
)

plt.xticks(
    list(x),
    df["scenario"],
    rotation=20,
    ha="right"
)

plt.ylabel("Score (%)")
plt.xlabel("Unseen Scenario")
plt.title(
    "XGBoost Cross-Scenario Robustness"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "unseen_combined_performance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Overall vs Random-Split comparison
# ============================================================

random_split_f1 = 0.998305
random_split_accuracy = 0.999442
random_split_recall = 0.998572

comparison = pd.DataFrame([
    {
        "Evaluation": "Random Flow-Level Test",
        "Accuracy": random_split_accuracy,
        "Recall": random_split_recall,
        "F1": random_split_f1,
    },
    {
        "Evaluation": "Friday Unseen-Day",
        "Accuracy": overall["accuracy"],
        "Recall": overall["recall"],
        "F1": overall["f1"],
    }
])

comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "random_vs_unseen_comparison.csv"
    ),
    index=False
)

plt.figure(figsize=(9, 6))

x = range(len(comparison))

width = 0.25

plt.bar(
    [i - width for i in x],
    comparison["Accuracy"] * 100,
    width=width,
    label="Accuracy"
)

plt.bar(
    x,
    comparison["Recall"] * 100,
    width=width,
    label="Recall"
)

plt.bar(
    [i + width for i in x],
    comparison["F1"] * 100,
    width=width,
    label="F1"
)

plt.xticks(
    list(x),
    comparison["Evaluation"]
)

plt.ylabel("Score (%)")

plt.title(
    "Random Flow-Level vs Unseen-Day Evaluation"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "random_vs_unseen_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Print final summary
# ============================================================

print("\n" + "=" * 80)
print("OVERALL FRIDAY RESULT")
print("=" * 80)

print(
    f"Accuracy : {overall['accuracy'] * 100:.4f}%"
)

print(
    f"Precision: {overall['precision'] * 100:.4f}%"
)

print(
    f"Recall   : {overall['recall'] * 100:.4f}%"
)

print(
    f"F1       : {overall['f1'] * 100:.4f}%"
)

print(
    f"ROC-AUC  : {overall['roc_auc'] * 100:.4f}%"
)

print(
    f"FPR      : {overall['fpr'] * 100:.4f}%"
)

print("\n" + "=" * 80)
print("ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for file in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {file}")
