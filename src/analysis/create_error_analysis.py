import os
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = "data/processed/results"
OUTPUT_DIR = "data/processed/results/error_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("ERROR ANALYSIS")
print("=" * 80)


# ============================================================
# 1. BINARY ERROR ANALYSIS
# ============================================================

binary_models = {
    "XGBoost": "xgboost_binary/predictions.csv",
    "Random Forest": "random_forest_binary/predictions.csv",
    "Neural Network": "neural_network_binary/predictions.csv",
}

binary_summary = []

for model_name, relative_path in binary_models.items():

    path = os.path.join(RESULTS_DIR, relative_path)

    if not os.path.exists(path):
        print(f"\nWARNING: Missing {path}")
        continue

    print(f"\nAnalyzing {model_name} binary predictions...")

    df = pd.read_csv(path)

    # Handle different column naming styles
    actual_col = next(
        (c for c in df.columns if c.lower() == "actual"),
        None
    )

    predicted_col = next(
        (c for c in df.columns if c.lower() == "predicted"),
        None
    )

    if actual_col is None or predicted_col is None:
        print(f"Skipping {model_name}: prediction columns not found.")
        continue

    actual = df[actual_col]
    predicted = df[predicted_col]

    # --------------------------------------------------------
    # False Positive / False Negative
    # --------------------------------------------------------

    false_positive = (
        (actual == 0) &
        (predicted == 1)
    )

    false_negative = (
        (actual == 1) &
        (predicted == 0)
    )

    true_positive = (
        (actual == 1) &
        (predicted == 1)
    )

    true_negative = (
        (actual == 0) &
        (predicted == 0)
    )

    fp = int(false_positive.sum())
    fn = int(false_negative.sum())
    tp = int(true_positive.sum())
    tn = int(true_negative.sum())

    total = len(df)

    binary_summary.append({
        "Model": model_name,
        "Total Samples": total,
        "True Negative": tn,
        "False Positive": fp,
        "False Negative": fn,
        "True Positive": tp,
        "Total Errors": fp + fn,
        "Error Rate": (fp + fn) / total,
    })

    # --------------------------------------------------------
    # Save FP and FN examples
    # --------------------------------------------------------

    fp_df = df[false_positive].copy()

    fp_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name.lower().replace(' ', '_')}_false_positives.csv"
        ),
        index=False
    )

    fn_df = df[false_negative].copy()

    fn_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name.lower().replace(' ', '_')}_false_negatives.csv"
        ),
        index=False
    )

    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Total Errors: {fp + fn}")


# ============================================================
# Save binary error summary
# ============================================================

binary_summary_df = pd.DataFrame(binary_summary)

binary_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "binary_error_summary.csv"
    ),
    index=False
)

print("\n" + "=" * 80)
print("BINARY ERROR SUMMARY")
print("=" * 80)

print(
    binary_summary_df.to_string(index=False)
)


# ============================================================
# 2. MULTICLASS ERROR ANALYSIS
# ============================================================

multiclass_models = {
    "XGBoost": "xgboost_multiclass/predictions.csv",
    "Random Forest": "random_forest_multiclass/predictions.csv",
    "Neural Network": "neural_network_multiclass/predictions.csv",
}

multiclass_summary = []

for model_name, relative_path in multiclass_models.items():

    path = os.path.join(RESULTS_DIR, relative_path)

    if not os.path.exists(path):
        print(f"\nWARNING: Missing {path}")
        continue

    print(f"\nAnalyzing {model_name} multiclass predictions...")

    df = pd.read_csv(path)

    actual_col = next(
        (c for c in df.columns if c.lower() == "actual"),
        None
    )

    predicted_col = next(
        (c for c in df.columns if c.lower() == "predicted"),
        None
    )

    if actual_col is None or predicted_col is None:
        print(f"Skipping {model_name}: prediction columns not found.")
        continue

    actual = df[actual_col].astype(str)
    predicted = df[predicted_col].astype(str)

    # --------------------------------------------------------
    # Overall multiclass errors
    # --------------------------------------------------------

    errors = actual != predicted

    total_errors = int(errors.sum())
    total_samples = len(df)

    multiclass_summary.append({
        "Model": model_name,
        "Total Samples": total_samples,
        "Correct Predictions": int((~errors).sum()),
        "Incorrect Predictions": total_errors,
        "Error Rate": total_errors / total_samples,
    })

    # --------------------------------------------------------
    # Confusion pairs
    # --------------------------------------------------------

    confusion_pairs = (
        df[errors]
        .groupby(
            [actual_col, predicted_col]
        )
        .size()
        .reset_index(name="Count")
        .sort_values(
            "Count",
            ascending=False
        )
    )

    confusion_pairs.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name.lower().replace(' ', '_')}_top_error_pairs.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Per-class error count
    # --------------------------------------------------------

    class_errors = (
        df[errors]
        .groupby(actual_col)
        .size()
        .reset_index(name="Error_Count")
        .sort_values(
            "Error_Count",
            ascending=False
        )
    )

    class_errors.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name.lower().replace(' ', '_')}_class_errors.csv"
        ),
        index=False
    )

    print(f"Total errors: {total_errors}")


# ============================================================
# Save multiclass summary
# ============================================================

multiclass_summary_df = pd.DataFrame(
    multiclass_summary
)

multiclass_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "multiclass_error_summary.csv"
    ),
    index=False
)

print("\n" + "=" * 80)
print("MULTICLASS ERROR SUMMARY")
print("=" * 80)

print(
    multiclass_summary_df.to_string(index=False)
)


# ============================================================
# 3. UNSEEN-DAY ERROR ANALYSIS
# ============================================================

unseen_path = (
    "data/processed/results/"
    "unseen_day_xgboost/overall_predictions.csv"
)

if os.path.exists(unseen_path):

    print("\n" + "=" * 80)
    print("UNSEEN-DAY ERROR ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(unseen_path)

    errors = df["Actual"] != df["Predicted"]

    unseen_errors = df[errors].copy()

    print(
        f"Total unseen-day samples: {len(df):,}"
    )

    print(
        f"Total errors: {len(unseen_errors):,}"
    )

    print(
        f"Error rate: "
        f"{len(unseen_errors) / len(df):.6f}"
    )

    # Errors by scenario
    scenario_errors = (
        unseen_errors
        .groupby("Scenario")
        .size()
        .reset_index(name="Error_Count")
        .sort_values(
            "Error_Count",
            ascending=False
        )
    )

    scenario_errors.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "unseen_day_scenario_errors.csv"
        ),
        index=False
    )

    unseen_errors.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "unseen_day_errors.csv"
        ),
        index=False
    )


# ============================================================
# 4. FINAL ERROR SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ERROR ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):
    print(f"  - {filename}")
