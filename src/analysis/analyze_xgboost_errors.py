import os
import pandas as pd

INPUT_DIR = "data/processed/results/error_analysis"
OUTPUT_DIR = "data/processed/results/error_analysis"

print("=" * 80)
print("XGBOOST MULTICLASS ERROR ANALYSIS")
print("=" * 80)

# ============================================================
# Load error pairs
# ============================================================

path = os.path.join(
    INPUT_DIR,
    "xgboost_top_error_pairs.csv"
)

df = pd.read_csv(path)

print("\nTop XGBoost multiclass error pairs:")
print(df.to_string(index=False))

# ============================================================
# Total errors by actual class
# ============================================================

class_error_path = os.path.join(
    INPUT_DIR,
    "xgboost_class_errors.csv"
)

class_errors = pd.read_csv(class_error_path)

print("\n" + "=" * 80)
print("ERRORS BY ACTUAL CLASS")
print("=" * 80)

print(
    class_errors.to_string(index=False)
)

# ============================================================
# Add percentage of total errors
# ============================================================

total_errors = df["Count"].sum()

df["Percentage_of_Errors"] = (
    df["Count"] / total_errors * 100
)

df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "xgboost_error_pairs_with_percentage.csv"
    ),
    index=False
)

# ============================================================
# Top 10 error pairs
# ============================================================

top10 = df.head(10).copy()

top10.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "xgboost_top10_error_pairs.csv"
    ),
    index=False
)

print("\n" + "=" * 80)
print("TOP 10 ERROR PAIRS")
print("=" * 80)

print(
    top10.to_string(index=False)
)

# ============================================================
# Most problematic actual classes
# ============================================================

class_errors["Percentage_of_Total_Errors"] = (
    class_errors["Error_Count"]
    / class_errors["Error_Count"].sum()
    * 100
)

class_errors.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "xgboost_class_errors_with_percentage.csv"
    ),
    index=False
)

print("\n" + "=" * 80)
print("MOST PROBLEMATIC ATTACK CLASSES")
print("=" * 80)

print(
    class_errors.head(10).to_string(index=False)
)

# ============================================================
# Final research observations
# ============================================================

print("\n" + "=" * 80)
print("RESEARCH ERROR SUMMARY")
print("=" * 80)

print(
    f"Total multiclass errors: {total_errors}"
)

if len(top10) > 0:
    top = top10.iloc[0]

    print(
        f"\nMost frequent error:"
    )

    print(
        f"Actual '{top.iloc[0]}' "
        f"-> Predicted '{top.iloc[1]}' "
        f"({int(top.iloc[2])} cases)"
    )

if len(class_errors) > 0:
    worst = class_errors.iloc[0]

    print(
        f"\nMost error-prone actual class:"
    )

    print(
        f"{worst.iloc[0]} "
        f"({int(worst.iloc[1])} errors)"
    )

print("\n" + "=" * 80)
print("ERROR ANALYSIS TABLES COMPLETE")
print("=" * 80)
