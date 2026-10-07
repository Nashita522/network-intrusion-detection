import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = "data/processed/binary_dataset.csv"
OUTPUT_DIR = "data/processed/feature_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 200_000
CORRELATION_THRESHOLD = 0.95


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("NETWORK INTRUSION DETECTION")
print("FEATURE QUALITY ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

target = "BinaryLabel"

feature_columns = [
    col for col in df.columns
    if col != target
]

X = df[feature_columns]

print(f"\nFeature count: {len(feature_columns)}")


# ============================================================
# 1. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

dtype_summary = (
    X.dtypes
    .value_counts()
)

print(dtype_summary)


# ============================================================
# 2. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = X.isna().sum()

missing_features = (
    missing[missing > 0]
    .sort_values(ascending=False)
)

if len(missing_features) == 0:
    print("No missing values found.")
else:
    print(missing_features)


# ============================================================
# 3. INFINITE VALUES
# ============================================================

print("\n" + "=" * 70)
print("INFINITE VALUES")
print("=" * 70)

numeric_X = X.select_dtypes(
    include=[np.number]
)

inf_counts = np.isinf(
    numeric_X.to_numpy()
).sum(axis=0)

inf_series = pd.Series(
    inf_counts,
    index=numeric_X.columns
)

inf_features = (
    inf_series[inf_series > 0]
    .sort_values(ascending=False)
)

if len(inf_features) == 0:
    print("No infinite values found.")
else:
    print(inf_features)


# ============================================================
# 4. CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CONSTANT FEATURES")
print("=" * 70)

nunique = X.nunique()

constant_features = (
    nunique[nunique <= 1]
)

if len(constant_features) == 0:
    print("No constant features found.")
else:
    print(constant_features)


# ============================================================
# 5. LOW VARIANCE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("LOW VARIANCE FEATURES")
print("=" * 70)

variance = numeric_X.var()

low_variance = (
    variance[
        variance < 1e-6
    ]
    .sort_values()
)

if len(low_variance) == 0:
    print("No extremely low-variance features found.")
else:
    print(low_variance)


# ============================================================
# 6. FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)

statistics = pd.DataFrame({

    "dtype": X.dtypes.astype(str),

    "missing": X.isna().sum(),

    "unique": X.nunique(),

    "mean": numeric_X.mean(),

    "std": numeric_X.std(),

    "min": numeric_X.min(),

    "max": numeric_X.max(),

})

statistics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_statistics.csv"
    )
)

print(
    "Saved:",
    os.path.join(
        OUTPUT_DIR,
        "feature_statistics.csv"
    )
)


# ============================================================
# 7. SAMPLE FOR CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

if len(numeric_X) > SAMPLE_SIZE:

    print(
        f"Using random sample of "
        f"{SAMPLE_SIZE:,} rows..."
    )

    correlation_data = numeric_X.sample(
        n=SAMPLE_SIZE,
        random_state=42
    )

else:

    correlation_data = numeric_X


# ============================================================
# CORRELATION MATRIX
# ============================================================

correlation_matrix = (
    correlation_data
    .corr()
)

correlation_matrix.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "correlation_matrix.csv"
    )
)


# ============================================================
# FIND HIGHLY CORRELATED FEATURES
# ============================================================

upper_triangle = correlation_matrix.where(
    np.triu(
        np.ones(
            correlation_matrix.shape
        ),
        k=1
    ).astype(bool)
)

high_correlation_pairs = []

for column in upper_triangle.columns:

    for row in upper_triangle.index:

        value = upper_triangle.loc[
            row,
            column
        ]

        if pd.notna(value) and abs(value) >= CORRELATION_THRESHOLD:

            high_correlation_pairs.append({

                "Feature_1": row,

                "Feature_2": column,

                "Correlation": value

            })


high_correlation_df = pd.DataFrame(
    high_correlation_pairs
)

if len(high_correlation_df) > 0:

    high_correlation_df = (
        high_correlation_df
        .sort_values(
            by="Correlation",
            key=lambda x: abs(x),
            ascending=False
        )
    )

    print(
        f"\nHighly correlated pairs "
        f"(threshold >= {CORRELATION_THRESHOLD}):"
    )

    print(
        high_correlation_df
        .head(30)
        .to_string(index=False)
    )

else:

    print(
        f"\nNo feature pairs found "
        f"above correlation threshold "
        f"{CORRELATION_THRESHOLD}."
    )


high_correlation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "high_correlation_pairs.csv"
    ),
    index=False
)


# ============================================================
# 8. EXTREME VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("EXTREME VALUE CHECK")
print("=" * 70)

extreme_statistics = pd.DataFrame({

    "mean": numeric_X.mean(),

    "std": numeric_X.std(),

    "min": numeric_X.min(),

    "max": numeric_X.max(),

    "median": numeric_X.median(),

    "q01": numeric_X.quantile(0.01),

    "q99": numeric_X.quantile(0.99)

})

extreme_statistics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "extreme_value_statistics.csv"
    )
)

print(
    "Extreme-value statistics saved."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nTotal features: {len(feature_columns)}"
)

print(
    f"Constant features: "
    f"{len(constant_features)}"
)

print(
    f"Low variance features: "
    f"{len(low_variance)}"
)

print(
    f"Highly correlated pairs: "
    f"{len(high_correlation_df)}"
)

print(
    "\nAnalysis files saved in:"
)

print(OUTPUT_DIR)