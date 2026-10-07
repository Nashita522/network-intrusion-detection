import os
import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/clean_dataset.csv"

BINARY_OUTPUT = "data/processed/binary_dataset.csv"
MULTICLASS_OUTPUT = "data/processed/multiclass_dataset.csv"
LABEL_MAP_OUTPUT = "data/processed/label_mapping.csv"


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: Input file not found: {INPUT_FILE}")
    exit(1)


print("=" * 70)
print("NETWORK INTRUSION DETECTION")
print("LABEL CLEANING AND ML DATASET PREPARATION")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)


# ============================================================
# CHECK LABEL COLUMN
# ============================================================

if "Label" not in df.columns:
    print("ERROR: Label column not found.")
    exit(1)


# ============================================================
# CLEAN LABEL VALUES
# ============================================================

df["Label"] = (
    df["Label"]
    .astype(str)
    .str.strip()
)


# ============================================================
# FIX COMMON ENCODING ISSUES
# ============================================================

label_replacements = {

    "Web Attack � Brute Force":
        "Web Attack - Brute Force",

    "Web Attack � XSS":
        "Web Attack - XSS",

    "Web Attack � Sql Injection":
        "Web Attack - Sql Injection",

    "Web Attack � SQL Injection":
        "Web Attack - Sql Injection",

}


df["Label"] = df["Label"].replace(
    label_replacements
)


# ============================================================
# STANDARDIZE LABEL CAPITALIZATION
# ============================================================

label_standardization = {

    "BENIGN": "BENIGN",

    "DoS Hulk": "DoS Hulk",

    "DDoS": "DDoS",

    "PortScan": "PortScan",

    "DoS GoldenEye": "DoS GoldenEye",

    "FTP-Patator": "FTP-Patator",

    "DoS slowloris": "DoS Slowloris",

    "DoS Slowhttptest": "DoS Slowhttptest",

    "SSH-Patator": "SSH-Patator",

    "Bot": "Bot",

    "Web Attack - Brute Force":
        "Web Attack - Brute Force",

    "Web Attack - XSS":
        "Web Attack - XSS",

    "Web Attack - Sql Injection":
        "Web Attack - SQL Injection",

    "Infiltration": "Infiltration",

    "Heartbleed": "Heartbleed",

}


df["Label"] = df["Label"].replace(
    label_standardization
)


# ============================================================
# DISPLAY FINAL LABELS
# ============================================================

print("\nFinal labels:")

for label in sorted(df["Label"].unique()):
    print(f" - {label}")


# ============================================================
# CREATE BINARY LABEL
# ============================================================

print("\nCreating binary classification label...")

df["BinaryLabel"] = np.where(
    df["Label"] == "BENIGN",
    0,
    1
)


# ============================================================
# CREATE MULTICLASS LABEL
# ============================================================

print("Creating multiclass encoded label...")

unique_labels = sorted(
    df["Label"].unique()
)

label_to_id = {
    label: index
    for index, label in enumerate(unique_labels)
}


df["MulticlassLabel"] = df["Label"].map(
    label_to_id
)


# ============================================================
# CREATE LABEL MAPPING TABLE
# ============================================================

mapping_rows = []

for label, label_id in label_to_id.items():

    mapping_rows.append(
        {
            "Label": label,
            "MulticlassID": label_id,
            "BinaryID": 0 if label == "BENIGN" else 1
        }
    )


label_mapping = pd.DataFrame(
    mapping_rows
)


label_mapping.to_csv(
    LABEL_MAP_OUTPUT,
    index=False
)


# ============================================================
# FEATURE VALIDATION
# ============================================================

print("\nValidating feature columns...")

feature_columns = [
    col
    for col in df.columns
    if col not in [
        "Label",
        "BinaryLabel",
        "MulticlassLabel"
    ]
]


print(f"Feature count: {len(feature_columns)}")


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

for col in feature_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# CHECK NaN
# ============================================================

nan_count = int(
    df[feature_columns]
    .isna()
    .sum()
    .sum()
)

print(f"Total NaN values: {nan_count:,}")


# ============================================================
# CHECK INFINITY
# ============================================================

inf_count = int(
    np.isinf(
        df[feature_columns].to_numpy()
    ).sum()
)

print(f"Total infinite values: {inf_count:,}")


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

if nan_count > 0 or inf_count > 0:

    print("\nRemoving invalid feature rows...")

    df[feature_columns] = (
        df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    before = len(df)

    df = df.dropna(
        subset=feature_columns
    )

    removed = before - len(df)

    print(f"Removed rows: {removed:,}")


# ============================================================
# BINARY DATASET
# ============================================================

print("\nPreparing binary dataset...")

binary_columns = (
    feature_columns
    + ["BinaryLabel"]
)

binary_df = df[
    binary_columns
].copy()


binary_df.to_csv(
    BINARY_OUTPUT,
    index=False
)


# ============================================================
# MULTICLASS DATASET
# ============================================================

print("Preparing multiclass dataset...")

multiclass_columns = (
    feature_columns
    + ["MulticlassLabel"]
)

multiclass_df = df[
    multiclass_columns
].copy()


multiclass_df.to_csv(
    MULTICLASS_OUTPUT,
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(f"\nFinal rows: {len(df):,}")
print(f"Features: {len(feature_columns)}")
print(f"Classes: {len(unique_labels)}")


# ============================================================
# BINARY DISTRIBUTION
# ============================================================

print("\nBinary distribution:")

binary_distribution = (
    df["BinaryLabel"]
    .value_counts()
    .sort_index()
)

print(binary_distribution)


# ============================================================
# MULTICLASS DISTRIBUTION
# ============================================================

print("\nMulticlass distribution:")

multiclass_distribution = (
    df["Label"]
    .value_counts()
)

print(multiclass_distribution)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(f"\nBinary dataset:")
print(BINARY_OUTPUT)

print("\nMulticlass dataset:")
print(MULTICLASS_OUTPUT)

print("\nLabel mapping:")
print(LABEL_MAP_OUTPUT)

print("\nDataset preparation completed successfully.")