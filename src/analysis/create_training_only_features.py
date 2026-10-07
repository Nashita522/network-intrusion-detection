import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

INPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed", "leakage_safe"
)

OUTPUT_DIR = os.path.join(
    INPUT_DIR, "model_ready"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(INPUT_DIR, "train.csv")
TEST_FILE = os.path.join(INPUT_DIR, "test.csv")

print("=" * 80)
print("TRAINING-ONLY FEATURE SELECTION")
print("=" * 80)

print("\nLoading training data...")
train = pd.read_csv(TRAIN_FILE)

print("Loading test data...")
test = pd.read_csv(TEST_FILE)

print(f"\nTrain shape: {train.shape}")
print(f"Test shape : {test.shape}")

# ------------------------------------------------------------------
# Separate labels
# ------------------------------------------------------------------

DROP_COLUMNS = ["Label", "BinaryLabel"]

feature_columns = [
    col for col in train.columns
    if col not in DROP_COLUMNS
]

X_train = train[feature_columns].copy()
X_test = test[feature_columns].copy()

y_train_binary = (train["Label"] != "BENIGN").astype(int)
y_test_binary = (test["Label"] != "BENIGN").astype(int)

y_train_multi = train["Label"].copy()
y_test_multi = test["Label"].copy()

# ------------------------------------------------------------------
# Replace infinite values
# ------------------------------------------------------------------

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Fill missing values using TRAINING statistics only
train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

# ------------------------------------------------------------------
# Remove constant features using TRAIN ONLY
# ------------------------------------------------------------------

print("\nChecking constant features using TRAIN ONLY...")

constant_features = [
    col for col in X_train.columns
    if X_train[col].nunique(dropna=False) <= 1
]

print(f"Constant features found: {len(constant_features)}")

if constant_features:
    for feature in constant_features:
        print(f"  Removing: {feature}")

    X_train = X_train.drop(columns=constant_features)
    X_test = X_test.drop(columns=constant_features)

# ------------------------------------------------------------------
# Correlation filtering using TRAIN ONLY
# ------------------------------------------------------------------

print("\nChecking highly correlated features using TRAIN ONLY...")

corr_matrix = X_train.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

CORRELATION_THRESHOLD = 0.95

high_corr_features = [
    column
    for column in upper.columns
    if any(upper[column] > CORRELATION_THRESHOLD)
]

print(
    f"Highly correlated features (> {CORRELATION_THRESHOLD}): "
    f"{len(high_corr_features)}"
)

if high_corr_features:
    for feature in high_corr_features:
        print(f"  Removing: {feature}")

    X_train = X_train.drop(columns=high_corr_features)
    X_test = X_test.drop(columns=high_corr_features)

# ------------------------------------------------------------------
# Final feature list
# ------------------------------------------------------------------

selected_features = list(X_train.columns)

print("\n" + "=" * 80)
print("FEATURE SELECTION RESULT")
print("=" * 80)

print(f"Original feature count : {len(feature_columns)}")
print(f"Constant features      : {len(constant_features)}")
print(f"High-correlation       : {len(high_corr_features)}")
print(f"Final feature count    : {len(selected_features)}")

# ------------------------------------------------------------------
# Save feature list
# ------------------------------------------------------------------

feature_list_file = os.path.join(
    OUTPUT_DIR, "selected_features.csv"
)

pd.DataFrame({
    "feature": selected_features
}).to_csv(feature_list_file, index=False)

# ------------------------------------------------------------------
# Save binary datasets
# ------------------------------------------------------------------

print("\nSaving binary datasets...")

X_train_binary_file = os.path.join(
    OUTPUT_DIR, "X_train_binary_selected.csv"
)

X_test_binary_file = os.path.join(
    OUTPUT_DIR, "X_test_binary_selected.csv"
)

y_train_binary_file = os.path.join(
    OUTPUT_DIR, "y_train_binary_selected.csv"
)

y_test_binary_file = os.path.join(
    OUTPUT_DIR, "y_test_binary_selected.csv"
)

X_train.to_csv(X_train_binary_file, index=False)
X_test.to_csv(X_test_binary_file, index=False)

pd.DataFrame({
    "BinaryLabel": y_train_binary
}).to_csv(y_train_binary_file, index=False)

pd.DataFrame({
    "BinaryLabel": y_test_binary
}).to_csv(y_test_binary_file, index=False)

# ------------------------------------------------------------------
# Save multiclass datasets
# ------------------------------------------------------------------

print("Saving multiclass datasets...")

X_train_multi_file = os.path.join(
    OUTPUT_DIR, "X_train_multiclass_selected.csv"
)

X_test_multi_file = os.path.join(
    OUTPUT_DIR, "X_test_multiclass_selected.csv"
)

y_train_multi_file = os.path.join(
    OUTPUT_DIR, "y_train_multiclass_selected.csv"
)

y_test_multi_file = os.path.join(
    OUTPUT_DIR, "y_test_multiclass_selected.csv"
)

X_train.to_csv(X_train_multi_file, index=False)
X_test.to_csv(X_test_multi_file, index=False)

pd.DataFrame({
    "Label": y_train_multi
}).to_csv(y_train_multi_file, index=False)

pd.DataFrame({
    "Label": y_test_multi
}).to_csv(y_test_multi_file, index=False)

# ------------------------------------------------------------------
# Final verification
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

print(f"Train X shape: {X_train.shape}")
print(f"Test X shape : {X_test.shape}")

print(
    f"\nSelected feature list saved to:\n"
    f"{feature_list_file}"
)

print("\nTraining-only preprocessing completed successfully.")
print("Test data was transformed using TRAINING statistics only.")