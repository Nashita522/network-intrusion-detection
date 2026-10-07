import os
import time
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "leakage_safe",
    "model_ready"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# Leakage-safe 47-feature datasets
X_TRAIN_PATH = os.path.join(
    DATA_DIR, "X_train_binary_selected.csv"
)

X_TEST_PATH = os.path.join(
    DATA_DIR, "X_test_binary_selected.csv"
)

Y_TRAIN_PATH = os.path.join(
    DATA_DIR, "y_train_binary_selected.csv"
)

Y_TEST_PATH = os.path.join(
    DATA_DIR, "y_test_binary_selected.csv"
)


# Output files
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_binary.joblib"
)

METRICS_PATH = os.path.join(
    RESULT_DIR,
    "random_forest_binary_metrics.csv"
)

CM_PATH = os.path.join(
    RESULT_DIR,
    "random_forest_binary_confusion_matrix.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("RANDOM FOREST - BINARY CLASSIFICATION")
print("=" * 80)

print("\nUsing leakage-safe TRAINING-ONLY selected features.")
print("Feature selection: 47 features")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")
y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")


print("\nDataset shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test : {y_test.shape}")


# ============================================================
# SAFETY CHECKS
# ============================================================

print("\n" + "=" * 80)
print("DATA VALIDATION")
print("=" * 80)

if X_train.shape[1] != 47:
    raise ValueError(
        f"Expected 47 features, but found {X_train.shape[1]}"
    )

if X_test.shape[1] != 47:
    raise ValueError(
        f"Expected 47 test features, but found {X_test.shape[1]}"
    )

if len(X_train) != len(y_train):
    raise ValueError(
        "Training X and y row counts do not match."
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "Test X and y row counts do not match."
    )

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Train and test feature columns do not match."
    )

print("Feature count       : 47")
print("Train X/y rows      :", len(X_train))
print("Test X/y rows       :", len(X_test))
print("Train/test columns  : MATCH")
print("Validation          : PASSED")


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTest label distribution:")
print(y_test.value_counts())


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST")
print("=" * 80)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(
    f"\nTraining completed in "
    f"{training_time:.2f} seconds."
)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating test predictions...")

prediction_start = time.time()

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

prediction_time = time.time() - prediction_start

print(
    f"Prediction completed in "
    f"{prediction_time:.2f} seconds."
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0.0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("RANDOM FOREST BINARY RESULTS")
print("=" * 80)

print(f"\nAccuracy           : {accuracy:.6f}")
print(f"Precision          : {precision:.6f}")
print(f"Recall             : {recall:.6f}")
print(f"F1-Score           : {f1:.6f}")
print(f"ROC-AUC            : {roc_auc:.6f}")
print(f"PR-AUC             : {pr_auc:.6f}")
print(f"MCC                : {mcc:.6f}")
print(f"Specificity        : {specificity:.6f}")
print(f"False Positive Rate: {false_positive_rate:.6f}")

print("\nConfusion Matrix:")
print(cm)

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=6,
        zero_division=0
    )
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
)

cm_df.to_csv(
    CM_PATH
)

print("\nConfusion matrix saved:")
print(CM_PATH)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "Random Forest",
    "classification": "Binary",
    "feature_count": X_train.shape[1],
    "train_samples": X_train.shape[0],
    "test_samples": X_test.shape[0],
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "mcc": mcc,
    "specificity": specificity,
    "false_positive_rate": false_positive_rate,
    "true_negative": tn,
    "false_positive": fp,
    "false_negative": fn,
    "true_positive": tp,
    "training_time_seconds": training_time,
    "prediction_time_seconds": prediction_time,
}

metrics_df = pd.DataFrame(
    [metrics]
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)

print("\nMetrics saved:")
print(METRICS_PATH)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    RESULT_DIR,
    "random_forest_binary_feature_importance.csv"
)

feature_importance.to_csv(
    FEATURE_IMPORTANCE_PATH,
    index=False
)

print("\nFeature importance saved:")
print(FEATURE_IMPORTANCE_PATH)

print("\nTop 15 important features:")

print(
    feature_importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving Random Forest model...")

joblib.dump(
    model,
    MODEL_PATH
)

print("Model saved:")
print(MODEL_PATH)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("RANDOM FOREST BINARY TRAINING COMPLETE")
print("=" * 80)

print("\nAll outputs generated successfully.")
