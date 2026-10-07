import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "results"
    / "xgboost_binary"
    / "xgboost_binary.json"
)

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "leakage_safe"
    / "model_ready"
    / "selected_features.csv"
)

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "leakage_safe"
    / "model_ready"
    / "X_train_binary_selected.csv"
)


# ============================================================
# LOAD SELECTED FEATURES
# ============================================================

def load_selected_features():
    df = pd.read_csv(FEATURE_PATH)

    # Support both possible formats:
    # 1. One column containing feature names
    # 2. A dataframe whose columns are the selected features

    if len(df.columns) == 1:
        features = df.iloc[:, 0].astype(str).tolist()
    else:
        features = df.columns.tolist()

    features = [f for f in features if f != "Label"]

    if not features:
        raise ValueError("No selected features found.")

    return features


# ============================================================
# LOAD TRAINING MEDIANS
# ============================================================

def load_training_medians(features):
    """
    Calculate fallback values from the training dataset only.
    The test/new-flow data is never used to calculate medians.
    """

    train_df = pd.read_csv(TRAIN_DATA_PATH, usecols=features)

    train_df = train_df.replace([np.inf, -np.inf], np.nan)

    medians = train_df.median(numeric_only=True)

    return medians.reindex(features)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(input_df, features, medians):

    # Remove columns that are not model features
    input_df = input_df.copy()

    # Remove common non-feature columns if present
    columns_to_drop = [
        "Label",
        "Source_Day",
        "Prediction",
        "Attack_Probability",
        "Confidence",
    ]

    for col in columns_to_drop:
        if col in input_df.columns:
            input_df = input_df.drop(columns=[col])

    # Check missing required features
    missing_features = [
        feature for feature in features
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features:\n"
            + "\n".join(missing_features)
        )

    # Keep exact feature order used during training
    X = input_df[features].copy()

    # Convert values to numeric
    for column in features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    # Replace infinite values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill missing values using training-only medians
    for column in features:
        X[column] = X[column].fillna(medians[column])

    return X


# ============================================================
# PREDICT
# ============================================================

def predict(input_path, output_path=None):

    print("=" * 70)
    print("NETWORK INTRUSION DETECTION - NEW FLOW PREDICTION")
    print("=" * 70)

    print("\nLoading model...")
    model = load_model()

    print("Loading selected features...")
    features = load_selected_features()

    print(f"Selected features: {len(features)}")

    print("\nLoading training medians...")
    medians = load_training_medians(features)

    print("Loading input CSV...")
    input_df = pd.read_csv(input_path)

    print(f"Input rows: {len(input_df)}")

    # Prepare data
    X = prepare_input(
        input_df,
        features,
        medians
    )

    print("\nRunning prediction...")

    probabilities = model.predict_proba(X)

    # XGBoost binary probability:
    # column 0 = BENIGN
    # column 1 = ATTACK
    attack_probability = probabilities[:, 1]

    predictions = (
        attack_probability >= 0.5
    ).astype(int)

    labels = np.where(
        predictions == 1,
        "ATTACK",
        "BENIGN"
    )

    confidence = np.maximum(
        probabilities[:, 0],
        probabilities[:, 1]
    )

    result = pd.DataFrame({
        "Prediction": labels,
        "Attack_Probability": attack_probability,
        "Confidence": confidence,
    })

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)

    print(
        f"\nBENIGN predictions : "
        f"{(labels == 'BENIGN').sum()}"
    )

    print(
        f"ATTACK predictions : "
        f"{(labels == 'ATTACK').sum()}"
    )

    print("\nResults:")
    print(result.to_string(index=False))

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    if output_path is None:
        output_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "results"
            / "prediction_results.csv"
        )

    result.to_csv(output_path, index=False)

    print(
        f"\nResults saved to:\n{output_path}"
    )

    print("\nPrediction completed successfully.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict BENIGN or ATTACK for new network flows."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )

    parser.add_argument(
        "--output",
        required=False,
        help="Optional output CSV path"
    )

    args = parser.parse_args()

    predict(
        input_path=args.input,
        output_path=args.output
    )