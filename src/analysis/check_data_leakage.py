import os
import numpy as np
import pandas as pd


BASE_DIR = "/Users/macbookairm2/Documents/network-intrusion-detection"

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "model_ready"
)

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "splits"
)


def load_data():
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    X_train = pd.read_csv(
        os.path.join(MODEL_DIR, "X_train_binary_selected.csv")
    )

    X_test = pd.read_csv(
        os.path.join(MODEL_DIR, "X_test_binary_selected.csv")
    )

    y_train = pd.read_csv(
        os.path.join(SPLIT_DIR, "y_train_binary.csv")
    )

    y_test = pd.read_csv(
        os.path.join(SPLIT_DIR, "y_test_binary.csv")
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")

    return X_train, X_test, y_train, y_test


def check_exact_duplicate_rows(X_train, X_test):
    print("\n" + "=" * 80)
    print("1. EXACT DUPLICATE ROW CHECK")
    print("=" * 80)

    train_hash = pd.util.hash_pandas_object(
        X_train,
        index=False
    )

    test_hash = pd.util.hash_pandas_object(
        X_test,
        index=False
    )

    train_hash_set = set(train_hash)

    overlap = sum(
        h in train_hash_set
        for h in test_hash
    )

    percentage = (overlap / len(X_test)) * 100

    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows : {len(X_test):,}")
    print(f"Exact duplicate rows between train/test: {overlap:,}")
    print(f"Percentage of test rows duplicated: {percentage:.6f}%")

    if overlap == 0:
        print("RESULT: PASS - No exact train/test row overlap.")
    else:
        print("WARNING: Exact duplicate rows exist between train/test.")


def check_duplicate_inside_datasets(X_train, X_test):
    print("\n" + "=" * 80)
    print("2. DUPLICATES INSIDE TRAIN / TEST")
    print("=" * 80)

    train_duplicates = X_train.duplicated().sum()
    test_duplicates = X_test.duplicated().sum()

    train_pct = train_duplicates / len(X_train) * 100
    test_pct = test_duplicates / len(X_test) * 100

    print(
        f"Train duplicate rows: {train_duplicates:,} "
        f"({train_pct:.4f}%)"
    )

    print(
        f"Test duplicate rows : {test_duplicates:,} "
        f"({test_pct:.4f}%)"
    )


def check_feature_distribution(X_train, X_test):
    print("\n" + "=" * 80)
    print("3. FEATURE DISTRIBUTION CHECK")
    print("=" * 80)

    results = []

    for column in X_train.columns:

        train_values = X_train[column].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        test_values = X_test[column].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(train_values) == 0 or len(test_values) == 0:
            continue

        train_min = train_values.min()
        train_max = train_values.max()

        test_min = test_values.min()
        test_max = test_values.max()

        results.append({
            "feature": column,
            "train_min": train_min,
            "train_max": train_max,
            "test_min": test_min,
            "test_max": test_max
        })

    result_df = pd.DataFrame(results)

    suspicious = result_df[
        (result_df["test_min"] < result_df["train_min"]) |
        (result_df["test_max"] > result_df["train_max"])
    ]

    print(f"Total features checked: {len(result_df)}")

    print(
        f"Features where test contains values outside "
        f"train range: {len(suspicious)}"
    )

    if len(suspicious) > 0:
        print("\nExamples:")
        print(suspicious.head(20).to_string(index=False))
    else:
        print("No major train/test range anomaly detected.")


def check_label_overlap():
    print("\n" + "=" * 80)
    print("4. LABEL DISTRIBUTION")
    print("=" * 80)

    y_train = pd.read_csv(
        os.path.join(SPLIT_DIR, "y_train_binary.csv")
    )

    y_test = pd.read_csv(
        os.path.join(SPLIT_DIR, "y_test_binary.csv")
    )

    print("\nTrain labels:")
    print(y_train.iloc[:, 0].value_counts())

    print("\nTest labels:")
    print(y_test.iloc[:, 0].value_counts())


def check_constant_features(X_train, X_test):
    print("\n" + "=" * 80)
    print("5. CONSTANT FEATURE CHECK")
    print("=" * 80)

    train_constant = [
        col for col in X_train.columns
        if X_train[col].nunique(dropna=False) <= 1
    ]

    test_constant = [
        col for col in X_test.columns
        if X_test[col].nunique(dropna=False) <= 1
    ]

    print(f"Constant features in train: {len(train_constant)}")
    print(f"Constant features in test : {len(test_constant)}")

    if train_constant:
        print("\nTrain constant features:")
        for col in train_constant:
            print(f"  - {col}")


def check_label_correlations(X_train, y_train):
    print("\n" + "=" * 80)
    print("6. FEATURE / LABEL CORRELATION CHECK")
    print("=" * 80)

    y = y_train.iloc[:, 0]

    correlations = []

    for column in X_train.columns:

        try:
            corr = X_train[column].corr(y)

            if pd.notna(corr):
                correlations.append(
                    (column, abs(corr), corr)
                )

        except Exception:
            pass

    correlations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop 20 feature correlations with label:")

    for feature, abs_corr, corr in correlations[:20]:
        print(
            f"{feature:40s} "
            f"corr={corr:.6f}"
        )

    very_high = [
        x for x in correlations
        if x[1] >= 0.95
    ]

    print(
        f"\nFeatures with |correlation| >= 0.95: "
        f"{len(very_high)}"
    )

    if very_high:
        print("\nWARNING - inspect these features:")
        for feature, abs_corr, corr in very_high:
            print(
                f"  {feature}: correlation={corr:.6f}"
            )
    else:
        print(
            "No feature has extremely high "
            "linear correlation with the label."
        )


def main():

    X_train, X_test, y_train, y_test = load_data()

    check_exact_duplicate_rows(
        X_train,
        X_test
    )

    check_duplicate_inside_datasets(
        X_train,
        X_test
    )

    check_feature_distribution(
        X_train,
        X_test
    )

    check_label_overlap()

    check_constant_features(
        X_train,
        X_test
    )

    check_label_correlations(
        X_train,
        y_train
    )

    print("\n" + "=" * 80)
    print("LEAKAGE CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()