import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


BASE_DIR = "/Users/macbookairm2/Documents/network-intrusion-detection"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "clean_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "leakage_safe"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():

    print("=" * 80)
    print("CREATING STRICT LEAKAGE-SAFE DATASET")
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ------------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. REPLACE INFINITY
    # ------------------------------------------------------------------

    print("\nReplacing infinite values...")

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ------------------------------------------------------------------
    # 3. REMOVE MISSING LABEL
    # ------------------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=["Label"]
    )

    print(
        f"Rows removed because of missing Label: "
        f"{before - len(df):,}"
    )

    # ------------------------------------------------------------------
    # 4. DEFINE FEATURES
    # ------------------------------------------------------------------

    feature_columns = [
        col
        for col in df.columns
        if col != "Label"
    ]

    print(
        f"\nNumber of feature columns: "
        f"{len(feature_columns)}"
    )

    # ------------------------------------------------------------------
    # 5. CHECK DUPLICATE FEATURE VECTORS
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CHECKING DUPLICATE FEATURE VECTORS")
    print("=" * 80)

    duplicate_feature_rows = df.duplicated(
        subset=feature_columns,
        keep=False
    )

    total_duplicate_feature_rows = (
        duplicate_feature_rows.sum()
    )

    print(
        f"Rows involved in duplicate feature vectors: "
        f"{total_duplicate_feature_rows:,}"
    )

    # ------------------------------------------------------------------
    # 6. CHECK CONFLICTING LABELS
    # ------------------------------------------------------------------

    print("\nChecking duplicate feature vectors with different labels...")

    duplicated_df = df.loc[
        duplicate_feature_rows,
        feature_columns + ["Label"]
    ]

    if len(duplicated_df) > 0:

        label_counts = (
            duplicated_df
            .groupby(feature_columns, dropna=False)["Label"]
            .nunique()
        )

        conflicting_groups = (
            label_counts > 1
        )

        conflicting_group_count = (
            conflicting_groups.sum()
        )

        conflicting_rows = (
            conflicting_groups.sum()
        )

        print(
            f"Feature vectors with conflicting labels: "
            f"{conflicting_group_count:,}"
        )

    else:

        conflicting_group_count = 0

        print(
            "Feature vectors with conflicting labels: 0"
        )

    # ------------------------------------------------------------------
    # 7. REMOVE ALL DUPLICATE FEATURE VECTORS
    #
    # This is intentionally strict.
    #
    # If the same feature vector occurs multiple times,
    # we keep only ONE copy.
    #
    # Therefore the same feature vector cannot appear
    # in both train and test.
    # ------------------------------------------------------------------

    before_dedup = len(df)

    df = df.drop_duplicates(
        subset=feature_columns,
        keep=False
    ).reset_index(drop=True)

    removed_feature_duplicates = (
        before_dedup - len(df)
    )

    print(
        f"\nRows removed due to duplicate feature vectors: "
        f"{removed_feature_duplicates:,}"
    )

    print(
        f"Shape after strict feature deduplication: "
        f"{df.shape}"
    )

    # ------------------------------------------------------------------
    # 8. CREATE BINARY LABEL
    # ------------------------------------------------------------------

    print("\nCreating binary labels...")

    df["BinaryLabel"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .ne("BENIGN")
        .astype(int)
    )

    print("\nBinary distribution:")

    print(
        df["BinaryLabel"]
        .value_counts()
        .sort_index()
    )

    # ------------------------------------------------------------------
    # 9. CHECK MULTICLASS LABELS
    # ------------------------------------------------------------------

    print("\nMulticlass distribution:")

    print(
        df["Label"]
        .value_counts()
    )

    # ------------------------------------------------------------------
    # 10. STRATIFIED TRAIN / TEST SPLIT
    # ------------------------------------------------------------------

    print("\nCreating stratified train/test split...")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Label"]
    )

    train_df = train_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )

    print(
        f"Train shape: {train_df.shape}"
    )

    print(
        f"Test shape : {test_df.shape}"
    )

    # ------------------------------------------------------------------
    # 11. STRICT OVERLAP VERIFICATION
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("STRICT TRAIN / TEST OVERLAP VERIFICATION")
    print("=" * 80)

    train_hash = pd.util.hash_pandas_object(
        train_df[feature_columns],
        index=False
    )

    test_hash = pd.util.hash_pandas_object(
        test_df[feature_columns],
        index=False
    )

    train_hash_set = set(
        train_hash
    )

    overlap = sum(
        h in train_hash_set
        for h in test_hash
    )

    print(
        f"Exact feature-row overlap: "
        f"{overlap:,}"
    )

    if overlap != 0:

        raise RuntimeError(
            "ERROR: Train/test feature overlap still exists!"
        )

    print(
        "PASS: Train/test exact feature overlap = 0"
    )

    # ------------------------------------------------------------------
    # 12. SAVE COMPLETE DATASETS
    # ------------------------------------------------------------------

    print("\nSaving complete train/test datasets...")

    train_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "train.csv"
        ),
        index=False
    )

    test_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test.csv"
        ),
        index=False
    )

    # ------------------------------------------------------------------
    # 13. BINARY DATASET
    # ------------------------------------------------------------------

    binary_dir = os.path.join(
        OUTPUT_DIR,
        "binary"
    )

    os.makedirs(
        binary_dir,
        exist_ok=True
    )

    train_df[feature_columns].to_csv(
        os.path.join(
            binary_dir,
            "X_train_binary.csv"
        ),
        index=False
    )

    test_df[feature_columns].to_csv(
        os.path.join(
            binary_dir,
            "X_test_binary.csv"
        ),
        index=False
    )

    train_df[["BinaryLabel"]].to_csv(
        os.path.join(
            binary_dir,
            "y_train_binary.csv"
        ),
        index=False
    )

    test_df[["BinaryLabel"]].to_csv(
        os.path.join(
            binary_dir,
            "y_test_binary.csv"
        ),
        index=False
    )

    # ------------------------------------------------------------------
    # 14. MULTICLASS DATASET
    # ------------------------------------------------------------------

    multiclass_dir = os.path.join(
        OUTPUT_DIR,
        "multiclass"
    )

    os.makedirs(
        multiclass_dir,
        exist_ok=True
    )

    train_df[feature_columns].to_csv(
        os.path.join(
            multiclass_dir,
            "X_train_multiclass.csv"
        ),
        index=False
    )

    test_df[feature_columns].to_csv(
        os.path.join(
            multiclass_dir,
            "X_test_multiclass.csv"
        ),
        index=False
    )

    train_df[["Label"]].to_csv(
        os.path.join(
            multiclass_dir,
            "y_train_multiclass.csv"
        ),
        index=False
    )

    test_df[["Label"]].to_csv(
        os.path.join(
            multiclass_dir,
            "y_test_multiclass.csv"
        ),
        index=False
    )

    # ------------------------------------------------------------------
    # 15. SAVE FEATURE LIST
    # ------------------------------------------------------------------

    pd.DataFrame(
        {
            "feature": feature_columns
        }
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_list.csv"
        ),
        index=False
    )

    # ------------------------------------------------------------------
    # 16. FINAL SUMMARY
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("LEAKAGE-SAFE SPLIT COMPLETED")
    print("=" * 80)

    print(
        f"\nOriginal rows: "
        f"{before:,}"
    )

    print(
        f"Final rows: "
        f"{len(df):,}"
    )

    print(
        f"Rows removed: "
        f"{before - len(df):,}"
    )

    print(
        f"\nTrain rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Test rows: "
        f"{len(test_df):,}"
    )

    print(
        "\nExact train/test feature overlap: 0"
    )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        "\nOriginal clean_dataset.csv was NOT modified."
    )


if __name__ == "__main__":
    main()