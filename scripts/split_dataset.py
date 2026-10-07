import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

INPUT_BINARY = "data/processed/binary_dataset.csv"
INPUT_MULTICLASS = "data/processed/multiclass_dataset.csv"

OUTPUT_DIR = "data/processed/splits"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# FUNCTION: BINARY SPLIT
# ============================================================

def split_binary_dataset():

    print("=" * 70)
    print("BINARY DATASET SPLIT")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_BINARY,
        low_memory=False
    )

    print(f"\nTotal rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    X = df.drop(
        columns=["BinaryLabel"]
    )

    y = df["BinaryLabel"]

    print("\nOriginal class distribution:")
    print(y.value_counts())
    print(
        y.value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    # --------------------------------------------------------
    # STRATIFIED TRAIN/TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # --------------------------------------------------------
    # SAVE FILES
    # --------------------------------------------------------

    X_train.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "X_train_binary.csv"
        ),
        index=False
    )

    X_test.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "X_test_binary.csv"
        ),
        index=False
    )

    y_train.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "y_train_binary.csv"
        ),
        index=False
    )

    y_test.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "y_test_binary.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\nTrain set:")
    print(f"Rows: {len(X_train):,}")

    print("\nTest set:")
    print(f"Rows: {len(X_test):,}")

    print("\nTrain class distribution:")
    print(y_train.value_counts())
    print(
        y_train.value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nTest class distribution:")
    print(y_test.value_counts())
    print(
        y_test.value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


# ============================================================
# FUNCTION: MULTICLASS SPLIT
# ============================================================

def split_multiclass_dataset():

    print("\n\n" + "=" * 70)
    print("MULTICLASS DATASET SPLIT")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_MULTICLASS,
        low_memory=False
    )

    print(f"\nTotal rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    X = df.drop(
        columns=["MulticlassLabel"]
    )

    y = df["MulticlassLabel"]

    print("\nOriginal class distribution:")
    print(y.value_counts().sort_index())

    # --------------------------------------------------------
    # STRATIFIED TRAIN/TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # --------------------------------------------------------
    # SAVE FILES
    # --------------------------------------------------------

    X_train.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "X_train_multiclass.csv"
        ),
        index=False
    )

    X_test.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "X_test_multiclass.csv"
        ),
        index=False
    )

    y_train.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "y_train_multiclass.csv"
        ),
        index=False
    )

    y_test.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "y_test_multiclass.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\nTrain set:")
    print(f"Rows: {len(X_train):,}")

    print("\nTest set:")
    print(f"Rows: {len(X_test):,}")

    print("\nTrain class distribution:")
    print(
        y_train
        .value_counts()
        .sort_index()
    )

    print("\nTest class distribution:")
    print(
        y_test
        .value_counts()
        .sort_index()
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NETWORK INTRUSION DETECTION")
    print("TRAIN / TEST DATASET SPLITTING")
    print("=" * 70)

    split_binary_dataset()

    split_multiclass_dataset()

    print("\n\n" + "=" * 70)
    print("DATASET SPLITTING COMPLETED")
    print("=" * 70)

    print("\nOutput directory:")
    print(OUTPUT_DIR)