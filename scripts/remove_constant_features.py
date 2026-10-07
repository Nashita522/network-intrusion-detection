from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = BASE_DIR / "data" / "processed" / "splits"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "model_ready"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONSTANT_FEATURES = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

DATASETS = [
    ("binary", "X_train_binary.csv", "X_test_binary.csv"),
    ("multiclass", "X_train_multiclass.csv", "X_test_multiclass.csv"),
]


def process_file(input_path, output_path, selected_features):
    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(
        input_path,
        usecols=selected_features,
        chunksize=100_000
    ):
        chunk.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False
        total_rows += len(chunk)

    return total_rows


def main():
    train_sample = pd.read_csv(
        SPLIT_DIR / "X_train_binary.csv",
        nrows=5
    )

    all_features = list(train_sample.columns)

    selected_features = [
        feature
        for feature in all_features
        if feature not in CONSTANT_FEATURES
    ]

    print("=" * 80)
    print("CONSTANT FEATURE REMOVAL")
    print("=" * 80)

    print(f"Original features: {len(all_features)}")
    print(f"Constant features removed: {len(CONSTANT_FEATURES)}")
    print(f"Selected features: {len(selected_features)}")

    print("\nRemoved features:")
    for feature in CONSTANT_FEATURES:
        print(f"  - {feature}")

    print("\nProcessing datasets...")

    for dataset_name, train_file, test_file in DATASETS:
        print(f"\n[{dataset_name.upper()}]")

        train_output = OUTPUT_DIR / f"X_train_{dataset_name}_selected.csv"
        test_output = OUTPUT_DIR / f"X_test_{dataset_name}_selected.csv"

        train_rows = process_file(
            SPLIT_DIR / train_file,
            train_output,
            selected_features
        )

        test_rows = process_file(
            SPLIT_DIR / test_file,
            test_output,
            selected_features
        )

        print(f"Train rows: {train_rows:,}")
        print(f"Test rows:  {test_rows:,}")

    feature_list = pd.DataFrame({
        "feature": selected_features
    })

    feature_list.to_csv(
        OUTPUT_DIR / "selected_features.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Final feature count: {len(selected_features)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()