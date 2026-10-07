import os
import pandas as pd

RAW_DIR = "data/raw"

files = [
    f for f in os.listdir(RAW_DIR)
    if f.lower().endswith(".csv")
]

files.sort()

print("=" * 80)
print("CICIDS2017 DATASET INSPECTION")
print("=" * 80)

print(f"\nTotal CSV files: {len(files)}")

total_rows = 0

for file in files:
    path = os.path.join(RAW_DIR, file)

    print("\n" + "=" * 80)
    print(f"FILE: {file}")
    print("=" * 80)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)

        rows, cols = df.shape
        total_rows += rows

        print(f"Rows: {rows:,}")
        print(f"Columns: {cols}")

        print("\nColumn names:")
        for col in df.columns:
            print(f"  - {col}")

        print("\nMissing values:")
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) == 0:
            print("  No missing values")
        else:
            print(missing.to_string())

        print("\nDuplicate rows:")
        print(f"  {df.duplicated().sum():,}")

        label_col = None

        for col in df.columns:
            if col.strip().lower() == "label":
                label_col = col
                break

        if label_col:
            print("\nLabel distribution:")
            print(df[label_col].value_counts(dropna=False).to_string())

        print("\nData types:")
        print(df.dtypes.value_counts().to_string())

        del df

    except Exception as e:
        print(f"ERROR reading file: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Total rows across all files: {total_rows:,}")
print(f"Total CSV files: {len(files)}")
print("=" * 80)


