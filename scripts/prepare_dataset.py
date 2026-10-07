import os
import glob
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "clean_dataset.csv"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 100_000


# ============================================================
# FIND CSV FILES
# ============================================================

csv_files = sorted(
    glob.glob(os.path.join(RAW_DIR, "*.csv"))
)

if not csv_files:
    print("❌ data/raw folder-এ কোনো CSV পাওয়া যায়নি।")
    exit(1)


print("=" * 70)
print("NETWORK INTRUSION DETECTION")
print("FULL DATASET PREPARATION")
print("=" * 70)

print(f"\nমোট CSV পাওয়া গেছে: {len(csv_files)}")

for file in csv_files:
    print(" -", os.path.basename(file))


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
    print("\nপুরোনো clean_dataset.csv মুছে দেওয়া হয়েছে।")


# ============================================================
# GLOBAL STATISTICS
# ============================================================

total_rows = 0
duplicate_rows = 0
invalid_rows = 0
missing_label_rows = 0

label_counts = {}

first_chunk = True


# ============================================================
# PROCESS EACH CSV
# ============================================================

for file_index, csv_file in enumerate(csv_files, start=1):

    print("\n" + "=" * 70)
    print(f"ফাইল {file_index}/{len(csv_files)}")
    print(os.path.basename(csv_file))
    print("=" * 70)

    file_rows = 0
    file_duplicates = 0
    file_invalid = 0
    file_missing_label = 0

    try:

        for chunk_index, chunk in enumerate(
            pd.read_csv(
                csv_file,
                chunksize=CHUNK_SIZE,
                low_memory=False
            ),
            start=1
        ):

            # ------------------------------------------------
            # COLUMN NAME CLEANING
            # ------------------------------------------------

            chunk.columns = (
                chunk.columns
                .str.strip()
                .str.replace("\ufeff", "", regex=False)
            )

            # ------------------------------------------------
            # REMOVE COMPLETELY EMPTY COLUMNS
            # ------------------------------------------------

            chunk = chunk.dropna(axis=1, how="all")

            # ------------------------------------------------
            # REMOVE DUPLICATE COLUMNS
            # ------------------------------------------------

            chunk = chunk.loc[
                :,
                ~chunk.columns.duplicated()
            ]

            # ------------------------------------------------
            # LABEL COLUMN CHECK
            # ------------------------------------------------

            if "Label" not in chunk.columns:

                possible_label = [
                    col for col in chunk.columns
                    if col.strip().lower() == "label"
                ]

                if possible_label:
                    chunk = chunk.rename(
                        columns={
                            possible_label[0]: "Label"
                        }
                    )

            if "Label" not in chunk.columns:
                print(
                    f"⚠️ Label column পাওয়া যায়নি: "
                    f"{os.path.basename(csv_file)}"
                )
                continue

            # ------------------------------------------------
            # STRIP LABEL VALUES
            # ------------------------------------------------

            chunk["Label"] = (
                chunk["Label"]
                .astype(str)
                .str.strip()
            )

            # ------------------------------------------------
            # REMOVE EMPTY / INVALID LABEL
            # ------------------------------------------------

            before_label = len(chunk)

            invalid_label_mask = (
                chunk["Label"].isna()
                | chunk["Label"].isin(
                    [
                        "",
                        "nan",
                        "NaN",
                        "None",
                        "null"
                    ]
                )
            )

            file_missing_label += int(
                invalid_label_mask.sum()
            )

            chunk = chunk[
                ~invalid_label_mask
            ].copy()

            file_invalid += (
                before_label - len(chunk)
            )

            # ------------------------------------------------
            # REMOVE DUPLICATE ROWS
            # ------------------------------------------------

            before_duplicate = len(chunk)

            chunk = chunk.drop_duplicates()

            file_duplicates += (
                before_duplicate - len(chunk)
            )

            # ------------------------------------------------
            # CONVERT FEATURE COLUMNS TO NUMERIC
            # ------------------------------------------------

            feature_columns = [
                col for col in chunk.columns
                if col != "Label"
            ]

            for col in feature_columns:

                chunk[col] = pd.to_numeric(
                    chunk[col],
                    errors="coerce"
                )

            # ------------------------------------------------
            # REPLACE INF VALUES
            # ------------------------------------------------

            chunk[feature_columns] = (
                chunk[feature_columns]
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
            )

            # ------------------------------------------------
            # REMOVE ROWS WITH INVALID FEATURES
            # ------------------------------------------------

            before_features = len(chunk)

            chunk = chunk.dropna(
                subset=feature_columns
            )

            file_invalid += (
                before_features - len(chunk)
            )

            # ------------------------------------------------
            # LABEL COUNTS
            # ------------------------------------------------

            counts = chunk["Label"].value_counts()

            for label, count in counts.items():

                label_counts[label] = (
                    label_counts.get(label, 0)
                    + int(count)
                )

            # ------------------------------------------------
            # SAVE CLEAN CHUNK
            # ------------------------------------------------

            if len(chunk) > 0:

                chunk.to_csv(
                    OUTPUT_FILE,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    index=False
                )

                first_chunk = False

                total_rows += len(chunk)
                file_rows += len(chunk)

            print(
                f"Chunk {chunk_index}: "
                f"{len(chunk):,} clean rows"
            )

    except Exception as e:

        print(
            f"\n❌ সমস্যা হয়েছে: "
            f"{os.path.basename(csv_file)}"
        )

        print("Error:", e)

        continue

    duplicate_rows += file_duplicates
    invalid_rows += file_invalid
    missing_label_rows += file_missing_label

    print("\nফাইলের summary:")
    print(f"Clean rows       : {file_rows:,}")
    print(f"Duplicate removed: {file_duplicates:,}")
    print(f"Invalid removed  : {file_invalid:,}")
    print(f"Missing label    : {file_missing_label:,}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("DATASET CLEANING COMPLETE")
print("=" * 70)

print(f"\nFinal clean rows       : {total_rows:,}")
print(f"Duplicate rows removed : {duplicate_rows:,}")
print(f"Invalid rows removed   : {invalid_rows:,}")
print(f"Missing labels removed : {missing_label_rows:,}")


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FINAL LABEL DISTRIBUTION")
print("=" * 70)

label_df = pd.DataFrame(
    list(label_counts.items()),
    columns=["Label", "Count"]
)

label_df = label_df.sort_values(
    by="Count",
    ascending=False
)

if total_rows > 0:

    label_df["Percentage"] = (
        label_df["Count"] / total_rows * 100
    )

    print(
        label_df.to_string(
            index=False,
            formatters={
                "Percentage": "{:.2f}%".format
            }
        )
    )


# ============================================================
# SAVE LABEL SUMMARY
# ============================================================

label_summary_file = os.path.join(
    PROCESSED_DIR,
    "label_distribution.csv"
)

label_df.to_csv(
    label_summary_file,
    index=False
)


# ============================================================
# FINAL FILE INFO
# ============================================================

if os.path.exists(OUTPUT_FILE):

    file_size_mb = (
        os.path.getsize(OUTPUT_FILE)
        / (1024 * 1024)
    )

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Clean dataset : {OUTPUT_FILE}"
    )

    print(
        f"File size     : {file_size_mb:.2f} MB"
    )

    print(
        f"Label summary : {label_summary_file}"
    )

else:

    print(
        "\n❌ Clean dataset তৈরি হয়নি।"
    )