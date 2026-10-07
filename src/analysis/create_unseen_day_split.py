import os
import pandas as pd

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed/unseen_day"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = {
    "Monday": "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday": "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday_WebAttacks": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Friday_DDoS": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday_PortScan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday_Morning": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
}

print("=" * 80)
print("UNSEEN-DAY / CROSS-SCENARIO DATA PREPARATION")
print("=" * 80)

all_data = []

for day, filename in files.items():
    path = os.path.join(RAW_DIR, filename)

    if not os.path.exists(path):
        print(f"\nWARNING: Missing file: {filename}")
        continue

    print(f"\nLoading: {filename}")

    df = pd.read_csv(path, low_memory=False)

    df.columns = df.columns.str.strip()

    df["Source_Day"] = day

    all_data.append(df)

    print(f"Rows: {len(df):,}")
    print(f"Labels: {df['Label'].nunique()}")

if not all_data:
    raise RuntimeError("No raw dataset files found.")

data = pd.concat(all_data, ignore_index=True)

print("\n" + "=" * 80)
print("COMBINED DATASET")
print("=" * 80)

print(f"Total rows: {len(data):,}")
print(f"Total columns: {len(data.columns)}")

print("\nRows by scenario:")
print(data["Source_Day"].value_counts())

print("\nLabel distribution:")
print(data["Label"].value_counts())

# ------------------------------------------------------------------
# Create binary label
# ------------------------------------------------------------------

data["Binary_Label"] = (
    data["Label"]
    .astype(str)
    .str.strip()
    .str.upper()
    .ne("BENIGN")
    .astype(int)
)

# ------------------------------------------------------------------
# Save scenario-wise datasets
# ------------------------------------------------------------------

for day in data["Source_Day"].unique():

    day_data = data[data["Source_Day"] == day].copy()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{day}_unseen_dataset.csv"
    )

    day_data.to_csv(output_file, index=False)

    print(
        f"\nSaved {day}: "
        f"{len(day_data):,} rows -> {output_file}"
    )

# ------------------------------------------------------------------
# Create summary
# ------------------------------------------------------------------

summary = (
    data.groupby(["Source_Day", "Label"])
    .size()
    .reset_index(name="Count")
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "unseen_day_label_distribution.csv"
)

summary.to_csv(summary_file, index=False)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Summary file: {summary_file}")
