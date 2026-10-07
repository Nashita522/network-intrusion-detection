from pathlib import Path
import urllib.request


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://www.unb.ca/cic/datasets/ids-2017.html"

print("Dataset source:")
print(URL)
print()
print("CICIDS2017 dataset files are large.")
print("Please download the required CSV files from the official CIC website")
print(f"and place them inside: {DATA_DIR}")