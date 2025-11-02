import os
import pandas as pd

# Get absolute path to the script directory
base_dir = os.path.dirname(os.path.abspath(__file__))
metadata_dir = os.path.join(base_dir, "../metadata")

# List of CSV files to check
csv_files = ["train.csv", "test.csv", "val.csv", "master_raw.csv"]

print("📊 Row count for each CSV file in 'metadata':\n")

for file in csv_files:
    file_path = os.path.join(metadata_dir, file)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"{file}: {len(df)} rows")
    else:
        print(f"{file}: ❌ File not found!")

print("\n✅ Counting completed.")
