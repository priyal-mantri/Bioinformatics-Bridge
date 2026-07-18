import os
import sys

# Resolve paths relative to the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

possible_paths = [
    os.path.join(script_dir, "Datasets", "DATASET"),
    os.path.join(script_dir, "DATASET"),
]

dataset_dir = None
for p in possible_paths:
    if os.path.exists(p) and os.path.isdir(p):
        dataset_dir = p
        break

# If not found immediately, search recursively starting from the script's directory
if not dataset_dir:
    for root, dirs, files in os.walk(script_dir):
        for d in dirs:
            if d.upper() == 'DATASET':
                dataset_dir = os.path.join(root, d)
                break
        if dataset_dir:
            break

if not dataset_dir:
    print("Error: Could not locate the 'DATASET' folder.")
    sys.exit(1)

print(f"Reading CSV files from: {os.path.abspath(dataset_dir)}\n")

# Find all CSV files in the folder
csv_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith('.csv')]

if not csv_files:
    print("No CSV files found in the dataset folder.")
    sys.exit(0)

# Check if pandas is available for nice tabular formatting
try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False
    import csv
    print("Tip: Install pandas ('pip install pandas') for enhanced tabular display of the datasets.\n")

for csv_file in csv_files:
    file_path = os.path.join(dataset_dir, csv_file)
    print("=" * 80)
    print(f"File: {csv_file}")
    print("=" * 80)
    try:
        if has_pandas:
            # Display first 10 rows using pandas DataFrame head
            df = pd.read_csv(file_path)
            print(df.head(10))
        else:
            # Fallback to standard csv library
            with open(file_path, mode='r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is not None:
                    # Print header
                    print(" | ".join(header))
                    print("-" * 80)
                    # Print first 10 data rows
                    for i, row in enumerate(reader):
                        if i >= 10:
                            break
                        print(" | ".join(row))
                else:
                    print("[Empty file]")
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
    print("\n")
