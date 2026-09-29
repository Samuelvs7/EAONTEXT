import csv
import os

data_dir = r"e:\Projects\EAONTEXT\data\raw\empatheticdialogues"

for fname in ["train.csv", "valid.csv", "test.csv"]:
    fpath = os.path.join(data_dir, fname)
    print(f"\n{'='*60}")
    print(f"FILE: {fname}")
    print(f"Size: {os.path.getsize(fpath) / (1024*1024):.2f} MB")
    
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    print(f"\nFirst 10 lines (raw):")
    for i in range(min(10, len(lines))):
        print(f"  LINE {i}: {repr(lines[i][:200])}")

print("\n\n=== TRYING PANDAS ===")
try:
    import pandas as pd
    for fname in ["train.csv", "valid.csv", "test.csv"]:
        fpath = os.path.join(data_dir, fname)
        # Try comma first
        try:
            df = pd.read_csv(fpath, nrows=5)
            print(f"\n{fname} columns (comma sep): {list(df.columns)}")
            print(f"Shape: {df.shape}")
            print(df.head(3).to_string())
        except Exception as e:
            print(f"{fname} comma failed: {e}")
            # Try tab
            try:
                df = pd.read_csv(fpath, sep="\t", nrows=5)
                print(f"\n{fname} columns (tab sep): {list(df.columns)}")
                print(f"Shape: {df.shape}")
                print(df.head(3).to_string())
            except Exception as e2:
                print(f"{fname} tab also failed: {e2}")
except ImportError:
    print("pandas not installed")
