import pandas as pd
from pathlib import Path

# Get all CSV files in NE_Data directory
ne_data_dir = Path("NE_Data")
csv_files = list(ne_data_dir.glob("*.csv"))

print("Checking missing values in NE_Data files...")
print("=" * 50)

for file_path in csv_files:
    print(f"\nFile: {file_path.name}")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    
    if total_missing == 0:
        print("✓ No missing values found")
    else:
        print(f"✗ Found {total_missing} missing values:")
        for col, missing in missing_values.items():
            if missing > 0:
                print(f"  - {col}: {missing} missing values")
    
    # Show basic info
    print(f"  Total records: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")

print("\n" + "=" * 50)
print("Missing values check completed!") 