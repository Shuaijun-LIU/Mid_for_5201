import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dataset_visualizations(all_data):
    """Create visualizations for the entire dataset overview"""
    # Create output directory if it doesn't exist
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('bmh')  

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.axisbelow'] = True
    
    # 1. Missing Values Comparison across all CSV files
    plt.figure(figsize=(15, 6))
    missing_data = []
    for file_name, df in all_data.items():
        missing_count = df.isnull().sum().sum()
        missing_data.append({'File': file_name, 'Missing Values': missing_count})
    
    missing_df = pd.DataFrame(missing_data)
    plt.bar(missing_df['File'], missing_df['Missing Values'])
    plt.title('Missing Values Across All CSV Files')
    plt.xlabel('CSV Files')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'missing_values_comparison.png')
    plt.close()
    
    # 2. Regional Distribution (Pie Chart)
    plt.figure(figsize=(12, 8))
    # Use the first dataframe that has region information
    first_df = next(df for df in all_data.values() if 'Region of HE provider' in df.columns)
    region_counts = first_df['Region of HE provider'].value_counts()
    plt.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Institutions by Region')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_dir / 'regional_distribution_pie.png')
    plt.close()
    
    # 3. North East Institutions Value Trends
    plt.figure(figsize=(15, 8))
    ne_institutions = ['University of Durham', 'Newcastle University', 
                      'University of Northumbria at Newcastle', 
                      'The University of Sunderland', 'Teesside University']
    
    for inst in ne_institutions:
        yearly_means = []
        for df in all_data.values():
            if 'HE Provider' in df.columns and 'Value' in df.columns and 'Academic Year' in df.columns:
                inst_data = df[df['HE Provider'] == inst]
                if not inst_data.empty:
                    yearly_mean = inst_data.groupby('Academic Year')['Value'].mean()
                    yearly_means.append(yearly_mean)
        
        if yearly_means:
            combined_means = pd.concat(yearly_means).groupby(level=0).mean()
            plt.plot(combined_means.index, combined_means.values, marker='o', label=inst)
    
    plt.title('Average Value Trends for North East Institutions')
    plt.xlabel('Academic Year')
    plt.ylabel('Average Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'ne_institutions_trends.png')
    plt.close()
    
    # 4. Overall Dataset Overview
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 4.1 Total Records per File
    records_count = [len(df) for df in all_data.values()]
    ax1.bar(all_data.keys(), records_count)
    ax1.set_title('Total Records per File')
    ax1.set_xticklabels(all_data.keys(), rotation=45)
    
    # 4.2 Average Values per File
    avg_values = [df['Value'].mean() if 'Value' in df.columns else 0 for df in all_data.values()]
    ax2.bar(all_data.keys(), avg_values)
    ax2.set_title('Average Values per File')
    ax2.set_xticklabels(all_data.keys(), rotation=45)
    
    # 4.3 Number of Unique Institutions per File
    unique_inst = [df['HE Provider'].nunique() if 'HE Provider' in df.columns else 0 
                  for df in all_data.values()]
    ax3.bar(all_data.keys(), unique_inst)
    ax3.set_title('Number of Unique Institutions per File')
    ax3.set_xticklabels(all_data.keys(), rotation=45)
    
    # 4.4 Year Coverage per File
    year_coverage = [df['Academic Year'].nunique() if 'Academic Year' in df.columns else 0 
                    for df in all_data.values()]
    ax4.bar(all_data.keys(), year_coverage)
    ax4.set_title('Year Coverage per File')
    ax4.set_xticklabels(all_data.keys(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_overview.png')
    plt.close()

def analyze_csv_file(file_path):
    print(f"\n{'='*50}")
    print(f"Analyzing file: {file_path}")
    print(f"{'='*50}")
    
    # header at line 12
    df = pd.read_csv(file_path, skiprows=11, header=0)
    
    # Display basic information
    print("\nColumn names:")
    print(df.columns.tolist())
    
    print("\nFirst 10 rows of actual data:")
    print(df.head(10))
    
    # Check for missing values
    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    has_missing = False
    for col, missing in missing_values.items():
        if missing > 0:
            has_missing = True
            print(f"{col}: {missing} missing values")
    
    # Display samples with missing values
    if has_missing:
        print("\nSample rows with missing values:")
        missing_rows = df[df.isnull().any(axis=1)]
        print(missing_rows.head())
    
    # Display basic statistics
    print("\nBasic statistics:")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Additional statistics
    if 'Academic Year' in df.columns:
        print("\nTime coverage:")
        print(f"Earliest year: {df['Academic Year'].min()}")
        print(f"Latest year: {df['Academic Year'].max()}")
        print(f"Number of years: {df['Academic Year'].nunique()}")
    
    if 'HE Provider' in df.columns:
        print("\nInstitution statistics:")
        print(f"Total number of institutions: {df['HE Provider'].nunique()}")
        
        if 'Region of HE provider' in df.columns:
            print("\nRegional distribution:")
            region_counts = df['Region of HE provider'].value_counts()
            print("\nNumber of institutions by region:")
            print(region_counts)
            
            # North East specific analysis
            ne_institutions = df[df['Region of HE provider'] == 'North East']['HE Provider'].unique()
            print(f"\nNorth East institutions ({len(ne_institutions)}):")
            for inst in ne_institutions:
                print(f"- {inst}")
            
            # Durham University specific analysis
            durham_data = df[df['HE Provider'] == 'Durham University']
            if not durham_data.empty:
                print("\nDurham University statistics:")
                if 'Value' in df.columns:
                    print(f"Total records: {len(durham_data)}")
                    if 'Unit' in df.columns:
                        print("\nValue distribution by unit:")
                        print(durham_data.groupby('Unit')['Value'].describe())
    
    return df

def handle_missing_values(df, file_name):
    """Handle missing values in the dataframe based on file type and column type"""
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Handle Value column missing values
    if 'Value' in df_processed.columns:
        # For financial data (table-1, table-2b, table-3, table-4c, table-4d)
        if file_name in ['table-1.csv', 'table-2b.csv', 'table-3.csv', 'table-4c.csv', 'table-4d.csv']:
            df_processed['Value'] = df_processed['Value'].fillna(0)
        # For count data (table-4a, table-4b, table-4e, table-5)
        else:
            df_processed['Value'] = df_processed['Value'].fillna(0)
    
    # Handle Unit column missing values (specific to table-2a)
    if 'Unit' in df_processed.columns and 'Number/Value Marker' in df_processed.columns:
        # Fill Unit based on Number/Value Marker type
        df_processed['Unit'] = df_processed.apply(
            lambda row: '£000s' if pd.notna(row['Number/Value']) and isinstance(row['Number/Value'], (int, float)) 
            else 'Count' if pd.notna(row['Number/Value']) 
            else row['Unit'], 
            axis=1
        )
    
    # Handle Number/Value column missing values in table-2a
    if file_name == 'table-2a.csv' and 'Number/Value' in df_processed.columns:
        df_processed['Number/Value'] = df_processed['Number/Value'].fillna(0)
    
    # Log the changes
    missing_before = df.isnull().sum().sum()
    missing_after = df_processed.isnull().sum().sum()
    logging.info(f"File: {file_name} - Missing values reduced from {missing_before} to {missing_after}")
    
    return df_processed

def main():
    # Get all CSV files in the Data directory
    data_dir = Path("Data")
    csv_files = list(data_dir.glob("table-*.csv"))
    
    # Sort files by name
    csv_files.sort()
    
    # Store all dataframes
    all_data = {}
    
    # Analyze each file
    for file_path in csv_files:
        with open(file_path, 'r') as f:
            header_lines = [next(f) for _ in range(11)]
        
        df = pd.read_csv(file_path, skiprows=11, header=0)
        all_data[file_path.name] = df  # Store original data
        
        df_processed = handle_missing_values(df, file_path.name)
        
        temp_path = file_path.parent / f"temp_{file_path.name}"
        df_processed.to_csv(temp_path, index=False)
        
        with open(file_path, 'w', encoding='utf-8') as f_out:
            f_out.writelines(header_lines)
            with open(temp_path, 'r', encoding='utf-8') as f_temp:
                f_out.writelines(f_temp.readlines())
        
        temp_path.unlink()
            
        logging.info(f"Updated missing values in {file_path}")
    
    # Create visualizations using original data
    create_dataset_visualizations(all_data)

if __name__ == "__main__":
    main() 