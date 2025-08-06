"""
Check the processed table files data
Used for comparison with SQL database creation statements
"""

import pandas as pd
import logging
import os
from pathlib import Path
import json

# Configure logging
os.makedirs("week1_data_preparation/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("week1_data_preparation/logs/check_tables.log"),
        logging.StreamHandler(),
    ],
)

# Define the list of files to check
TABLE_FILES = [
    'orders.csv',
    'order_items.csv',
    'order_payments.csv',
    'order_reviews.csv',
    'customers.csv',
    'sellers.csv',
    'products.csv',
    'product_categories.csv',
    'geolocations.csv',
    'dim_date.csv',
    'dim_location.csv',
    'region_sales_stats.csv',
    'customer_profiles.csv',
    'product_lifecycle.csv',
    'product_category_hierarchy.csv',
    'seller_performance.csv'
]

def get_table_info(file_path: Path) -> dict:
    """
    Get basic information about the table file
    Args:
        file_path: Path to the table file
    Returns:
        Dictionary containing table information
    """
    try:
        # Read first 1000 rows of the file
        df = pd.read_csv(file_path, nrows=1000)
        
        # Get basic information
        info = {
            'file_name': file_path.name,
            'total_rows': sum(1 for _ in open(file_path)),  # Get total number of rows
            'columns': list(df.columns),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'sample_data': df.head(5).to_dict('records'),
            'null_counts': df.isnull().sum().to_dict(),
            'unique_counts': {col: df[col].nunique() for col in df.columns}
        }
        
        # For numeric columns, add basic statistics
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe()
            info['numeric_stats'] = {
                col: {
                    'min': stats[col]['min'],
                    'max': stats[col]['max'],
                    'mean': stats[col]['mean'],
                    'std': stats[col]['std']
                } for col in numeric_cols
            }
        
        return info
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return {
            'file_name': file_path.name,
            'error': str(e)
        }

def main():
    """Main function"""
    try:
        print("\n=== Starting Table Check ===")
        logging.info("=== Starting Table Check ===")
        
        # Set data directory
        data_dir = Path("data/cleaned")
        
        # Store all table information
        tables_info = {}
        
        # Check each table file
        for file_name in TABLE_FILES:
            print(f"\nChecking {file_name}...")
            logging.info(f"Checking {file_name}...")
            
            file_path = data_dir / file_name
            if not file_path.exists():
                print(f"File not found: {file_name}")
                logging.warning(f"File not found: {file_name}")
                continue
                
            # Get table information
            table_info = get_table_info(file_path)
            tables_info[file_name] = table_info
            
            # Output basic information
            print(f"Total rows: {table_info['total_rows']}")
            print(f"Columns: {', '.join(table_info['columns'])}")
            print(f"Data types: {json.dumps(table_info['dtypes'], indent=2)}")
            
        # Save all table information to JSON file
        output_file = data_dir / 'tables_info.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tables_info, f, indent=2, ensure_ascii=False)
            
        print(f"\nTable information saved to {output_file}")
        logging.info(f"Table information saved to {output_file}")
        
        print("\n=== Table Check Completed ===")
        logging.info("=== Table Check Completed ===")
        
    except Exception as e:
        print(f"\nERROR: Table check failed: {str(e)}")
        logging.error(f"Table check failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 