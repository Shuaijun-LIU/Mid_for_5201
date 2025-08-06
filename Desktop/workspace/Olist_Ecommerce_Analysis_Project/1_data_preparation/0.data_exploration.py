import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("week1_data_preparation/logs/data_exploration.log"),
        logging.StreamHandler(),
    ],
)

# set up path
DATA_DIR = Path('data/raw')

# file name
FILES = {
    'orders': 'olist_orders_dataset.csv',
    'order_items': 'olist_order_items_dataset.csv',
    'products': 'olist_products_dataset.csv',
    'customers': 'olist_customers_dataset.csv',
    'reviews': 'olist_order_reviews_dataset.csv',
    'payments': 'olist_order_payments_dataset.csv',
    'sellers': 'olist_sellers_dataset.csv',
    'category_translation': 'product_category_name_translation.csv',
    'geolocation': 'olist_geolocation_dataset.csv'
}

def load_data(file_name):
    """Load a CSV file and return a pandas DataFrame"""
    try:
        df = pd.read_csv(DATA_DIR / file_name)
        print(f"\nSuccessfully loaded {file_name}")
        logging.info(f"\nSuccessfully loaded {file_name}")
        return df
    except Exception as e:
        print(f"Error loading {file_name}: {str(e)}")
        logging.error(f"Error loading {file_name}: {str(e)}")
        return None

def analyze_missing_values(df, table_name):
    """Analyze missing values in a DataFrame"""
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    print(f"\nMissing Values Analysis for {table_name}:")
    logging.info(f"\nMissing Values Analysis for {table_name}:")
    print("-" * 50)
    logging.info("-" * 50)
    for col in df.columns:
        if missing_values[col] > 0:
            print(f"{col}: {missing_values[col]} missing values ({missing_percentage[col]:.2f}%)")
            logging.info(f"{col}: {missing_values[col]} missing values ({missing_percentage[col]:.2f}%)")
            # show the missing samples
            missing_samples = df[df[col].isnull()].head(5)
            if not missing_samples.empty:
                print("\nSample rows with missing values:")
                logging.info("\nSample rows with missing values:")
                print(missing_samples)
                logging.info(f"\n{missing_samples}")
                print("-" * 50)
                logging.info("-" * 50)

def analyze_data_types(df, table_name):
    """Analyze data types of columns"""
    print(f"\nData Types Analysis for {table_name}:")
    logging.info(f"\nData Types Analysis for {table_name}:")
    print("-" * 50)
    logging.info("-" * 50)
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
        logging.info(f"{col}: {df[col].dtype}")

def analyze_unique_values(df, table_name):
    """Analyze unique values in categorical columns"""
    print(f"\nUnique Values Analysis for {table_name}:")
    logging.info(f"\nUnique Values Analysis for {table_name}:")
    print("-" * 50)
    logging.info("-" * 50)
    for col in df.select_dtypes(include=['object']).columns:
        unique_count = df[col].nunique()
        if unique_count < 10:  # Only show if there are few unique values
            print(f"\n{col} unique values ({unique_count}):")
            logging.info(f"\n{col} unique values ({unique_count}):")
            print(df[col].value_counts().head())
            logging.info(f"\n{df[col].value_counts().head()}")

def analyze_numeric_columns(df, table_name):
    """Analyze numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric Columns Analysis for {table_name}:")
        logging.info(f"\nNumeric Columns Analysis for {table_name}:")
        print("-" * 50)
        logging.info("-" * 50)
        print(df[numeric_cols].describe())
        logging.info(f"\n{df[numeric_cols].describe()}")

def analyze_date_columns(df, table_name):
    """Analyze date columns"""
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
    if date_cols:
        print(f"\nDate Columns Analysis for {table_name}:")
        logging.info(f"\nDate Columns Analysis for {table_name}:")
        print("-" * 50)
        logging.info("-" * 50)
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col])
                print(f"\n{col}:")
                logging.info(f"\n{col}:")
                print(f"Range: {df[col].min()} to {df[col].max()}")
                logging.info(f"Range: {df[col].min()} to {df[col].max()}")
            except:
                print(f"Could not convert {col} to datetime")
                logging.warning(f"Could not convert {col} to datetime")

def analyze_table_relationships(df, table_name):
    """Analyze potential relationships between tables"""
    print(f"\nPotential Relationships Analysis for {table_name}:")
    logging.info(f"\nPotential Relationships Analysis for {table_name}:")
    print("-" * 50)
    logging.info("-" * 50)
    
    # Check for common ID columns
    id_columns = [col for col in df.columns if 'id' in col.lower()]
    if id_columns:
        print("ID columns found:")
        logging.info("ID columns found:")
        for col in id_columns:
            print(f"- {col}: {df[col].nunique()} unique values")
            logging.info(f"- {col}: {df[col].nunique()} unique values")

def main():
    """Main function to run all analyses"""
    print("Starting data exploration...")
    logging.info("Starting data exploration...")
    
    datasets = {}
    for name, file in FILES.items():
        datasets[name] = load_data(file)
    
    # Analyze 
    for name, df in datasets.items():
        if df is not None:
            print(f"\n{'='*80}")
            logging.info(f"\n{'='*80}")
            print(f"Analyzing {name} dataset")
            logging.info(f"Analyzing {name} dataset")
            print(f"{'='*80}")
            logging.info(f"{'='*80}")
            
            # Basic information
            print(f"\nDataset Shape: {df.shape}")
            logging.info(f"\nDataset Shape: {df.shape}")
            
            analyze_missing_values(df, name)
            analyze_data_types(df, name)
            analyze_unique_values(df, name)
            analyze_numeric_columns(df, name)
            analyze_date_columns(df, name)
            analyze_table_relationships(df, name)

if __name__ == "__main__":
    main() 