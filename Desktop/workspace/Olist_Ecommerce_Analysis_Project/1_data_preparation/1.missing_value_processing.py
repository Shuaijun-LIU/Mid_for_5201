import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import logging
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# paths
DATA_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed_missing')

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# File configuration
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
    """Load CSV file and return pandas DataFrame"""
    try:
        df = pd.read_csv(DATA_DIR / file_name)
        print(f"\nSuccessfully loaded {file_name}")
        logging.info(f"\nSuccessfully loaded {file_name}")
        return df
    except Exception as e:
        print(f"Error loading {file_name}: {str(e)}")
        logging.error(f"Error loading {file_name}: {str(e)}")
        return None

def check_missing_values(df, file_name):
    """Check for any remaining missing values in the DataFrame"""
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]
    
    if len(missing_columns) > 0:
        print(f"\nWarning: {file_name} still has missing values:")
        for col in missing_columns.index:
            print(f"- {col}: {missing_columns[col]} missing values")
    else:
        print(f"\n{file_name}: No missing values found")

def process_orders(df):
    """Process missing values in orders dataset"""
    print("\nProcessing missing values in orders dataset...")
    logging.info("\nProcessing missing values in orders dataset...")
    
    # Columns to process
    timestamp_columns = [
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date'
    ]
    
    # Fill missing values
    for col in timestamp_columns:
        df[col] = df[col].fillna("0000-00-00 00:00:00")
        print(f"Filled missing values in {col}")
        logging.info(f"Filled missing values in {col}")
    
    return df

def process_products(df):
    """Process missing values in products dataset"""
    print("\nProcessing missing values in products dataset...")
    logging.info("\nProcessing missing values in products dataset...")
    
    # Fill text column
    df['product_category_name'] = df['product_category_name'].fillna("unknown")
    print("Filled missing values in product_category_name")
    logging.info("Filled missing values in product_category_name")
    
    # Fill numeric columns
    numeric_columns = [
        'product_name_lenght',
        'product_description_lenght',
        'product_photos_qty',
        'product_weight_g',
        'product_length_cm',
        'product_height_cm',
        'product_width_cm'
    ]
    
    for col in numeric_columns:
        df[col] = df[col].fillna(-1)
        print(f"Filled missing values in {col}")
        logging.info(f"Filled missing values in {col}")
    
    return df

def process_reviews(df):
    """Process missing values in reviews dataset"""
    print("\nProcessing missing values in reviews dataset...")
    logging.info("\nProcessing missing values in reviews dataset...")
    
    # Fill text columns
    text_columns = [
        'review_comment_title',
        'review_comment_message'
    ]
    
    for col in text_columns:
        df[col] = df[col].fillna("unknown")
        print(f"Filled missing values in {col}")
        logging.info(f"Filled missing values in {col}")
    
    return df

def copy_file(file_name):
    """Copy files that don't need processing to target directory"""
    source_path = DATA_DIR / file_name
    target_path = PROCESSED_DIR / file_name
    
    try:
        shutil.copy2(source_path, target_path)
        print(f"\nCopied {file_name} to target directory")
        logging.info(f"\nCopied {file_name} to target directory")
    except Exception as e:
        print(f"Error copying {file_name}: {str(e)}")
        logging.error(f"Error copying {file_name}: {str(e)}")

def main():
    """Main function to process missing values in all datasets"""
    print("Starting missing value processing...")
    logging.info("Starting missing value processing...")
    
    # Process files that need special handling
    orders_df = load_data(FILES['orders'])
    if orders_df is not None:
        processed_orders = process_orders(orders_df)
        processed_orders.to_csv(PROCESSED_DIR / FILES['orders'], index=False)
        check_missing_values(processed_orders, FILES['orders'])
    
    products_df = load_data(FILES['products'])
    if products_df is not None:
        processed_products = process_products(products_df)
        processed_products.to_csv(PROCESSED_DIR / FILES['products'], index=False)
        check_missing_values(processed_products, FILES['products'])
    
    reviews_df = load_data(FILES['reviews'])
    if reviews_df is not None:
        processed_reviews = process_reviews(reviews_df)
        processed_reviews.to_csv(PROCESSED_DIR / FILES['reviews'], index=False)
        check_missing_values(processed_reviews, FILES['reviews'])
    
    # Copy files that don't need processing
    files_to_copy = [
        FILES['order_items'],
        FILES['customers'],
        FILES['payments'],
        FILES['sellers'],
        FILES['category_translation'],
        FILES['geolocation']
    ]
    
    for file_name in files_to_copy:
        copy_file(file_name)
        # Check copied files for missing values
        df = load_data(file_name)
        if df is not None:
            check_missing_values(df, file_name)
    
    print("\nMissing value processing completed!")
    logging.info("\nMissing value processing completed!")

if __name__ == "__main__":
    main() 