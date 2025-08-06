"""
Generate customer profiles for Olist E-commerce dataset
"""
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from pathlib import Path
import json

# import configuration
from config import (
    RAW_DATA_DIR, CLEANED_DATA_DIR, SAMPLE_DATA_DIR,
    DTYPE_DICT, DATE_COLUMNS, FORCE_REPROCESS, SAMPLE_RATIO
)

# create logs directory
os.makedirs("week1_data_preparation/logs", exist_ok=True)

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("week1_data_preparation/logs/derived_customer_profile.log"),
        logging.StreamHandler(),
    ],
)

def validate_customer_profile(df: pd.DataFrame) -> bool:
    """
    Validate customer profile data
    Args:
        df: Customer profile dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required fields
        assert not df['customer_id'].isna().any(), "Missing customer_id"
        assert not df['total_orders'].isna().any(), "Missing total_orders"
        assert not df['total_spent'].isna().any(), "Missing total_spent"
        assert not df['avg_order_value'].isna().any(), "Missing avg_order_value"
        assert not df['first_order_date'].isna().any(), "Missing first_order_date"
        assert not df['last_order_date'].isna().any(), "Missing last_order_date"
        
        # Check numeric fields
        numeric_cols = [
            'total_orders', 'total_spent', 'avg_order_value',
            'avg_delivery_time', 'customer_lifetime_days'
        ]
        for col in numeric_cols:
            assert df[col].dtype in ['float64', 'Int64'], f"Invalid {col} type"
        
        # Check value ranges
        assert df['total_orders'].min() >= 0, "Negative total orders found"
        assert df['total_spent'].min() >= 0, "Negative total spent found"
        assert df['avg_order_value'].min() >= 0, "Negative average order value found"
        assert df['avg_delivery_time'].min() >= 0, "Negative delivery time found"
        assert df['customer_lifetime_days'].min() >= 0, "Negative customer lifetime found"
        
        # Check logical relationships
        assert (df['total_spent'] >= df['total_orders']).all(), "Total spent should be greater than or equal to total orders"
        assert (df['last_order_date'] >= df['first_order_date']).all(), "Last order date should be after first order date"
        
        # Check behavior profile format
        assert df['behavior_profile'].apply(lambda x: isinstance(x, str)).all(), "Invalid behavior profile format"
        for profile in df['behavior_profile']:
            try:
                json.loads(profile)
            except json.JSONDecodeError:
                raise AssertionError("Invalid JSON in behavior profile")
        
        # Check for duplicates
        assert not df['customer_id'].duplicated().any(), "Duplicate customer IDs found"
        
        logging.info("Customer profile validation passed")
        return True
        
    except AssertionError as e:
        logging.error(f"Customer profile validation failed: {str(e)}")
        return False

def check_file_exists_and_valid(file_path: Path) -> bool:
    """
    Check if a file exists and has been successfully processed
    Args:
        file_path: Path to the file
    Returns:
        True if file exists and is valid, False otherwise
    """
    if not file_path.exists():
        return False
    
    try:
        # Check if file is not empty
        if file_path.stat().st_size == 0:
            return False
            
        # Try to read the file to check if it's valid
        df = pd.read_csv(
            file_path,
            dtype={
                'total_orders': 'Int64',
                'total_spent': 'float64',
                'avg_order_value': 'float64',
                'avg_delivery_time': 'float64',
                'customer_lifetime_days': 'Int64'
            },
            parse_dates=['first_order_date', 'last_order_date']
        )
            
        # Check if dataframe has data
        if len(df) == 0:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Error checking file {file_path}: {str(e)}")
        return False

def generate_customer_profile(orders_df: pd.DataFrame, order_items_df: pd.DataFrame, 
                            reviews_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate customer profiles
    Args:
        orders_df: Orders dataframe
        order_items_df: Order items dataframe
        reviews_df: Reviews dataframe
        products_df: Products dataframe
    Returns:
        Customer profiles dataframe
    """
    # Merge order items with products to get category information
    customer_orders = order_items_df.merge(
        products_df[['product_id', 'product_category_name']],
        on='product_id',
        how='left'
    ).merge(
        orders_df,
        on='order_id',
        how='left'
    )
    
    # Calculate customer metrics
    customer_metrics = pd.DataFrame()
    
    # Basic metrics
    customer_metrics['total_orders'] = customer_orders.groupby('customer_id')['order_id'].nunique()
    customer_metrics['total_items'] = customer_orders.groupby('customer_id')['order_item_id'].count()
    customer_metrics['total_spent'] = customer_orders.groupby('customer_id')['price'].sum()
    customer_metrics['avg_order_value'] = customer_metrics['total_spent'] / customer_metrics['total_orders']
    
    # Category metrics
    customer_metrics['total_categories'] = customer_orders.groupby('customer_id')['product_category_name'].nunique()
    
    # Get top category safely
    def get_top_category(x):
        if len(x) == 0:
            return None
        value_counts = x.value_counts()
        if len(value_counts) == 0:
            return None
        return value_counts.index[0]
    
    customer_metrics['top_category'] = customer_orders.groupby('customer_id')['product_category_name'].agg(get_top_category)
    
    # Review metrics
    customer_reviews = reviews_df.merge(orders_df[['order_id', 'customer_id']], on='order_id', how='left')
    customer_metrics['avg_review_score'] = customer_reviews.groupby('customer_id')['review_score'].mean()
    customer_metrics['total_reviews'] = customer_reviews.groupby('customer_id')['review_id'].count()
    
    # Behavior profile
    def create_behavior_profile(group):
        profile = {
            'purchase_frequency': len(group) / (
                (group['order_purchase_timestamp'].max() - group['order_purchase_timestamp'].min()).days + 1
            ) if len(group) > 0 else 0,
            'total_categories': group['product_category_name'].nunique(),
            'preferred_categories': group['product_category_name'].value_counts().head(3).to_dict()
        }
        return json.dumps(profile)
    
    customer_metrics['behavior_profile'] = customer_orders.groupby('customer_id').apply(create_behavior_profile)
    
    return customer_metrics.reset_index()

def main():
    """Main function to generate customer profiles"""
    try:
        print("\n=== Starting Customer Profile Generation ===")
        logging.info("=== Starting Customer Profile Generation ===")
        
        # 1. Load raw data
        print("\n1. Loading raw data...")
        logging.info("1. Loading raw data...")
        
        # Load orders data
        orders_file = RAW_DATA_DIR / 'olist_orders_dataset.csv'
        orders_df = pd.read_csv(orders_file)
        # Convert timestamp columns to datetime
        orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
        orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
        print(f"Loaded {len(orders_df)} order records")
        logging.info(f"Loaded {len(orders_df)} order records")
        
        # Load order items data
        order_items_file = RAW_DATA_DIR / 'olist_order_items_dataset.csv'
        order_items_df = pd.read_csv(order_items_file)
        print(f"Loaded {len(order_items_df)} order item records")
        logging.info(f"Loaded {len(order_items_df)} order item records")
        
        # Load reviews data
        reviews_file = RAW_DATA_DIR / 'olist_order_reviews_dataset.csv'
        reviews_df = pd.read_csv(reviews_file)
        print(f"Loaded {len(reviews_df)} review records")
        logging.info(f"Loaded {len(reviews_df)} review records")
        
        # Load products data
        products_file = RAW_DATA_DIR / 'olist_products_dataset.csv'
        products_df = pd.read_csv(products_file)
        print(f"Loaded {len(products_df)} product records")
        logging.info(f"Loaded {len(products_df)} product records")
        
        # 2. Generate customer profiles
        print("\n2. Generating customer profiles...")
        logging.info("2. Generating customer profiles...")
        customer_profiles = generate_customer_profile(orders_df, order_items_df, reviews_df, products_df)
        print(f"Generated {len(customer_profiles)} customer profiles")
        logging.info(f"Generated {len(customer_profiles)} customer profiles")
        
        # 3. Export profiles
        print("\n3. Exporting customer profiles...")
        logging.info("3. Exporting customer profiles...")
        output_file = CLEANED_DATA_DIR / 'customer_profiles.csv'
        customer_profiles.to_csv(output_file, index=False)
        print(f"Customer profiles exported to {output_file}")
        logging.info(f"Customer profiles exported to {output_file}")
        
        # 4. Generate sample
        print("\n4. Generating sample data...")
        logging.info("4. Generating sample data...")
        sample = customer_profiles.sample(frac=0.05, random_state=42)
        sample_file = SAMPLE_DATA_DIR / 'customer_profiles_sample.csv'
        sample.to_csv(sample_file, index=False)
        print(f"Sample data exported to {sample_file}")
        logging.info(f"Sample data exported to {sample_file}")
        
        print("\n=== Customer Profile Generation Completed Successfully ===")
        logging.info("=== Customer Profile Generation Completed Successfully ===")
        
    except Exception as e:
        print(f"\nERROR: Customer profile generation failed: {str(e)}")
        logging.error(f"Customer profile generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 