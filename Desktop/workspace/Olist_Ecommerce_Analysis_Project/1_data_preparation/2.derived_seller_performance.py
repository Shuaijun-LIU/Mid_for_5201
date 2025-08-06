"""
Generate seller performance metrics for Olist E-commerce dataset
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from pathlib import Path

# import configuration
from config import (
    RAW_DATA_DIR, CLEANED_DATA_DIR, SAMPLE_DATA_DIR,
    DTYPE_DICT, DATE_COLUMNS, FORCE_REPROCESS, SAMPLE_RATIO
)

# create logs directory if not exists
os.makedirs("week1_data_preparation/logs", exist_ok=True)

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("week1_data_preparation/logs/derived_seller_performance.log"),
        logging.StreamHandler(),
    ],
)

def validate_seller_performance(df: pd.DataFrame) -> bool:
    """
    Validate seller performance data
    Args:
        df: Seller performance dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required columns
        required_cols = [
            'seller_id', 'total_sales', 'total_orders',
            'delivery_time_avg', 'avg_rating', 'customer_satisfaction_score'
        ]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            logging.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check data types
        if not pd.api.types.is_string_dtype(df['seller_id']):
            logging.error("seller_id must be string type")
            return False
        
        # Check numeric ranges
        if not (df['total_sales'] >= 0).all():
            logging.error("total_sales must be non-negative")
            return False
        
        if not (df['total_orders'] > 0).all():
            logging.error("total_orders must be positive")
            return False
        
        if not (df['delivery_time_avg'] >= 0).all():
            logging.error("delivery_time_avg must be non-negative")
            return False
        
        # Check rating range (1-5)
        if not ((df['avg_rating'] >= 1) & (df['avg_rating'] <= 5)).all():
            logging.error("avg_rating must be between 1 and 5")
            return False
        
        # Check customer satisfaction score range (0-1)
        if not ((df['customer_satisfaction_score'] >= 0) & (df['customer_satisfaction_score'] <= 1)).all():
            logging.error("customer_satisfaction_score must be between 0 and 1")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
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
                'total_sales': 'float64',
                'total_orders': 'Int64',
                'avg_rating': 'float64',
                'delivery_time_avg': 'float64',
                'customer_satisfaction_score': 'float64'
            }
        )
            
        # Check if dataframe has data
        if len(df) == 0:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Error checking file {file_path}: {str(e)}")
        return False

def generate_seller_performance(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    reviews_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate seller performance metrics
    Args:
        orders_df: Orders dataframe
        order_items_df: Order items dataframe
        reviews_df: Reviews dataframe
    Returns:
        Seller performance dataframe
    """
    # Merge dataframes
    df = order_items_df.merge(orders_df[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date']], on='order_id')
    df = df.merge(reviews_df[['order_id', 'review_score']], on='order_id')
    
    # Calculate delivery time
    df['delivery_time'] = (
        pd.to_datetime(df['order_delivered_customer_date']) - 
        pd.to_datetime(df['order_purchase_timestamp'])
    ).dt.total_seconds() / (24 * 3600)  # Convert to days
    
    # Calculate seller metrics
    seller_metrics = df.groupby('seller_id').agg({
        'price': 'sum',  # Total sales
        'order_id': 'nunique',  # Total orders
        'delivery_time': 'mean',  # Average delivery time
        'review_score': 'mean'  # Average rating
    }).rename(columns={
        'price': 'total_sales',
        'order_id': 'total_orders',
        'delivery_time': 'delivery_time_avg',
        'review_score': 'avg_rating'
    })
    
    # Calculate customer satisfaction score (normalized between 0 and 1)
    seller_metrics['customer_satisfaction_score'] = (
        seller_metrics['avg_rating'] - 1
    ) / 4  # Convert 1-5 scale to 0-1 scale
    
    # Reset index to make seller_id a column
    seller_metrics = seller_metrics.reset_index()
    
    return seller_metrics

def main():
    """Main function to generate seller performance metrics"""
    try:
        print("\n=== Starting Seller Performance Generation ===")
        
        # 1. Load raw data
        print("\n1. Loading raw data...")
        orders_file = CLEANED_DATA_DIR / 'orders.csv'
        order_items_file = CLEANED_DATA_DIR / 'order_items.csv'
        reviews_file = CLEANED_DATA_DIR / 'order_reviews.csv'
        
        orders_df = pd.read_csv(
            orders_file,
            parse_dates=['order_purchase_timestamp', 'order_delivered_customer_date']
        )
        order_items_df = pd.read_csv(order_items_file)
        reviews_df = pd.read_csv(reviews_file)
        
        print(f"Loaded {len(orders_df)} order records")
        print(f"Loaded {len(order_items_df)} order item records")
        print(f"Loaded {len(reviews_df)} review records")
        
        # 2. Generate performance metrics
        print("\n2. Generating performance metrics...")
        seller_performance = generate_seller_performance(orders_df, order_items_df, reviews_df)
        print(f"Generated {len(seller_performance)} seller performance records")
        
        # 3. Validate data
        print("\n3. Validating data...")
        if not validate_seller_performance(seller_performance):
            raise ValueError("Seller performance validation failed")
        print("Data validation passed")
        
        # 4. Export data
        print("\n4. Exporting data...")
        output_file = CLEANED_DATA_DIR / 'seller_performance.csv'
        seller_performance.to_csv(output_file, index=False)
        print(f"Data exported to {output_file}")
        
        # 5. Generate sample
        print("\n5. Generating sample data...")
        sample = seller_performance.sample(frac=0.05, random_state=42)
        sample_file = SAMPLE_DATA_DIR / 'seller_performance_sample.csv'
        sample.to_csv(sample_file, index=False)
        print(f"Sample data exported to {sample_file}")
        
        print("\n=== Seller Performance Generation Completed Successfully ===")
        
    except Exception as e:
        print(f"\nERROR: Seller performance generation failed: {str(e)}")
        logging.error(f"Seller performance generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 