"""
Generate region sales statistics for Olist E-commerce dataset
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
        logging.FileHandler("week1_data_preparation/logs/derived_region_sales_stats.log"),
        logging.StreamHandler(),
    ],
)

def load_required_data() -> dict:
    """
    Load all required datasets for regional sales statistics
    Returns:
        Dictionary containing all required dataframes
    """
    try:
        # Load orders data
        orders_df = pd.read_csv(
            CLEANED_DATA_DIR / 'orders.csv',
            parse_dates=[
                'order_purchase_timestamp',
                'order_approved_at',
                'order_delivered_carrier_date',
                'order_delivered_customer_date',
                'order_estimated_delivery_date'
            ]
        )
        
        # Load order items data
        order_items_df = pd.read_csv(CLEANED_DATA_DIR / 'order_items.csv')
        
        # Load sellers data
        sellers_df = pd.read_csv(CLEANED_DATA_DIR / 'sellers.csv')
        
        # Load geolocations data
        geolocations_df = pd.read_csv(CLEANED_DATA_DIR / 'geolocations.csv')
        
        return {
            'orders': orders_df,
            'order_items': order_items_df,
            'sellers': sellers_df,
            'geolocations': geolocations_df
        }
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def calculate_regional_sales_stats(data: dict) -> pd.DataFrame:
    """
    Calculate regional sales statistics
    Args:
        data: Dictionary containing all required dataframes
    Returns:
        DataFrame with regional sales statistics
    """
    try:
        # 1. Merge orders with order items to get sales data
        sales_df = pd.merge(
            data['orders'],
            data['order_items'],
            on='order_id',
            how='inner'
        )
        
        # 2. Merge with sellers to get seller location
        sales_df = pd.merge(
            sales_df,
            data['sellers'],
            on='seller_id',
            how='inner'
        )
        
        # 3. Merge with geolocations to get detailed location data
        sales_df = pd.merge(
            sales_df,
            data['geolocations'],
            left_on='seller_zip_code_prefix',
            right_on='geolocation_zip_code_prefix',
            how='inner'
        )
        
        # 4. Calculate monthly statistics by region
        sales_df['year_month'] = sales_df['order_purchase_timestamp'].dt.to_period('M')
        
        # Group by region and calculate statistics
        regional_stats = sales_df.groupby(['geolocation_state', 'year_month']).agg({
            'order_id': 'count',
            'price': 'sum',
            'freight_value': 'sum',
            'seller_id': 'nunique',
            'customer_id': 'nunique'
        }).reset_index()
        
        # Rename columns
        regional_stats.columns = [
            'state',
            'year_month',
            'order_count',
            'total_sales',
            'total_freight',
            'seller_count',
            'customer_count'
        ]
        
        # Calculate additional metrics
        regional_stats['average_order_value'] = regional_stats['total_sales'] / regional_stats['order_count']
        regional_stats['sales_with_freight'] = regional_stats['total_sales'] + regional_stats['total_freight']
        
        # Calculate month-over-month growth
        regional_stats = regional_stats.sort_values(['state', 'year_month'])
        regional_stats['sales_growth'] = regional_stats.groupby('state')['total_sales'].pct_change()
        
        # Handle missing values
        # Fill first month's growth with 0
        regional_stats['sales_growth'] = regional_stats['sales_growth'].fillna(0)
        
        # Convert year_month to string format for better compatibility
        regional_stats['year_month'] = regional_stats['year_month'].astype(str)
        
        # Ensure all numeric columns are properly formatted
        numeric_columns = [
            'order_count', 'total_sales', 'total_freight',
            'seller_count', 'customer_count', 'average_order_value',
            'sales_with_freight', 'sales_growth'
        ]
        for col in numeric_columns:
            regional_stats[col] = pd.to_numeric(regional_stats[col], errors='coerce')
        
        return regional_stats
        
    except Exception as e:
        logging.error(f"Error calculating regional sales statistics: {str(e)}")
        raise

def validate_regional_stats(df: pd.DataFrame) -> bool:
    """
    Validate regional sales statistics
    Args:
        df: Regional sales statistics dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check for required columns
        required_columns = [
            'state', 'year_month', 'order_count', 'total_sales',
            'total_freight', 'seller_count', 'customer_count',
            'average_order_value', 'sales_with_freight', 'sales_growth'
        ]
        assert all(col in df.columns for col in required_columns), "Missing required columns"
        
        # Check for negative values in non-growth metrics
        non_growth_columns = [
            'order_count', 'total_sales', 'total_freight',
            'seller_count', 'customer_count', 'average_order_value',
            'sales_with_freight'
        ]
        assert not (df[non_growth_columns] < 0).any().any(), "Found negative values in non-growth metrics"
        
        # Check for missing values in non-growth metrics
        assert not df[non_growth_columns].isna().any().any(), "Found missing values in non-growth metrics"
        
        # Check for infinite values
        assert not np.isinf(df[non_growth_columns]).any().any(), "Found infinite values in numeric columns"
        
        logging.info("Regional sales statistics validation passed")
        return True
        
    except AssertionError as e:
        logging.error(f"Regional sales statistics validation failed: {str(e)}")
        return False

def generate_region_sales_stats(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    geolocations_df: pd.DataFrame,
    sellers_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate regional sales statistics
    Args:
        orders_df: Orders dataframe
        order_items_df: Order items dataframe
        geolocations_df: Geolocations dataframe
        sellers_df: Sellers dataframe
    Returns:
        Regional sales statistics dataframe
    """
    # Print column names for debugging
    print("\nDebugging Information:")
    print("orders_df columns:", orders_df.columns.tolist())
    print("order_items_df columns:", order_items_df.columns.tolist())
    print("geolocations_df columns:", geolocations_df.columns.tolist())
    print("sellers_df columns:", sellers_df.columns.tolist())
    
    # 1. Merge orders with order items to get sales data
    df = pd.merge(orders_df, order_items_df, on='order_id', how='inner')
    print("\nAfter merging with order items, df columns:", df.columns.tolist())
    
    # 2. Merge with sellers to get seller location
    df = pd.merge(df, sellers_df, on='seller_id', how='inner')
    print("\nAfter merging with sellers, df columns:", df.columns.tolist())
    
    # 3. Merge with geolocations to get detailed location data
    df = pd.merge(
        df,
        geolocations_df,
        left_on='seller_zip_code_prefix',
        right_on='geolocation_zip_code_prefix',
        how='inner'
    )
    print("\nAfter merging with geolocations, df columns:", df.columns.tolist())
    
    # Convert timestamp to datetime
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # Add year_month column
    df['year_month'] = df['order_purchase_timestamp'].dt.to_period('M')
    
    # Calculate regional metrics
    region_stats = df.groupby(['geolocation_state', 'year_month']).agg({
        'order_id': 'count',  # Total orders
        'price': 'sum',  # Total sales
        'freight_value': 'sum',  # Total freight
        'seller_id': 'nunique',  # Total sellers
        'customer_id': 'nunique'  # Total customers
    }).rename(columns={
        'order_id': 'order_count',
        'price': 'total_sales',
        'freight_value': 'total_freight',
        'seller_id': 'seller_count',
        'customer_id': 'customer_count'
    })
    
    # Reset index
    region_stats = region_stats.reset_index()
    region_stats = region_stats.rename(columns={'geolocation_state': 'state'})
    
    # Calculate additional metrics
    region_stats['average_order_value'] = region_stats['total_sales'] / region_stats['order_count']
    region_stats['sales_with_freight'] = region_stats['total_sales'] + region_stats['total_freight']
    
    # Calculate month-over-month growth
    region_stats = region_stats.sort_values(['state', 'year_month'])
    region_stats['sales_growth'] = region_stats.groupby('state')['total_sales'].pct_change()
    region_stats['sales_growth'] = region_stats['sales_growth'].fillna(0)
    
    # Convert year_month to string
    region_stats['year_month'] = region_stats['year_month'].astype(str)
    
    print("\nFinal region_stats columns:", region_stats.columns.tolist())
    print("First few rows of region_stats:")
    print(region_stats.head())
    
    return region_stats

def main():
    """
    Main function to generate regional sales statistics
    """
    try:
        print("\n=== Starting Regional Sales Statistics Generation ===\n")
        
        # Load data
        data = load_required_data()
        print(f"\nSuccessfully loaded data for {len(data['orders'])} orders\n")
        
        print("2. Generating regional sales statistics...")
        region_stats = generate_region_sales_stats(
            data['orders'],
            data['order_items'],
            data['geolocations'],
            data['sellers']
        )
        
        # Validate the generated statistics
        if validate_regional_stats(region_stats):
            # Save the results
            output_path = CLEANED_DATA_DIR / 'region_sales_stats.csv'
            region_stats.to_csv(output_path, index=False)
            print(f"\nRegional sales statistics saved to: {output_path}")
            print(f"Generated statistics for {len(region_stats)} region-month combinations")
        else:
            raise ValueError("Regional sales statistics validation failed")
            
    except Exception as e:
        logging.error(f"Regional sales statistics generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 