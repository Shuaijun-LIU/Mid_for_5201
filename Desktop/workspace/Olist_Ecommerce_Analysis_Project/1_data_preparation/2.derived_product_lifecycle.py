"""
Generate product lifecycle metrics for Olist E-commerce dataset
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
        logging.FileHandler("week1_data_preparation/logs/derived_product_lifecycle.log"),
        logging.StreamHandler(),
    ],
)

def load_required_data() -> dict:
    """
    Load all required datasets for product lifecycle
    Returns:
        Dictionary containing all required dataframes
    """
    try:
        # Load products data
        products_df = pd.read_csv(CLEANED_DATA_DIR / 'products.csv')
        
        # Load order items data
        order_items_df = pd.read_csv(CLEANED_DATA_DIR / 'order_items.csv')
        
        # Load orders data
        orders_df = pd.read_csv(CLEANED_DATA_DIR / 'orders.csv')
        
        return {
            'products': products_df,
            'order_items': order_items_df,
            'orders': orders_df
        }
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def calculate_product_lifecycle(data: dict) -> pd.DataFrame:
    """
    Calculate product lifecycle metrics
    Args:
        data: Dictionary containing all required dataframes
    Returns:
        DataFrame with product lifecycle metrics
    """
    try:
        # 1. Merge order items with orders to get dates
        sales_data = pd.merge(
            data['order_items'],
            data['orders'][['order_id', 'order_purchase_timestamp']],
            on='order_id',
            how='inner'
        )
        
        # Convert timestamp to datetime
        sales_data['order_purchase_timestamp'] = pd.to_datetime(sales_data['order_purchase_timestamp'])
        
        # 2. Calculate first and last sale dates
        first_sale = sales_data.groupby('product_id').agg({
            'order_purchase_timestamp': 'min'
        }).reset_index()
        first_sale.columns = ['product_id', 'first_sale_date']
        
        last_sale = sales_data.groupby('product_id').agg({
            'order_purchase_timestamp': 'max'
        }).reset_index()
        last_sale.columns = ['product_id', 'last_sale_date']
        
        # 3. Calculate sales frequency
        sales_frequency = sales_data.groupby('product_id').agg({
            'order_id': 'count',
            'price': 'sum',
            'freight_value': 'sum'
        }).reset_index()
        sales_frequency.columns = ['product_id', 'total_orders', 'total_revenue', 'total_freight']
        
        # 4. Calculate monthly sales
        sales_data['year_month'] = sales_data['order_purchase_timestamp'].dt.strftime('%Y-%m')
        monthly_sales = sales_data.groupby(['product_id', 'year_month']).agg({
            'order_id': 'count',
            'price': 'sum'
        }).reset_index()
        
        # Calculate months with sales
        months_with_sales = monthly_sales.groupby('product_id').size().reset_index(name='months_with_sales')
        
        # 5. Calculate lifecycle metrics
        lifecycle_df = pd.merge(first_sale, last_sale, on='product_id', how='outer')
        lifecycle_df = pd.merge(lifecycle_df, sales_frequency, on='product_id', how='outer')
        lifecycle_df = pd.merge(lifecycle_df, months_with_sales, on='product_id', how='outer')
        
        # Calculate lifecycle duration in days
        lifecycle_df['lifecycle_duration_days'] = (
            lifecycle_df['last_sale_date'] - lifecycle_df['first_sale_date']
        ).dt.days
        
        # Calculate average monthly orders (with protection against division by zero)
        lifecycle_df['avg_monthly_orders'] = np.where(
            lifecycle_df['months_with_sales'] > 0,
            lifecycle_df['total_orders'] / lifecycle_df['months_with_sales'],
            0
        )
        
        # Calculate average order value (with protection against division by zero)
        lifecycle_df['avg_order_value'] = np.where(
            lifecycle_df['total_orders'] > 0,
            lifecycle_df['total_revenue'] / lifecycle_df['total_orders'],
            0
        )
        
        # Calculate freight ratio (with protection against division by zero)
        lifecycle_df['freight_ratio'] = np.where(
            lifecycle_df['total_revenue'] > 0,
            lifecycle_df['total_freight'] / lifecycle_df['total_revenue'],
            0
        )
        
        # 6. Add product information
        lifecycle_df = pd.merge(
            lifecycle_df,
            data['products'][['product_id', 'product_category_name', 'product_name_length', 'product_description_length', 'product_photos_qty']],
            on='product_id',
            how='left'
        )
        
        # 7. Calculate product lifecycle stage
        def get_lifecycle_stage(row):
            if pd.isna(row['lifecycle_duration_days']) or row['lifecycle_duration_days'] < 0:
                return 'Unknown'
            elif row['lifecycle_duration_days'] <= 30:
                return 'New'
            elif row['lifecycle_duration_days'] <= 90:
                return 'Growth'
            elif row['lifecycle_duration_days'] <= 180:
                return 'Mature'
            else:
                return 'Decline'
        
        lifecycle_df['lifecycle_stage'] = lifecycle_df.apply(get_lifecycle_stage, axis=1)
        
        # 8. Calculate product performance metrics (with protection against division by zero)
        lifecycle_df['sales_velocity'] = np.where(
            lifecycle_df['lifecycle_duration_days'] > 0,
            lifecycle_df['total_orders'] / lifecycle_df['lifecycle_duration_days'],
            0
        )
        
        lifecycle_df['revenue_velocity'] = np.where(
            lifecycle_df['lifecycle_duration_days'] > 0,
            lifecycle_df['total_revenue'] / lifecycle_df['lifecycle_duration_days'],
            0
        )
        
        # 9. Select and order columns
        lifecycle_dim = lifecycle_df[[
            'product_id',
            'product_category_name',
            'product_name_length',
            'product_description_length',
            'product_photos_qty',
            'first_sale_date',
            'last_sale_date',
            'lifecycle_duration_days',
            'lifecycle_stage',
            'total_orders',
            'total_revenue',
            'total_freight',
            'months_with_sales',
            'avg_monthly_orders',
            'avg_order_value',
            'freight_ratio',
            'sales_velocity',
            'revenue_velocity'
        ]].copy()
        
        return lifecycle_dim
        
    except Exception as e:
        logging.error(f"Error calculating product lifecycle: {str(e)}")
        raise

def validate_product_lifecycle(df: pd.DataFrame) -> bool:
    """
    Validate product lifecycle table
    Args:
        df: Product lifecycle dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check for required columns
        required_columns = [
            'product_id', 'product_category_name', 'first_sale_date', 'last_sale_date',
            'lifecycle_duration_days', 'lifecycle_stage', 'total_orders', 'total_revenue',
            'total_freight', 'months_with_sales', 'avg_monthly_orders', 'avg_order_value',
            'freight_ratio', 'sales_velocity', 'revenue_velocity'
        ]
        assert all(col in df.columns for col in required_columns), "Missing required columns"
        
        # Check for missing values in key columns
        key_columns = ['product_id', 'first_sale_date', 'last_sale_date']
        assert not df[key_columns].isna().any().any(), "Found missing values in key columns"
        
        # Check lifecycle stage values
        valid_stages = ['New', 'Growth', 'Mature', 'Decline', 'Unknown']
        assert df['lifecycle_stage'].isin(valid_stages).all(), "Invalid lifecycle stage values"
        
        # Check numeric ranges
        numeric_columns = [
            'lifecycle_duration_days', 'total_orders', 'total_revenue', 'total_freight',
            'months_with_sales', 'avg_monthly_orders', 'avg_order_value', 'freight_ratio',
            'sales_velocity', 'revenue_velocity'
        ]
        assert (df[numeric_columns] >= 0).all().all(), "Numeric columns should be non-negative"
        
        # Check date relationships
        assert (df['last_sale_date'] >= df['first_sale_date']).all(), "Last sale date should be after first sale date"
        
        logging.info("Product lifecycle validation passed")
        return True
        
    except AssertionError as e:
        logging.error(f"Product lifecycle validation failed: {str(e)}")
        return False

def main():
    """Main function to generate product lifecycle table"""
    try:
        print("\n=== Starting Product Lifecycle Generation ===")
        
        # 1. Load data
        print("\n1. Loading required data...")
        data = load_required_data()
        print("Data loaded successfully")
        
        # 2. Generate product lifecycle
        print("\n2. Generating product lifecycle table...")
        lifecycle_dim = calculate_product_lifecycle(data)
        print("Product lifecycle table generated successfully")
        
        # 3. Validate results
        print("\n3. Validating results...")
        if not validate_product_lifecycle(lifecycle_dim):
            raise ValueError("Product lifecycle validation failed")
        print("Validation completed successfully")
        
        # 4. Export results
        print("\n4. Exporting results...")
        output_file = CLEANED_DATA_DIR / 'product_lifecycle.csv'
        lifecycle_dim.to_csv(output_file, index=False)
        print(f"Results exported to {output_file}")
        
        # 5. Generate sample
        print("\n5. Generating sample data...")
        sample = lifecycle_dim.sample(frac=0.05, random_state=42)
        sample_file = SAMPLE_DATA_DIR / 'product_lifecycle_sample.csv'
        sample.to_csv(sample_file, index=False)
        print(f"Sample data exported to {sample_file}")
        
        print("\n=== Product Lifecycle Generation Completed Successfully ===")
        
    except Exception as e:
        print(f"\nERROR: Product lifecycle generation failed: {str(e)}")
        logging.error(f"Product lifecycle generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 