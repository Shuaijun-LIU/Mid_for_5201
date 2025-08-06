"""
Generate location dimension table for Olist E-commerce Dataset
Purpose: Create a comprehensive location dimension table for geographic analysis
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from pathlib import Path
import re
import unicodedata

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
        logging.FileHandler("week1_data_preparation/logs/dim_location.log"),
        logging.StreamHandler(),
    ],
)

# Define Brazilian regions and their states
BRAZIL_REGIONS = {
    'Norte': ['AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO'],
    'Nordeste': ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],
    'Centro-Oeste': ['DF', 'GO', 'MT', 'MS'],
    'Sudeste': ['ES', 'MG', 'RJ', 'SP'],
    'Sul': ['PR', 'RS', 'SC']
}

def remove_accents(text: str) -> str:
    """
    Remove accents from text
    Args:
        text: Input text
    Returns:
        Text without accents
    """
    if not isinstance(text, str):
        return text
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                  if unicodedata.category(c) != 'Mn')

def standardize_location_name(name: str) -> str:
    """
    Standardize location name
    Args:
        name: Location name
    Returns:
        Standardized location name
    """
    if not isinstance(name, str):
        return name
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove accents
    name = remove_accents(name)
    
    # Remove special characters
    name = re.sub(r'[^\w\s]', '', name)
    
    # Remove extra spaces
    name = ' '.join(name.split())
    
    return name

def load_required_data() -> dict:
    """
    Load all required datasets for location dimension
    Returns:
        Dictionary containing all required dataframes
    """
    try:
        # Load geolocations data
        geolocations_df = pd.read_csv(CLEANED_DATA_DIR / 'geolocations.csv')
        
        # Load customers data
        customers_df = pd.read_csv(CLEANED_DATA_DIR / 'customers.csv')
        
        # Load sellers data
        sellers_df = pd.read_csv(CLEANED_DATA_DIR / 'sellers.csv')
        
        # Load orders data
        orders_df = pd.read_csv(CLEANED_DATA_DIR / 'orders.csv')
        
        return {
            'geolocations': geolocations_df,
            'customers': customers_df,
            'sellers': sellers_df,
            'orders': orders_df
        }
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def generate_location_dimension(data: dict) -> pd.DataFrame:
    """
    Generate location dimension table
    Args:
        data: Dictionary containing all required dataframes
    Returns:
        DataFrame with location dimension
    """
    try:
        # 1. Get unique locations from geolocations
        locations_df = data['geolocations'].copy()
        
        # 2. Standardize location names
        locations_df['geolocation_city'] = locations_df['geolocation_city'].apply(standardize_location_name)
        locations_df['geolocation_state'] = locations_df['geolocation_state'].str.upper()
        
        # 3. Add region information
        def get_region(state: str) -> str:
            for region, states in BRAZIL_REGIONS.items():
                if state in states:
                    return region
            return 'Unknown'
        
        locations_df['region'] = locations_df['geolocation_state'].apply(get_region)
        
        # 4. Calculate location statistics
        # Customer counts by location
        customer_counts = data['customers'].groupby('customer_zip_code_prefix').size().reset_index(name='customer_count')
        
        # Seller counts by location
        seller_counts = data['sellers'].groupby('seller_zip_code_prefix').size().reset_index(name='seller_count')
        
        # Order counts by location (using customer zip codes)
        order_counts = pd.merge(
            data['orders'],
            data['customers'][['customer_id', 'customer_zip_code_prefix']],
            on='customer_id',
            how='inner'
        ).groupby('customer_zip_code_prefix').size().reset_index(name='order_count')
        
        # 5. Merge statistics with locations
        locations_df = pd.merge(
            locations_df,
            customer_counts,
            left_on='geolocation_zip_code_prefix',
            right_on='customer_zip_code_prefix',
            how='left'
        )
        
        locations_df = pd.merge(
            locations_df,
            seller_counts,
            left_on='geolocation_zip_code_prefix',
            right_on='seller_zip_code_prefix',
            how='left'
        )
        
        locations_df = pd.merge(
            locations_df,
            order_counts,
            left_on='geolocation_zip_code_prefix',
            right_on='customer_zip_code_prefix',
            how='left'
        )
        
        # 6. Fill missing values with 0
        locations_df['customer_count'] = locations_df['customer_count'].fillna(0)
        locations_df['seller_count'] = locations_df['seller_count'].fillna(0)
        locations_df['order_count'] = locations_df['order_count'].fillna(0)
        
        # 7. Create location hierarchy
        locations_df['location_id'] = locations_df['geolocation_zip_code_prefix'].astype(str).str.zfill(5)
        locations_df['location_level'] = 'zip_code'
        locations_df['parent_location_id'] = locations_df['geolocation_state']
        
        # 8. Select and rename columns
        location_dim = locations_df[[
            'location_id',
            'parent_location_id',
            'location_level',
            'region',
            'geolocation_state',
            'geolocation_city',
            'geolocation_zip_code_prefix',
            'geolocation_lat',
            'geolocation_lng',
            'customer_count',
            'seller_count',
            'order_count'
        ]].copy()
        
        location_dim.columns = [
            'location_id',
            'parent_location_id',
            'location_level',
            'region',
            'state',
            'city',
            'zip_code_prefix',
            'latitude',
            'longitude',
            'customer_count',
            'seller_count',
            'order_count'
        ]
        
        return location_dim
        
    except Exception as e:
        logging.error(f"Error generating location dimension: {str(e)}")
        raise

def validate_location_dimension(df: pd.DataFrame) -> bool:
    """
    Validate location dimension table
    Args:
        df: Location dimension dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check for required columns
        required_columns = [
            'location_id', 'parent_location_id', 'location_level',
            'region', 'state', 'city', 'zip_code_prefix',
            'latitude', 'longitude',
            'customer_count', 'seller_count', 'order_count'
        ]
        assert all(col in df.columns for col in required_columns), "Missing required columns"
        
        # Check for missing values in key columns
        key_columns = ['location_id', 'state', 'city', 'zip_code_prefix']
        assert not df[key_columns].isna().any().any(), "Found missing values in key columns"
        
        # Check location_id format
        assert df['location_id'].str.len().eq(5).all(), "location_id should be 5 digits"
        
        # Check state format
        assert df['state'].str.len().eq(2).all(), "state should be 2 characters"
        
        # Check region values
        valid_regions = list(BRAZIL_REGIONS.keys()) + ['Unknown']
        assert df['region'].isin(valid_regions).all(), "Invalid region values"
        
        # Check numeric ranges
        assert df['latitude'].between(-90, 90).all(), "Latitude out of range"
        assert df['longitude'].between(-180, 180).all(), "Longitude out of range"
        
        # Check count columns
        count_columns = ['customer_count', 'seller_count', 'order_count']
        assert (df[count_columns] >= 0).all().all(), "Count columns should be non-negative"
        
        logging.info("Location dimension validation passed")
        return True
        
    except AssertionError as e:
        logging.error(f"Location dimension validation failed: {str(e)}")
        return False

def main():
    """Main function to generate location dimension table"""
    try:
        print("\n=== Starting Location Dimension Generation ===")
        
        # 1. Load data
        print("\n1. Loading required data...")
        data = load_required_data()
        print("Data loaded successfully")
        
        # 2. Generate location dimension
        print("\n2. Generating location dimension table...")
        location_dim = generate_location_dimension(data)
        print("Location dimension table generated successfully")
        
        # 3. Validate results
        print("\n3. Validating results...")
        if not validate_location_dimension(location_dim):
            raise ValueError("Location dimension validation failed")
        print("Validation completed successfully")
        
        # 4. Export results
        print("\n4. Exporting results...")
        output_file = CLEANED_DATA_DIR / 'dim_location.csv'
        location_dim.to_csv(output_file, index=False)
        print(f"Results exported to {output_file}")
        
        # 5. Generate sample
        print("\n5. Generating sample data...")
        sample = location_dim.sample(frac=0.05, random_state=42)
        sample_file = SAMPLE_DATA_DIR / 'dim_location_sample.csv'
        sample.to_csv(sample_file, index=False)
        print(f"Sample data exported to {sample_file}")
        
        print("\n=== Location Dimension Generation Completed Successfully ===")
        
    except Exception as e:
        print(f"\nERROR: Location dimension generation failed: {str(e)}")
        logging.error(f"Location dimension generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 