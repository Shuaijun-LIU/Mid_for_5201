"""
Generate product category hierarchy for Olist E-commerce dataset
"""
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional, Union
import uuid

# import configuration
from config import (
    BASE_DIR, CLEANED_DATA_DIR, SAMPLE_DATA_DIR, RAW_DATA_DIR,
    DTYPE_DICT, DATE_COLUMNS, FORCE_REPROCESS, SAMPLE_RATIO
)

# create logs directory if not exists
os.makedirs("week1_data_preparation/logs", exist_ok=True)

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("week1_data_preparation/logs/derived_product_category_hierarchy.log"),
        logging.StreamHandler(),
    ],
)

def validate_product_category_hierarchy(df: pd.DataFrame) -> bool:
    """
    Validate product category hierarchy data
    Args:
        df: Product category hierarchy dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required fields
        assert not df['category_id'].isna().any(), "Missing category_id"
        assert not df['category_level'].isna().any(), "Missing category_level"
        assert not df['category_path'].isna().any(), "Missing category_path"
        
        # Check category level range
        assert df['category_level'].min() >= 1, "Invalid category level found"
        
        # Check parent-child relationships
        if df['parent_category_id'].notna().any():
            assert all(df['parent_category_id'].isin(df['category_id'])), "Invalid parent category reference"
        
        # Check category path format
        assert df['category_path'].str.match(r'^[A-Z_]+(\/[A-Z_]+)*$').all(), "Invalid category path format"
        
        # Check for duplicates
        assert not df['category_id'].duplicated().any(), "Duplicate category IDs found"
        
        logging.info("Product category hierarchy validation passed")
        return True
        
    except AssertionError as e:
        logging.error(f"Product category hierarchy validation failed: {str(e)}")
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
        df = pd.read_csv(file_path)
            
        # Check if dataframe has data
        if len(df) == 0:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Error checking file {file_path}: {str(e)}")
        return False

def generate_product_category_hierarchy(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate product category hierarchy
    Args:
        products_df: Cleaned products dataframe
    Returns:
        Product category hierarchy dataframe
    """
    # Create a copy to avoid modifying the original dataframe
    df = products_df.copy()
    
    # Clean up category names
    df['name'] = df['product_category_name_english'].str.replace('_', ' ').str.title()
    
    # Fix spelling errors and inconsistencies
    name_corrections = {
        'Costruction': 'Construction',
        'Fashio': 'Fashion',
        'Confort': 'Comfort',
        'Cine': 'Cinema',
        'Craftmanship': 'Craftsmanship',
        'Cds': 'CDs',
        'Dvds': 'DVDs',
        'Blu Ray': 'Blu-ray'
    }
    df['name'] = df['name'].replace(name_corrections)
    
    # Handle duplicate categories
    duplicates = df[df.duplicated(['name'], keep=False)]
    if not duplicates.empty:
        logging.warning(f"Found duplicate categories: {duplicates['name'].tolist()}")
        # Merge duplicates instead of adding suffix
        df = df.drop_duplicates(['name'])
    
    # Create hierarchy dataframe
    hierarchy_df = df[['name']].drop_duplicates()
    hierarchy_df['id'] = range(1, len(hierarchy_df) + 1)
    hierarchy_df['parent_id'] = None
    hierarchy_df['level'] = 1
    
    # Sort categories alphabetically
    hierarchy_df = hierarchy_df.sort_values('name')
    
    # Log hierarchy information
    logging.info(f"Generated hierarchy with {len(hierarchy_df)} categories")
    logging.info(f"Categories: {hierarchy_df['name'].tolist()}")
    
    return hierarchy_df

def validate_hierarchy(df: pd.DataFrame) -> bool:
    """
    Validate product category hierarchy
    Args:
        df: Product category hierarchy dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required fields
        assert not df['id'].isna().any(), "Missing category ID"
        assert not df['name'].isna().any(), "Missing category name"
        assert not df['level'].isna().any(), "Missing level"
        
        # Check ID uniqueness
        assert not df['id'].duplicated().any(), "Duplicate category ID found"
        
        # Check parent references
        # Level 1 categories should not have parents
        level1_mask = df['level'] == 1
        assert df.loc[level1_mask, 'parent_id'].isna().all(), "Level 1 categories should not have parents"
        
        # Other levels should have valid parent references
        other_levels_mask = df['level'] > 1
        parent_ids = df.loc[other_levels_mask, 'parent_id']
        assert parent_ids.notna().all(), "Missing parent references"
        assert parent_ids.isin(df['id']).all(), "Invalid parent references"
        
        # Check level values
        assert df['level'].between(1, 3).all(), "Invalid level values"
        
        logging.info("Product category hierarchy validation passed")
        return True
        
    except AssertionError as e:
        logging.error(f"Product category hierarchy validation failed: {str(e)}")
        return False

def main():
    """Main function to generate product category hierarchy"""
    try:
        print("\n=== Starting Product Category Hierarchy Generation ===")
        
        # 1. Load raw data
        print("\n1. Loading raw data...")
        
        # Load product categories data
        categories_file = RAW_DATA_DIR / 'product_category_name_translation.csv'
        categories_df = pd.read_csv(categories_file)
        print(f"Loaded {len(categories_df)} product category records")
        
        # 2. Generate hierarchy
        print("\n2. Generating hierarchy...")
        hierarchy_df = generate_product_category_hierarchy(categories_df)
        print(f"Generated {len(hierarchy_df)} category hierarchy records")
        
        # 3. Validate data
        print("\n3. Validating data...")
        if not validate_hierarchy(hierarchy_df):
            raise ValueError("Product category hierarchy validation failed")
        
        # 4. Export hierarchy
        print("\n4. Exporting hierarchy...")
        output_file = CLEANED_DATA_DIR / 'product_category_hierarchy.csv'
        hierarchy_df.to_csv(output_file, index=False)
        print(f"Product category hierarchy exported to {output_file}")
        
        # 5. Generate sample
        print("\n5. Generating sample data...")
        sample = hierarchy_df.sample(frac=SAMPLE_RATIO, random_state=42)
        sample_file = SAMPLE_DATA_DIR / 'product_category_hierarchy_sample.csv'
        sample.to_csv(sample_file, index=False)
        print(f"Sample data exported to {sample_file}")
        
        print("\n=== Product Category Hierarchy Generation Completed Successfully ===")
        
    except Exception as e:
        print(f"\nERROR: Product category hierarchy generation failed: {str(e)}")
        logging.error(f"Product category hierarchy generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 