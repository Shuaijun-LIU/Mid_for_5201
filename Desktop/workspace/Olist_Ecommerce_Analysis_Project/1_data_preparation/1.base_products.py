"""
Clean products data for Olist E-commerce dataset
Purpose: Clean and standardize product data
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from pathlib import Path

# import configuration
from config import (
    RAW_DATA_DIR,
    CLEANED_DATA_DIR,
    SAMPLE_DATA_DIR,
    DTYPE_DICT,
    DATE_COLUMNS,
    FORCE_REPROCESS,
    SAMPLE_RATIO,
)

# create logs directory if not exists
os.makedirs("week1_data_preparation/logs", exist_ok=True)

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("week1_data_preparation/logs/base_products.log"),
        logging.StreamHandler(),
    ],
)


def validate_products(df: pd.DataFrame) -> bool:
    """
    Validate products data
    Args:
        df: Products dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required fields
        assert not df["product_id"].isna().any(), "Missing product_id"
        assert not df["product_category_name"].isna().any(), "Missing category name"

        # Check numeric fields
        assert df["product_weight_g"].dtype in [
            "float64",
            "Int64",
        ], "Invalid weight type"
        assert df["product_length_cm"].dtype in [
            "float64",
            "Int64",
        ], "Invalid length type"
        assert df["product_height_cm"].dtype in [
            "float64",
            "Int64",
        ], "Invalid height type"
        assert df["product_width_cm"].dtype in [
            "float64",
            "Int64",
        ], "Invalid width type"

        # Check value ranges
        assert df["product_weight_g"].min() >= 0, "Negative weight found"
        assert df["product_length_cm"].min() >= 0, "Negative length found"
        assert df["product_height_cm"].min() >= 0, "Negative height found"
        assert df["product_width_cm"].min() >= 0, "Negative width found"

        # Check logical relationships
        assert (
            df["product_length_cm"] >= df["product_width_cm"]
        ).all(), "Length should be greater than or equal to width"

        logging.info("Products data validation passed")
        return True

    except AssertionError as e:
        logging.error(f"Products data validation failed: {str(e)}")
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
                "product_weight_g": "float64",
                "product_length_cm": "float64",
                "product_height_cm": "float64",
                "product_width_cm": "float64",
            },
        )

        # Check if dataframe has data
        if len(df) == 0:
            return False

        return True
    except Exception as e:
        logging.warning(f"Error checking file {file_path}: {str(e)}")
        return False


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean products data
    Args:
        df: Raw products dataframe
    Returns:
        Cleaned products dataframe
    """
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()

    # 1. Fix column name typos
    df_clean = df_clean.rename(
        columns={
            "product_name_lenght": "product_name_length",
            "product_description_lenght": "product_description_length",
        }
    )

    # 2. Convert numeric columns to appropriate types
    numeric_cols = [
        "product_name_length",
        "product_description_length",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").astype("Int64")

    # 3. Handle missing values
    # Convert category column to string type first
    df_clean["product_category_name"] = df_clean["product_category_name"].astype(
        "string"
    )
    # Fill missing category names with 'UNKNOWN'
    df_clean["product_category_name"] = df_clean["product_category_name"].fillna(
        "UNKNOWN"
    )
    # Convert back to category type
    df_clean["product_category_name"] = df_clean["product_category_name"].astype(
        "category"
    )

    # Fill missing numeric values with 0
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

    # 4. Ensure logical relationships
    # If length is less than width, swap them
    mask = df_clean["product_length_cm"] < df_clean["product_width_cm"]
    df_clean.loc[mask, ["product_length_cm", "product_width_cm"]] = df_clean.loc[
        mask, ["product_width_cm", "product_length_cm"]
    ].values

    return df_clean


def main():
    """Main function to clean products data"""
    try:
        print("\n=== Starting Products Data Cleaning ===")

        # 1. Load raw data
        print("\n1. Loading raw data...")
        input_file = RAW_DATA_DIR / "olist_products_dataset.csv"
        products_df = pd.read_csv(
            input_file, dtype=DTYPE_DICT["olist_products_dataset"]
        )
        print(f"Loaded {len(products_df)} product records")

        # 2. Clean data
        output_file = CLEANED_DATA_DIR / "products.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(output_file):
            print("\n2. Cleaning data...")
            cleaned_products = clean_products(products_df)
            print(f"Cleaned {len(cleaned_products)} product records")
        else:
            print("\nProducts data already cleaned, loading from file...")
            cleaned_products = pd.read_csv(
                output_file, dtype=DTYPE_DICT["olist_products_dataset"]
            )

        # 3. Validate data
        print("\n3. Validating data...")
        if not validate_products(cleaned_products):
            raise ValueError("Products data validation failed")
        print("Data validation passed")

        # 4. Export cleaned data
        print("\n4. Exporting cleaned data...")
        cleaned_products.to_csv(output_file, index=False)
        print(f"Cleaned data exported to {output_file}")

        # 5. Generate sample
        print("\n5. Generating sample data...")
        sample_file = SAMPLE_DATA_DIR / "products_sample.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(sample_file):
            sample = cleaned_products.sample(frac=SAMPLE_RATIO, random_state=42)
            sample.to_csv(sample_file, index=False)
            print(f"Sample data exported to {sample_file}")
        else:
            print("Sample data already exists, skipping...")

        print("\n=== Products Data Cleaning Completed Successfully ===")

    except Exception as e:
        print(f"\nERROR: Products data cleaning failed: {str(e)}")
        logging.error(f"Products data cleaning failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
