"""
Clean sellers data for Olist E-commerce dataset
Purpose: Clean and standardize seller data
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
        logging.FileHandler("week1_data_preparation/logs/base_sellers.log"),
        logging.StreamHandler(),
    ],
)

# Configure paths
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CLEANED_DATA_DIR = BASE_DIR / "data" / "cleaned"
SAMPLE_DATA_DIR = BASE_DIR / "data" / "samples"


def validate_sellers(df: pd.DataFrame) -> bool:
    """
    Validate sellers data
    Args:
        df: Sellers dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required fields
        assert not df["seller_id"].isna().any(), "Missing seller_id"
        assert not df["seller_zip_code_prefix"].isna().any(), "Missing zip code prefix"
        assert not df["seller_city"].isna().any(), "Missing city"
        assert not df["seller_state"].isna().any(), "Missing state"

        # Check string fields
        assert df["seller_id"].dtype == "string", "Invalid seller_id type"
        assert (
            df["seller_zip_code_prefix"].dtype == "string"
        ), "Invalid zip code prefix type"
        assert df["seller_city"].dtype == "category", "Invalid city type"
        assert df["seller_state"].dtype == "category", "Invalid state type"

        # Check value formats
        assert (
            df["seller_zip_code_prefix"].str.match(r"^\d{5}$").all()
        ), "Invalid zip code prefix format"
        assert df["seller_state"].str.match(r"^[A-Z]{2}$").all(), "Invalid state format"

        # Check for duplicates
        assert not df["seller_id"].duplicated().any(), "Duplicate seller IDs found"

        logging.info("Sellers data validation passed")
        return True

    except AssertionError as e:
        logging.error(f"Sellers data validation failed: {str(e)}")
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
        df = pd.read_csv(file_path, dtype=DTYPE_DICT["olist_sellers_dataset"])

        # Check if dataframe has data
        if len(df) == 0:
            return False

        return True
    except Exception as e:
        logging.warning(f"Error checking file {file_path}: {str(e)}")
        return False


def clean_sellers(
    df: pd.DataFrame, valid_seller_ids: set, valid_zip_codes: set
) -> pd.DataFrame:
    """
    Clean sellers data
    Args:
        df: Raw sellers dataframe
        valid_seller_ids: Set of valid seller IDs from order_items.csv
        valid_zip_codes: Set of valid zip codes from geolocations.csv
    Returns:
        Cleaned sellers dataframe
    """
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()

    # 1. Clean seller_id
    # Ensure it's a string and remove any whitespace
    df_clean["seller_id"] = df_clean["seller_id"].astype("string").str.strip()

    # Filter out sellers not in order_items
    df_clean = df_clean[df_clean["seller_id"].isin(valid_seller_ids)]

    # 2. Clean zip code prefix
    # Ensure it's a 5-digit string
    df_clean["seller_zip_code_prefix"] = (
        df_clean["seller_zip_code_prefix"].astype("string").str.strip()
    )
    # Pad with zeros if less than 5 digits
    df_clean["seller_zip_code_prefix"] = df_clean["seller_zip_code_prefix"].str.zfill(5)

    # Filter out sellers with invalid zip codes
    df_clean = df_clean[df_clean["seller_zip_code_prefix"].isin(valid_zip_codes)]

    # 3. Clean city names
    # Convert to lowercase and remove accents
    df_clean["seller_city"] = df_clean["seller_city"].str.lower()
    # Remove any special characters
    df_clean["seller_city"] = df_clean["seller_city"].str.replace(
        r"[^\w\s]", "", regex=True
    )

    # 4. Clean state codes
    # Convert to uppercase and ensure 2 characters
    df_clean["seller_state"] = df_clean["seller_state"].str.upper()
    df_clean["seller_state"] = df_clean["seller_state"].str[:2]

    # 5. Convert to categorical
    df_clean["seller_city"] = df_clean["seller_city"].astype("category")
    df_clean["seller_state"] = df_clean["seller_state"].astype("category")

    # 6. Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=["seller_id"])

    return df_clean


def main():
    """Main function to clean sellers data"""
    try:
        print("\n=== Starting Sellers Data Cleaning ===")

        # 1. Load cleaned order items data to get valid seller IDs
        print("\n1. Loading cleaned order items data...")
        order_items_file = CLEANED_DATA_DIR / "order_items.csv"
        order_items_df = pd.read_csv(
            order_items_file, dtype=DTYPE_DICT["olist_order_items_dataset"]
        )
        valid_seller_ids = set(order_items_df["seller_id"].unique())
        print(f"Found {len(valid_seller_ids)} valid seller IDs")

        # 2. Load cleaned geolocations data to get valid zip codes
        print("\n2. Loading cleaned geolocations data...")
        geolocations_file = CLEANED_DATA_DIR / "geolocations.csv"
        geolocations_df = pd.read_csv(
            geolocations_file, dtype=DTYPE_DICT["olist_geolocation_dataset"]
        )
        valid_zip_codes = set(geolocations_df["geolocation_zip_code_prefix"].unique())
        print(f"Found {len(valid_zip_codes)} valid zip codes")

        # 3. Load raw seller data
        print("\n3. Loading raw seller data...")
        input_file = RAW_DATA_DIR / "olist_sellers_dataset.csv"
        sellers_df = pd.read_csv(input_file, dtype=DTYPE_DICT["olist_sellers_dataset"])
        print(f"Loaded {len(sellers_df)} seller records")

        # 4. Clean data
        output_file = CLEANED_DATA_DIR / "sellers.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(output_file):
            print("\n4. Cleaning data...")
            cleaned_sellers = clean_sellers(
                sellers_df, valid_seller_ids, valid_zip_codes
            )
            print(f"Cleaned {len(cleaned_sellers)} seller records")

            # 5. Validate data
            print("\n5. Validating data...")
            if not validate_sellers(cleaned_sellers):
                raise ValueError("Sellers data validation failed")
            print("Data validation passed")

            # 6. Export cleaned data
            print("\n6. Exporting cleaned data...")
            cleaned_sellers.to_csv(output_file, index=False)
            print(f"Cleaned data exported to {output_file}")
        else:
            print("\nSellers data already cleaned, loading from file...")
            cleaned_sellers = pd.read_csv(
                output_file, dtype=DTYPE_DICT["olist_sellers_dataset"]
            )

        # 7. Generate sample
        print("\n7. Generating sample data...")
        sample_file = SAMPLE_DATA_DIR / "sellers_sample.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(sample_file):
            sample = cleaned_sellers.sample(frac=SAMPLE_RATIO, random_state=42)
            sample.to_csv(sample_file, index=False)
            print(f"Sample data exported to {sample_file}")
        else:
            print("Sample data already exists, skipping...")

        print("\n=== Sellers Data Cleaning Completed Successfully ===")

    except Exception as e:
        print(f"\nERROR: Sellers data cleaning failed: {str(e)}")
        logging.error(f"Sellers data cleaning failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
