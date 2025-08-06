"""
Clean customers data for Olist E-commerce Dataset
Purpose: Clean and standardize customer data
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
        logging.FileHandler("week1_data_preparation/logs/base_customers.log"),
        logging.StreamHandler(),
    ],
)


def validate_customers(df: pd.DataFrame) -> bool:
    """
    Validate customers data
    Args:
        df: Customers dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required fields
        assert not df["customer_id"].isna().any(), "Missing customer_id"
        assert not df["customer_unique_id"].isna().any(), "Missing customer_unique_id"
        assert (
            not df["customer_zip_code_prefix"].isna().any()
        ), "Missing zip code prefix"
        assert not df["customer_city"].isna().any(), "Missing city"
        assert not df["customer_state"].isna().any(), "Missing state"

        # Check string fields
        assert df["customer_id"].dtype == "string", "Invalid customer_id type"
        assert (
            df["customer_unique_id"].dtype == "string"
        ), "Invalid customer_unique_id type"
        assert (
            df["customer_zip_code_prefix"].dtype == "string"
        ), "Invalid zip code prefix type"
        assert df["customer_city"].dtype == "category", "Invalid city type"
        assert df["customer_state"].dtype == "category", "Invalid state type"

        # Check value formats
        assert (
            df["customer_zip_code_prefix"].str.match(r"^\d{5}$").all()
        ), "Invalid zip code prefix format"
        assert (
            df["customer_state"].str.match(r"^[A-Z]{2}$").all()
        ), "Invalid state format"

        # Check for duplicates
        assert not df["customer_id"].duplicated().any(), "Duplicate customer IDs found"

        logging.info("Customers data validation passed")
        return True

    except AssertionError as e:
        logging.error(f"Customers data validation failed: {str(e)}")
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


def clean_customers(
    df: pd.DataFrame, valid_customer_ids: set, valid_zip_codes: set
) -> pd.DataFrame:
    """
    Clean customers data
    Args:
        df: Raw customers dataframe
        valid_customer_ids: Set of valid customer IDs from orders.csv
        valid_zip_codes: Set of valid zip codes from geolocations.csv
    Returns:
        Cleaned customers dataframe
    """
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()

    # 1. Clean customer_id and customer_unique_id
    # Ensure they are strings and remove any whitespace
    df_clean["customer_id"] = df_clean["customer_id"].astype("string").str.strip()
    df_clean["customer_unique_id"] = (
        df_clean["customer_unique_id"].astype("string").str.strip()
    )

    # Filter out customers not in orders
    df_clean = df_clean[df_clean["customer_id"].isin(valid_customer_ids)]

    # 2. Clean zip code prefix
    # Ensure it's a 5-digit string
    df_clean["customer_zip_code_prefix"] = (
        df_clean["customer_zip_code_prefix"].astype("string").str.strip()
    )
    # Pad with zeros if less than 5 digits
    df_clean["customer_zip_code_prefix"] = df_clean[
        "customer_zip_code_prefix"
    ].str.zfill(5)

    # Filter out customers with invalid zip codes
    df_clean = df_clean[df_clean["customer_zip_code_prefix"].isin(valid_zip_codes)]

    # 3. Clean city names
    # Convert to lowercase and remove accents
    df_clean["customer_city"] = df_clean["customer_city"].str.lower()
    # Remove any special characters
    df_clean["customer_city"] = df_clean["customer_city"].str.replace(
        r"[^\w\s]", "", regex=True
    )

    # 4. Clean state codes
    # Convert to uppercase and ensure 2 characters
    df_clean["customer_state"] = df_clean["customer_state"].str.upper()
    df_clean["customer_state"] = df_clean["customer_state"].str[:2]

    # 5. Convert to categorical
    df_clean["customer_city"] = df_clean["customer_city"].astype("category")
    df_clean["customer_state"] = df_clean["customer_state"].astype("category")

    # 6. Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=["customer_id"])

    return df_clean


def main():
    """Main function to clean customers data"""
    try:
        print("\n=== Starting Customers Data Cleaning ===")

        # 1. Load cleaned orders data to get valid order IDs
        print("\n1. Loading cleaned orders data...")
        orders_file = CLEANED_DATA_DIR / "orders.csv"
        orders_df = pd.read_csv(orders_file, dtype=DTYPE_DICT["olist_orders_dataset"])
        valid_customer_ids = set(orders_df["customer_id"].unique())
        print(f"Found {len(valid_customer_ids)} valid customer IDs")
        print(f"Sample customer IDs: {list(valid_customer_ids)[:5]}")

        # 2. Load cleaned geolocations data to get valid zip codes
        print("\n2. Loading cleaned geolocations data...")
        geolocations_file = CLEANED_DATA_DIR / "geolocations.csv"
        geolocations_df = pd.read_csv(
            geolocations_file, dtype=DTYPE_DICT["olist_geolocation_dataset"]
        )
        valid_zip_codes = set(geolocations_df["geolocation_zip_code_prefix"].unique())
        print(f"Found {len(valid_zip_codes)} valid zip codes")
        print(f"Sample zip codes: {list(valid_zip_codes)[:5]}")

        # 3. Load raw customer data
        print("\n3. Loading raw customer data...")
        input_file = RAW_DATA_DIR / "olist_customers_dataset.csv"
        customers_df = pd.read_csv(
            input_file, dtype=DTYPE_DICT["olist_customers_dataset"]
        )
        print(f"Loaded {len(customers_df)} customer records")
        print(f"Sample customer IDs: {customers_df['customer_id'].head().tolist()}")
        print(
            f"Sample zip codes: {customers_df['customer_zip_code_prefix'].head().tolist()}"
        )

        # 4. Clean data
        output_file = CLEANED_DATA_DIR / "customers.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(output_file):
            print("\n4. Cleaning data...")
            cleaned_customers = clean_customers(
                customers_df, valid_customer_ids, valid_zip_codes
            )
            print(f"Cleaned {len(cleaned_customers)} customer records")

            if len(cleaned_customers) == 0:
                print("WARNING: No records after cleaning!")
                print("Checking filter conditions:")
                print(
                    f"Records with valid customer IDs: {len(customers_df[customers_df['customer_id'].isin(valid_customer_ids)])}"
                )
                print(
                    f"Records with valid zip codes: {len(customers_df[customers_df['customer_zip_code_prefix'].isin(valid_zip_codes)])}"
                )

            # 5. Validate data
            print("\n5. Validating data...")
            if not validate_customers(cleaned_customers):
                raise ValueError("Customers data validation failed")
            print("Data validation passed")

            # 6. Export cleaned data
            print("\n6. Exporting cleaned data...")
            cleaned_customers.to_csv(output_file, index=False)
            print(f"Cleaned data exported to {output_file}")
        else:
            print("\nCustomers data already cleaned, loading from file...")
            cleaned_customers = pd.read_csv(
                output_file, dtype=DTYPE_DICT["olist_customers_dataset"]
            )

        # 7. Generate sample
        print("\n7. Generating sample data...")
        sample_file = SAMPLE_DATA_DIR / "customers_sample.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(sample_file):
            sample = cleaned_customers.sample(frac=SAMPLE_RATIO, random_state=42)
            sample.to_csv(sample_file, index=False)
            print(f"Sample data exported to {sample_file}")
        else:
            print("Sample data already exists, skipping...")

        print("\n=== Customers Data Cleaning Completed Successfully ===")

    except Exception as e:
        print(f"\nERROR: Customers data cleaning failed: {str(e)}")
        logging.error(f"Customers data cleaning failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
