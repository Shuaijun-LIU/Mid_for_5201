"""
Clean order items data for Olist E-commerce dataset
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
        logging.FileHandler("week1_data_preparation/logs/base_order_items.log"),
        logging.StreamHandler(),
    ],
)


def validate_order_items(df: pd.DataFrame) -> bool:
    """
    Validate order items data
    Args:
        df: Order items dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required fields
        assert not df["order_id"].isna().any(), "Missing order_id"
        assert not df["order_item_id"].isna().any(), "Missing order_item_id"
        assert not df["product_id"].isna().any(), "Missing product_id"
        assert not df["seller_id"].isna().any(), "Missing seller_id"
        assert not df["shipping_limit_date"].isna().any(), "Missing shipping_limit_date"
        assert not df["price"].isna().any(), "Missing price"
        assert not df["freight_value"].isna().any(), "Missing freight_value"

        # Check timestamp formats
        assert (
            df["shipping_limit_date"].dtype == "datetime64[ns]"
        ), "Invalid shipping limit date format"

        # Check numeric fields
        assert df["price"].dtype == "float64", "Invalid price type"
        assert df["freight_value"].dtype == "float64", "Invalid freight value type"

        # Check value ranges
        assert df["price"].min() >= 0, "Negative price found"
        assert df["freight_value"].min() >= 0, "Negative freight value found"

        # Check logical relationships
        assert (df["price"] > 0).all(), "Zero or negative price found"
        assert (df["freight_value"] >= 0).all(), "Negative freight value found"

        # Check for duplicates
        assert not df.duplicated(
            subset=["order_id", "order_item_id"]
        ).any(), "Duplicate order items found"

        logging.info("Order items data validation passed")
        return True

    except AssertionError as e:
        logging.error(f"Order items data validation failed: {str(e)}")
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
            parse_dates=["shipping_limit_date"],
            dtype={"price": "float64", "freight_value": "float64"},
        )

        # Check if dataframe has data
        if len(df) == 0:
            return False

        return True
    except Exception as e:
        logging.warning(f"Error checking file {file_path}: {str(e)}")
        return False


def clean_order_items(
    df: pd.DataFrame, valid_order_ids: set = None, valid_product_ids: set = None
) -> pd.DataFrame:
    """
    Clean order items data
    Args:
        df: Raw order items dataframe
        valid_order_ids: Set of valid order IDs from orders.csv
        valid_product_ids: Set of valid product IDs from products.csv
    Returns:
        Cleaned order items dataframe
    """
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()

    # 1. Clean order_id
    # Ensure it's a string and remove any whitespace
    df_clean["order_id"] = df_clean["order_id"].astype(str).str.strip()

    # Filter out items for non-existent orders if valid_order_ids provided
    if valid_order_ids:
        df_clean = df_clean[df_clean["order_id"].isin(valid_order_ids)]

    # 2. Clean order_item_id
    # Ensure it's an integer
    df_clean["order_item_id"] = pd.to_numeric(
        df_clean["order_item_id"], errors="coerce"
    )

    # 3. Clean product_id
    # Ensure it's a string and remove any whitespace
    df_clean["product_id"] = df_clean["product_id"].astype(str).str.strip()

    # Filter out non-existent products if valid_product_ids provided
    if valid_product_ids:
        df_clean = df_clean[df_clean["product_id"].isin(valid_product_ids)]

    # 4. Clean seller_id
    # Ensure it's a string and remove any whitespace
    df_clean["seller_id"] = df_clean["seller_id"].astype(str).str.strip()

    # 5. Clean shipping_limit_date
    # Convert to datetime and handle invalid values
    df_clean["shipping_limit_date"] = pd.to_datetime(
        df_clean["shipping_limit_date"], errors="coerce"
    )

    # 6. Clean price and freight_value
    # Convert to float and handle invalid values
    df_clean["price"] = pd.to_numeric(df_clean["price"], errors="coerce")
    df_clean["freight_value"] = pd.to_numeric(
        df_clean["freight_value"], errors="coerce"
    )

    # Calculate total value
    df_clean["total_value"] = df_clean["price"] + df_clean["freight_value"]

    # Remove rows with invalid values
    df_clean = df_clean.dropna(
        subset=[
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "shipping_limit_date",
            "price",
            "freight_value",
        ]
    )

    # Ensure order_item_id is unique within each order
    df_clean["order_item_id"] = df_clean.groupby("order_id").cumcount() + 1

    return df_clean


def main():
    """Main function to clean order items data"""
    try:
        print("\n=== Starting Order Items Data Cleaning ===")

        # 1. Load raw data
        print("\n1. Loading raw data...")
        input_file = RAW_DATA_DIR / "olist_order_items_dataset.csv"
        order_items_df = pd.read_csv(
            input_file,
            dtype=DTYPE_DICT["olist_order_items_dataset"],
            parse_dates=DATE_COLUMNS.get("olist_order_items_dataset", []),
        )
        print(f"Loaded {len(order_items_df)} order item records")

        # 2. Load valid order IDs and product IDs if available
        valid_order_ids = None
        valid_product_ids = None

        orders_file = CLEANED_DATA_DIR / "orders.csv"
        if orders_file.exists():
            print("\n2. Loading valid order IDs...")
            orders_df = pd.read_csv(orders_file)
            valid_order_ids = set(orders_df["order_id"].unique())
            print(f"Found {len(valid_order_ids)} valid order IDs")

        products_file = CLEANED_DATA_DIR / "products.csv"
        if products_file.exists():
            print("\n3. Loading valid product IDs...")
            products_df = pd.read_csv(products_file)
            valid_product_ids = set(products_df["product_id"].unique())
            print(f"Found {len(valid_product_ids)} valid product IDs")

        # 3. Clean data
        output_file = CLEANED_DATA_DIR / "order_items.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(output_file):
            print("\n4. Cleaning data...")
            cleaned_order_items = clean_order_items(
                order_items_df, valid_order_ids, valid_product_ids
            )
        else:
            print("\nOrder items data already cleaned, loading from file...")
            cleaned_order_items = pd.read_csv(output_file)

        # 5. Validate data
        print("\n5. Validating data...")
        if not validate_order_items(cleaned_order_items):
            raise ValueError("Order items data validation failed")
        print("Data validation passed")

        # 6. Export cleaned data
        print("\n6. Exporting cleaned data...")
        cleaned_order_items.to_csv(output_file, index=False)
        print(f"Cleaned data exported to {output_file}")

        # 7. Generate sample
        print("\n7. Generating sample data...")
        sample_file = SAMPLE_DATA_DIR / "order_items_sample.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(sample_file):
            sample = cleaned_order_items.sample(frac=SAMPLE_RATIO, random_state=42)
            sample.to_csv(sample_file, index=False)
            print(f"Sample data exported to {sample_file}")
        else:
            print("Sample data already exists, skipping...")

        print("\n=== Order Items Data Cleaning Completed Successfully ===")

    except Exception as e:
        print(f"\nERROR: Order items data cleaning failed: {str(e)}")
        logging.error(f"Order items data cleaning failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
