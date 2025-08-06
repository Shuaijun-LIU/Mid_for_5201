"""
Clean order payments data for Olist E-commerce dataset
Purpose: Clean and standardize order payments data
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
        logging.FileHandler("week1_data_preparation/logs/base_order_payments.log"),
        logging.StreamHandler(),
    ],
)


def validate_order_payments(df: pd.DataFrame) -> bool:
    """
    Validate order payments data
    Args:
        df: Order payments dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required fields
        assert not df["order_id"].isna().any(), "Missing order_id"
        assert not df["payment_sequential"].isna().any(), "Missing payment sequential"
        assert not df["payment_type"].isna().any(), "Missing payment type"
        assert (
            not df["payment_installments"].isna().any()
        ), "Missing payment installments"
        assert not df["payment_value"].isna().any(), "Missing payment value"

        # Check numeric fields
        assert (
            df["payment_sequential"].dtype == "Int64"
        ), "Invalid payment sequential type"
        assert (
            df["payment_installments"].dtype == "Int64"
        ), "Invalid payment installments type"
        assert df["payment_value"].dtype == "float64", "Invalid payment value type"

        # Check value ranges
        assert (
            df["payment_sequential"].min() > 0
        ), "Non-positive payment sequential found"
        assert (
            df["payment_installments"].min() >= 0
        ), "Negative payment installments found"
        assert df["payment_value"].min() > 0, "Zero or negative payment value found"

        # Check logical relationships
        assert (
            df["payment_installments"] >= 1
        ).all(), "Payment installments less than 1 found"

        logging.info("Order payments data validation passed")
        return True

    except AssertionError as e:
        logging.error(f"Order payments data validation failed: {str(e)}")
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
                "payment_sequential": "Int64",
                "payment_installments": "Int64",
                "payment_value": "float64",
            },
        )

        # Check if dataframe has data
        if len(df) == 0:
            return False

        return True
    except Exception as e:
        logging.warning(f"Error checking file {file_path}: {str(e)}")
        return False


def clean_order_payments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean order payments data
    Args:
        df: Raw order payments dataframe
    Returns:
        Cleaned order payments dataframe
    """
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()

    # 1. Convert numeric columns to appropriate types
    df_clean["payment_sequential"] = pd.to_numeric(
        df_clean["payment_sequential"], errors="coerce"
    ).astype("Int64")
    df_clean["payment_installments"] = pd.to_numeric(
        df_clean["payment_installments"], errors="coerce"
    ).astype("Int64")
    df_clean["payment_value"] = pd.to_numeric(
        df_clean["payment_value"], errors="coerce"
    )

    # check payment_installments
    print("\nPayment installments distribution before cleaning:")
    print(df_clean["payment_installments"].value_counts().sort_index())
    print("\nPayment installments with values < 1:")
    print(
        df_clean[df_clean["payment_installments"] < 1][
            ["order_id", "payment_sequential", "payment_installments", "payment_value"]
        ]
    )

    # 2. Handle missing values and invalid values
    df_clean["payment_sequential"] = df_clean["payment_sequential"].fillna(1)
    df_clean["payment_installments"] = df_clean["payment_installments"].fillna(1)

    # 0 to 1
    df_clean.loc[df_clean["payment_installments"] < 1, "payment_installments"] = 1

    # 3. Handle invalid payment values
    # Replace zero or negative values with the mean of positive values
    positive_mean = df_clean[df_clean["payment_value"] > 0]["payment_value"].mean()
    df_clean.loc[df_clean["payment_value"] <= 0, "payment_value"] = positive_mean

    # 4. Convert payment type to uppercase and handle missing values
    df_clean["payment_type"] = df_clean["payment_type"].str.upper()
    df_clean["payment_type"] = df_clean["payment_type"].fillna("UNKNOWN")

    # 5. Remove any remaining rows with missing values
    df_clean = df_clean.dropna()

    # 检查清理后的 payment_installments 分布
    print("\nPayment installments distribution after cleaning:")
    print(df_clean["payment_installments"].value_counts().sort_index())

    return df_clean


def main():
    """Main function to clean order payments data"""
    try:
        print("\n=== Starting Order Payments Data Cleaning ===")

        # 1. Load raw data
        print("\n1. Loading raw data...")
        input_file = RAW_DATA_DIR / "olist_order_payments_dataset.csv"
        order_payments_df = pd.read_csv(
            input_file, dtype=DTYPE_DICT["olist_order_payments_dataset"]
        )
        print(f"Loaded {len(order_payments_df)} order payment records")

        # 2. Clean data
        output_file = CLEANED_DATA_DIR / "order_payments.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(output_file):
            print("\n2. Cleaning data...")
            cleaned_order_payments = clean_order_payments(order_payments_df)
            print(f"Cleaned {len(cleaned_order_payments)} order payment records")

            # 3. Validate data
            print("\n3. Validating data...")
            if not validate_order_payments(cleaned_order_payments):
                raise ValueError("Order payments data validation failed")
            print("Data validation passed")

            # 4. Export cleaned data
            print("\n4. Exporting cleaned data...")
            cleaned_order_payments.to_csv(output_file, index=False)
            print(f"Cleaned data exported to {output_file}")
        else:
            print("\nOrder payments data already cleaned, loading from file...")
            cleaned_order_payments = pd.read_csv(
                output_file,
                dtype={
                    "payment_sequential": "Int64",
                    "payment_installments": "Int64",
                    "payment_value": "float64",
                },
            )

        # 5. Generate sample
        print("\n5. Generating sample data...")
        sample_file = SAMPLE_DATA_DIR / "order_payments_sample.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(sample_file):
            sample = cleaned_order_payments.sample(frac=SAMPLE_RATIO, random_state=42)
            sample.to_csv(sample_file, index=False)
            print(f"Sample data exported to {sample_file}")
        else:
            print("Sample data already exists, skipping...")

        print("\n=== Order Payments Data Cleaning Completed Successfully ===")

    except Exception as e:
        print(f"\nERROR: Order payments data cleaning failed: {str(e)}")
        logging.error(f"Order payments data cleaning failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
