"""
Clean orders data for Olist E-commerce dataset
Purpose: Clean and standardize order data
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
        logging.FileHandler("week1_data_preparation/logs/base_orders.log"),
        logging.StreamHandler(),
    ],
)


def validate_orders(df: pd.DataFrame) -> bool:
    """
    Validate orders data
    Args:
        df: Orders dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required fields
        assert not df["order_id"].isna().any(), "Missing order_id"
        assert not df["customer_id"].isna().any(), "Missing customer_id"
        assert (
            not df["order_purchase_timestamp"].isna().any()
        ), "Missing purchase timestamp"

        # Check timestamp formats
        assert (
            df["order_purchase_timestamp"].dtype == "datetime64[ns]"
        ), "Invalid timestamp format"

        # Check numeric fields
        assert (
            df["delivery_time_days"].dtype == "Int64"
        ), "Invalid delivery_time_days type"
        assert (
            df["processing_time_days"].dtype == "Int64"
        ), "Invalid processing_time_days type"

        # Check value ranges
        assert df["delivery_time_days"].min() >= 0, "Negative delivery time found"
        assert df["processing_time_days"].min() >= 0, "Negative processing time found"

        # Check logical relationships
        assert (
            df["delivery_time_days"] >= df["processing_time_days"]
        ).all(), "Delivery time should be greater than or equal to processing time"

        logging.info("Orders data validation passed")
        return True

    except AssertionError as e:
        logging.error(f"Orders data validation failed: {str(e)}")
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
            parse_dates=[
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
            dtype={"delivery_time_days": "Int64", "processing_time_days": "Int64"},
        )

        # Check if dataframe has data
        if len(df) == 0:
            return False

        return True
    except Exception as e:
        logging.warning(f"Error checking file {file_path}: {str(e)}")
        return False


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean orders data
    Args:
        df: Raw orders dataframe
    Returns:
        Cleaned orders dataframe
    """
    # Convert timestamp columns to datetime
    timestamp_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in timestamp_cols:
        df[col] = pd.to_datetime(df[col])

    # Calculate delivery and processing times
    df["delivery_time_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 3600)

    df["processing_time_days"] = (
        df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 3600)

    # Handle negative times
    # If processing time is negative, it means the carrier date is before purchase date
    # In this case, we'll set it to 0
    df.loc[df["processing_time_days"] < 0, "processing_time_days"] = 0

    # If delivery time is negative, it means the customer date is before purchase date
    # In this case, we'll set it to the processing time
    mask = df["delivery_time_days"] < 0
    df.loc[mask, "delivery_time_days"] = df.loc[mask, "processing_time_days"]

    # Ensure delivery time is always greater than or equal to processing time
    mask = df["delivery_time_days"] < df["processing_time_days"]
    df.loc[mask, "delivery_time_days"] = df.loc[mask, "processing_time_days"]

    # Convert to integer
    df["delivery_time_days"] = df["delivery_time_days"].round().astype("Int64")
    df["processing_time_days"] = df["processing_time_days"].round().astype("Int64")

    return df


def handle_missing_timestamps(
    df: pd.DataFrame, customers_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Handle missing timestamps in orders data
    Args:
        df: Orders dataframe
        customers_df: Customers dataframe
    Returns:
        Dataframe with filled timestamps
    """
    # Ensure timestamp columns are datetime
    timestamp_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in timestamp_cols:
        df[col] = pd.to_datetime(df[col])

    # Log initial state
    logging.info("Initial state of timestamps:")
    for col in timestamp_cols:
        missing_count = df[col].isna().sum()
        logging.info(f"{col}: {missing_count} missing values")

    # Merge with customers to get state information
    df = df.merge(
        customers_df[["customer_id", "customer_state"]], on="customer_id", how="left"
    )

    # Calculate average processing times by state
    state_processing_times = df.groupby("customer_state", observed=True)[
        "processing_time_days"
    ].mean()
    state_delivery_times = df.groupby("customer_state", observed=True)[
        "delivery_time_days"
    ].mean()

    # Log state averages
    logging.info("Average processing times by state:")
    for state, time in state_processing_times.items():
        logging.info(f"{state}: {time:.2f} days")

    # Fill missing order_approved_at
    mask = df["order_approved_at"].isna()
    df.loc[mask, "order_approved_at"] = df.loc[mask].apply(
        lambda row: row["order_purchase_timestamp"]
        + pd.Timedelta(
            days=float(state_processing_times.get(row["customer_state"], 0))
        ),
        axis=1,
    )

    # Fill missing order_delivered_carrier_date
    mask = df["order_delivered_carrier_date"].isna()
    df.loc[mask, "order_delivered_carrier_date"] = df.loc[mask].apply(
        lambda row: row["order_approved_at"]
        + pd.Timedelta(
            days=float(state_processing_times.get(row["customer_state"], 0))
        ),
        axis=1,
    )

    # Fill missing order_delivered_customer_date
    mask = df["order_delivered_customer_date"].isna()
    df.loc[mask, "order_delivered_customer_date"] = df.loc[mask].apply(
        lambda row: row["order_delivered_carrier_date"]
        + pd.Timedelta(days=float(state_delivery_times.get(row["customer_state"], 0))),
        axis=1,
    )

    # Log state after filling missing values
    logging.info("State after filling missing values:")
    for col in timestamp_cols:
        missing_count = df[col].isna().sum()
        logging.info(f"{col}: {missing_count} missing values")

    # Drop the temporary customer_state column
    df = df.drop("customer_state", axis=1)

    # Recalculate delivery and processing times after filling missing values
    df["delivery_time_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 3600)

    df["processing_time_days"] = (
        df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 3600)

    # Log time statistics before adjustment
    logging.info("Time statistics before adjustment:")
    logging.info(
        f"Processing time - min: {df['processing_time_days'].min():.2f}, max: {df['processing_time_days'].max():.2f}"
    )
    logging.info(
        f"Delivery time - min: {df['delivery_time_days'].min():.2f}, max: {df['delivery_time_days'].max():.2f}"
    )

    # Handle negative times
    neg_processing = (df["processing_time_days"] < 0).sum()
    neg_delivery = (df["delivery_time_days"] < 0).sum()
    logging.info(
        f"Found {neg_processing} negative processing times and {neg_delivery} negative delivery times"
    )

    # Set negative times to 0
    df.loc[df["processing_time_days"] < 0, "processing_time_days"] = 0
    df.loc[df["delivery_time_days"] < 0, "delivery_time_days"] = 0

    # Ensure delivery time is always greater than or equal to processing time
    invalid_times = (df["delivery_time_days"] < df["processing_time_days"]).sum()
    logging.info(
        f"Found {invalid_times} cases where delivery time is less than processing time"
    )

    mask = df["delivery_time_days"] < df["processing_time_days"]
    df.loc[mask, "delivery_time_days"] = df.loc[mask, "processing_time_days"]

    # Convert to integer
    df["delivery_time_days"] = df["delivery_time_days"].round().astype("Int64")
    df["processing_time_days"] = df["processing_time_days"].round().astype("Int64")

    # Log final statistics
    logging.info("Final time statistics:")
    logging.info(
        f"Processing time - min: {df['processing_time_days'].min()}, max: {df['processing_time_days'].max()}"
    )
    logging.info(
        f"Delivery time - min: {df['delivery_time_days'].min()}, max: {df['delivery_time_days'].max()}"
    )

    return df


def main():
    """Main function to clean orders data"""
    try:
        print("\n=== Starting Orders Data Cleaning ===")

        # 1. Load raw data
        print("\n1. Loading raw data...")
        input_file = RAW_DATA_DIR / "olist_orders_dataset.csv"
        orders_df = pd.read_csv(
            input_file,
            dtype=DTYPE_DICT["olist_orders_dataset"],
            parse_dates=DATE_COLUMNS["olist_orders_dataset"],
        )
        print(f"Loaded {len(orders_df)} order records")

        # Load customers data for timestamp handling
        customers_file = RAW_DATA_DIR / "olist_customers_dataset.csv"
        customers_df = pd.read_csv(
            customers_file, dtype=DTYPE_DICT["olist_customers_dataset"]
        )

        # 2. Clean data
        output_file = CLEANED_DATA_DIR / "orders.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(output_file):
            print("\n2. Cleaning data...")
            cleaned_orders = clean_orders(orders_df)
            print(f"Cleaned {len(cleaned_orders)} order records")

            # Handle missing timestamps
            print("\n3. Handling missing timestamps...")
            cleaned_orders = handle_missing_timestamps(cleaned_orders, customers_df)
            print("Missing timestamps handled")

            # 4. Validate data
            print("\n4. Validating data...")
            if not validate_orders(cleaned_orders):
                raise ValueError("Orders data validation failed")
            print("Data validation passed")

            # 5. Export cleaned data
            print("\n5. Exporting cleaned data...")
            cleaned_orders.to_csv(output_file, index=False)
            print(f"Cleaned data exported to {output_file}")
        else:
            print("\nOrders data already cleaned, loading from file...")
            cleaned_orders = pd.read_csv(
                output_file,
                parse_dates=DATE_COLUMNS["olist_orders_dataset"],
                dtype={"delivery_time_days": "Int64", "processing_time_days": "Int64"},
            )

        # 6. Generate sample
        print("\n6. Generating sample data...")
        sample_file = SAMPLE_DATA_DIR / "orders_sample.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(sample_file):
            sample = cleaned_orders.sample(frac=SAMPLE_RATIO, random_state=42)
            sample.to_csv(sample_file, index=False)
            print(f"Sample data exported to {sample_file}")
        else:
            print("Sample data already exists, skipping...")

        print("\n=== Orders Data Cleaning Completed Successfully ===")

    except Exception as e:
        print(f"\nERROR: Orders data cleaning failed: {str(e)}")
        logging.error(f"Orders data cleaning failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
