"""
Clean reviews data for Olist E-commerce dataset
Purpose: Clean and standardize review data
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
        logging.FileHandler("week1_data_preparation/logs/base_reviews.log"),
        logging.StreamHandler(),
    ],
)


def validate_reviews(df: pd.DataFrame) -> bool:
    """
    Validate reviews data
    Args:
        df: Reviews dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required fields
        assert not df["review_id"].isna().any(), "Missing review ID"
        assert not df["order_id"].isna().any(), "Missing order ID"
        assert not df["review_score"].isna().any(), "Missing review score"

        # Check string fields
        assert df["review_id"].dtype in ["object", "string"], "Invalid review ID type"
        assert df["order_id"].dtype in ["object", "string"], "Invalid order ID type"
        assert df["review_comment_title"].dtype in [
            "object",
            "string",
        ], "Invalid comment title type"
        assert df["review_comment_message"].dtype in [
            "object",
            "string",
        ], "Invalid comment message type"

        # Check numeric fields
        assert df["review_score"].dtype in [
            "int64",
            "Int8",
            "Int32",
            "Int64",
        ], "Invalid review score type"

        # Check value ranges
        assert df["review_score"].between(1, 5).all(), "Review score out of range"

        # Check for duplicates
        assert not df["review_id"].duplicated().any(), "Duplicate review IDs found"

        logging.info("Reviews data validation passed")
        return True

    except AssertionError as e:
        logging.error(f"Reviews data validation failed: {str(e)}")
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
            parse_dates=["review_creation_date", "review_answer_timestamp"],
            dtype={"review_score": "Int64"},
        )

        # Check if dataframe has data
        if len(df) == 0:
            return False

        return True
    except Exception as e:
        logging.warning(f"Error checking file {file_path}: {str(e)}")
        return False


def clean_reviews(df: pd.DataFrame, valid_order_ids: set = None) -> pd.DataFrame:
    """
    Clean reviews data
    Args:
        df: Raw reviews dataframe
        valid_order_ids: Set of valid order IDs
    Returns:
        Cleaned reviews dataframe
    """
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()

    # 1. Convert timestamp columns to datetime
    df_clean["review_creation_date"] = pd.to_datetime(df_clean["review_creation_date"])
    df_clean["review_answer_timestamp"] = pd.to_datetime(
        df_clean["review_answer_timestamp"]
    )

    # 2. Convert review_score to Int8
    df_clean["review_score"] = pd.to_numeric(
        df_clean["review_score"], errors="coerce"
    ).astype("Int8")

    # 3. Handle missing values
    # Fill missing comment title and message with empty string
    df_clean["review_comment_title"] = df_clean["review_comment_title"].fillna("")
    df_clean["review_comment_message"] = df_clean["review_comment_message"].fillna("")

    # 4. Clean comment text
    # Remove extra whitespace
    df_clean["review_comment_title"] = df_clean["review_comment_title"].str.strip()
    df_clean["review_comment_message"] = df_clean["review_comment_message"].str.strip()

    # 5. Handle duplicate review IDs
    # Keep the latest review for each review_id
    df_clean = df_clean.sort_values("review_creation_date", ascending=False)
    df_clean = df_clean.drop_duplicates(subset=["review_id"], keep="first")

    # 6. Ensure logical relationships
    # If answer timestamp is before creation date, set it to creation date
    mask = df_clean["review_answer_timestamp"] < df_clean["review_creation_date"]
    df_clean.loc[mask, "review_answer_timestamp"] = df_clean.loc[
        mask, "review_creation_date"
    ]

    # 7. Filter by valid order IDs
    if valid_order_ids:
        df_clean = df_clean[df_clean["order_id"].isin(valid_order_ids)]

    return df_clean


def main():
    """Main function to clean reviews data"""
    try:
        print("\n=== Starting Reviews Data Cleaning ===")

        # 1. Load raw data
        print("\n1. Loading raw data...")
        input_file = RAW_DATA_DIR / "olist_order_reviews_dataset.csv"
        reviews_df = pd.read_csv(
            input_file,
            dtype=DTYPE_DICT["olist_order_reviews_dataset"],
            parse_dates=DATE_COLUMNS.get("olist_order_reviews_dataset", []),
        )
        print(f"Loaded {len(reviews_df)} review records")

        # 2. Load valid order IDs if available
        valid_order_ids = None
        orders_file = CLEANED_DATA_DIR / "orders.csv"
        if orders_file.exists():
            print("\n2. Loading valid order IDs...")
            orders_df = pd.read_csv(orders_file)
            valid_order_ids = set(orders_df["order_id"].unique())
            print(f"Found {len(valid_order_ids)} valid order IDs")

        # 3. Clean data
        output_file = CLEANED_DATA_DIR / "order_reviews.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(output_file):
            print("\n3. Cleaning data...")
            cleaned_reviews = clean_reviews(reviews_df, valid_order_ids)
            print(f"Cleaned {len(cleaned_reviews)} review records")

            # 4. Validate data
            print("\n4. Validating data...")
            if not validate_reviews(cleaned_reviews):
                raise ValueError("Reviews data validation failed")
            print("Data validation passed")

            # 5. Export cleaned data
            print("\n5. Exporting cleaned data...")
            cleaned_reviews.to_csv(output_file, index=False)
            print(f"Cleaned data exported to {output_file}")
        else:
            print("\nReviews data already cleaned, loading from file...")
            cleaned_reviews = pd.read_csv(
                output_file,
                dtype=DTYPE_DICT["olist_order_reviews_dataset"],
                parse_dates=DATE_COLUMNS.get("olist_order_reviews_dataset", []),
            )

        # 6. Generate sample
        print("\n6. Generating sample data...")
        sample_file = SAMPLE_DATA_DIR / "reviews_sample.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(sample_file):
            sample = cleaned_reviews.sample(frac=SAMPLE_RATIO, random_state=42)
            sample.to_csv(sample_file, index=False)
            print(f"Sample data exported to {sample_file}")
        else:
            print("Sample data already exists, skipping...")

        print("\n=== Reviews Data Cleaning Completed Successfully ===")

    except Exception as e:
        print(f"\nERROR: Reviews data cleaning failed: {str(e)}")
        logging.error(f"Reviews data cleaning failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
