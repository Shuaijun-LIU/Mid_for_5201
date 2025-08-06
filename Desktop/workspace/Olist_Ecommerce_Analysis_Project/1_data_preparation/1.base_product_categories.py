"""
Clean product categories data for Olist E-commerce Dataset
Purpose: Clean and standardize product category data
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
        logging.FileHandler("week1_data_preparation/logs/base_product_categories.log"),
        logging.StreamHandler(),
    ],
)


def validate_product_categories(df: pd.DataFrame) -> bool:
    """
    Validate product categories data
    Args:
        df: Product categories dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required fields
        assert not df["product_category_name"].isna().any(), "Missing category name"
        assert (
            not df["product_category_name_english"].isna().any()
        ), "Missing English category name"

        # Check string fields
        assert df["product_category_name"].dtype in [
            "object",
            "string",
        ], "Invalid category name type"
        assert df["product_category_name_english"].dtype in [
            "object",
            "string",
        ], "Invalid English category name type"

        # Check value formats
        assert (
            df["product_category_name"].str.match(r"^[A-Z_]+$").all()
        ), "Invalid category name format"
        assert (
            df["product_category_name_english"].str.match(r"^[A-Za-z\s]+$").all()
        ), "Invalid English category name format"

        # Check for empty strings
        assert not (
            df["product_category_name"] == ""
        ).any(), "Empty category names found"
        assert not (
            df["product_category_name_english"] == ""
        ).any(), "Empty English category names found"

        # Check for duplicates
        assert (
            not df["product_category_name"].duplicated().any()
        ), "Duplicate category names found"
        assert (
            not df["product_category_name_english"].duplicated().any()
        ), "Duplicate English category names found"

        logging.info("Product categories data validation passed")
        return True

    except AssertionError as e:
        logging.error(f"Product categories data validation failed: {str(e)}")
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


def clean_product_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean product categories data
    Args:
        df: Raw product categories dataframe
    Returns:
        Cleaned product categories dataframe
    """
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()

    # 1. Clean product_category_name
    # Convert to uppercase and remove accents
    df_clean["product_category_name"] = df_clean["product_category_name"].str.upper()
    # Remove any special characters except underscore
    df_clean["product_category_name"] = df_clean["product_category_name"].str.replace(
        r"[^A-Z_]", "", regex=True
    )
    # Replace spaces with underscores
    df_clean["product_category_name"] = df_clean["product_category_name"].str.replace(
        r"\s+", "_", regex=True
    )

    # 2. Clean product_category_name_english
    # Convert to title case and remove accents
    df_clean["product_category_name_english"] = df_clean[
        "product_category_name_english"
    ].str.title()
    # Remove any special characters except letters and spaces
    df_clean["product_category_name_english"] = df_clean[
        "product_category_name_english"
    ].str.replace(r"[^A-Za-z\s]", "", regex=True)

    # 3. Handle missing values
    df_clean["product_category_name"] = df_clean["product_category_name"].fillna(
        "UNKNOWN"
    )
    df_clean["product_category_name_english"] = df_clean[
        "product_category_name_english"
    ].fillna("Unknown")

    # 4. Remove empty strings
    df_clean = df_clean[df_clean["product_category_name"] != ""]
    df_clean = df_clean[df_clean["product_category_name_english"] != ""]

    # 5. Ensure category names are unique
    df_clean = df_clean.drop_duplicates(subset=["product_category_name"])
    df_clean = df_clean.drop_duplicates(subset=["product_category_name_english"])

    return df_clean


def main():
    """Main function to clean product categories data"""
    try:
        print("\n=== Starting Product Categories Data Cleaning ===")

        # 1. Load raw data
        print("\n1. Loading raw data...")
        input_file = RAW_DATA_DIR / "product_category_name_translation.csv"
        product_categories_df = pd.read_csv(
            input_file, dtype=DTYPE_DICT["product_category_name_translation"]
        )
        print(f"Loaded {len(product_categories_df)} product category records")

        # 2. Clean data
        output_file = CLEANED_DATA_DIR / "product_categories.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(output_file):
            print("\n2. Cleaning data...")
            cleaned_product_categories = clean_product_categories(product_categories_df)
            print(f"Cleaned {len(cleaned_product_categories)} product category records")

            # 3. Validate data
            print("\n3. Validating data...")
            if not validate_product_categories(cleaned_product_categories):
                raise ValueError("Product categories data validation failed")
            print("Data validation passed")

            # 4. Export cleaned data
            print("\n4. Exporting cleaned data...")
            cleaned_product_categories.to_csv(output_file, index=False)
            print(f"Cleaned data exported to {output_file}")
        else:
            print("\nProduct categories data already cleaned, loading from file...")
            cleaned_product_categories = pd.read_csv(
                output_file, dtype=DTYPE_DICT["product_category_name_translation"]
            )

        # 5. Generate sample
        print("\n5. Generating sample data...")
        sample_file = SAMPLE_DATA_DIR / "product_categories_sample.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(sample_file):
            sample = cleaned_product_categories.sample(
                frac=SAMPLE_RATIO, random_state=42
            )
            sample.to_csv(sample_file, index=False)
            print(f"Sample data exported to {sample_file}")
        else:
            print("Sample data already exists, skipping...")

        print("\n=== Product Categories Data Cleaning Completed Successfully ===")

    except Exception as e:
        print(f"\nERROR: Product categories data cleaning failed: {str(e)}")
        logging.error(f"Product categories data cleaning failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
