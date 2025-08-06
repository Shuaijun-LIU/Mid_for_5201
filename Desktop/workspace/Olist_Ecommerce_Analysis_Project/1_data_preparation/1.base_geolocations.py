"""
Clean geolocations data for Olist E-commerce Dataset
Purpose: Clean and standardize geolocation data
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
        logging.FileHandler("week1_data_preparation/logs/base_geolocations.log"),
        logging.StreamHandler(),
    ],
)


def validate_geolocations(df: pd.DataFrame) -> bool:
    """
    Validate geolocations data
    Args:
        df: Geolocations dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required fields
        assert (
            not df["geolocation_zip_code_prefix"].isna().any()
        ), "Missing zip code prefix"
        assert not df["geolocation_lat"].isna().any(), "Missing latitude"
        assert not df["geolocation_lng"].isna().any(), "Missing longitude"
        assert not df["geolocation_city"].isna().any(), "Missing city"
        assert not df["geolocation_state"].isna().any(), "Missing state"

        # Check string fields
        assert (
            df["geolocation_zip_code_prefix"].dtype == "object"
        ), "Invalid zip code prefix type"
        assert df["geolocation_city"].dtype == "object", "Invalid city type"
        assert df["geolocation_state"].dtype == "object", "Invalid state type"

        # Check numeric fields
        assert df["geolocation_lat"].dtype == "float64", "Invalid latitude type"
        assert df["geolocation_lng"].dtype == "float64", "Invalid longitude type"

        # Check value formats
        assert (
            df["geolocation_zip_code_prefix"].str.match(r"^\d{5}$").all()
        ), "Invalid zip code prefix format"
        assert (
            df["geolocation_state"].str.match(r"^[A-Z]{2}$").all()
        ), "Invalid state format"

        # Check value ranges
        assert df["geolocation_lat"].between(-90, 90).all(), "Latitude out of range"
        assert df["geolocation_lng"].between(-180, 180).all(), "Longitude out of range"

        logging.info("Geolocations data validation passed")
        return True

    except AssertionError as e:
        logging.error(f"Geolocations data validation failed: {str(e)}")
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
            dtype={"geolocation_lat": "float64", "geolocation_lng": "float64"},
        )

        # Check if dataframe has data
        if len(df) == 0:
            return False

        return True
    except Exception as e:
        logging.warning(f"Error checking file {file_path}: {str(e)}")
        return False


def clean_geolocations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean geolocations data
    Args:
        df: Raw geolocations dataframe
    Returns:
        Cleaned geolocations dataframe
    """
    # Convert zip code prefix to string and pad with zeros
    df["geolocation_zip_code_prefix"] = (
        df["geolocation_zip_code_prefix"].astype(str).str.zfill(5)
    )

    # Convert city and state to uppercase and strip whitespace
    df["geolocation_city"] = df["geolocation_city"].str.upper().str.strip()
    df["geolocation_state"] = df["geolocation_state"].str.upper().str.strip()

    # Remove special characters from city names
    df["geolocation_city"] = df["geolocation_city"].str.replace(
        r"[^A-Z\s]", "", regex=True
    )

    # Fill missing values with 'UNKNOWN'
    df["geolocation_city"] = df["geolocation_city"].fillna("UNKNOWN")
    df["geolocation_state"] = df["geolocation_state"].fillna("UNKNOWN")

    # Convert coordinates to float
    df["geolocation_lat"] = pd.to_numeric(df["geolocation_lat"], errors="coerce")
    df["geolocation_lng"] = pd.to_numeric(df["geolocation_lng"], errors="coerce")

    # Clip coordinates to valid ranges
    df["geolocation_lat"] = df["geolocation_lat"].clip(-90, 90)
    df["geolocation_lng"] = df["geolocation_lng"].clip(-180, 180)

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def main():
    """Main function to clean geolocations data"""
    try:
        print("\n=== Starting Geolocations Data Cleaning ===")

        # 1. Load raw data
        print("\n1. Loading raw data...")
        input_file = RAW_DATA_DIR / "olist_geolocation_dataset.csv"
        geolocations_df = pd.read_csv(
            input_file, dtype=DTYPE_DICT["olist_geolocation_dataset"]
        )
        print(f"Loaded {len(geolocations_df)} geolocation records")

        # 2. Clean data
        output_file = CLEANED_DATA_DIR / "geolocations.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(output_file):
            print("\n2. Cleaning data...")
            cleaned_geolocations = clean_geolocations(geolocations_df)
            print(f"Cleaned {len(cleaned_geolocations)} geolocation records")

            # 3. Validate data
            print("\n3. Validating data...")
            if not validate_geolocations(cleaned_geolocations):
                raise ValueError("Geolocations data validation failed")
            print("Data validation passed")

            # 4. Export cleaned data
            print("\n4. Exporting cleaned data...")
            cleaned_geolocations.to_csv(output_file, index=False)
            print(f"Cleaned data exported to {output_file}")
        else:
            print("\nGeolocations data already cleaned, loading from file...")
            cleaned_geolocations = pd.read_csv(
                output_file, dtype=DTYPE_DICT["olist_geolocation_dataset"]
            )

        # 5. Generate sample
        print("\n5. Generating sample data...")
        sample_file = SAMPLE_DATA_DIR / "geolocations_sample.csv"
        if FORCE_REPROCESS or not check_file_exists_and_valid(sample_file):
            sample = cleaned_geolocations.sample(frac=SAMPLE_RATIO, random_state=42)
            sample.to_csv(sample_file, index=False)
            print(f"Sample data exported to {sample_file}")
        else:
            print("Sample data already exists, skipping...")

        print("\n=== Geolocations Data Cleaning Completed Successfully ===")

    except Exception as e:
        print(f"\nERROR: Geolocations data cleaning failed: {str(e)}")
        logging.error(f"Geolocations data cleaning failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
