"""
Generate date dimension table for Olist E-commerce dataset
Purpose: Create a comprehensive date dimension table for time-based analysis
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
import holidays
import re
from pathlib import Path

# import configuration
from config import (
    RAW_DATA_DIR, CLEANED_DATA_DIR, SAMPLE_DATA_DIR,
    DTYPE_DICT, DATE_COLUMNS, FORCE_REPROCESS, SAMPLE_RATIO
)

# create logs directory if not exists
os.makedirs("week1_data_preparation/logs", exist_ok=True)

# configure loggging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("week1_data_preparation/logs/dim_date.log"),
        logging.StreamHandler(),
    ],
)

def get_date_range() -> tuple:
    """
    Get the date range from orders data
    Returns:
        Tuple of (start_date, end_date)
    """
    try:
        # Load orders data
        orders_df = pd.read_csv(
            CLEANED_DATA_DIR / 'orders.csv',
            parse_dates=['order_purchase_timestamp']
        )
        
        # Get min and max dates
        start_date = orders_df['order_purchase_timestamp'].min()
        end_date = orders_df['order_purchase_timestamp'].max()
        
        # Round to start of month and end of month
        start_date = start_date.replace(day=1)
        end_date = (end_date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        
        return start_date, end_date
        
    except Exception as e:
        logging.error(f"Error getting date range: {str(e)}")
        raise

def get_season(month: int) -> str:
    """
    Get season name based on month (Southern Hemisphere)
    Args:
        month: Month number (1-12)
    Returns:
        Season name
    """
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:  # 9, 10, 11
        return 'Spring'

def generate_date_dimension(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Generate date dimension table
    Args:
        start_date: Start date
        end_date: End date
    Returns:
        DataFrame with date dimension
    """
    try:
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create base dataframe
        df = pd.DataFrame({'date': date_range})
        
        # Extract basic date components
        df['date_id'] = df['date'].dt.strftime('%Y%m%d').astype(int)
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek + 1  # 1-7 (Monday-Sunday)
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Add month name
        df['month_name'] = df['date'].dt.strftime('%B')
        
        # Add quarter name
        df['quarter_name'] = 'Q' + df['quarter'].astype(str)
        
        # Add day name
        df['day_name'] = df['date'].dt.strftime('%A')
        
        # Add year-month
        df['year_month'] = df['date'].dt.strftime('%Y-%m')
        
        # Add year-quarter
        df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
        
        # Add is_weekend flag
        df['is_weekend'] = df['day_of_week'].isin([6, 7]).astype(int)
        
        # Add is_holiday flag (Brazil holidays)
        br_holidays = holidays.BR()
        df['is_holiday'] = df['date'].apply(lambda x: x in br_holidays).astype(int)
        
        # Add holiday name
        df['holiday_name'] = df['date'].apply(lambda x: br_holidays.get(x, ''))
        
        # Add is_black_friday flag (last Friday of November)
        df['is_black_friday'] = (
            (df['month'] == 11) & 
            (df['day_of_week'] == 5) & 
            (df['day'] > 21)
        ).astype(int)
        
        # Add is_cyber_monday flag (first Monday after Black Friday)
        df['is_cyber_monday'] = (
            (df['month'] == 11) & 
            (df['day_of_week'] == 1) & 
            (df['day'] > 24)
        ).astype(int)
        
        # Add is_promotion_day flag
        df['is_promotion_day'] = (
            df['is_black_friday'] | 
            df['is_cyber_monday'] | 
            df['is_holiday']
        ).astype(int)
        
        # Add season (Southern Hemisphere)
        df['season'] = df['month'].apply(get_season)
        
        # Add fiscal year (assuming fiscal year starts in January)
        df['fiscal_year'] = df['year']
        df['fiscal_quarter'] = df['quarter']
        
        # Add fiscal period
        df['fiscal_period'] = df['fiscal_year'].astype(str) + '-Q' + df['fiscal_quarter'].astype(str)
        
        return df
        
    except Exception as e:
        logging.error(f"Error generating date dimension: {str(e)}")
        raise

def validate_date_dimension(df: pd.DataFrame) -> bool:
    """
    Validate date dimension table
    Args:
        df: Date dimension dataframe
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check for required columns
        required_columns = [
            'date_id', 'date', 'year', 'quarter', 'month', 'day',
            'day_of_week', 'day_of_year', 'week_of_year',
            'month_name', 'quarter_name', 'day_name',
            'year_month', 'year_quarter',
            'is_weekend', 'is_holiday', 'holiday_name',
            'is_black_friday', 'is_cyber_monday', 'is_promotion_day',
            'season', 'fiscal_year', 'fiscal_quarter', 'fiscal_period'
        ]
        assert all(col in df.columns for col in required_columns), "Missing required columns"
        
        # Check for missing values
        assert not df[required_columns].isna().any().any(), "Found missing values in required columns"
        
        # Check date_id format
        assert df['date_id'].dtype == 'int64', "date_id should be integer"
        assert len(df['date_id'].unique()) == len(df), "date_id should be unique"
        
        # Check date range
        assert df['date'].min() <= df['date'].max(), "Invalid date range"
        
        # Check numeric ranges
        assert df['year'].between(2016, 2020).all(), "Year out of expected range"
        assert df['quarter'].between(1, 4).all(), "Quarter out of range"
        assert df['month'].between(1, 12).all(), "Month out of range"
        assert df['day'].between(1, 31).all(), "Day out of range"
        assert df['day_of_week'].between(1, 7).all(), "Day of week out of range"
        
        # Check boolean flags
        boolean_columns = ['is_weekend', 'is_holiday', 'is_black_friday', 'is_cyber_monday', 'is_promotion_day']
        assert df[boolean_columns].isin([0, 1]).all().all(), "Boolean flags should be 0 or 1"
        
        # Check season values
        valid_seasons = ['Summer', 'Autumn', 'Winter', 'Spring']
        assert df['season'].isin(valid_seasons).all(), "Invalid season values"
        
        logging.info("Date dimension validation passed")
        return True
        
    except AssertionError as e:
        logging.error(f"Date dimension validation failed: {str(e)}")
        return False

def generate_dim_date(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate date dimension table
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    Returns:
        Date dimension dataframe
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create date dimension table
    date_dim = pd.DataFrame({
        'date_id': range(1, len(date_range) + 1),
        'date': date_range,
        'year': date_range.year,
        'month': date_range.month,
        'day': date_range.day,
        'quarter': date_range.quarter,
        'day_of_week': date_range.dayofweek,
        'day_of_year': date_range.dayofyear,
        'is_weekend': date_range.dayofweek.isin([5, 6]).astype(int),
        'is_holiday': 0  # Placeholder for holiday flag
    })
    
    # Add month name and day name
    date_dim['month_name'] = date_dim['date'].dt.strftime('%B')
    date_dim['day_name'] = date_dim['date'].dt.strftime('%A')
    
    # Add fiscal year and quarter
    date_dim['fiscal_year'] = date_dim['year'] + (date_dim['month'] >= 4).astype(int)
    date_dim['fiscal_quarter'] = ((date_dim['month'] - 1) // 3 + 1).astype(int)
    
    # Add week number
    date_dim['week_number'] = date_dim['date'].dt.isocalendar().week
    
    # Reorder columns
    date_dim = date_dim[[
        'date_id', 'date', 'year', 'month', 'day',
        'quarter', 'day_of_week', 'day_of_year',
        'is_weekend', 'is_holiday', 'month_name',
        'day_name', 'fiscal_year', 'fiscal_quarter',
        'week_number'
    ]]
    
    return date_dim

def main():
    """Main function to generate date dimension table"""
    try:
        print("\n=== Starting Date Dimension Table Generation ===")
        
        # 1. Set date range
        print("\n1. Setting date range...")
        start_date = '2016-01-01'
        end_date = '2018-12-31'
        print(f"Date range: {start_date} to {end_date}")
        
        # 2. Generate date dimension
        print("\n2. Generating date dimension table...")
        date_dim = generate_dim_date(start_date, end_date)
        print(f"Generated date dimension with {len(date_dim)} dates")
        
        # 3. Export date dimension
        print("\n3. Exporting date dimension data...")
        output_file = CLEANED_DATA_DIR / 'dim_date.csv'
        date_dim.to_csv(output_file, index=False)
        print(f"Date dimension data exported to {output_file}")
        
        # 4. Generate sample
        print("\n4. Generating sample data...")
        sample = date_dim.sample(frac=0.05, random_state=42)
        sample_file = SAMPLE_DATA_DIR / 'dim_date_sample.csv'
        sample.to_csv(sample_file, index=False)
        print(f"Sample data exported to {sample_file}")
        
        print("\n=== Date Dimension Table Generation Completed Successfully ===")
        
    except Exception as e:
        print(f"\nERROR: Date dimension table generation failed: {str(e)}")
        logging.error(f"Date dimension table generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 