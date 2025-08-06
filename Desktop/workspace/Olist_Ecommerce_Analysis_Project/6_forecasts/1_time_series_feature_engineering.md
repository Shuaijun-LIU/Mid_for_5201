# Week 6 - Task 1: Time Series Feature Engineering

## Overview

Task 1 prepares advanced time series features for prediction models in Week6, based on Week5's four-dimensional cross-analysis results and Week2's pre-processed time series data. Creates prediction-ready features including lag variables, moving averages, seasonal decomposition, and machine learning transformations.

## Core Features

### 1. Data Integration
- Load Week5 four-dimensional analysis results
- Use Week2 pre-processed holiday data (avoiding re-calculation)
- Create seller-product-geography-time base connections
- Handle missing values with median imputation

### 2. Advanced Time Series Features
- **Lag Features**: order_count, total_sales, avg_price, avg_freight (lags: 1,2,3,7,14,30)
- **Moving Averages**: 3,7,14,30,60,90-day windows + weighted moving averages
- **Seasonal Decomposition**: STL decomposition with trend, seasonal, residual components
- **Differential Features**: 1st/2nd order differences, 7/30-day differences

### 3. Holiday Features (Week6-specific)
- Holiday effect windows: 1,3,7,14 days
- Holiday intensity by seller-product-state combinations
- Uses Week2 pre-processed holiday data

### 4. Cyclical Features (Week6-specific)
- Monthly/quarterly cyclical encoding (sine/cosine)
- Seasonal one-hot encoding (Winter/Spring/Summer/Fall)
- Time-based cyclical transformations

### 5. Machine Learning Features
- Logarithmic, square root, square transformations
- Standardization (Z-score) and normalization (Min-Max)
- Ratio features: sales_per_order, price_to_freight_ratio
- Feature importance analysis and correlation matrix

## Input Data Sources

### Week5 Results
- `four_d_main_analysis.csv` - Four-dimensional main data
- `seller_summary_analysis.csv` - Seller dimension summary
- `product_summary_analysis.csv` - Product dimension summary
- `geo_summary_analysis.csv` - Geography dimension summary
- `time_summary_analysis.csv` - Time dimension summary

### Week2 Pre-processed Data
- `weekend_holiday_analysis/orders_with_weekend_holiday.csv` - Holiday annotations

## Output Files

### Main Outputs
- `week6_time_series_features_matrix.csv` - Complete feature matrix
- `week6_feature_importance_ranking.csv` - Feature importance by correlation
- `week6_feature_correlation_matrix.csv` - Feature correlation matrix
- `week6_feature_statistics_summary.csv` - Descriptive statistics
- `week6_high_correlation_pairs.csv` - High correlation feature pairs

### Visualizations
- `visualizations/week6_time_series_features.png` - Feature analysis charts
- `visualizations/week6_feature_correlation_heatmap.png` - Correlation heatmap

## Usage

```bash
cd week6_forecasts_and_accurate_recommendations
python time_series_feature_engineering.py
```

## Feature Matrix Structure

```python
{
    'seller_id': 'seller_123',
    'product_category': 'electronics',
    'state': 'SP',
    'date': '2024-01-01',
    
    # Target variables
    'order_count': 45,
    'total_sales': 1250.50,
    
    # Week6-specific features
    'order_count_lag7': 40,
    'order_count_ma7': 43.2,
    'seasonal_component': 0.08,
    'holiday_intensity': 0.15,
    'month_sin': 0.5,
    'order_count_diff7': 5,
    'total_sales_log': 7.13,
    'sales_per_order': 27.79
}
```

## Quality Control

### Data Quality Checks
- Missing value ratio < 5%
- Time series continuity validation
- Feature distribution analysis

### Feature Validation
- Correlation with target variables
- Multicollinearity detection
- Feature importance ranking
- Stability testing

## Performance Metrics

- Feature creation time
- Memory usage
- Feature count statistics
- Data completeness metrics

## Error Handling & Logging

### Error Handling
- Missing file detection
- Memory management
- Calculation exception handling
- Output file error handling

### Logging
- Data loading progress
- Feature creation progress
- Quality check results
- Performance statistics

Log file: `output/logs/time_series_feature_engineering.log`

## Key Optimizations

1. **Eliminated Redundancy**: Uses Week2 pre-processed holiday data
2. **Simplified Processing**: Removed duplicate time series aggregations
3. **Focused Features**: Concentrates on Week6-specific advanced features
4. **Efficient Computation**: Avoids re-calculating existing metrics

## Actual Run Output (2025-07-20)

**Data loaded:**
- four_d_main: (56031, 14)
- seller_summary: (3095, 9)
- product_summary: (74, 8)
- geo_summary: (27, 7)
- time_summary: (24, 7)
- weekend_holiday: (99441, 13)

**Missing value ratio:**
seller_id                  0.000000
product_category           0.000000
state                      0.000000
month                      0.000000
order_count                0.000000
total_sales                0.000000
avg_price                  0.000000
avg_freight                0.000000
same_state_ratio           0.000000
dominant_lifecycle_x       0.000000
holiday_ratio              0.000000
complexity_score           0.002891
retention_rate             0.002481
growth_potential           0.000000
date                       0.000000
product_diversity          0.000000
geographic_coverage        0.000000
temporal_coverage_x        0.000000
avg_complexity             0.002891
avg_retention              0.002481
seller_diversity           0.000000
geographic_distribution    0.000000
temporal_coverage_y        0.000000
dominant_lifecycle_y       0.000000
seller_concentration       0.000000
product_mix                0.000000
temporal_coverage          0.000000
active_sellers             0.000000
active_categories          0.000000
active_states              0.000000

time series data preparation completed: (56031, 30)

**Output files:**
- week6_time_series_features_matrix.csv (48M, 117 columns, 56031 rows)
- week6_feature_importance_ranking.csv
- week6_feature_correlation_matrix.csv
- week6_feature_statistics_summary.csv
- week6_high_correlation_pairs.csv
- visualizations/week6_time_series_features.png
- visualizations/week6_feature_correlation_heatmap.png

**Feature count:** 117
**Sample count:** 56031
