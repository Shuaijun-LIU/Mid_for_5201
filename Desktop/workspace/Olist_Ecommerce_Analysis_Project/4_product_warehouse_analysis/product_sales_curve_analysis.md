# Product Sales Curve Analysis

## Overview
- **Execution Date**: 2025-06-25
- **Update Date**: 2025-06-30
- **Main Purpose**: Analyze product sales patterns over time to identify trends, growth rates, and seasonal variations for inventory planning
- **Input Data**: Order items (112,650 records), Orders (99,441 records), Products (32,951 records)
- **Output Files**: monthly_product_sales.csv (139,173 records), product_sales_summary.csv (6,051 products), 10 sales curve plots

## Analysis Process

### 1. Data Preprocessing and Integration
Integrated three key datasets to create a comprehensive sales analysis foundation:
- **Order items data**: 112,650 records with product and order details
- **Orders data**: 99,441 records with timestamps and delivery status
- **Products data**: 32,951 records with product categories

After filtering for delivered orders only, the final merged dataset contained **110,197 records** representing actual sales transactions.

### 2. Sales Aggregation and Time Series Construction
Created a complete product-month matrix to analyze sales patterns over time:
- **Aggregated sales** by product and month (volume and value)
- **Created complete matrix** with zero values for months without sales
- **Filtered products** with at least 3 months of sales history for meaningful analysis
- **Result**: 6,051 products with 139,173 product-month combinations

### 3. Time Series Smoothing and Growth Analysis
Applied advanced time series techniques to identify underlying trends:
- **3-month moving average smoothing** to reduce noise and highlight trends
- **Monthly growth rate calculation** to identify growth/decline patterns
- **Handled infinite values** in growth rate calculations (27,677 valid records out of 139,173)

### 4. Product Performance Summary Statistics
Generated comprehensive product-level metrics for inventory planning:
- **Total months active**: Average 23.0 months per product
- **Sales volume statistics**: Total, average, maximum, and standard deviation
- **Revenue analysis**: Total revenue per product
- **Peak performance identification**: Month of maximum sales for each product

### 5. Top Product Identification and Visualization
Created detailed sales curve visualizations for the top 10 products:
- **Identified top performers** by total sales volume
- **Generated individual plots** showing raw vs. smoothed sales curves
- **Annotated significant growth/decline points** (>20% or <-20% monthly change)
- **Added statistical summaries** on each plot for quick insights

## Key Results

### Product Performance Landscape
The analysis revealed a diverse product performance landscape with significant variation:

- **Total products analyzed**: 6,051 products with meaningful sales history
- **Average performance**: 11.5 units sold per product over 23.0 months
- **Performance distribution**:
  - Products with >10 units sold: 1,580 (26.1%)
  - Products with >50 units sold: 165 (2.7%)
  - Products with >100 units sold: 56 (0.9%)

### Top 10 Products by Sales Volume
The analysis identified the highest-performing products for inventory prioritization:

1. **aca2eb7d00ea1a7b8ebd4e68314663af**: 520 units (peak performer)
2. **422879e10f46682990de24d770e7f83d**: 484 units
3. **99a4788cb24856965c36a24e339b6058**: 477 units
4. **389d119b48cf3043d311335e499d9c6b**: 390 units
5. **368c6c730842d78016ad823897a372db**: 388 units
6. **53759a2ecddad2bb87a079a1f1519f73**: 373 units
7. **d1c427060a0f73f6b889a5c7c61f2ac4**: 332 units
8. **53b36df67ebb7c41585e8d54d6772e08**: 321 units
9. **154e7e31ebfa092203795c972e5804a6**: 274 units
10. **3dd2a17168ec895c781a9191c1e95ad7**: 272 units

### Time Series Analysis Insights
The smoothing and growth analysis revealed important patterns:

- **Average smoothed volume**: 0.50 units per month per product
- **Average growth rate**: -30.46% (indicating overall declining trend)
- **Growth rate validity**: 19.9% of records had valid growth rate calculations
- **Significant volatility**: High standard deviation in sales patterns

### Sales Curve Characteristics
Individual product analysis showed diverse sales curve patterns:
- **Growth patterns**: Some products showed consistent growth trends
- **Seasonal variations**: Clear seasonal patterns in many products
- **Volatility levels**: High variation in monthly sales volumes
- **Peak identification**: Distinct peak sales months for each product

## Business Insights

### Insight 1: Product Performance Concentration
The analysis reveals a significant concentration in product performance with only 26.1% of products selling more than 10 units. This suggests the need for a **tiered inventory strategy** that prioritizes high-performing products while managing low-volume items efficiently.

### Insight 2: Overall Declining Sales Trend
The average growth rate of -30.46% indicates an **overall declining trend** in product sales. This critical insight suggests the need for:
- **Product lifecycle management** strategies
- **New product introduction** programs
- **Marketing and promotion** efforts to reverse the trend

### Insight 3: High Volatility in Sales Patterns
The high standard deviation in sales volumes indicates **significant volatility** in product demand. This volatility creates challenges for inventory planning and suggests the need for:
- **Flexible inventory policies**
- **Safety stock adjustments**
- **Dynamic replenishment strategies**

### Insight 4: Long Product Lifecycles
The average of 23.0 months of activity per product suggests **extended product lifecycles**. This provides opportunities for:
- **Long-term inventory planning**
- **Seasonal pattern optimization**
- **Predictive demand forecasting**

### Insight 5: Top Product Performance Gap
The significant gap between top performers (520 units) and average performance (11.5 units) highlights the **importance of product selection** in inventory management. The top 10 products represent the core inventory that should receive priority attention.

## Technical Details

### Data Processing Methodology
The analysis employed a comprehensive data processing approach:
- **Complete matrix construction**: Ensured all product-month combinations were represented
- **Moving average smoothing**: Applied 3-month window to reduce noise
- **Growth rate calculation**: Used percentage change with proper handling of edge cases
- **Statistical aggregation**: Generated multiple performance metrics per product

### Time Series Analysis Techniques
Advanced time series techniques were applied:
- **Smoothing algorithms**: 3-month moving average for trend identification
- **Growth rate analysis**: Monthly percentage change calculations
- **Volatility measurement**: Standard deviation analysis
- **Peak detection**: Identification of maximum performance periods

### Visualization Approach
Comprehensive visualization strategy was implemented:
- **Individual product plots**: Detailed sales curves for top performers
- **Statistical annotations**: Key metrics displayed on each plot
- **Growth point highlighting**: Significant changes marked with annotations
- **Comparative analysis**: Raw vs. smoothed data comparison

### Quality Control Measures
Multiple quality control measures were implemented:
- **Data filtering**: Only delivered orders included
- **Minimum activity threshold**: Products with <3 months excluded
- **Growth rate validation**: Infinite values properly handled
- **Statistical rounding**: Consistent decimal precision

## Output File Description

### monthly_product_sales.csv
Complete time series dataset with 139,173 records containing:
- **Product identification**: Product ID for each record
- **Time dimension**: Order month in YYYY-MM format
- **Sales metrics**: Monthly sales volume and value
- **Smoothed data**: 3-month moving average values
- **Growth analysis**: Monthly growth rate percentages

### product_sales_summary.csv
Product-level summary statistics for 6,051 products including:
- **Activity metrics**: Total months active, total units sold
- **Performance statistics**: Average, maximum, and standard deviation of sales
- **Revenue analysis**: Total revenue per product
- **Peak performance**: Month of maximum sales for each product

### Sales Curve Plots (10 files)
Individual visualization files for top 10 products showing:
- **Raw sales data**: Original monthly sales volumes
- **Smoothed trends**: 3-month moving average lines
- **Growth annotations**: Significant growth/decline points
- **Statistical summaries**: Key performance metrics
- **Time axis**: Monthly progression over the analysis period

## Recommendations

### Inventory Management
1. **Implement tiered inventory strategy** based on sales performance
2. **Increase safety stock** for top-performing products
3. **Reduce inventory levels** for low-volume products
4. **Monitor declining products** for potential discontinuation

### Demand Planning
1. **Develop seasonal forecasting models** based on identified patterns
2. **Implement dynamic replenishment** for volatile products
3. **Create growth strategies** for declining product categories
4. **Optimize product mix** based on performance analysis

### Business Strategy
1. **Focus marketing efforts** on top-performing products
2. **Investigate declining trends** to identify root causes
3. **Develop new product introduction** strategies
4. **Optimize pricing strategies** based on demand patterns 