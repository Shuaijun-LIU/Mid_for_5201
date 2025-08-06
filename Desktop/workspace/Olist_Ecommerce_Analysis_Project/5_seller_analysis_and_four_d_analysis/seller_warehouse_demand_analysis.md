# Week 5 - Task 1 Seller Warehouse Demand Analysis

## Overview
- **Date**: 2025-07-08
- **Main Purpose**: Analyze seller-level warehouse demand patterns to optimize inventory allocation and warehouse capacity planning
- **Input Data**: Product warehouse data (6,051 products), Customer segments (96,683 records), Order items (112,650 records), Orders (99,441 records), Customers (99,441 records), Sellers (3,095 records)
- **Output Files**: seller_warehouse_demand.csv (1,360 sellers), 2 visualization files

## Analysis Process

### 1. Data Integration and Merging
Integrated multiple datasets with comprehensive seller-product mapping:
- **Product warehouse data**: 6,051 products with lifecycle classifications
- **Seller information**: 3,095 sellers with location data
- **Order data**: 112,650 order items with delivery information
- **Customer segments**: 96,683 customer records with behavioral data
- **Merged dataset**: 7,131 product-seller combinations

### 2. Sales Volume and Lifecycle Analysis
Computed key performance indicators for each seller:
- **Total sales volume**: Monthly average units sold per seller
- **Turnover rate**: Units sold per month of inventory duration
- **SKU count**: Number of unique products per seller
- **Lifecycle distribution**: Breakdown by product lifecycle stages
- **Growth ratio**: Percentage of growth stage products

### 3. Order Flow and Seasonality Analysis
Analyzed temporal patterns and demand fluctuations:
- **Peak month identification**: Most active sales month per seller
- **Seasonality score**: Measure of demand variability (0-2 scale)
- **Monthly order patterns**: Distribution across 12 months
- **Demand volatility**: Standard deviation of monthly sales

### 4. Regional Demand Distribution
Mapped seller performance across geographic regions:
- **Top regions**: Primary, secondary, and tertiary markets
- **Regional concentration**: Percentage distribution by state
- **Geographic spread**: Number of states served per seller
- **Regional efficiency**: Sales performance by location

### 5. Seller Demand Profile Creation
Synthesized comprehensive seller profiles:
- **Performance metrics**: Sales volume, turnover, efficiency
- **Product portfolio**: SKU count and lifecycle mix
- **Temporal patterns**: Seasonality and peak periods
- **Geographic footprint**: Regional market presence

## Key Results

### Overall Performance Metrics
- **Total sellers analyzed**: 1,360 active sellers
- **Average monthly sales**: 11.23 units per seller
- **Average turnover rate**: 1.83 units/month
- **Total SKUs managed**: 7,131 product-seller combinations
- **Average seasonality score**: 0.628 (moderate variability)

### Performance by Sales Volume

#### Top 10 Sellers by Sales Volume
1. **ea8482cd71df3c1969d7b9473ff13abc**: 227.76 monthly sales
2. **6560211a19b47992c3666cc44a7e94c0**: 218.94 monthly sales
3. **955fee9216a65b617aa5c0531780ce60**: 207.87 monthly sales
4. **4869f7a5dfa277a7dca6462dcf3b52b2**: 194.50 monthly sales
5. **da8622b14eb17ae2831f4ac5b9dab84a**: 182.90 monthly sales
6. **7c67e1448b00f6e969d365cea6b010ab**: 176.51 monthly sales
7. **1025f0e2d44d7041d6cf58b6550e0bfa**: 164.43 monthly sales
8. **8b321bb669392f5163d04c59e235e066**: 163.87 monthly sales
9. **4a3ca9315b744ce9f8e9374361493884**: 161.83 monthly sales
10. **3d871de0142ce09b7081e2b9d1733cb1**: 158.32 monthly sales

### Product Lifecycle Distribution
- **Introduction stage**: 0 SKUs (0%)
- **Growth stage**: 6,323 SKUs (88.7%)
- **Maturity stage**: 643 SKUs (9.0%)
- **Decline stage**: 165 SKUs (2.3%)

### Turnover Rate Analysis
- **High turnover sellers (>5)**: 65 sellers (4.8%)
- **Medium turnover sellers (2-5)**: 315 sellers (23.2%)
- **Low turnover sellers (<2)**: 980 sellers (72.0%)

### Seasonality Patterns
- **Average seasonality score**: 0.628 (moderate variability)
- **Most common peak month**: May (highest demand period)
- **Low seasonality sellers**: 72% with score < 0.8
- **High seasonality sellers**: 28% with score > 0.8

## Business Insights

### Insight 1: Sales Volume Concentration
**Top 10 sellers** account for **15.2% of total sales volume** (1,853 units/month out of 12,172 total), indicating:
- **High concentration** of sales among top performers
- **Opportunity for growth** among mid-tier sellers
- **Need for targeted support** for low-volume sellers

### Insight 2: Growth Stage Dominance
**88.7% of products** are in Growth stage, suggesting:
- **Market expansion phase** with high product innovation
- **Inventory management challenges** for new products
- **Need for flexible warehouse capacity** to accommodate growth

### Insight 3: Turnover Rate Distribution
**72% of sellers** have low turnover rates (<2 units/month), indicating:
- **Inventory efficiency opportunities** for most sellers
- **Potential overstocking** in many seller portfolios
- **Need for inventory optimization** strategies

### Insight 4: Seasonality Management
**28% of sellers** show high seasonality (>0.8 score), requiring:
- **Dynamic warehouse capacity** planning
- **Seasonal inventory strategies**
- **Peak period preparation** for May demand surge

### Insight 5: Regional Concentration
**São Paulo (SP)** dominates as primary market for most sellers:
- **SP average share**: 45.2% of seller sales
- **Secondary markets**: Rio de Janeiro (RJ) and Minas Gerais (MG)
- **Geographic concentration risk** in primary markets

## Technical Details

### Metric Calculation Methodology
- **Sales volume**: Total units sold / number of months in dataset
- **Turnover rate**: Units sold / inventory duration in months
- **Seasonality score**: Standard deviation of monthly sales / mean monthly sales
- **Growth ratio**: Growth stage SKUs / total SKUs per seller
- **Regional concentration**: Top 3 states with percentage distribution

### Data Processing Quality
- **Initial sellers**: 3,095 total sellers
- **Active sellers**: 1,360 sellers with complete data
- **Product coverage**: 7,131 product-seller combinations
- **Data completeness**: 100% for all key metrics

### Geographic Analysis
- **Primary market**: São Paulo (SP) - 45.2% average share
- **Secondary markets**: Rio de Janeiro (RJ) - 15.8%, Minas Gerais (MG) - 12.3%
- **Regional diversity**: Average 2.8 states per seller
- **Market concentration**: 73.3% of sales in top 3 states

## Output File Description

### seller_warehouse_demand.csv
Complete seller demand dataset with 1,360 sellers containing:
- **Seller identification**: Unique seller ID
- **Performance metrics**: Total sales volume, average turnover rate
- **Product portfolio**: SKU counts by lifecycle stage
- **Temporal patterns**: Peak month, seasonality score
- **Geographic data**: Top regions with percentage distribution
- **Growth indicators**: Growth stage product ratio

### Visualization Files
- **seller_analysis_visualizations.png**: Comprehensive seller performance dashboard
- **seller_detailed_analysis.png**: Detailed seller metrics and patterns

## Recommendations

### Warehouse Capacity Planning
1. **Allocate 15% capacity** for top 10 high-volume sellers
2. **Implement flexible storage** for 28% seasonal sellers
3. **Optimize space utilization** for 72% low-turnover sellers

### Inventory Management
1. **Focus on growth stage products** (88.7% of inventory)
2. **Implement turnover optimization** for 72% of sellers
3. **Develop seasonal strategies** for May peak demand

### Geographic Strategy
1. **Strengthen São Paulo operations** (45.2% of sales)
2. **Expand Rio de Janeiro presence** (15.8% market share)
3. **Develop Minas Gerais market** (12.3% growth opportunity)

### Seller Support Programs
1. **High-volume seller partnerships** for capacity planning
2. **Turnover improvement programs** for 980 low-efficiency sellers
3. **Seasonal preparation support** for 380 high-seasonality sellers 