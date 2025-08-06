# Product Lifecycle Classification

## Overview
- **Execution Date**: 2025-06-26
- **Update Date**: 2025-06-30
- **Main Purpose**: Classify products into lifecycle stages (Introduction, Growth, Maturity, Decline) based on sales performance patterns
- **Input Data**: Monthly product sales (139,173 records), Product summary (6,051 products), Product categories (32,951 records)
- **Output Files**: product_lifecycle_labels.csv (6,051 products), 2 visualization files

## Analysis Process

### 1. Feature Engineering
Calculated five key lifecycle indicators for each product:
- **Initial growth rate**: Average monthly growth in first 3 months (29.56% average)
- **Saturation duration**: Months with stable growth (-5% to +5% change)
- **Decline months**: Consecutive negative growth months from recent period
- **Sales peak position**: Early, mid, or late in product timeline
- **Volatility**: Standard deviation of monthly growth rates (76.83 average)

### 2. Lifecycle Classification Logic
Applied business rules to classify products:
- **Growth**: High initial growth (>20%) OR high volatility (>30%) OR mid-peak position
- **Maturity**: Long saturation duration (≥4 months) OR late peak OR low volatility (<15%)
- **Decline**: Sustained negative growth (≥3 months) OR negative initial growth (<-10%)
- **Introduction**: Short history (≤6 months) with positive growth and low sales

### 3. Visualization Creation
Generated two key visualizations:
- **Stage distribution bar chart**: Shows product count by lifecycle stage
- **Sales boxplot**: Compares monthly sales volume distribution across stages

## Key Results

### Lifecycle Stage Distribution
The classification revealed a **growth-dominated** product portfolio:

- **Growth**: 5,355 products (88.5%) - Dominant stage
- **Maturity**: 550 products (9.1%) - Stable products
- **Decline**: 146 products (2.4%) - Declining products
- **Introduction**: 0 products (0.0%) - No new products identified

### Performance Metrics by Stage

#### Growth Stage (88.5% of products)
- **Avg months active**: 23.0 months
- **Avg monthly sales**: 0.5 units
- **Avg volatility**: 86.0 (high volatility)
- **Avg saturation duration**: 0.9 months

#### Maturity Stage (9.1% of products)
- **Avg months active**: 23.0 months
- **Avg monthly sales**: 0.2 units (lowest)
- **Avg volatility**: 1.2 (lowest volatility)
- **Avg saturation duration**: 0.1 months

#### Decline Stage (2.4% of products)
- **Avg months active**: 23.0 months
- **Avg monthly sales**: 0.3 units
- **Avg volatility**: 23.7 (moderate)
- **Avg saturation duration**: 0.0 months

### Peak Position Analysis
Sales peak timing distribution:
- **Mid-peak**: 2,895 products (47.8%) - Most common
- **Late-peak**: 2,056 products (34.0%) - Second most common
- **Early-peak**: 1,100 products (18.2%) - Least common

## Business Insights

### Insight 1: Growth-Dominated Portfolio
With 88.5% of products in the Growth stage, the portfolio shows **high volatility and rapid change**. This suggests:
- **Dynamic market conditions** requiring flexible inventory strategies
- **Need for frequent monitoring** and rapid response to changes
- **Opportunity for growth optimization** across most products

### Insight 2: Lack of New Products
The absence of Introduction stage products (0%) indicates:
- **No recent product launches** in the analysis period
- **Potential need for new product development**
- **Risk of portfolio aging** without fresh introductions

### Insight 3: Low Maturity Rate
Only 9.1% of products reached Maturity stage, suggesting:
- **Most products maintain high volatility** rather than stabilizing
- **Market conditions prevent products from reaching stable maturity**
- **Need for strategies to extend product lifecycles**

### Insight 4: Minimal Decline Stage
Only 2.4% of products in Decline stage indicates:
- **Strong product retention** or **early discontinuation** before decline
- **Effective lifecycle management** preventing widespread decline
- **Potential for revival strategies** for declining products

## Technical Details

### Classification Methodology
- **Multi-feature approach**: Combined 5 different lifecycle indicators
- **Business rule-based**: Applied domain-specific classification logic
- **Threshold optimization**: Used realistic thresholds based on data distribution
- **Handled edge cases**: Properly managed infinite values and missing data

### Feature Engineering Quality
- **Initial growth calculation**: Focused on first 3 months for early-stage identification
- **Saturation detection**: Used ±5% threshold for stability measurement
- **Decline tracking**: Counted consecutive negative months from recent period
- **Volatility measurement**: Standard deviation of growth rates with outlier handling

## Output File Description

### product_lifecycle_labels.csv
Complete lifecycle classification dataset with 6,051 products containing:
- **Product identification**: Product ID and category information
- **Lifecycle features**: All 5 calculated lifecycle indicators
- **Stage classification**: Assigned lifecycle stage (Growth/Maturity/Decline)
- **Performance metrics**: Sales and activity statistics

### Visualization Files
- **lifecycle_stage_counts.png**: Bar chart showing product distribution by stage
- **lifecycle_sales_boxplot.png**: Boxplot comparing sales volume across stages

## Recommendations

### Inventory Strategy
1. **High volatility management** for 88.5% Growth stage products
2. **Stability optimization** for 9.1% Maturity stage products
3. **Revival strategies** for 2.4% Decline stage products

### Product Development
1. **New product introduction** to address 0% Introduction stage
2. **Lifecycle extension** strategies for Maturity stage products
3. **Growth optimization** for dominant Growth stage products

### Risk Management
1. **Volatility monitoring** for high-risk Growth products
2. **Decline prevention** strategies for at-risk products
3. **Portfolio diversification** to reduce concentration risk 