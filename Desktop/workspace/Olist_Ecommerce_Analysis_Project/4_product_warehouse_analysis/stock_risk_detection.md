# Stock Risk Detection

## Overview
- **Execution Date**: 2025-06-27
- **Update Date**: 2025-07-01
- **Main Purpose**: Identify products at risk of overstock or stockout based on sales velocity, turnover rates, and lifecycle stage analysis
- **Input Data**: Inventory efficiency metrics (6,051 products), Lifecycle labels (6,051 products), Order items (112,650 records)
- **Output Files**: product_stock_risk_flags.csv (6,051 products), 3 visualization files

## Analysis Process

### 1. Risk Threshold Calculation
Established data-driven risk thresholds using quantile-based analysis:
- **Overstock turnover threshold**: 2.00 (low turnover indicator)
- **Overstock holding cost threshold**: $61.17 (90th percentile)
- **Stockout turnover threshold**: 3.47 (90th percentile)
- **Stockout delivery time threshold**: 11.25 days (median)

### 2. Risk Classification Logic
Applied business rules to categorize products into three risk categories:

#### Overstock Risk Criteria
- **Lifecycle stage**: Maturity or Decline
- **Turnover rate**: < 2.00 units/month
- **Holding cost**: > $61.17

#### Stockout Risk Criteria
- **Lifecycle stage**: Growth or Introduction
- **Turnover rate**: > 3.47 units/month
- **Delivery time**: < 11.25 days

#### Stable Category
- Products not meeting either risk criteria

### 3. Risk Analysis and Visualization
Generated comprehensive risk analysis with:
- **Risk distribution charts**: Overall product risk categorization
- **Lifecycle stage heatmap**: Risk patterns across product stages
- **Scatter plot analysis**: Turnover vs. holding cost with risk coloring

## Key Results

### Overall Risk Distribution
The analysis revealed a **predominantly stable** product portfolio:

- **Stable**: 5,699 products (94.2%) - Low risk
- **Stockout Risk**: 284 products (4.7%) - High demand risk
- **Overstock Risk**: 68 products (1.1%) - Excess inventory risk

### Risk Distribution by Lifecycle Stage

#### Growth Stage (5,355 products)
- **Stable**: 5,071 products (94.7%)
- **Stockout Risk**: 284 products (5.3%)
- **Overstock Risk**: 0 products (0.0%)

#### Maturity Stage (550 products)
- **Stable**: 496 products (90.2%)
- **Overstock Risk**: 54 products (9.8%)
- **Stockout Risk**: 0 products (0.0%)

#### Decline Stage (146 products)
- **Stable**: 132 products (90.4%)
- **Overstock Risk**: 14 products (9.6%)
- **Stockout Risk**: 0 products (0.0%)

### High-Risk Product Analysis

#### Top Overstock Risk Products
Highest holding cost products at overstock risk:
1. **0c8862859cb952ee81428f80dc8140d9**: $499.49 holding cost, 0.45 turnover
2. **68fe8893052a044100e60cd0c8c6b274**: $326.40 holding cost, 0.36 turnover
3. **c62dee961914cc2e49239963b04258ec**: $288.28 holding cost, 0.84 turnover

#### Top Stockout Risk Products
Highest turnover products at stockout risk:
1. **f1c7f353075ce59d8a6f3cf58f419c9c**: 37.86 turnover, 9.2 days delivery
2. **437c05a395e9e47f9762e677a7068ce7**: 25.82 turnover, 11.0 days delivery
3. **fbce4c4cb307679d89a3bf3d3bb353b9**: 19.85 turnover, 9.5 days delivery

### Risk Performance Metrics
- **Overstock risk products**: 68 products (avg holding cost: $124.41)
- **Stockout risk products**: 284 products (avg turnover: 6.13)
- **Risk concentration**: 94.2% of products are stable

## Business Insights

### Insight 1: Low Overall Risk Profile
With 94.2% of products classified as stable, the portfolio shows **strong inventory management practices**. This suggests:
- **Effective demand forecasting** across most products
- **Balanced inventory levels** preventing major risks
- **Robust supply chain management** maintaining stability

### Insight 2: Growth Stage Stockout Risk
All 284 stockout risk products are in the Growth stage, indicating:
- **High demand volatility** for growing products
- **Supply chain challenges** meeting rapid growth
- **Need for dynamic inventory adjustment** for growth products

### Insight 3: Maturity Stage Overstock Concentration
54 out of 68 overstock risk products (79.4%) are in Maturity stage, suggesting:
- **Conservative inventory policies** for mature products
- **Declining demand** not reflected in inventory levels
- **Opportunity for inventory reduction** in mature categories

### Insight 4: Decline Stage Risk Pattern
14 overstock risk products in Decline stage (9.6% of decline products) indicates:
- **Delayed inventory adjustment** to declining demand
- **Need for faster liquidation** strategies
- **Potential for early discontinuation** decisions

### Insight 5: Risk Threshold Effectiveness
The quantile-based thresholds effectively identified:
- **High-value risk products** requiring immediate attention
- **Clear risk patterns** across lifecycle stages
- **Actionable risk categories** for inventory management

## Technical Details

### Risk Calculation Methodology
- **Quantile-based thresholds**: Used 90th percentile and median for risk boundaries
- **Multi-criteria approach**: Combined lifecycle stage, turnover, and cost metrics
- **Business rule validation**: Applied domain-specific logic for risk classification
- **Data quality assurance**: Handled missing values and outliers appropriately

### Risk Classification Quality
- **Comprehensive coverage**: All 6,051 products classified
- **Balanced distribution**: 94.2% stable, 5.8% at risk
- **Stage-specific patterns**: Clear risk concentration by lifecycle stage
- **Actionable results**: Specific products identified for intervention

## Output File Description

### product_stock_risk_flags.csv
Complete risk classification dataset with 6,051 products containing:
- **Product identification**: Product ID and category information
- **Lifecycle classification**: Assigned lifecycle stage
- **Performance metrics**: Turnover rate, delivery time, holding cost
- **Risk classification**: Overstock Risk, Stockout Risk, or Stable
- **Risk indicators**: All calculated risk metrics and thresholds

### Visualization Files
- **risk_distribution.png**: Bar chart showing overall risk distribution
- **risk_summary_by_lifecycle.png**: Heatmap of risk distribution by lifecycle stage
- **stock_risk_scatter.png**: Scatter plot of turnover vs. holding cost with risk coloring

## Recommendations

### Immediate Actions
1. **Address 68 overstock risk products** with inventory reduction strategies
2. **Monitor 284 stockout risk products** for supply chain optimization
3. **Review Maturity stage inventory** policies for overstock prevention

### Strategic Planning
1. **Develop dynamic inventory policies** for Growth stage products
2. **Implement faster liquidation** for Decline stage overstock
3. **Establish risk monitoring** systems for early detection

### Process Improvement
1. **Optimize demand forecasting** for high-turnover products
2. **Improve supply chain responsiveness** for growth products
3. **Enhance inventory planning** for lifecycle stage transitions 