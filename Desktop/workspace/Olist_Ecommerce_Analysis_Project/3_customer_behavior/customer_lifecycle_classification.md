# Customer Lifecycle Classification - 客户生命周期与流失分类

## Overview
- **Execution Date**: 2025-06-20
- **Update Date**: 2025-06-23
- **Main Purpose**: Classify customers into lifecycle stages and churn risk levels based on recency and frequency patterns
- **Input Data**: customer_logistics_features.csv, rfm_segmented_customers.csv
- **Output Files**: customer_lifecycle.csv, lifecycle visualizations

## Analysis Process

### 1. Data Integration and Recency Calculation
I merged customer logistics features with RFM segments to create a comprehensive customer profile. Using the reference date of 2018-08-29, I calculated recency_days for each customer, which ranged from 0 to 713 days since their last purchase.

### 2. Lifecycle Stage Classification
I implemented a time-based lifecycle classification system:
- **New**: ≤30 days since last order
- **Active**: ≤90 days with multiple orders
- **At-Risk**: 90-180 days since last order
- **Churned**: >180 days since last order
- **Unknown**: Edge cases requiring special attention

### 3. Churn Risk Assessment
I mapped lifecycle stages to churn risk levels:
- **Low Risk**: New and Active customers
- **Medium Risk**: At-Risk customers
- **High Risk**: Churned customers
- **Unknown**: Customers requiring further analysis

### 4. Logistics Priority Tagging
I identified customers with high freight values (>$20) who are in Active or At-Risk stages, creating a priority list for logistics optimization and retention efforts.

## Key Results

### Lifecycle Stage Distribution
The analysis revealed a critical customer retention challenge:

1. **Churned Customers** (58.9% - 55,006 customers)
   - High-risk segment requiring reactivation
   - Average 1.5 days since first purchase
   - Long inactive period (>180 days)

2. **At-Risk Customers** (21.2% - 19,762 customers)
   - Medium-risk segment needing immediate attention
   - Average 3.5 days since first purchase
   - Recent activity but declining engagement

3. **Unknown Customers** (12.0% - 11,184 customers)
   - Edge cases requiring special analysis
   - 0 days since first purchase (data quality issue)
   - Need manual review and classification

4. **New Customers** (7.5% - 7,002 customers)
   - Low-risk segment with recent activity
   - Average 5.0 days since first purchase
   - High potential for retention

5. **Active Customers** (0.4% - 404 customers)
   - Lowest-risk segment
   - Average 145.1 days since first purchase
   - Strong engagement and loyalty

### Churn Risk Distribution
- **High Risk**: 58.9% (55,006 customers) - Churned segment
- **Medium Risk**: 21.2% (19,762 customers) - At-Risk segment
- **Low Risk**: 7.9% (7,406 customers) - New and Active segments
- **Unknown**: 12.0% (11,184 customers) - Requires investigation

### Logistics Priority Analysis
- **Freight Priority Customers**: 7.9% (7,341 customers)
- These customers have high freight values and are in Active or At-Risk stages
- Represent immediate opportunities for logistics optimization and retention

## Business Insights

### Insight 1: Critical Customer Retention Challenge
The analysis reveals a severe customer retention problem with 58.9% of customers already churned and 21.2% at risk. This means 80.1% of the customer base is either lost or at risk of churning, representing a massive opportunity cost and highlighting the urgent need for retention strategies.

### Insight 2: Short Customer Lifespan
The extremely short average time since first purchase (1.5-5.0 days) across most segments indicates that customers are churning very quickly after their initial purchase. This suggests fundamental issues with the post-purchase experience, product quality, or customer service.

### Insight 3: Limited Active Customer Base
Only 0.4% of customers (404) are classified as Active, representing a tiny fraction of the customer base. This indicates that Olist struggles to convert one-time buyers into repeat customers, which is essential for sustainable business growth.

### Insight 4: Data Quality Issues
The 12.0% Unknown segment with 0 days since first purchase suggests data quality issues that need to be addressed. This could be due to missing timestamps, data processing errors, or edge cases in the classification logic.

## Technical Details

### Classification Methodology
I used a time-based classification system that considers both recency and frequency patterns. The system was designed to handle the highly skewed distribution where most customers have only one order, while still providing meaningful segmentation for business action.

### Risk Assessment Logic
The churn risk mapping was designed to align with business priorities:
- **Low Risk**: Customers with recent activity and potential for retention
- **Medium Risk**: Customers showing signs of decline but still reachable
- **High Risk**: Customers requiring reactivation campaigns

### Freight Priority Identification
I identified customers with high freight values who are still in active or at-risk stages, creating a targeted list for logistics optimization that could improve retention rates.

## Output File Description

- `customer_lifecycle.csv`: Complete dataset with lifecycle stages, churn risk levels, and logistics tags
- `lifecycle_distribution.png`: Visual distribution of customer lifecycle stages
- `lifecycle_freight_boxplot.png`: Freight value analysis by lifecycle stage

