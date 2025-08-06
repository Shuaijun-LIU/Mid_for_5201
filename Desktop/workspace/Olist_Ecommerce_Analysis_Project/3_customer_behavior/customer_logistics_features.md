# Customer Logistics Features - Customer Behavior Feature Engineering

## Overview
- **Execution Date**: 2025-06-18
- **Update Date**: 2025-06-22
- **Main Purpose**: Extract customer-level behavioral features from order, product, and customer data
- **Input Data**: olist_orders_dataset.csv, olist_order_items_dataset.csv, olist_products_dataset.csv, olist_customers_dataset.csv
- **Output Files**: customer_logistics_features.csv

## Analysis Process

### 1. Data Loading and Preprocessing
I started by loading four core datasets: orders, order items, products, and customers. The data preprocessing involved filtering for delivered orders only, which gave me a solid 97.02% delivery rate. I then merged order items with product category information and linked everything to customer data to build a comprehensive dataset.

### 2. Order-Level Feature Calculation
At the order level, calculated total order value, freight costs, and freight percentages. The results showed an average order value of $137.04 with freight costs averaging $22.79 per order.

### 3. Customer-Level Feature Engineering
I built 13 customer-level features covering multiple dimensions:
- **Order behavior**: order count, total spending, average order value
- **Temporal features**: first/last purchase dates, days between orders
- **Geographic features**: number of states purchased from
- **Freight features**: high freight order ratios
- **Category features**: product category diversity

## Key Results

### Data Statistics
- **Total Customers**: 93,358
- **Total Orders**: 96,478
- **Data Time Range**: 2016-09-15 to 2018-08-29
- **Product Categories**: 74 unique categories

### Customer Behavior Patterns
- **Average Order Count**: 1.03 (indicating mostly one-time buyers)
- **Average Total Spending**: $141.62
- **Average Order Value**: $137.51
- **Average Freight Percentage**: 30.84%
- **Average Category Diversity**: 1.03 (highly concentrated)

### Customer Distribution Insights
- **One-time Buyers**: 90,557 (97.0%)
- **Repeat Buyers**: 2,801 (3.0%)
- **High Freight Customers**: 52,225 (55.9%)
- **Multi-state Buyers**: 37 (0.04%)

## Business Insights

### Insight 1: Highly Concentrated Purchase Behavior
The data reveals a significant customer retention challenge - 97% of customers make only one purchase. This suggests that Olist needs to focus heavily on customer retention strategies. Additionally, the average category diversity of just 1.03 indicates customers tend to stick within single product categories, limiting cross-selling opportunities.

### Insight 2: Freight Costs Are a Major Factor
Freight costs represent 30.84% of average order value, which is quite substantial and likely influences customer purchase decisions significantly. More than half of customers (55.9%) have experienced orders with high freight ratios, making freight optimization a critical factor for improving customer experience.

### Insight 3: High-Value Customer Characteristics
I identified some interesting patterns among high-value customers. The highest-spending customer made a single purchase of $13,440, while the most frequent buyer made 15 purchases totaling $714.63. Interestingly, frequent buyers tend to have lower average order values, suggesting they might be price-sensitive customers.

### Insight 4: Strong Data Quality
The data quality is excellent with a 97.02% delivery rate, indicating stable business operations. The two-year time span provides good representation, and coverage across 74 product categories shows comprehensive category coverage.

## Technical Details

### Feature Engineering Approach
I used time-series based customer behavior analysis with multi-dimensional feature aggregation across orders, time, geography, and categories. The approach involved statistical calculations including means, proportions, and diversity measures.

### Key Parameters
- High freight threshold: 20%
- Time window: 2016-2018
- Data filter: Delivered orders only

### Data Quality Assessment
- Data completeness: 97.02% delivery rate
- Feature coverage: 13 customer-level features
- Customer coverage: 93,358 unique customers

## Output File Description

- `customer_logistics_features.csv`: Complete dataset with 13 customer-level behavioral features
  - Basic features: order count, total spending, average order value
  - Temporal features: order intervals, first/last purchase dates
  - Geographic features: number of purchase states
  - Freight features: high freight order ratios
  - Category features: product category diversity
