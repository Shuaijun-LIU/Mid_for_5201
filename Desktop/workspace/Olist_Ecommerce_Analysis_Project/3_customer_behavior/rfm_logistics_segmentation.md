# RFM Logistics Segmentation - RFM评分与物流感知分群

## Overview
- **Execution Date**: 2025-06-20
- **Update Date**: 2025-06-23
- **Main Purpose**: Perform RFM scoring and logistics-aware customer segmentation using K-means clustering
- **Input Data**: customer_logistics_features.csv (from Step 1)
- **Output Files**: rfm_segmented_customers.csv, cluster visualizations

## Analysis Process

### 1. RFM Metrics Calculation
I calculated the three core RFM metrics for each customer:
- **Recency**: Days since last order (0-713 days range)
- **Frequency**: Number of orders (1-15 orders range)
- **Monetary**: Total spending ($0.85-$13,440 range)

The reference date was set to 2018-08-29, the latest order date in the dataset.

### 2. RFM Scoring System
I implemented a sophisticated scoring system that handled the highly skewed distribution:
- **R-score**: 1-5 scale based on recency (lower days = higher score)
- **F-score**: Custom scoring based on actual order distribution (1 order=1, 2 orders=2, etc.)
- **M-score**: 1-5 scale based on spending quintiles
- **RFM Score**: Sum of individual scores (3-15 range)

### 3. Logistics Tags Addition
I added five logistics-focused behavioral tags:
- **Freight Sensitive**: Customers with >15% freight costs (70.0% of customers)
- **High Order Frequency**: Multi-order customers with <30 days between orders (1.5%)
- **Multi-Region**: Customers ordering from multiple states (0.04%)
- **High Value**: Customers above 75th percentile spending (25.0%)
- **Recent Customers**: Last order within 30 days (7.5%)

### 4. Clustering Analysis
I used K-means clustering with optimal cluster determination:
- **Optimal Clusters**: 4 (enforced minimum for business value)
- **Silhouette Score**: 0.415 (good cluster separation)
- **Features**: Standardized R, F, M scores

## Key Results

### Customer Segmentation
I identified four distinct customer segments:

1. **Recent Customers** (37.7% - 35,224 customers)
   - High recency, high monetary value
   - Average spending: $262.65
   - Low freight sensitivity (38.1%)
   - 59.0% are high-value customers

2. **About to Sleep** (31.6% - 29,501 customers)
   - Medium recency, low monetary value
   - Average spending: $51.59
   - High freight sensitivity (92.3%)
   - Low-value, price-sensitive segment

3. **Lost** (27.7% - 25,832 customers)
   - Low recency, low monetary value
   - Average spending: $66.57
   - Very high freight sensitivity (86.9%)
   - Long inactive period (404 days average)

4. **At Risk** (3.0% - 2,801 customers)
   - Medium recency, high frequency, high value
   - Average spending: $260.05
   - High freight sensitivity (80.0%)
   - Most valuable but declining segment

### RFM Score Distribution
- **R-score**: Evenly distributed (18,639-18,728 per score)
- **F-score**: Highly skewed (90,557 single-order customers)
- **M-score**: Relatively balanced (17,451-19,437 per score)
- **Total RFM**: Range 3-15, average 7.2

## Business Insights

### Insight 1: Freight Costs Drive Customer Behavior
The analysis reveals that freight costs significantly impact customer segments. 70% of customers are freight-sensitive, with the "About to Sleep" and "Lost" segments showing 92.3% and 86.9% freight sensitivity respectively. This suggests that freight optimization could be a key lever for customer retention.

### Insight 2: High-Value Customers Are Concentrated
The "Recent Customers" segment contains 59% of high-value customers despite representing only 37.7% of the total customer base. This segment has the lowest freight sensitivity (38.1%) and highest average spending ($262.65), making them the most attractive customer group.

### Insight 3: At-Risk Segment Requires Immediate Attention
The "At Risk" segment, though small (3.0%), represents the most valuable customers with multiple orders and high spending. However, their declining recency (219 days) and high freight sensitivity (80.0%) suggest they need immediate retention efforts.

### Insight 4: Customer Retention Challenge
The data shows a clear customer retention challenge with 27.7% of customers already "Lost" (404 days average recency) and 31.6% "About to Sleep" (123 days average recency). This represents nearly 60% of customers at risk of churning.

## Technical Details

### Clustering Methodology
I used K-means clustering with standardized RFM scores to ensure equal feature weighting. The optimal number of clusters was determined using silhouette analysis, with a minimum of 4 clusters enforced for business interpretability.

### Scoring System Design
The scoring system was carefully designed to handle the highly skewed frequency distribution, where 97% of customers have only 1 order. Custom frequency scoring ensured meaningful differentiation while maintaining business relevance.

### Logistics Integration
I integrated logistics-specific features to create actionable segments that consider both customer value and operational factors like freight sensitivity and geographic distribution.

## Output File Description

- `rfm_segmented_customers.csv`: Complete dataset with RFM scores, cluster assignments, and logistics tags
- `rfm_cluster_plot.png`: 3D and 2D visualizations of customer segments
- `rfm_metrics_distribution.png`: Distribution plots of RFM scores
- `rfm_cluster_evaluation.png`: Elbow curve and silhouette analysis
- `rfm_segment_comparison.png`: Comparative analysis of segment characteristics
