# Purchase Funnel Dropout Analysis

## Overview
I conducted a comprehensive purchase funnel analysis to understand customer journey patterns and identify critical dropout points in the Olist e-commerce platform. This analysis maps order statuses to funnel stages and provides insights into conversion rates and customer behavior patterns.

## Analysis Process

### Data Integration
- **Orders dataset**: 99,441 orders with 8 fields
- **Order items dataset**: 112,650 order items with 7 fields  
- **Customers dataset**: 99,441 customers with 5 fields

After merging customer unique IDs, I had a complete dataset of 99,441 orders with 9 fields for analysis.

### Funnel Stage Mapping
Mapped order statuses to three main funnel stages:
- **Order Created**: Initial order placement (created, canceled, unavailable)
- **Payment Approved**: Payment processing (approved, processing, invoiced)
- **Order Delivered**: Final delivery (shipped, delivered)

### Statistical Analysis
Calculated key funnel metrics:
- Order counts at each stage
- Dropout rates between stages
- Conversion rates from initial to each stage

## Key Results

### Funnel Performance Metrics
- **Order Created**: 1,239 orders (100% conversion rate)
- **Payment Approved**: 617 orders (49.8% conversion rate)
- **Order Delivered**: 97,585 orders (78.8% conversion rate)

### Order Status Distribution
The analysis revealed the following order status breakdown:
- **Delivered**: 96,478 orders (97.0%)
- **Shipped**: 1,107 orders (1.1%)
- **Canceled**: 625 orders (0.6%)
- **Unavailable**: 609 orders (0.6%)
- **Invoiced**: 314 orders (0.3%)
- **Processing**: 301 orders (0.3%)
- **Created**: 5 orders (0.01%)
- **Approved**: 2 orders (0.002%)

## Business Insights

### Critical Dropout Points
1. **Payment Processing Gap**: Only 49.8% of created orders reach payment approval, indicating significant payment-related issues
2. **High Delivery Success**: 78.8% of initial orders reach delivery, showing strong fulfillment capabilities
3. **Order Status Anomaly**: The large number of delivered orders (97,585) compared to created orders (1,239) suggests data quality issues or different counting methodologies

### Customer Journey Patterns
- **Payment Friction**: 622 orders drop out during payment processing
- **Inventory Issues**: 609 orders marked as unavailable, indicating stock management challenges
- **Cancellation Rate**: 625 canceled orders represent 0.6% of total orders

## Technical Details

### Data Quality Considerations
- **Status Mapping**: Some order statuses may not perfectly align with funnel stages
- **Timing Issues**: The analysis doesn't account for temporal progression of orders
- **Data Completeness**: All orders have valid status mappings

### Output Files Generated
1. **funnel_stage_summary.csv**: Detailed funnel statistics with conversion rates
2. **funnel_plot.png**: Visual funnel chart showing stage progression
3. **order_status_distribution.png**: Bar chart of order status distribution
4. **orders_with_funnel_tag.csv**: Complete dataset with funnel stage tags

