# Product Insights Merge Summary

## Overview
- **Script**: `merge_product_insights.py`
- **Purpose**: Merge outputs from 6 sub-tasks into unified product warehouse dataset
- **Products**: 6,051 total products
- **Columns**: 10 key metrics

## Output File

### `product_warehouse_summary.csv`
Comprehensive dataset combining sales analysis, lifecycle classification, inventory efficiency, risk assessment, and policy recommendations.

**Key Fields:**
1. `product_id` - Product unique identifier
2. `lifecycle_stage` - Product lifecycle stage (Growth/Maturity/Decline)
3. `avg_monthly_sales` - Average monthly sales volume
4. `volatility_score` - Sales volatility measure
5. `inventory_turnover_rate` - Inventory turnover efficiency
6. `avg_holding_cost_per_unit` - Unit holding cost
7. `stock_risk_flag` - Stock risk classification (Normal/Stockout/Overstock)
8. `suggested_policy_type` - Recommended inventory policy
9. `fulfillment_success_rate` - Order fulfillment rate
10. `backlog_volume` - Backlog order volume

## Key Statistics
- **Growth Stage**: 5,355 products (88.5%)
- **Maturity Stage**: 550 products (9.1%)
- **Decline Stage**: 146 products (2.4%)
- **High-frequency Policy**: 3,985 products (65.9%)
- **Risk Products**: 352 products (5.8%)
