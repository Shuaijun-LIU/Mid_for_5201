# Week 2 - Step 1: Time Series Aggregation Analysis

This section analyzes the evolution of key performance indicators (KPIs) on the Olist platform over time. The goal is to understand growth trends, customer dynamics, and order value stability through daily, weekly, and monthly aggregations.

## Dataset Overview

- Source files:
  - `olist_orders_dataset.csv`
  - `olist_order_items_dataset.csv`
  - `olist_order_payments_dataset.csv`
- Filtered to include only `delivered` orders
- Timeframe covered: September 2016 to August 2018

## Daily Aggregation Summary

### Order Count
- Mean: 161.12 orders/day
- Maximum: 1,386 orders on 2017-11-24
- Observation: Order count grew from near zero in late 2016 to over 1,000 per day by 2018.

### Total Revenue
- Mean: BRL 19,347.29/day
- Maximum: BRL 152,751.87 on 2017-11-24
- Observation: Revenue growth closely mirrored order count.

### Average Order Value (AOV)
- Mean: BRL 121.70
- Range: Stable between BRL 110–130
- Maximum: BRL 432.55 on 2017-01-06 (possible outlier)

### Unique Customers
- Mean: 135.12 customers/day
- Maximum: 1,147 on 2017-11-24

Conclusion: November 24, 2017 was a key high-traffic day, likely corresponding to Black Friday. It is a candidate for further holiday-based analysis.

## Weekly Aggregation Summary

### Order Count
- Mean: 1,116.87 orders/week
- Maximum: 3,565 orders during the week of 2017-11-26

### Revenue
- Mean: BRL 134,116.15/week
- Maximum: BRL 410,624.32

### AOV
- Mean: BRL 118.13
- Maximum: BRL 158.93 during the week of 2017-09-03

### Unique Customers
- Mean: 936.68/week
- Maximum: 2,915 during the week of 2017-11-26

## Monthly Aggregation Summary

### Order Count
- Mean: 4,793 orders/month
- Maximum: 8,813 on 2017-11-30

### Revenue
- Mean: BRL 575,581.82/month
- Maximum: BRL 1,022,688.86

### AOV
- Mean: BRL 113.17
- Maximum: BRL 133.42 on 2017-04-30

### Unique Customers
- Mean: 4,019.92 customers/month
- Maximum: 7,289 on 2017-11-30

## Key Observations

1. November 2017 represents a significant sales and engagement peak across all metrics. It likely aligns with Black Friday promotions.
2. Average Order Value (AOV) remains remarkably stable, showing that pricing strategy and product mix did not fluctuate despite platform growth.
3. The number of unique customers scales proportionally with order count, suggesting that platform growth was largely driven by new user acquisition rather than repeat customers.
4. Initial low-volume periods in 2016–2017 are consistent with a platform ramp-up or launch phase.

## Visual Output Summary

| File | Description |
|------|-------------|
| `order_count_daily.png` | Daily order volume trend |
| `total_revenue_daily.png` | Daily revenue growth |
| `aov_weekly.png` | Weekly average order value |
| `unique_customers_monthly.png` | Monthly customer growth |

All figures are saved in the `output/` directory for inclusion in dashboards and reports.

## Deliverables

- Aggregated statistics printed to the console for daily, weekly, and monthly frequencies
- Visualizations saved in `.png` format
- Results aligned with business questions about order behavior, platform growth, and customer engagement