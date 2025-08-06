# Week 2 - Step 2: Sales Distribution Analysis

This section investigates how sales are distributed across geographic regions, product categories, and sellers. The goal is to understand concentration patterns, top-performing segments, and long-tail dynamics in the Olist platform.

## Dataset Overview

- Source files:
  - `olist_orders_dataset.csv`
  - `olist_order_items_dataset.csv`
  - `olist_customers_dataset.csv`
  - `olist_products_dataset.csv`
  - `product_category_name_translation.csv`
- Filtered to include only `delivered` orders
- Joined on keys: `order_id`, `customer_id`, `product_id`, and `product_category_name`
- Timeframe covered: September 2016 to August 2018

## Sales by State

### Key Metrics
- **Top State by Revenue**: São Paulo (SP) with 38.33% of total revenue
- **Top State by Order Count**: São Paulo (SP) with 41.98% of total orders
- **Other High-Contribution States**:
  - Rio de Janeiro (RJ): 13.31% revenue
  - Minas Gerais (MG): 11.74% revenue
- **Low-Contribution States**: Roraima (RR), Amapá (AP), and Acre (AC) each with < 0.1% revenue

### Observation
Sales are heavily concentrated in Brazil’s Southeast region, particularly SP and RJ. The platform has limited reach in the North and Northeast.

## Sales by Product Category

### Key Metrics
- **Top Categories by Revenue**:
  - `health_beauty`: 9.45%
  - `watches_gifts`: 8.94%
  - `bed_bath_table`: 7.85%
- **Revenue Skew**: The top 10 categories account for a majority of sales.
- **Category Diversity**: Over 70 unique categories with varied average order sizes.

### Observation
Product demand is skewed toward personal and home goods. Categories like `computers` and `telephony` have lower volumes but higher per-order values.

## Sales by Seller

### Key Metrics
- **Top Seller Revenue**: R$226,987.93 from 1,124 orders
- **Revenue Skew**: Long-tail effect; most sellers had <10 orders
- **Bottom 10 Sellers**: Generated < R$12 total revenue

### Observation
The seller distribution follows a power law: a small set of sellers dominate, while most generate minimal sales. There may be opportunities to consolidate or better support small sellers.

## Key Observations

1. The Southeast dominates in both revenue and customer base, suggesting regionally-focused logistics and marketing strategies may yield better returns.
2. Product categories show varying sales profiles: some high-volume/low-value, others low-volume/high-value.
3. Seller distribution is highly skewed, indicating potential inefficiencies in seller onboarding, support, or market exposure.

## Visual Output Summary

| File | Description |
|------|-------------|
| `top10_state_sales.png` | Bar chart of top 10 states by revenue |
| `top10_category_state_heatmap.png` | Heatmap: product category sales by state |
| `top_sellers.png` | Top 10 sellers by revenue |
| `bottom_sellers.png` | Bottom 10 sellers by revenue |

All figures are saved in the `output/sales_distribution/` directory for integration into dashboards and the final white paper.