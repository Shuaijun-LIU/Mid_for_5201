# Week 2 - Step 3: Product Performance Overview

## Dataset Overview

This step integrates and analyzes three datasets:

- `olist_order_items_dataset.csv`: Contains item-level order information, including price and freight cost.
- `olist_products_dataset.csv`: Provides product metadata such as category.
- `product_category_name_translation.csv`: Offers English translations of product categories.

These are merged to compute per-product performance metrics.

## Product Performance Metrics

The following indicators are calculated per product:

- `order_count`: Number of orders containing the product.
- `total_sales`: Sum of prices across all orders.
- `avg_price`: Mean unit price.
- `price_std`: Standard deviation of product prices.
- `avg_freight`: Average freight cost per item.
- `category`: Product category in English.

Products are then categorized into `high`, `low`, or `average` performance groups based on the 25th and 75th percentiles of `order_count` and `total_sales`.

## Outlier Detection

Outliers are defined based on z-scores > 3 for `avg_price` or `order_count`. Key findings:

- **896 products** identified as outliers.
- Top price outliers exceed **6,000 BRL**, often with only one order.
- Top order count outliers have **over 480 orders**, concentrated in `garden_tools`, `bed_bath_table`, and `computers_accessories`.

## Key Observations

### Sales Leaders

- Top product by total sales: `1dec4c88c685d5a07bf01dcb0f8bf9f8` from **auto** category with **19,965 BRL** in sales across 35 orders.
- Other high performers include **watches_gifts**, **consoles_games**, and **computers**.

### Order Volume Leaders

- Several products had **exactly 35 orders**, likely due to platform limits or batch effects.
- Most popular categories include **furniture_decor**, **bed_bath_table**, and **cool_stuff**.

### Price Variability

- Products with high `price_std` often had **very low order volume**, suggesting promotional pricing or inconsistent catalog entries.

### Category-Level Insights

- **computers** lead with the highest average total sales per product (~3,770 BRL).
- **small_appliances_home_oven_and_coffee** and **home_appliances_2** also show strong performance with fewer products.
- **cds_dvds_musicals** shows high order count despite limited product count.

### Performance Classification Distribution

| Performance Group | Count   |
|-------------------|---------|
| High              | 5,021   |
| Average           | 20,035  |
| Low               | 6,999   |

### Visual Outputs (Saved)

- `top10_sales.png`: Top products by total sales
- `performance_distribution.png`: Distribution of high/low/average performers
- `top10_categories.png`: Best-performing categories by avg sales
- `price_distribution.png`: Overall price distribution
- `price_vs_orders.png`: Price vs order volume scatterplot
- `outlier_scatter.png`, `outlier_price_distribution.png`: Outlier diagnostics

## Insights & Implications

- A small subset of products contributes disproportionately to total revenue.
- High price variance among low-volume items may indicate catalog or supplier inconsistencies.
- Certain categories such as `garden_tools` and `computers_accessories` consistently appear in high-order or outlier lists, making them suitable for focused marketing or inventory strategies.

These results form the foundation for further customer segmentation and demand forecasting in upcoming weeks.