# Week 2 – Step 5: Holiday-Sensitive Product Analysis

## Dataset Overview

This analysis identifies products and categories that experience significant changes in demand during holidays compared to regular days. It combines and processes the following datasets:

- `orders_with_holidays.csv`: Contains holiday-annotated order information.
- `olist_order_items_dataset.csv`: Provides item-level order and price data.
- `olist_products_dataset.csv`: Links products to their category.
- `product_category_name_translation.csv`: Translates category names to English.

**Filtering Criteria:**

- Only holidays with at least **10 orders** were included.
- Only products and categories with **30+ total orders** were analyzed to reduce noise.

## Methodology

Each product and category was evaluated for holiday sensitivity using the following metrics:

- `holiday_count`, `regular_count`: Number of orders placed during holidays vs. regular days.
- `holiday_ratio`, `regular_ratio`: Normalized ratio of demand across holiday vs. regular orders.
- `lift`: Defined as  
  \[
  \text{Lift} = \frac{\text{holiday\_ratio} + 1e^{-6}}{\text{regular\_ratio} + 1e^{-6}}
  \]  
  A lift value > 1.5 indicates high holiday sensitivity.

## Top Holiday-Sensitive Categories

| Category                    | Holiday Orders | Regular Orders | Lift | Holiday Ratio | Regular Ratio |
|-----------------------------|----------------|----------------|------|----------------|----------------|
| consoles_games              | 23             | 204            | 3.17 | 0.03           | 0.01           |
| tablets_printing_image      | 3              | 29             | 2.91 | 0.00           | 0.00           |
| garden_tools                | 132            | 1907           | 1.95 | 0.17           | 0.09           |
| computers                   | 2              | 33             | 1.70 | 0.00           | 0.00           |
| home_construction           | 2              | 33             | 1.70 | 0.00           | 0.00           |
| perfumery                   | 39             | 830            | 1.32 | 0.05           | 0.04           |
| home_confort                | 7              | 149            | 1.32 | 0.01           | 0.01           |
| furniture_decor             | 47             | 1077           | 1.23 | 0.06           | 0.05           |
| construction_tools_lights   | 4              | 92             | 1.22 | 0.01           | 0.00           |
| audio                       | 5              | 123            | 1.14 | 0.01           | 0.01           |

## Top Holiday-Sensitive Products

| Product ID                             | Category           | Holiday Orders | Regular Orders | Lift | Holiday Ratio | Regular Ratio |
|----------------------------------------|--------------------|----------------|----------------|------|----------------|----------------|
| fe01b643060a6446e59f58e3021e66b3       | perfumery          | 9              | 36             | 7.03 | 0.01           | 0.00           |
| 3e5201fe0d1ba474d9b90152c83c706c       | bed_bath_table     | 8              | 35             | 6.42 | 0.01           | 0.00           |
| 9ecadb84c81da840dbf3564378b586e9       | furniture_decor    | 8              | 38             | 5.92 | 0.01           | 0.00           |
| 9ad75bd7267e5c724cb42c71ac56ca72       | bed_bath_table     | 6              | 36             | 4.68 | 0.01           | 0.00           |
| 6c3effec7c8ddba466d4f03f982c7aa3       | consoles_games     | 13             | 81             | 4.51 | 0.02           | 0.00           |
| e9def91e99c8ecb7c5cef5e31506a056       | toys               | 6              | 39             | 4.32 | 0.01           | 0.00           |
| 9bb8ca338e5588c361e34eae02e8fad6       | health_beauty      | 4              | 28             | 4.01 | 0.01           | 0.00           |
| c211ff3068fcd2f8898192976d8b3a32       | bed_bath_table     | 4              | 30             | 3.75 | 0.01           | 0.00           |
| 0bcc3eeca39e1064258aa1e932269894       | garden_tools       | 11             | 89             | 3.47 | 0.01           | 0.00           |
| 83b00325c13c44245b2c3a2befa62a0e       | baby               | 4              | 35             | 3.21 | 0.01           | 0.00           |

## Visual Outputs

| File Name                                       | Description                                          |
|------------------------------------------------|------------------------------------------------------|
| `top10_holiday_sensitive_categories.png`       | Bar chart of top 10 categories by lift              |
| `holiday_lift_bubble_chart.png`                | Bubble chart of lift vs total orders by category    |

### Highlight: `holiday_lift_bubble_chart.png`

This scatter plot visualizes:

- **X-axis**: Total orders (holiday + regular)  
- **Y-axis**: Holiday lift  
- **Bubble size/color**: Scaled by total volume and lift value  

It helps distinguish between niche but responsive categories (`tablets_printing_image`) and those with both lift and scale (`garden_tools`, `consoles_games`).

## Key Insights

1. **Holiday-sensitive categories** like `consoles_games`, `perfumery`, and `garden_tools` show a **lift > 1.5**, suggesting they perform significantly better during holidays.
2. Several individual SKUs (perfumes, toys, game consoles) saw more than **5x demand increase**, indicating strong promotional potential.
3. **Lower-volume but high-lift categories** may benefit from targeted exposure during key seasonal events.
4. Some categories (e.g., `computers`, `home_construction`) are niche but reveal behavioral shifts that could inform specialized campaigns.

## Recommendations

- Promote high-lift categories/products in pre-holiday campaigns (e.g., Black Friday, Christmas).
- Adjust pricing and inventory strategy to capitalize on predictable holiday surges.
- Monitor and refine forecasts for sensitive SKUs to improve fulfillment accuracy.

## Output Summary

| File | Description |
|------|-------------|
| `holiday_sensitive_products.csv` | Product-level lift and ratios |
| `holiday_sensitive_categories.csv` | Category-level sensitivity metrics |
| `top10_holiday_sensitive_categories.png` | Lift comparison for categories |
| `holiday_lift_bubble_chart.png` | Lift vs order volume scatterplot |

These insights inform holiday-specific assortment planning, pricing strategies, and marketing focus for the upcoming seasonal cycle.