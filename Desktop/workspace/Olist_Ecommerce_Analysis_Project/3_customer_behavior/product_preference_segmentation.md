# Product Preference Segmentation 

## Overview
- **Execution Date**: 2025-06-23
- **Update Date**: 2025-06-23
- **Main Purpose**: Segment customers based on their product category preferences, loyalty patterns, and purchasing behavior
- **Input Data**: olist_orders_dataset.csv, olist_order_items_dataset.csv, olist_products_dataset.csv, olist_customers_dataset.csv
- **Output Files**: customer_product_preferences.csv, preference visualizations

## Analysis Process

### 1. Data Integration and Filtering
I merged order, product, and customer data to create a comprehensive purchase dataset. I filtered for delivered orders only to ensure data quality, resulting in 110,197 order items across 93,358 unique customers.

### 2. Product Preference Metrics Calculation
calculated sophisticated customer-level product preference metrics:
- **Dominant Category**: Most frequently purchased product category
- **Category Loyalty Score**: Proportion of purchases in the dominant category (0.200-1.000 range)
- **Product Category Diversity**: Number of unique categories purchased (1-5 range)
- **Category Entropy**: Information theory measure of category distribution diversity
- **Repeat Category Orders**: Number of categories purchased multiple times
- **Average Price per Category**: Mean price across all purchased categories

### 3. Customer Segmentation
I used K-means clustering with rule-based refinement to create four distinct segments:
- **Specialist**: High loyalty (1.000), low diversity, focused category buyers
- **Premium Niche**: High prices ($187.74), low diversity, luxury category buyers
- **Price-Focused**: Low prices ($34.61), price-sensitive buyers
- **Other**: Mixed characteristics, diverse buying patterns

### 4. Validation and Analysis
Conducted comprehensive data validation to understand the underlying purchase behavior patterns and ensure the segmentation was meaningful and actionable.

## Key Results

### Customer Segmentation Distribution
The analysis revealed a highly concentrated customer base:

1. **Specialist** (97.7% - 91,212 customers)
   - Perfect category loyalty (1.000 score)
   - Average price: $126.44
   - Single-category focused buyers
   - Dominates the customer base

2. **Other** (0.9% - 875 customers)
   - Moderate loyalty (0.525 score)
   - Average price: $73.70
   - Mixed buying patterns
   - Small but diverse segment

3. **Premium Niche** (0.8% - 720 customers)
   - Moderate loyalty (0.521 score)
   - High average price: $187.74
   - Luxury/premium category buyers
   - High-value but small segment

4. **Price-Focused** (0.6% - 551 customers)
   - Moderate loyalty (0.522 score)
   - Low average price: $34.61
   - Price-sensitive buyers
   - Budget-conscious segment

### Product Category Preferences
The top 10 dominant categories reveal customer preferences:
1. **cama_mesa_banho** (8,757 customers) - Home & Garden
2. **beleza_saude** (8,352 customers) - Beauty & Health
3. **esporte_lazer** (7,191 customers) - Sports & Leisure
4. **informatica_acessorios** (6,293 customers) - Computer Accessories
5. **moveis_decoracao** (5,931 customers) - Furniture & Decoration
6. **utilidades_domesticas** (5,503 customers) - Home Utilities
7. **relogios_presentes** (5,326 customers) - Watches & Gifts
8. **telefonia** (3,986 customers) - Telephony
9. **automotivo** (3,703 customers) - Automotive
10. **brinquedos** (3,678 customers) - Toys

### Purchase Behavior Patterns
- **Category Loyalty**: Extremely high average loyalty (0.989)
- **Category Diversity**: Very low average diversity (1.0)
- **Single-Category Buyers**: 97.6% of customers purchase from only one category
- **Multi-Category Buyers**: Only 2.4% of customers purchase from multiple categories

## Business Insights

### Insight 1: Extreme Category Specialization
The analysis reveals that 97.6% of customers are single-category buyers, indicating extreme specialization in product preferences. This suggests that customers have very specific needs and don't cross-shop across categories, presenting both challenges and opportunities for cross-selling strategies.

### Insight 2: High Category Loyalty Reflects Limited Engagement
The average loyalty score of 0.989 indicates that customers are highly loyal to their chosen categories, but this likely reflects limited engagement rather than strong brand loyalty. Most customers make only one purchase in their preferred category, suggesting they're not returning for additional purchases.

### Insight 3: Clear Price Sensitivity Segments
The segmentation reveals distinct price sensitivity patterns:
- **Premium Niche** customers spend 5.4x more than **Price-Focused** customers
- **Specialist** customers represent the mainstream market with moderate pricing
- **Other** segment shows mixed pricing patterns

### Insight 4: Category Concentration Opportunities
The top categories (Home & Garden, Beauty & Health, Sports & Leisure) represent significant market opportunities. These categories have the highest customer concentrations and could be leveraged for targeted marketing and inventory optimization.

## Technical Details

### Segmentation Methodology
I used a hybrid approach combining K-means clustering with rule-based refinement to handle the highly skewed distribution. The methodology was designed to create meaningful segments despite the overwhelming dominance of single-category buyers.

### Feature Engineering
I created sophisticated metrics including:
- **Category loyalty score**: Measures customer focus on specific categories
- **Category entropy**: Information theory measure of purchase diversity
- **Price sensitivity indicators**: Average price per category analysis
- **Repeat purchase patterns**: Frequency of category repurchases

### Data Quality Assessment
The analysis included comprehensive validation to understand the underlying purchase behavior:
- **97.6% single-category buyers** confirmed the extreme specialization
- **91,088 customers with 0.9-1.0 loyalty scores** validated the high loyalty pattern
- **Only 2,270 multi-category buyers** highlighted the cross-selling challenge

## Output File Description

- `customer_product_preferences.csv`: Complete dataset with product preference metrics and segment assignments
- `product_segments_summary.png`: Visual distribution of customer segments
- `segment_vs_category_heatmap.png`: Heatmap showing segment-category relationships
