# Olist E-commerce Dataset Schema Summary

## Overview
This document provides a comprehensive overview of the Olist E-commerce dataset structure, including table relationships, field descriptions, and data types. The dataset represents a Brazilian e-commerce platform's operations from 2016 to 2018, containing over 100,000 orders with associated customer, product, seller, and review information.

## Database Tables

### 1. Orders (olist_orders_dataset.csv)
**Primary Key**: order_id
**Description**: Core table containing order information and status tracking

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| order_id | VARCHAR | Unique order identifier | Primary Key |
| customer_id | VARCHAR | Customer identifier | Foreign Key to customers |
| order_status | VARCHAR | Current order status | e.g., delivered, invoiced |
| order_purchase_timestamp | TIMESTAMP | Order purchase date and time | Format: YYYY-MM-DD HH:MM:SS |
| order_approved_at | TIMESTAMP | Order approval date and time | Format: YYYY-MM-DD HH:MM:SS |
| order_delivered_carrier_date | TIMESTAMP | Date when order was delivered to carrier | Format: YYYY-MM-DD HH:MM:SS |
| order_delivered_customer_date | TIMESTAMP | Date when order was delivered to customer | Format: YYYY-MM-DD HH:MM:SS |
| order_estimated_delivery_date | TIMESTAMP | Estimated delivery date | Format: YYYY-MM-DD HH:MM:SS |

### 2. Order Items (olist_order_items_dataset.csv)
**Primary Key**: (order_id, order_item_id)
**Description**: Detailed information about items in each order

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| order_id | VARCHAR | Order identifier | Foreign Key to orders |
| order_item_id | INTEGER | Item sequence number within order | Composite Primary Key |
| product_id | VARCHAR | Product identifier | Foreign Key to products |
| seller_id | VARCHAR | Seller identifier | Foreign Key to sellers |
| shipping_limit_date | TIMESTAMP | Shipping deadline | Format: YYYY-MM-DD HH:MM:SS |
| price | DECIMAL | Item price | 2 decimal places |
| freight_value | DECIMAL | Shipping cost | 2 decimal places |

### 3. Products (olist_products_dataset.csv)
**Primary Key**: product_id
**Description**: Product catalog information

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| product_id | VARCHAR | Unique product identifier | Primary Key |
| product_category_name | VARCHAR | Product category in Portuguese | Foreign Key to category translation |
| product_name_lenght | INTEGER | Length of product name | Character count |
| product_description_lenght | INTEGER | Length of product description | Character count |
| product_photos_qty | INTEGER | Number of product photos | Range: 1-4 |
| product_weight_g | INTEGER | Product weight in grams | |
| product_length_cm | INTEGER | Product length in centimeters | |
| product_height_cm | INTEGER | Product height in centimeters | |
| product_width_cm | INTEGER | Product width in centimeters | |

### 4. Customers (olist_customers_dataset.csv)
**Primary Key**: customer_id
**Description**: Customer information

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| customer_id | VARCHAR | Unique customer identifier | Primary Key |
| customer_unique_id | VARCHAR | Anonymous customer identifier | For privacy protection |
| customer_zip_code_prefix | VARCHAR | Customer ZIP code prefix | 5 digits |
| customer_city | VARCHAR | Customer city | Lowercase, no accents |
| customer_state | VARCHAR | Customer state | 2-letter code |

### 5. Order Reviews (olist_order_reviews_dataset.csv)
**Primary Key**: review_id
**Description**: Customer reviews and ratings

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| review_id | VARCHAR | Unique review identifier | Primary Key |
| order_id | VARCHAR | Order identifier | Foreign Key to orders |
| review_score | INTEGER | Rating score | Range: 1-5 |
| review_comment_title | VARCHAR | Review title | Optional |
| review_comment_message | TEXT | Review text | Optional, in Portuguese |
| review_creation_date | TIMESTAMP | Review creation date | Format: YYYY-MM-DD HH:MM:SS |
| review_answer_timestamp | TIMESTAMP | Review response date | Format: YYYY-MM-DD HH:MM:SS |

### 6. Order Payments (olist_order_payments_dataset.csv)
**Primary Key**: (order_id, payment_sequential)
**Description**: Payment information for orders

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| order_id | VARCHAR | Order identifier | Foreign Key to orders |
| payment_sequential | INTEGER | Payment sequence number | Composite Primary Key |
| payment_type | VARCHAR | Payment method | e.g., credit_card, boleto |
| payment_installments | INTEGER | Number of installments | Range: 1-8 |
| payment_value | DECIMAL | Payment amount | 2 decimal places |

### 7. Sellers (olist_sellers_dataset.csv)
**Primary Key**: seller_id
**Description**: Seller information

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| seller_id | VARCHAR | Unique seller identifier | Primary Key |
| seller_zip_code_prefix | VARCHAR | Seller ZIP code prefix | 5 digits |
| seller_city | VARCHAR | Seller city | Lowercase, no accents |
| seller_state | VARCHAR | Seller state | 2-letter code |

### 8. Product Category Translation (product_category_name_translation.csv)
**Primary Key**: product_category_name
**Description**: Product category name translations

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| product_category_name | VARCHAR | Category name in Portuguese | Primary Key |
| product_category_name_english | VARCHAR | Category name in English | |

### 9. Geolocation (olist_geolocation_dataset.csv)
**Description**: Brazilian ZIP code prefix geolocation data

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| geolocation_zip_code_prefix | VARCHAR | ZIP code prefix | 5 digits |
| geolocation_lat | DECIMAL | Latitude | 6 decimal places |
| geolocation_lng | DECIMAL | Longitude | 6 decimal places |
| geolocation_city | VARCHAR | City name | May include accents |
| geolocation_state | VARCHAR | State code | 2-letter code |

## Key Relationships

1. **Orders → Customers**
   - One-to-one relationship
   - Linked by customer_id

2. **Orders → Order Items**
   - One-to-many relationship
   - One order can have multiple items

3. **Order Items → Products**
   - Many-to-one relationship
   - Multiple items can reference the same product

4. **Order Items → Sellers**
   - Many-to-one relationship
   - Multiple items can be sold by the same seller

5. **Orders → Order Reviews**
   - One-to-one relationship
   - Each order can have one review

6. **Orders → Order Payments**
   - One-to-many relationship
   - One order can have multiple payment records

7. **Products → Category Translation**
   - Many-to-one relationship
   - Multiple products can belong to the same category

## Data Quality Notes

1. **Time Formats**
   - All timestamps are in YYYY-MM-DD HH:MM:SS format
   - Some timestamps may be NULL for certain order statuses

2. **Missing Values**
   - Review comments and titles can be NULL
   - Some delivery dates may be NULL for non-delivered orders

3. **Data Consistency**
   - All monetary values use 2 decimal places
   - All measurements use metric system
   - City names are stored in lowercase
   - State codes are consistently 2 letters

4. **Language Considerations**
   - Product categories are in Portuguese
   - Review comments are in Portuguese
   - Category translations available in English

## Usage Considerations

1. **Privacy**
   - Customer and seller information is anonymized
   - ZIP codes are stored as prefixes only

2. **Performance**
   - Large number of records in orders and order items
   - Consider indexing on frequently joined fields

3. **Analysis Potential**
   - Rich temporal data for time series analysis
   - Geographic data for spatial analysis
   - Text data for sentiment analysis
   - Comprehensive order lifecycle tracking

## Data Exploration Findings

### 1. Dataset Size Overview
| Table Name | Number of Records | Number of Columns |
|------------|-------------------|-------------------|
| Orders | 99,441 | 8 |
| Order Items | 112,650 | 7 |
| Products | 32,951 | 9 |
| Customers | 99,441 | 5 |
| Reviews | 99,224 | 7 |
| Payments | 103,886 | 5 |
| Sellers | 3,095 | 4 |
| Category Translation | 71 | 2 |
| Geolocation | 1,000,163 | 5 |

### 2. Missing Values Analysis
| Dataset                      | Column Name                     | Missing Count | Missing Rate | Variable Type             | Description                                             |
|-----------------------------|----------------------------------|----------------|---------------|----------------------------|---------------------------------------------------------|
| olist_orders_dataset.csv    | order_approved_at               | 160            | 0.16%         | Timestamp (order process) | Order not approved                                       |
|                             | order_delivered_carrier_date    | 1,783          | 1.79%         | Timestamp (shipping)      | Order not shipped or was canceled                       |
|                             | order_delivered_customer_date   | 2,965          | 2.98%         | Timestamp (delivery)      | Order not yet delivered to customer                     |
| olist_products_dataset.csv  | product_category_name           | 610            | 1.85%         | Categorical (string)      | Missing product category, can fill with `'unknown'`     |
|                             | product_name_lenght             | 610            | 1.85%         | Numeric (text length)     | Same rows as category, reflects structural missingness  |
|                             | product_description_lenght      | 610            | 1.85%         | Numeric (description len) | Same rows as above                                      |
|                             | product_photos_qty              | 610            | 1.85%         | Numeric (image count)     | Same rows as above                                      |
|                             | product_weight_g                | 2              | 0.01%         | Numeric (weight)          | Can fill with median or by product category             |
|                             | product_length_cm               | 2              | 0.01%         | Numeric (dimension)       | Same as above                                           |
|                             | product_height_cm               | 2              | 0.01%         | Numeric (dimension)       | Same as above                                           |
|                             | product_width_cm                | 2              | 0.01%         | Numeric (dimension)       | Same as above                                           |
| olist_order_reviews_dataset.csv | review_comment_title        | 87,656         | 88.34%        | Text                      | Buyer left title blank; common behavior                 |
|                             | review_comment_message          | 58,247         | 58.70%        | Text                      | No review comment written by buyer                      |

### 3. Key Metrics

#### Order Items
- Price Statistics:
  - Mean: 120.65
  - Median: 74.99
  - Max: 6,735.00
  - Min: 0.85
- Freight Value Statistics:
  - Mean: 19.99
  - Median: 16.26
  - Max: 409.68
  - Min: 0.00

#### Reviews
- Score Statistics:
  - Mean: 4.09
  - Median: 5.00
  - Distribution: 1-5 stars

#### Payments
- Payment Value Statistics:
  - Mean: 154.10
  - Median: 100.00
  - Max: 13,664.08
  - Min: 0.00
- Installments:
  - Range: 1-24
  - Mean: 2.85
  - Median: 1.00

### 4. Categorical Data Analysis

#### Order Status Distribution
- delivered: 96,478 (97.0%)
- shipped: 1,107 (1.1%)
- canceled: 625 (0.6%)
- unavailable: 609 (0.6%)
- invoiced: 314 (0.3%)
- Others: 308 (0.3%)

#### Payment Types
- credit_card: 76,795 (73.9%)
- boleto: 19,784 (19.0%)
- voucher: 5,775 (5.6%)
- debit_card: 1,529 (1.5%)
- not_defined: 3 (<0.1%)

### 5. Time Range Analysis
- Orders: 2016-09-04 to 2018-10-17
- Reviews: 2016-10-02 to 2018-08-31
- Shipping Limits: 2016-09-19 to 2020-04-09

### 6. Relationship Analysis
- Orders to Items: ~1.2 items per order on average
- Orders to Payments: ~1.09 payments per order on average
- Unique Customers: 96,096
- Unique Sellers: 3,095
- Unique Products: 32,951

### 7. Data Quality Recommendations

1. **Missing Value Treatment**
   - Implement business rules for order status-related missing values
   - Consider imputation for product physical attributes
   - Review strategy for missing review comments
   - Validate missing category names

2. **Data Validation**
   - Check for outliers in payment values
   - Verify shipping limit dates beyond order period
   - Validate product dimensions and weights
   - Review duplicate entries in geolocation data

3. **ETL Considerations**
   - Standardize timestamp formats
   - Implement data quality checks
   - Create appropriate indexes
   - Consider time-based partitioning
   - Normalize geographic data

4. **Performance Optimization**
   - Index frequently joined fields
   - Consider materialized views for common queries
   - Implement appropriate data types
   - Plan for data archiving strategy

### 8. Business Insights

1. **Order Processing**
   - High delivery success rate (97%)
   - Average order processing time can be analyzed
   - Multiple payment methods indicate diverse customer preferences

2. **Customer Behavior**
   - High review scores indicate good customer satisfaction
   - Credit card is the dominant payment method
   - Significant number of unique customers

3. **Product Management**
   - Large product catalog (32,951 unique products)
   - Some products lack complete information
   - Category system needs maintenance

4. **Geographic Coverage**
   - Extensive geolocation data
   - Wide coverage of Brazilian regions
   - Potential for regional analysis
