# Olist E-Commerce Data Processing and ETL Summary
## Overview of Raw Data

The raw dataset consists of 9 core tables. The original data presented challenges such as inconsistent formats, missing values, and varying data types.

| Dataset                           | Rows     | Missing Fields                            | Notes                                          |
|----------------------------------|----------|-------------------------------------------|------------------------------------------------|
| `olist_orders_dataset`           | 99,441   | `order_approved_at`, `delivery_date`      | Some missing timestamps                       |
| `olist_order_items_dataset`      | 112,650  | None                                      | One-to-many relationships per order           |
| `olist_products_dataset`         | 32,951   | Several descriptive and dimension fields  | Many float64 fields with nulls                |
| `olist_customers_dataset`        | 99,441   | None                                      | Duplicates in `customer_unique_id`            |
| `olist_order_reviews_dataset`    | 99,224   | High missing rate in comment fields       | Scores are complete and usable                |
| `olist_order_payments_dataset`   | 103,886  | None                                      | Contains multiple payment types and parts     |
| `olist_sellers_dataset`          | 3,095    | None                                      | Seller ID clearly linked to ZIP code regions  |
| `product_category_name_translation.csv` | 71 | None                            | Required for mapping category names to English |
| `olist_geolocation_dataset`      | 1,000,163| None                                      | Contains many duplicate ZIP coordinates       |

---

## ETL Strategy

### General Cleaning Steps

- Convert all timestamp columns to ISO-standard datetime format
- Fill or retain nulls where appropriate (e.g., for missing comments)
- Validate ranges for ZIP codes, ratings, and numeric attributes
- Standardize column naming: lowercase with underscores

### Table-Specific Processing

#### `orders.csv`
- Added `delivery_time_days` and `processing_time_days` as derived fields
- Standardized all date formats
- Removed invalid or extreme timestamps

#### `order_items.csv`
- Calculated `total_value = price + freight_value`
- Enforced composite primary key `(order_id, order_item_id)`

#### `products.csv`
- Removed rows with critical missing fields (~2% of data)
- Converted float64 fields to int64 where appropriate
- Retained uncategorized products for later classification

#### `customers.csv`
- Used `customer_id` as the primary key
- Retained `customer_unique_id` for future repeat purchase analysis

#### `order_reviews.csv`
- Retained missing comment fields as null for future text analysis
- Deduplicated reviews by keeping the most recent per order

#### `order_payments.csv`
- Normalized `payment_type` text formats
- Merged multiple payments per order while preserving installment info

#### `sellers.csv`
- Mapped ZIP code and city/state into a unified location dimension
- Later joined with the `dim_location` table for regional analysis

#### `geolocations.csv`
- Aggregated by ZIP code prefix to derive average coordinates
- Retained only unique, complete ZIP code entries

#### `product_category_name_translation.csv`
- Mapped category names to English
- Replaced missing values with "unknown"

---

## Output Structure (Cleaned Data)

After transformation, 16 normalized tables were produced and validated. These are structured to match the PostgreSQL schema and are ready for import.

| Table Name                      | Description                                      |
|--------------------------------|--------------------------------------------------|
| `orders.csv`                   | Includes derived delivery and processing fields  |
| `order_items.csv`              | Includes `total_value`                           |
| `order_payments.csv`           | Consolidated payment records                     |
| `order_reviews.csv`            | Cleaned and deduplicated review data             |
| `customers.csv`                | Primary customer data                            |
| `sellers.csv`                  | Primary seller data                              |
| `products.csv`                 | Product details and attributes                   |
| `product_categories.csv`       | Category names and translations                  |
| `geolocations.csv`             | Clean ZIP-level coordinates                      |
| `dim_date.csv`                 | Date dimension table                             |
| `dim_location.csv`             | Location hierarchy with regional roll-up         |
| `region_sales_stats.csv`       | Monthly sales statistics by state                |
| `customer_profiles.csv`        | Behavioral profiling of customers                |
| `product_lifecycle.csv`        | Product lifecycle analytics                      |
| `product_category_hierarchy.csv` | Hierarchical structure of product categories   |
| `seller_performance.csv`       | Seller performance metrics                       |

---

## Data Normalization Considerations (Third Normal Form - 3NF)

The design and transformation of the dataset strictly adhered to the principles of Third Normal Form (3NF) to ensure data integrity, eliminate redundancy, and support query.

### Key Normalization Decisions

1. **Atomic Fields**
   - All columns store atomic (indivisible) values. For example, address and location data were split into `zip_code_prefix`, `city`, and `state`, avoiding compound fields.

2. **Primary Keys and Foreign Keys**
   - Each table has a clearly defined primary key (`order_id`, `product_id`, `customer_id`, etc.).
   - Foreign key relationships were established between related entities (e.g., `orders.customer_id` → `customers.customer_id`), avoiding duplication of descriptive attributes.

3. **Eliminating Transitive Dependencies**
   - Data that could be derived from other non-key attributes was moved into separate tables. For instance:
     - Product category translations were stored in `product_categories`, rather than embedding English names into the `products` table.
     - Geographical data was factored out into `dim_location`, separating region-level descriptors from fact tables.

4. **Separate Fact and Dimension Tables**
   - Fact tables such as `orders`, `order_items`, and `order_reviews` contain measurable or transactional data.
   - Dimension tables like `dim_date`, `dim_location`, and `product_categories` provide context and descriptive metadata, supporting analytical flexibility.

5. **Avoiding Partial Dependencies**
   - Composite keys were carefully handled (e.g., `(order_id, order_item_id)` in `order_items`) to ensure that no non-key column is dependent on part of a key only.
