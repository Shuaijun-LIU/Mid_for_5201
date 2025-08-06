-- Create base table structures
-- Execution date: 2025-05-31
-- Purpose: Create all necessary data tables

-- Customer table
CREATE TABLE customers (
    customer_id VARCHAR(32) PRIMARY KEY,
    customer_unique_id VARCHAR(32),
    customer_zip_code_prefix INTEGER,
    customer_city TEXT,
    customer_state VARCHAR(2)
);

-- Seller table
CREATE TABLE sellers (
    seller_id VARCHAR(32) PRIMARY KEY,
    seller_zip_code_prefix INTEGER,
    seller_city TEXT,
    seller_state VARCHAR(2)
);

-- Geolocation table
CREATE TABLE geolocations (
    geolocation_zip_code_prefix INTEGER,
    geolocation_lat FLOAT,
    geolocation_lng FLOAT,
    geolocation_city TEXT,
    geolocation_state VARCHAR(2)
);

-- Order table
CREATE TABLE orders (
    order_id VARCHAR(32) PRIMARY KEY,
    customer_id VARCHAR(32) REFERENCES customers(customer_id),
    order_status TEXT,
    order_purchase_timestamp TIMESTAMP,
    order_approved_at TIMESTAMP,
    order_delivered_carrier_date TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP,
    delivery_time_days INTEGER,
    processing_time_days INTEGER
);

-- Order items table
CREATE TABLE order_items (
    order_id VARCHAR(32) REFERENCES orders(order_id),
    order_item_id INTEGER,
    product_id VARCHAR(32) REFERENCES products(product_id),
    seller_id VARCHAR(32) REFERENCES sellers(seller_id),
    shipping_limit_date TIMESTAMP,
    price FLOAT,
    freight_value FLOAT,
    total_value FLOAT,
    PRIMARY KEY (order_id, order_item_id)
);

-- Product table
CREATE TABLE products (
    product_id VARCHAR(32) PRIMARY KEY,
    product_category_name TEXT REFERENCES product_categories(product_category_name),
    product_name_length INTEGER,
    product_description_length INTEGER,
    product_photos_qty INTEGER,
    product_weight_g INTEGER,
    product_length_cm INTEGER,
    product_height_cm INTEGER,
    product_width_cm INTEGER
);

-- Product category table
CREATE TABLE product_categories (
    product_category_name TEXT PRIMARY KEY,
    product_category_name_english TEXT
);

-- Product category hierarchy table
CREATE TABLE product_category_hierarchy (
    name TEXT,
    id INTEGER,
    parent_id FLOAT,
    level INTEGER
);

-- Order review table
CREATE TABLE order_reviews (
    review_id VARCHAR(32) PRIMARY KEY,
    order_id VARCHAR(32) REFERENCES orders(order_id),
    review_score INTEGER,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date TIMESTAMP,
    review_answer_timestamp TIMESTAMP
);

-- Order payment table
CREATE TABLE order_payments (
    order_id VARCHAR(32) REFERENCES orders(order_id),
    payment_sequential INTEGER,
    payment_type TEXT,
    payment_installments INTEGER,
    payment_value FLOAT,
    PRIMARY KEY (order_id, payment_sequential)
);

-- Seller performance table
CREATE TABLE seller_performance (
    seller_id VARCHAR(32) PRIMARY KEY REFERENCES sellers(seller_id),
    total_sales FLOAT,
    total_orders INTEGER,
    delivery_time_avg FLOAT,
    avg_rating FLOAT,
    customer_satisfaction_score FLOAT
);

-- Regional sales statistics table
CREATE TABLE region_sales_stats (
    state VARCHAR(2),
    year_month TEXT,
    order_count INTEGER,
    total_sales FLOAT,
    total_freight FLOAT,
    seller_count INTEGER,
    customer_count INTEGER,
    average_order_value FLOAT,
    sales_with_freight FLOAT,
    sales_growth FLOAT
);

-- Customer profile table
CREATE TABLE customer_profiles (
    customer_id VARCHAR(32) PRIMARY KEY REFERENCES customers(customer_id),
    total_orders INTEGER,
    total_items INTEGER,
    total_spent FLOAT,
    avg_order_value FLOAT,
    total_categories INTEGER,
    top_category TEXT,
    avg_review_score FLOAT,
    total_reviews FLOAT,
    behavior_profile TEXT
);

-- Date dimension table
CREATE TABLE dim_date (
    date_id INTEGER PRIMARY KEY,
    date DATE,
    year INTEGER,
    month INTEGER,
    day INTEGER,
    quarter INTEGER,
    day_of_week INTEGER,
    day_of_year INTEGER,
    is_weekend INTEGER,
    is_holiday INTEGER,
    month_name TEXT,
    day_name TEXT,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    week_number INTEGER
);

-- Location dimension table
CREATE TABLE dim_location (
    location_id INTEGER PRIMARY KEY,
    parent_location_id TEXT,
    location_level TEXT,
    region TEXT,
    state TEXT,
    city TEXT,
    zip_code_prefix INTEGER,
    latitude FLOAT,
    longitude FLOAT,
    customer_count FLOAT,
    seller_count FLOAT,
    order_count FLOAT
);

-- Product lifecycle table
CREATE TABLE product_lifecycle (
    product_id VARCHAR(32) PRIMARY KEY REFERENCES products(product_id),
    product_category_name TEXT,
    product_name_length INTEGER,
    product_description_length INTEGER,
    product_photos_qty INTEGER,
    first_sale_date TIMESTAMP,
    last_sale_date TIMESTAMP,
    lifecycle_duration_days INTEGER,
    lifecycle_stage TEXT,
    total_orders INTEGER,
    total_revenue FLOAT,
    total_freight FLOAT,
    months_with_sales INTEGER,
    avg_monthly_orders FLOAT,
    avg_order_value FLOAT,
    freight_ratio FLOAT,
    sales_velocity FLOAT,
    revenue_velocity FLOAT
); 


-- orders.customer_id → customers
ALTER TABLE orders
ADD CONSTRAINT fk_orders_customer
FOREIGN KEY (customer_id) REFERENCES customers(customer_id);

-- order_items.order_id → orders
ALTER TABLE order_items
ADD CONSTRAINT fk_order_items_order
FOREIGN KEY (order_id) REFERENCES orders(order_id);

-- order_items.product_id → products
ALTER TABLE order_items
ADD CONSTRAINT fk_order_items_product
FOREIGN KEY (product_id) REFERENCES products(product_id);

-- order_items.seller_id → sellers
ALTER TABLE order_items
ADD CONSTRAINT fk_order_items_seller
FOREIGN KEY (seller_id) REFERENCES sellers(seller_id);

-- products.product_category_name → product_categories
ALTER TABLE products
ADD CONSTRAINT fk_products_category
FOREIGN KEY (product_category_name) REFERENCES product_categories(product_category_name);

-- order_reviews.order_id → orders
ALTER TABLE order_reviews
ADD CONSTRAINT fk_order_reviews_order
FOREIGN KEY (order_id) REFERENCES orders(order_id);

-- order_payments.order_id → orders
ALTER TABLE order_payments
ADD CONSTRAINT fk_order_payments_order
FOREIGN KEY (order_id) REFERENCES orders(order_id);

-- seller_performance.seller_id → sellers
-- exist
-- customer_profiles.customer_id → customers
-- exist
-- product_lifecycle.product_id → products
-- exist