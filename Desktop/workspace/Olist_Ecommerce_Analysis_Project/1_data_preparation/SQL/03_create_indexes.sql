-- Create indexes
-- Execution date: 2025-06-02 upodate at 06-05
-- Purpose: Create necessary indexes to improve query performance

-- Order table indexes
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_purchase_timestamp ON orders(order_purchase_timestamp);
CREATE INDEX idx_orders_status ON orders(order_status);
CREATE INDEX idx_orders_delivery_time ON orders(delivery_time_days);

-- Order items table indexes
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
CREATE INDEX idx_order_items_seller_id ON order_items(seller_id);
CREATE INDEX idx_order_items_shipping_date ON order_items(shipping_limit_date);
CREATE INDEX idx_order_items_price ON order_items(price);

-- Product table indexes
CREATE INDEX idx_products_category ON products(product_category_name);
CREATE INDEX idx_products_weight ON products(product_weight_g);
CREATE INDEX idx_products_dimensions ON products(product_length_cm, product_height_cm, product_width_cm);

-- Review table indexes
CREATE INDEX idx_reviews_order_id ON order_reviews(order_id);
CREATE INDEX idx_reviews_score ON order_reviews(review_score);
CREATE INDEX idx_reviews_creation_date ON order_reviews(review_creation_date);

-- Payment table indexes
CREATE INDEX idx_payments_order_id ON order_payments(order_id);
CREATE INDEX idx_payments_type ON order_payments(payment_type);
CREATE INDEX idx_payments_installments ON order_payments(payment_installments);
CREATE INDEX idx_payments_value ON order_payments(payment_value);

-- Geolocation indexes
CREATE INDEX idx_geolocation_zip ON geolocations(geolocation_zip_code_prefix);
CREATE INDEX idx_geolocation_state ON geolocations(geolocation_state);
CREATE INDEX idx_geolocation_city ON geolocations(geolocation_city);

-- Seller performance indexes
CREATE INDEX idx_seller_performance_rating ON seller_performance(avg_rating);
CREATE INDEX idx_seller_performance_sales ON seller_performance(total_sales);
CREATE INDEX idx_seller_performance_delivery ON seller_performance(delivery_time_avg);

-- Regional sales statistics indexes
CREATE INDEX idx_region_sales_state ON region_sales_stats(state);
CREATE INDEX idx_region_sales_year_month ON region_sales_stats(year_month);
CREATE INDEX idx_region_sales_total ON region_sales_stats(total_sales);

-- Customer profiles indexes
CREATE INDEX idx_customer_profiles_orders ON customer_profiles(total_orders);
CREATE INDEX idx_customer_profiles_spent ON customer_profiles(total_spent);
CREATE INDEX idx_customer_profiles_category ON customer_profiles(top_category);

-- Date dimension indexes
CREATE INDEX idx_dim_date_date ON dim_date(date);
CREATE INDEX idx_dim_date_year_month ON dim_date(year, month);
CREATE INDEX idx_dim_date_holiday ON dim_date(is_holiday);

-- Location dimension indexes
CREATE INDEX idx_dim_location_state ON dim_location(state);
CREATE INDEX idx_dim_location_region ON dim_location(region);
CREATE INDEX idx_dim_location_city ON dim_location(city);

-- Product lifecycle indexes
CREATE INDEX idx_product_lifecycle_stage ON product_lifecycle(lifecycle_stage);
CREATE INDEX idx_product_lifecycle_dates ON product_lifecycle(first_sale_date, last_sale_date);
CREATE INDEX idx_product_lifecycle_revenue ON product_lifecycle(total_revenue);