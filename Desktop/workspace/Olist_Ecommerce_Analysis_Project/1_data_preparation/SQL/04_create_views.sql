-- Create materialized views
-- Execution date: 2025-06-03
-- Purpose: Create materialized views to support data analysis

-- Order summary view
CREATE MATERIALIZED VIEW mv_order_summary AS
SELECT 
    o.order_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    o.delivery_time_days,
    o.processing_time_days,
    COUNT(DISTINCT oi.product_id) as unique_products,
    COUNT(DISTINCT oi.seller_id) as unique_sellers,
    SUM(oi.price) as total_price,
    SUM(oi.freight_value) as total_freight,
    AVG(r.review_score) as avg_review_score,
    MAX(p.payment_installments) as max_installments
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN order_reviews r ON o.order_id = r.order_id
LEFT JOIN order_payments p ON o.order_id = p.order_id
GROUP BY o.order_id, o.customer_id, o.order_status, 
         o.order_purchase_timestamp, o.order_delivered_customer_date,
         o.delivery_time_days, o.processing_time_days;

-- Product sales statistics view
CREATE MATERIALIZED VIEW mv_product_sales_stats AS
SELECT 
    p.product_id,
    p.product_category_name,
    COUNT(DISTINCT oi.order_id) as total_orders,
    SUM(oi.price) as total_revenue,
    COUNT(DISTINCT oi.seller_id) as unique_sellers,
    AVG(r.review_score) as avg_rating,
    MIN(o.order_purchase_timestamp) as first_sale_date,
    MAX(o.order_purchase_timestamp) as last_sale_date
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN order_reviews r ON o.order_id = r.order_id
GROUP BY p.product_id, p.product_category_name;

-- Seller performance view
CREATE MATERIALIZED VIEW mv_seller_performance AS
SELECT 
    s.seller_id,
    s.seller_state,
    s.seller_city,
    COUNT(DISTINCT oi.order_id) as total_orders,
    SUM(oi.price) as total_revenue,
    AVG(r.review_score) as avg_rating,
    AVG(o.delivery_time_days) as avg_delivery_time,
    COUNT(DISTINCT oi.product_id) as unique_products
FROM sellers s
LEFT JOIN order_items oi ON s.seller_id = oi.seller_id
LEFT JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN order_reviews r ON o.order_id = r.order_id
GROUP BY s.seller_id, s.seller_state, s.seller_city;

-- Customer behavior view
CREATE MATERIALIZED VIEW mv_customer_behavior AS
SELECT 
    c.customer_id,
    c.customer_state,
    c.customer_city,
    COUNT(DISTINCT o.order_id) as total_orders,
    SUM(oi.price) as total_spent,
    AVG(oi.price) as avg_order_value,
    COUNT(DISTINCT oi.product_id) as unique_products,
    COUNT(DISTINCT oi.seller_id) as unique_sellers,
    AVG(r.review_score) as avg_rating
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN order_reviews r ON o.order_id = r.order_id
GROUP BY c.customer_id, c.customer_state, c.customer_city;

-- Regional sales analysis view
CREATE MATERIALIZED VIEW mv_regional_sales AS
SELECT 
    g.geolocation_state as state,
    g.geolocation_region as region,
    COUNT(DISTINCT o.order_id) as total_orders,
    SUM(oi.price) as total_revenue,
    COUNT(DISTINCT o.customer_id) as unique_customers,
    COUNT(DISTINCT oi.seller_id) as unique_sellers,
    AVG(r.review_score) as avg_review_score,
    AVG(o.delivery_time_days) as avg_delivery_time
FROM geolocations g
LEFT JOIN customers c ON g.geolocation_zip_code_prefix = c.customer_zip_code_prefix
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN order_reviews r ON o.order_id = r.order_id
GROUP BY g.geolocation_state, g.geolocation_region; 