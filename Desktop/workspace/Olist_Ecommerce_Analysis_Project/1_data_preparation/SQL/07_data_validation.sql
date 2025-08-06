-- Data validation
-- Execution date: 2025-06-05
-- Purpose: Validate data integrity and quality

-- Basic data integrity check
SELECT 'customers' as table_name, COUNT(*) as record_count FROM customers
UNION ALL
SELECT 'sellers', COUNT(*) FROM sellers
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'orders', COUNT(*) FROM orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL
SELECT 'order_reviews', COUNT(*) FROM order_reviews
UNION ALL
SELECT 'order_payments', COUNT(*) FROM order_payments
UNION ALL
SELECT 'geolocations', COUNT(*) FROM geolocations
UNION ALL
SELECT 'seller_performance', COUNT(*) FROM seller_performance
UNION ALL
SELECT 'customer_profiles', COUNT(*) FROM customer_profiles
UNION ALL
SELECT 'region_sales_stats', COUNT(*) FROM region_sales_stats;

-- Foreign key constraint check
SELECT 'orders without customers' as check_type, COUNT(*) as count
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL
UNION ALL
SELECT 'order_items without orders', COUNT(*)
FROM order_items oi
LEFT JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_id IS NULL
UNION ALL
SELECT 'order_items without products', COUNT(*)
FROM order_items oi
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE p.product_id IS NULL
UNION ALL
SELECT 'order_items without sellers', COUNT(*)
FROM order_items oi
LEFT JOIN sellers s ON oi.seller_id = s.seller_id
WHERE s.seller_id IS NULL;

-- Data quality check
SELECT 'order_status' as check_type, order_status, COUNT(*) as count
FROM orders
GROUP BY order_status
UNION ALL
SELECT 'payment_type', payment_type, COUNT(*)
FROM order_payments
GROUP BY payment_type
UNION ALL
SELECT 'review_score', review_score, COUNT(*)
FROM order_reviews
GROUP BY review_score;

-- Time range check
SELECT 
    'orders' as table_name,
    MIN(order_purchase_timestamp) as min_timestamp,
    MAX(order_purchase_timestamp) as max_timestamp
FROM orders
UNION ALL
SELECT 
    'reviews',
    MIN(review_creation_date),
    MAX(review_creation_date)
FROM order_reviews
UNION ALL
SELECT 
    'shipping',
    MIN(shipping_limit_date),
    MAX(shipping_limit_date)
FROM order_items;

-- Numeric range check
SELECT 
    'price' as metric,
    MIN(price) as min_value,
    MAX(price) as max_value,
    AVG(price) as avg_value,
    STDDEV(price) as stddev_value
FROM order_items
UNION ALL
SELECT 
    'freight_value',
    MIN(freight_value),
    MAX(freight_value),
    AVG(freight_value),
    STDDEV(freight_value)
FROM order_items
UNION ALL
SELECT 
    'payment_value',
    MIN(payment_value),
    MAX(payment_value),
    AVG(payment_value),
    STDDEV(payment_value)
FROM order_payments;

-- Geographic data check
SELECT 
    COUNT(DISTINCT geolocation_zip_code_prefix) as unique_zip_codes,
    COUNT(DISTINCT geolocation_city) as unique_cities,
    COUNT(DISTINCT geolocation_state) as unique_states,
    COUNT(DISTINCT geolocation_region) as unique_regions
FROM geolocations;

-- Seller performance check
SELECT 
    AVG(avg_rating) as avg_seller_rating,
    AVG(total_sales) as avg_seller_sales,
    AVG(delivery_time_avg) as avg_delivery_time,
    AVG(customer_satisfaction_score) as avg_satisfaction
FROM seller_performance;

-- Customer behavior check
SELECT 
    AVG(total_orders) as avg_orders_per_customer,
    AVG(total_spent) as avg_spent_per_customer,
    AVG(avg_order_value) as avg_order_value,
    COUNT(DISTINCT top_category) as unique_top_categories
FROM customer_profiles;

-- Regional sales check
SELECT 
    state,
    region,
    COUNT(DISTINCT year_month) as months_with_sales,
    AVG(total_orders) as avg_orders_per_month,
    AVG(total_sales) as avg_sales_per_month,
    AVG(avg_review_score) as avg_region_rating,
    AVG(avg_delivery_time) as avg_region_delivery_time
FROM region_sales_stats
GROUP BY state, region
ORDER BY avg_sales_per_month DESC;

-- Materialized view check
SELECT 'mv_order_summary' as view_name, COUNT(*) as record_count FROM mv_order_summary
UNION ALL
SELECT 'mv_product_sales_stats', COUNT(*) FROM mv_product_sales_stats
UNION ALL
SELECT 'mv_seller_performance', COUNT(*) FROM mv_seller_performance
UNION ALL
SELECT 'mv_customer_behavior', COUNT(*) FROM mv_customer_behavior
UNION ALL
SELECT 'mv_regional_sales', COUNT(*) FROM mv_regional_sales; 