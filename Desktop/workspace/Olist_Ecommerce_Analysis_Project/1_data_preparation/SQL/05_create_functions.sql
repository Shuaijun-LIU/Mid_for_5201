-- Create functions and triggers
-- Execution date: 2025-06-03
-- Purpose: Create necessary functions and triggers for data processing

-- Function to calculate delivery time
CREATE OR REPLACE FUNCTION calculate_delivery_time()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.order_delivered_customer_date IS NOT NULL AND NEW.order_purchase_timestamp IS NOT NULL THEN
        NEW.delivery_time_days := EXTRACT(EPOCH FROM (NEW.order_delivered_customer_date - NEW.order_purchase_timestamp))/86400;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate processing time
CREATE OR REPLACE FUNCTION calculate_processing_time()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.order_delivered_carrier_date IS NOT NULL AND NEW.order_purchase_timestamp IS NOT NULL THEN
        NEW.processing_time_days := EXTRACT(EPOCH FROM (NEW.order_delivered_carrier_date - NEW.order_purchase_timestamp))/86400;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to update seller performance
CREATE OR REPLACE FUNCTION update_seller_performance()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO seller_performance (
        seller_id,
        total_orders,
        total_sales,
        avg_rating,
        delivery_time_avg,
        customer_satisfaction_score
    )
    SELECT 
        s.seller_id,
        COUNT(DISTINCT oi.order_id) as total_orders,
        SUM(oi.price) as total_sales,
        AVG(r.review_score) as avg_rating,
        AVG(o.delivery_time_days) as delivery_time_avg,
        AVG(r.review_score) * (1 - AVG(o.delivery_time_days)/30) as customer_satisfaction_score
    FROM sellers s
    LEFT JOIN order_items oi ON s.seller_id = oi.seller_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
    LEFT JOIN order_reviews r ON o.order_id = r.order_id
    WHERE s.seller_id = NEW.seller_id
    GROUP BY s.seller_id
    ON CONFLICT (seller_id) DO UPDATE SET
        total_orders = EXCLUDED.total_orders,
        total_sales = EXCLUDED.total_sales,
        avg_rating = EXCLUDED.avg_rating,
        delivery_time_avg = EXCLUDED.delivery_time_avg,
        customer_satisfaction_score = EXCLUDED.customer_satisfaction_score;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to update customer profile
CREATE OR REPLACE FUNCTION update_customer_profile()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO customer_profiles (
        customer_id,
        total_orders,
        total_spent,
        avg_order_value,
        top_category,
        last_order_date
    )
    SELECT 
        c.customer_id,
        COUNT(DISTINCT o.order_id) as total_orders,
        SUM(oi.price) as total_spent,
        AVG(oi.price) as avg_order_value,
        (
            SELECT p.product_category_name
            FROM order_items oi2
            JOIN products p ON oi2.product_id = p.product_id
            WHERE oi2.order_id IN (
                SELECT order_id 
                FROM orders 
                WHERE customer_id = c.customer_id
            )
            GROUP BY p.product_category_name
            ORDER BY COUNT(*) DESC
            LIMIT 1
        ) as top_category,
        MAX(o.order_purchase_timestamp) as last_order_date
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    WHERE c.customer_id = NEW.customer_id
    GROUP BY c.customer_id
    ON CONFLICT (customer_id) DO UPDATE SET
        total_orders = EXCLUDED.total_orders,
        total_spent = EXCLUDED.total_spent,
        avg_order_value = EXCLUDED.avg_order_value,
        top_category = EXCLUDED.top_category,
        last_order_date = EXCLUDED.last_order_date;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE TRIGGER trg_calculate_delivery_time
    BEFORE INSERT OR UPDATE ON orders
    FOR EACH ROW
    EXECUTE FUNCTION calculate_delivery_time();

CREATE TRIGGER trg_calculate_processing_time
    BEFORE INSERT OR UPDATE ON orders
    FOR EACH ROW
    EXECUTE FUNCTION calculate_processing_time();

CREATE TRIGGER trg_update_seller_performance
    AFTER INSERT OR UPDATE ON order_items
    FOR EACH ROW
    EXECUTE FUNCTION update_seller_performance();

CREATE TRIGGER trg_update_customer_profile
    AFTER INSERT OR UPDATE ON orders
    FOR EACH ROW
    EXECUTE FUNCTION update_customer_profile(); 