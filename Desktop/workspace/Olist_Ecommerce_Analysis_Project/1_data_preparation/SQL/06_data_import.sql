BEGIN;

-- customers
COPY customers FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/customers.csv' WITH (FORMAT csv, HEADER true);

-- sellers
COPY sellers FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/sellers.csv' WITH (FORMAT csv, HEADER true);

-- geolocations
COPY geolocations FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/geolocations.csv' WITH (FORMAT csv, HEADER true);

-- orders
COPY orders FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/orders.csv' WITH (FORMAT csv, HEADER true);

-- order_items
COPY order_items FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/order_items.csv' WITH (FORMAT csv, HEADER true);

-- products
COPY products FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/products.csv' WITH (FORMAT csv, HEADER true);

-- product_categories
COPY product_categories FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/product_categories.csv' WITH (FORMAT csv, HEADER true);

-- product_category_hierarchy
COPY product_category_hierarchy FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/product_category_hierarchy.csv' WITH (FORMAT csv, HEADER true);

-- order_reviews
COPY order_reviews FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/order_reviews.csv' WITH (FORMAT csv, HEADER true);

-- order_payments
COPY order_payments FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/order_payments.csv' WITH (FORMAT csv, HEADER true);

-- seller_performance
COPY seller_performance FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/seller_performance.csv' WITH (FORMAT csv, HEADER true);

-- region_sales_stats
COPY region_sales_stats FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/region_sales_stats.csv' WITH (FORMAT csv, HEADER true);

-- customer_profiles
COPY customer_profiles FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/customer_profiles.csv' WITH (FORMAT csv, HEADER true);

-- dim_date
COPY dim_date FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/dim_date.csv' WITH (FORMAT csv, HEADER true);

-- dim_location
COPY dim_location FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/dim_location.csv' WITH (FORMAT csv, HEADER true);

-- product_lifecycle
COPY product_lifecycle FROM '/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project/data/cleaned/product_lifecycle.csv' WITH (FORMAT csv, HEADER true);

COMMIT;