 #!/bin/bash

echo "Starting data processing pipeline..."

# Step 1: Process base tables
echo "Processing base tables..."
python3 base_orders.py
python3 base_products.py
python3 base_reviews.py
python3 base_order_items.py
python3 base_order_payments.py
python3 base_customers.py
python3 base_sellers.py
python3 base_geolocations.py
python3 base_product_categories.py

# Step 2: Generate derived tables
echo "Generating derived tables..."
python3 derived_product_category_hierarchy.py
python3 derived_seller_performance.py
python3 derived_customer_profile.py
python3 derived_product_lifecycle.py

# Step 3: Generate dimension tables
echo "Generating dimension tables..."
python3 dim_date.py
python3 dim_location.py
python3 dim_region_sales_stats.py

echo "Data processing pipeline completed!"