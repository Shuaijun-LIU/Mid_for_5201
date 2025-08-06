"""
Configuration for ETL pipeline
"""

from pathlib import Path

# Configure paths
BASE_DIR = Path('/Users/a1234/Desktop/workspace/Olist_Ecommerce_Analysis_Project')
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
CLEANED_DATA_DIR = BASE_DIR / 'data' / 'cleaned'
SAMPLE_DATA_DIR = BASE_DIR / 'data' / 'samples'

# Create necessary directories
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configure sampling ratio
SAMPLE_RATIO = 0.05

# Configure force reprocess flag
FORCE_REPROCESS = True  # Set to True to force reprocessing of all files

# Define data types for each table to optimize memory usage
DTYPE_DICT = {
    'olist_orders_dataset': {
        'order_id': 'string',
        'customer_id': 'string',
        'order_status': 'category'
    },
    'olist_products_dataset': {
        'product_id': 'string',
        'product_category_name': 'category',
        'product_name_length': 'Int32',
        'product_description_length': 'Int32',
        'product_photos_qty': 'Int32',
        'product_weight_g': 'Int32',
        'product_length_cm': 'Int32',
        'product_height_cm': 'Int32',
        'product_width_cm': 'Int32'
    },
    'olist_customers_dataset': {
        'customer_id': 'string',
        'customer_unique_id': 'string',
        'customer_zip_code_prefix': 'string',
        'customer_city': 'category',
        'customer_state': 'category'
    },
    'olist_sellers_dataset': {
        'seller_id': 'string',
        'seller_zip_code_prefix': 'string',
        'seller_city': 'category',
        'seller_state': 'category'
    },
    'olist_geolocation_dataset': {
        'geolocation_zip_code_prefix': 'string',
        'geolocation_lat': 'float64',
        'geolocation_lng': 'float64',
        'geolocation_city': 'category',
        'geolocation_state': 'category'
    },
    'olist_order_items_dataset': {
        'order_id': 'string',
        'order_item_id': 'Int32',
        'product_id': 'string',
        'seller_id': 'string',
        'shipping_limit_date': 'string',
        'price': 'float64',
        'freight_value': 'float64'
    },
    'olist_order_payments_dataset': {
        'order_id': 'string',
        'payment_sequential': 'Int64',
        'payment_type': 'category',
        'payment_installments': 'Int64',
        'payment_value': 'float64'
    },
    'olist_order_reviews_dataset': {
        'review_id': 'string',
        'order_id': 'string',
        'review_score': 'Int32',
        'review_comment_title': 'string',
        'review_comment_message': 'string',
        'review_creation_date': 'string',
        'review_answer_timestamp': 'string'
    },
    'product_category_name_translation': {
        'product_category_name': 'string',
        'product_category_name_english': 'string'
    }
}

# Define date columns for each table
DATE_COLUMNS = {
    'olist_orders_dataset': [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ],
    'olist_order_reviews_dataset': [
        'review_creation_date',
        'review_answer_timestamp'
    ],
    'olist_order_items_dataset': [
        'shipping_limit_date'
    ]
} 