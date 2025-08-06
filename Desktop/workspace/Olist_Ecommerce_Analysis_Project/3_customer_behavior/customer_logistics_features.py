"""
Step 1: Customer Behavior Feature Engineering
Execution date: 2025-06-18
Update date: 2025-06-22
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_data():
    print("Loading datasets...")
    
    orders_df = pd.read_csv('../data/processed_missing/olist_orders_dataset.csv')
    print(f"Orders dataset shape: {orders_df.shape}")
    
    order_items_df = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    print(f"Order items dataset shape: {order_items_df.shape}")
    
    products_df = pd.read_csv('../data/processed_missing/olist_products_dataset.csv')
    print(f"Products dataset shape: {products_df.shape}")
    
    customers_df = pd.read_csv('../data/processed_missing/olist_customers_dataset.csv')
    print(f"Customers dataset shape: {customers_df.shape}")
    
    return orders_df, order_items_df, products_df, customers_df

def preprocess_data(orders_df, order_items_df, products_df, customers_df):
    """preprocess and merge datasets"""
    print("\nPreprocessing data...")
    
    # filter only delivered orders
    delivered_orders = orders_df[orders_df['order_status'] == 'delivered'].copy()
    print(f"Delivered orders: {len(delivered_orders)} out of {len(orders_df)}")
    print(f"Delivery rate: {len(delivered_orders)/len(orders_df)*100:.2f}%")
    
    # convert timestamp columns to datetime
    delivered_orders['order_purchase_timestamp'] = pd.to_datetime(delivered_orders['order_purchase_timestamp'])
    print(f"Date range: {delivered_orders['order_purchase_timestamp'].min()} to {delivered_orders['order_purchase_timestamp'].max()}")
    
    # merge order items with products to get category information
    items_with_categories = order_items_df.merge(
        products_df[['product_id', 'product_category_name']], 
        on='product_id', 
        how='left'
    )
    print(f"Items with categories: {len(items_with_categories)}")
    print(f"Categories covered: {items_with_categories['product_category_name'].nunique()}")
    
    # merge with orders to get customer and timestamp information
    customer_orders = delivered_orders.merge(
        items_with_categories, 
        on='order_id', 
        how='inner'
    )
    print(f"Customer orders: {len(customer_orders)}")
    
    # merge with customers to get customer information
    final_df = customer_orders.merge(
        customers_df, 
        on='customer_id', 
        how='inner'
    )
    
    # sort by customer and purchase timestamp
    final_df = final_df.sort_values(['customer_unique_id', 'order_purchase_timestamp'])
    
    print(f"Final merged dataset shape: {final_df.shape}")
    print(f"Unique customers: {final_df['customer_unique_id'].nunique()}")
    print(f"Unique orders: {final_df['order_id'].nunique()}")
    print(f"Unique products: {final_df['product_id'].nunique()}")
    
    return final_df

def calculate_order_level_features(df):
    print("\nCalculating order-level features...")
    
    # group by order to calculate order totals
    order_features = df.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum',
        'customer_unique_id': 'first',
        'order_purchase_timestamp': 'first',
        'customer_state': 'first'
    }).reset_index()
    
    # calculate freight percentage
    order_features['freight_percent'] = (order_features['freight_value'] / order_features['price']) * 100
    
    # rename columns for clarity
    order_features = order_features.rename(columns={
        'price': 'order_total_price',
        'freight_value': 'order_freight_value'
    })
    
    print(f"Order features calculated: {len(order_features)} orders")
    print(f"Average order value: ${order_features['order_total_price'].mean():.2f}")
    print(f"Average freight value: ${order_features['order_freight_value'].mean():.2f}")
    print(f"Average freight percentage: {order_features['freight_percent'].mean():.2f}%")
    
    return order_features

def calculate_customer_features(order_features):
    """calculate customer-level features"""
    print("\nCalculating customer-level features...")
    
    customer_features = []
    
    for customer_id in order_features['customer_unique_id'].unique():
        customer_orders = order_features[order_features['customer_unique_id'] == customer_id].copy()
        customer_orders = customer_orders.sort_values('order_purchase_timestamp')
        
        # basic order metrics
        order_count = len(customer_orders)
        total_spending = customer_orders['order_total_price'].sum()
        avg_order_value = customer_orders['order_total_price'].mean()
        avg_freight_value = customer_orders['order_freight_value'].mean()
        avg_freight_percent = customer_orders['freight_percent'].mean()
        
        # date features
        first_order_date = customer_orders['order_purchase_timestamp'].min()
        last_order_date = customer_orders['order_purchase_timestamp'].max()
        
        # time between orders (for customers with multiple orders)
        if order_count > 1:
            time_diffs = customer_orders['order_purchase_timestamp'].diff().dt.days
            avg_days_between_orders = time_diffs.mean()
        else:
            avg_days_between_orders = np.nan
        
        # geographic diversity
        num_distinct_states = customer_orders['customer_state'].nunique()
        
        # high freight flag (freight > 20% of order value)
        high_freight_orders = len(customer_orders[customer_orders['freight_percent'] > 20])
        high_freight_ratio = high_freight_orders / order_count if order_count > 0 else 0
        
        customer_features.append({
            'customer_unique_id': customer_id,
            'order_count': order_count,
            'total_spending': total_spending,
            'avg_order_value': avg_order_value,
            'avg_freight_value': avg_freight_value,
            'avg_freight_percent': avg_freight_percent,
            'avg_days_between_orders': avg_days_between_orders,
            'first_order_date': first_order_date,
            'last_order_date': last_order_date,
            'num_distinct_states_ordered_to': num_distinct_states,
            'high_freight_orders': high_freight_orders,
            'high_freight_ratio': high_freight_ratio
        })
    
    customer_features_df = pd.DataFrame(customer_features)
    
    print(f"Customer features calculated for {len(customer_features_df)} customers")
    print(f"Average customer order count: {customer_features_df['order_count'].mean():.2f}")
    print(f"Average customer total spending: ${customer_features_df['total_spending'].mean():.2f}")
    print(f"Average customer order value: ${customer_features_df['avg_order_value'].mean():.2f}")
    
    return customer_features_df

def calculate_product_diversity(df, customer_features):
    """calculate product category diversity for each customer"""
    print("\nCalculating product category diversity...")
    
    # get unique categories per customer
    customer_categories = df.groupby('customer_unique_id')['product_category_name'].nunique().reset_index()
    customer_categories = customer_categories.rename(columns={'product_category_name': 'product_category_diversity'})
    
    # merge with customer features
    customer_features = customer_features.merge(customer_categories, on='customer_unique_id', how='left')
    
    print(f"Product diversity calculated")
    print(f"Average category diversity: {customer_features['product_category_diversity'].mean():.2f}")
    print(f"Max category diversity: {customer_features['product_category_diversity'].max()}")
    
    return customer_features

def create_output_directory():
    """create output directory if it doesn't exist"""
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def main():
    print("=" * 60)
    print("CUSTOMER BEHAVIOR FEATURE ENGINEERING")
    print("=" * 60)

    orders_df, order_items_df, products_df, customers_df = load_data()
    
    # preprocess and merge data
    merged_df = preprocess_data(orders_df, order_items_df, products_df, customers_df)
    
    # calculate order-level features
    order_features = calculate_order_level_features(merged_df)
    
    # calculate customer-level features
    customer_features = calculate_customer_features(order_features)
    
    # calculate product diversity
    customer_features = calculate_product_diversity(merged_df, customer_features)

    create_output_directory()

    output_file = 'output/customer_logistics_features.csv'
    customer_features.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"Output file: {output_file}")
    print(f"Total customers processed: {len(customer_features)}")
    print(f"Features calculated: {len(customer_features.columns)}")
    
    print("\nFeature list:")
    for i, col in enumerate(customer_features.columns, 1):
        print(f"{i:2d}. {col}")
    
    print("\nSample of customer features:")
    print(customer_features.head())
    
    print("\nFeature summary statistics:")
    print(customer_features.describe())
    
    print("\nHigh freight customers (freight > 20% of order value):")
    high_freight_customers = customer_features[customer_features['high_freight_ratio'] > 0]
    print(f"Count: {len(high_freight_customers)} ({len(high_freight_customers)/len(customer_features)*100:.1f}%)")
    
    print("\nCustomers with multiple states:")
    multi_state_customers = customer_features[customer_features['num_distinct_states_ordered_to'] > 1]
    print(f"Count: {len(multi_state_customers)} ({len(multi_state_customers)/len(customer_features)*100:.1f}%)")
    
    print("\nCustomer order count distribution:")
    order_count_stats = customer_features['order_count'].value_counts().sort_index()
    print(order_count_stats.head(10))
    
    print("\nTop 5 customers by total spending:")
    top_spenders = customer_features.nlargest(5, 'total_spending')[['customer_unique_id', 'total_spending', 'order_count']]
    print(top_spenders)
    
    print("\nTop 5 customers by order count:")
    top_orderers = customer_features.nlargest(5, 'order_count')[['customer_unique_id', 'order_count', 'total_spending']]
    print(top_orderers)
    
    print("\nFeature engineering completed successfully!")

if __name__ == "__main__":
    main() 