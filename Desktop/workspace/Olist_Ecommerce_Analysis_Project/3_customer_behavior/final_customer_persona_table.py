"""
Step 6: Unified Customer Persona Table Construction
Execution date: 2025-06-23
Update date: 2025-06-24
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_all_data():

    logistics = pd.read_csv('output/customer_logistics_features.csv')
    rfm = pd.read_csv('output/rfm_segmented_customers.csv')
    geo = pd.read_csv('output/customer_cluster_geolocation.csv')
    lifecycle = pd.read_csv('output/customer_lifecycle.csv')
    funnel = pd.read_csv('output/orders_with_funnel_tag.csv')
    orders = pd.read_csv('../data/processed_missing/olist_orders_dataset.csv', parse_dates=['order_purchase_timestamp'])
    order_items = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    products = pd.read_csv('../data/processed_missing/olist_products_dataset.csv')
    customers = pd.read_csv('../data/processed_missing/olist_customers_dataset.csv')
    
    # merge customer_unique_id to orders
    orders = orders.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id', how='left')
    print(f"Orders shape after merge: {orders.shape}")
    
    return logistics, rfm, geo, lifecycle, funnel, orders, order_items, products

def merge_customer_tables(logistics, rfm, geo, lifecycle, funnel):
    # merge all customer-level tables on customer_unique_id
    df = logistics.merge(rfm, on='customer_unique_id', how='inner', suffixes=('', '_rfm'))
    df = df.merge(geo[['customer_unique_id', 'customer_state']], on='customer_unique_id', how='left')
    df = df.merge(lifecycle[['customer_unique_id', 'lifecycle_stage', 'churn_risk_level']], on='customer_unique_id', how='left')
    # get last funnel stage for each customer
    funnel_last = funnel.groupby('customer_unique_id')['funnel_position'].last().reset_index()
    df = df.merge(funnel_last, on='customer_unique_id', how='left')
    print(f"Merged persona table shape: {df.shape}")
    print(f"Unique customers: {df['customer_unique_id'].nunique()}")
    print("Missing data summary:\n", df.isnull().sum())
    return df

def enrich_temporal_behavior(df, orders):
    # most common order hour
    order_hours = orders.groupby('customer_unique_id')['order_purchase_timestamp'].apply(lambda x: x.dt.hour.mode()[0] if not x.dt.hour.mode().empty else np.nan)
    df = df.merge(order_hours.rename('most_common_order_hour'), on='customer_unique_id', how='left')
    # weekend order ratio
    orders['is_weekend'] = orders['order_purchase_timestamp'].dt.weekday >= 5
    weekend_ratio = orders.groupby('customer_unique_id')['is_weekend'].mean().rename('weekend_order_ratio')
    df = df.merge(weekend_ratio, on='customer_unique_id', how='left')
    # holiday order ratio (TODO: fill with actual holiday dates)
    holiday_dates = set()  # TODO: fill with actual holiday dates as strings 'YYYY-MM-DD'
    orders['is_holiday'] = orders['order_purchase_timestamp'].dt.date.astype(str).isin(holiday_dates)
    holiday_ratio = orders.groupby('customer_unique_id')['is_holiday'].mean().rename('holiday_order_ratio')
    df = df.merge(holiday_ratio, on='customer_unique_id', how='left')
    # peak order season
    def get_season(dt):
        month = dt.month
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'
    orders['season'] = orders['order_purchase_timestamp'].apply(get_season)
    peak_season = orders.groupby('customer_unique_id')['season'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).rename('peak_order_season')
    df = df.merge(peak_season, on='customer_unique_id', how='left')
    return df

def enrich_category_behavior(df, orders, order_items, products):
    # merge orders with order_items and products to get category per order
    order_items = order_items.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
    order_items = order_items.merge(orders[['order_id', 'customer_unique_id']], on='order_id', how='left')
    # top 3 categories per customer
    def top_categories(x):
        return ', '.join([cat for cat, _ in Counter(x).most_common(3)])
    top3 = order_items.groupby('customer_unique_id')['product_category_name'].agg(top_categories).rename('top_3_categories')
    df = df.merge(top3, on='customer_unique_id', how='left')
    return df

def create_persona_label(row):
    # combine key labels for persona summary
    return f"{row['cluster_label']} {row['lifecycle_stage']} - {row['customer_state']}"

def main():
    print("=" * 80)
    print("UNIFIED CUSTOMER PERSONA TABLE CONSTRUCTION")
    print("=" * 80)

    logistics, rfm, geo, lifecycle, funnel, orders, order_items, products = load_all_data()
    # merge all customer-level tables
    df = merge_customer_tables(logistics, rfm, geo, lifecycle, funnel)
    # enrich with temporal behavior
    df = enrich_temporal_behavior(df, orders)
    # enrich with product category behavior
    df = enrich_category_behavior(df, orders, order_items, products)
    # create persona summary label
    df['persona_summary_label'] = df.apply(create_persona_label, axis=1)
    
    print("\n" + "=" * 50)
    print("DETAILED STATISTICS")
    print("=" * 50)
    
    print("\nRFM Cluster Distribution:")
    print(df['cluster_label'].value_counts())
    
    print("\nLifecycle Stage Distribution:")
    print(df['lifecycle_stage'].value_counts())
    
    print("\nTop 10 Customer States:")
    print(df['customer_state'].value_counts().head(10))
    
    print("\nTemporal Behavior Summary:")
    print(f"Average weekend order ratio: {df['weekend_order_ratio'].mean():.3f}")
    print(f"Most common order hour: {df['most_common_order_hour'].mode()[0] if not df['most_common_order_hour'].mode().empty else 'N/A'}")
    print("\nPeak order season distribution:")
    print(df['peak_order_season'].value_counts())
    
    print("\nTop 10 Persona Labels:")
    print(df['persona_summary_label'].value_counts().head(10))
    
    # data quality warnings
    print("\n" + "=" * 50)
    print("DATA QUALITY & BUSINESS INSIGHTS")
    print("=" * 50)
    
    missing_avg_days = df['avg_days_between_orders'].isnull().sum()
    missing_ratio = missing_avg_days / len(df) * 100
    print(f"\n Data Quality Warning:")
    print(f"   - {missing_avg_days:,} customers ({missing_ratio:.1f}%) have missing avg_days_between_orders")
    print(f"   - These are likely single-order customers")
    
    churned_ratio = (df['lifecycle_stage'] == 'Churned').sum() / len(df) * 100
    print(f"\n Business Insights:")
    print(f"   - Customer churn rate: {churned_ratio:.1f}%")
    print(f"   - Top 3 states account for {df['customer_state'].value_counts().head(3).sum() / len(df) * 100:.1f}% of customers")
    print(f"   - Weekend order ratio: {df['weekend_order_ratio'].mean():.1%}")
    
    # save final persona table
    output_file = 'output/final_customer_segments.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    print(f"Final persona table shape: {df.shape}")
    print(f"Unique customers: {df['customer_unique_id'].nunique()}")
    print("Missing data summary:\n", df.isnull().sum())
    print("\nPersona table construction complete!")

if __name__ == "__main__":
    main() 