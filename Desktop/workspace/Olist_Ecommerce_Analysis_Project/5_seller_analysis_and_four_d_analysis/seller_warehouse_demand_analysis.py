"""
Week 5 - Task 1: Seller-Level Warehouse Demand Summary
Analyze seller-level historical sales, product lifecycle stage, and stock turnover
Execution date: 2025-07-01
Update date: 2025-07-04
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    print("Loading data files...")
    
    # load product warehouse summary data
    product_warehouse = pd.read_csv('../week4_product_warehouse_analysis/output/product_warehouse_summary.csv')
    print(f"Product warehouse data: {product_warehouse.shape}")
    
    # load customer segments data
    customer_segments = pd.read_csv('../week3_customer_behavior/output/final_customer_segments.csv')
    print(f"Customer segments data: {customer_segments.shape}")
    
    # load order items data
    order_items = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    print(f"Order items data: {order_items.shape}")
    
    # load orders data
    orders = pd.read_csv('../data/processed_missing/olist_orders_dataset.csv')
    print(f"Orders data: {orders.shape}")
    
    # load customers data for regional analysis
    customers = pd.read_csv('../data/processed_missing/olist_customers_dataset.csv')
    print(f"Customers data: {customers.shape}")
    
    # load sellers data
    sellers = pd.read_csv('../data/processed_missing/olist_sellers_dataset.csv')
    print(f"Sellers data: {sellers.shape}")
    
    return product_warehouse, customer_segments, order_items, orders, customers, sellers

def merge_seller_info(product_warehouse, order_items):
    """merge product warehouse data with seller information"""
    print("Merging product and seller information...")
    
    # get product_id to seller_id mapping from order items
    product_seller_mapping = order_items[['product_id', 'seller_id']].drop_duplicates()
    
    # merge product warehouse data with seller info
    product_warehouse_with_seller = product_warehouse.merge(
        product_seller_mapping, 
        on='product_id', 
        how='left'
    )
    
    # check for missing seller_id
    missing_seller = product_warehouse_with_seller['seller_id'].isna().sum()
    if missing_seller > 0:
        print(f"Warning: {missing_seller} products missing seller_id")
        product_warehouse_with_seller = product_warehouse_with_seller.dropna(subset=['seller_id'])
    
    print(f"Merged product warehouse data: {product_warehouse_with_seller.shape}")
    return product_warehouse_with_seller

def analyze_sales_volume_and_lifecycle(product_warehouse_with_seller):
    """analyze sales volume and lifecycle distribution"""
    print("Analyzing sales volume and lifecycle distribution...")
    
    # aggregate data by seller
    seller_summary = product_warehouse_with_seller.groupby('seller_id').agg({
        'avg_monthly_sales': 'sum',  # total monthly sales
        'inventory_turnover_rate': 'mean',  # average turnover rate
        'product_id': 'count'  # total SKUs
    }).rename(columns={
        'avg_monthly_sales': 'total_monthly_sales',
        'inventory_turnover_rate': 'avg_turnover_rate',
        'product_id': 'total_skus'
    })
    
    # calculate SKU counts by lifecycle stage
    lifecycle_counts = product_warehouse_with_seller.groupby(['seller_id', 'lifecycle_stage']).size().unstack(fill_value=0)
    
    # rename columns and handle missing lifecycle stages
    lifecycle_counts.columns = [f"{col.lower()}_sku" for col in lifecycle_counts.columns]
    
    # ensure all lifecycle stages are present
    required_stages = ['introduction_sku', 'growth_sku', 'maturity_sku', 'decline_sku']
    for stage in required_stages:
        if stage not in lifecycle_counts.columns:
            lifecycle_counts[stage] = 0
    
    # merge data
    seller_summary = seller_summary.merge(lifecycle_counts, left_index=True, right_index=True, how='left')
    
    # fill missing values
    lifecycle_columns = [col for col in seller_summary.columns if col.endswith('_sku')]
    seller_summary[lifecycle_columns] = seller_summary[lifecycle_columns].fillna(0)
    
    print(f"Seller summary data: {seller_summary.shape}")
    return seller_summary

def analyze_order_flow(order_items, orders):
    """analyze order flow and seasonality"""
    print("Analyzing order flow and seasonality...")
    
    # Merge order items and orders data
    order_flow = order_items.merge(
        orders[['order_id', 'order_purchase_timestamp']], 
        on='order_id', 
        how='left'
    )
    
    # convert timestamps
    order_flow['order_purchase_timestamp'] = pd.to_datetime(order_flow['order_purchase_timestamp'])
    order_flow['month'] = order_flow['order_purchase_timestamp'].dt.month
    order_flow['year'] = order_flow['order_purchase_timestamp'].dt.year
    
    # count orders by seller and month
    monthly_orders = order_flow.groupby(['seller_id', 'year', 'month']).size().reset_index(name='order_count')
    
    # calculate peak sales month for each seller
    peak_months = monthly_orders.groupby('seller_id').apply(
        lambda x: x.loc[x['order_count'].idxmax(), ['year', 'month']]
    ).reset_index()
    
    # calculate seasonality score (sales fluctuation across months)
    seasonality_scores = monthly_orders.groupby('seller_id')['order_count'].agg([
        'std', 'mean'
    ]).reset_index()
    seasonality_scores['seasonality_score'] = seasonality_scores['std'] / seasonality_scores['mean']
    
    # merge peak months and seasonality scores
    order_flow_summary = peak_months.merge(seasonality_scores, on='seller_id', how='left')
    
    # convert months to readable format
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
        7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    order_flow_summary['peak_month_name'] = order_flow_summary['month'].map(month_names)
    
    print(f"Order flow analysis completed: {order_flow_summary.shape}")
    return order_flow_summary

def analyze_regional_demand(order_items, orders, customers):
    """analyze regional demand distribution"""
    print("Analyzing regional demand distribution...")
    
    # merge order items, orders, and customers data
    regional_data = order_items.merge(
        orders[['order_id', 'customer_id']], 
        on='order_id', 
        how='left'
    ).merge(
        customers[['customer_id', 'customer_state']], 
        on='customer_id', 
        how='left'
    )
    
    # count orders by seller and customer state
    regional_orders = regional_data.groupby(['seller_id', 'customer_state']).size().reset_index(name='order_count')
    
    # calculate total orders per seller
    seller_total_orders = regional_orders.groupby('seller_id')['order_count'].sum().reset_index(name='total_orders')
    
    # calculate percentage of orders per state for each seller
    regional_orders = regional_orders.merge(seller_total_orders, on='seller_id', how='left')
    regional_orders['state_percentage'] = (regional_orders['order_count'] / regional_orders['total_orders'] * 100).round(1)
    
    # get top 3 regions for each seller
    top_regions = regional_orders.groupby('seller_id').apply(
        lambda x: x.nlargest(3, 'state_percentage')[['customer_state', 'state_percentage']]
    ).reset_index()
    
    # format top regions as string
    regional_summary = top_regions.groupby('seller_id').apply(
        lambda x: ', '.join([f"{row['customer_state']}:{row['state_percentage']}%" for _, row in x.iterrows()])
    ).reset_index(name='top_regions')
    
    print(f"Regional demand analysis completed: {regional_summary.shape}")
    return regional_summary

def create_seller_demand_profile(seller_summary, order_flow_summary, regional_summary):
    """create seller demand profile"""
    print("Creating seller demand profile...")
    
    # merge all analysis results
    seller_profile = seller_summary.reset_index().merge(
        order_flow_summary[['seller_id', 'peak_month_name', 'seasonality_score']], 
        on='seller_id', 
        how='left'
    ).merge(
        regional_summary[['seller_id', 'top_regions']], 
        on='seller_id', 
        how='left'
    )
    
    # reorder columns
    column_order = [
        'seller_id', 'total_monthly_sales', 'avg_turnover_rate', 'total_skus',
        'introduction_sku', 'growth_sku', 'maturity_sku', 'decline_sku',
        'peak_month_name', 'seasonality_score', 'top_regions'
    ]
    
    # ensure all columns exist
    for col in column_order:
        if col not in seller_profile.columns:
            seller_profile[col] = 0
    
    seller_profile = seller_profile[column_order]
    
    # rename columns to match output requirements
    seller_profile = seller_profile.rename(columns={
        'total_monthly_sales': 'total_sales_volume',
        'avg_turnover_rate': 'avg_turnover_rate'
    })
    
    print(f"Seller demand profile created: {seller_profile.shape}")
    return seller_profile

def create_visualizations(seller_profile):
    print("Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Seller Warehouse Demand Analysis', fontsize=16, fontweight='bold')
    
    # 1. Top 10 sellers by sales volume
    top_sellers = seller_profile.nlargest(10, 'total_sales_volume')
    axes[0, 0].barh(range(len(top_sellers)), top_sellers['total_sales_volume'])
    axes[0, 0].set_yticks(range(len(top_sellers)))
    axes[0, 0].set_yticklabels([f"S{seller_id[:8]}..." for seller_id in top_sellers['seller_id']])
    axes[0, 0].set_xlabel('Total Sales Volume')
    axes[0, 0].set_title('Top 10 Sellers by Sales Volume')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. lifecycle stage distribution
    lifecycle_cols = ['introduction_sku', 'growth_sku', 'maturity_sku', 'decline_sku']
    lifecycle_totals = seller_profile[lifecycle_cols].sum()
    axes[0, 1].pie(lifecycle_totals.values, labels=[col.replace('_sku', '').title() for col in lifecycle_cols], 
                   autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Product Lifecycle Distribution')
    
    # 3. turnover rate distribution
    axes[1, 0].hist(seller_profile['avg_turnover_rate'].dropna(), bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Average Turnover Rate')
    axes[1, 0].set_ylabel('Number of Sellers')
    axes[1, 0].set_title('Distribution of Average Turnover Rates')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Sales volume vs turnover rate scatter
    axes[1, 1].scatter(seller_profile['avg_turnover_rate'], seller_profile['total_sales_volume'], alpha=0.6)
    axes[1, 1].set_xlabel('Average Turnover Rate')
    axes[1, 1].set_ylabel('Total Sales Volume')
    axes[1, 1].set_title('Sales Volume vs Turnover Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'seller_analysis_visualizations.png'), dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to: {os.path.join(output_dir, 'seller_analysis_visualizations.png')}")
    
    # create additional detailed visualization
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    
    # growth stage product ratio distribution
    seller_profile['growth_ratio'] = seller_profile['growth_sku'] / seller_profile['total_skus']
    axes2[0].hist(seller_profile['growth_ratio'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    axes2[0].set_xlabel('Growth Stage Product Ratio')
    axes2[0].set_ylabel('Number of Sellers')
    axes2[0].set_title('Distribution of Growth Stage Product Ratios')
    axes2[0].grid(True, alpha=0.3)
    
    # seasonality score distribution
    axes2[1].hist(seller_profile['seasonality_score'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    axes2[1].set_xlabel('Seasonality Score')
    axes2[1].set_ylabel('Number of Sellers')
    axes2[1].set_title('Distribution of Seasonality Scores')
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seller_detailed_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Detailed analysis saved to: {os.path.join(output_dir, 'seller_detailed_analysis.png')}")

def print_summary_statistics(seller_profile):
    print("\n" + "="*60)
    print("SELLER WAREHOUSE DEMAND ANALYSIS SUMMARY")
    print("="*60)
    
    # top 10 sellers by sales volume
    print("\nTOP 10 SELLERS BY SALES VOLUME:")
    top_sellers = seller_profile.nlargest(10, 'total_sales_volume')
    for idx, row in top_sellers.iterrows():
        print(f"  {row['seller_id']}: {row['total_sales_volume']:.2f} monthly sales")
    
    # sellers with highest growth stage product ratio
    print("\nSELLERS WITH HIGHEST GROWTH STAGE PRODUCT RATIO:")
    seller_profile['growth_ratio'] = seller_profile['growth_sku'] / seller_profile['total_skus']
    growth_sellers = seller_profile.nlargest(5, 'growth_ratio')
    for idx, row in growth_sellers.iterrows():
        print(f"  {row['seller_id']}: {row['growth_ratio']:.2%} growth stage products")
    
    # regional concentration analysis
    print("\nREGIONAL CONCENTRATION ANALYSIS:")
    print(f"  Total sellers: {len(seller_profile)}")
    print(f"  Average monthly sales: {seller_profile['total_sales_volume'].mean():.2f}")
    print(f"  Average turnover rate: {seller_profile['avg_turnover_rate'].mean():.2f}")
    
    # lifecycle distribution
    print("\nPRODUCT LIFECYCLE DISTRIBUTION:")
    lifecycle_cols = ['introduction_sku', 'growth_sku', 'maturity_sku', 'decline_sku']
    for col in lifecycle_cols:
        total_skus = seller_profile[col].sum()
        print(f"  {col.replace('_sku', '').title()}: {total_skus} SKUs")
    
    # seasonality analysis
    print("\nSEASONALITY ANALYSIS:")
    print(f"  Average seasonality score: {seller_profile['seasonality_score'].mean():.3f}")
    print(f"  Most common peak month: {seller_profile['peak_month_name'].mode().iloc[0] if not seller_profile['peak_month_name'].mode().empty else 'N/A'}")
    
    # turnover rate analysis
    print("\nTURNOVER RATE ANALYSIS:")
    print(f"  High turnover sellers (>5): {(seller_profile['avg_turnover_rate'] > 5).sum()}")
    print(f"  Medium turnover sellers (2-5): {((seller_profile['avg_turnover_rate'] >= 2) & (seller_profile['avg_turnover_rate'] <= 5)).sum()}")
    print(f"  Low turnover sellers (<2): {(seller_profile['avg_turnover_rate'] < 2).sum()}")

def save_outputs(seller_profile):
    print("\nSaving output files...")
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # save seller warehouse demand profile
    output_file = os.path.join(output_dir, 'seller_warehouse_demand.csv')
    seller_profile.to_csv(output_file, index=False)
    print(f"Seller warehouse demand profile saved to: {output_file}")

def main():
    print("Starting seller-level warehouse demand analysis...")
    
    try:
        # 1. 
        product_warehouse, customer_segments, order_items, orders, customers, sellers = load_data()
        
        # 2. Merge seller information
        product_warehouse_with_seller = merge_seller_info(product_warehouse, order_items)
        
        # 3. Analyze sales volume and lifecycle
        seller_summary = analyze_sales_volume_and_lifecycle(product_warehouse_with_seller)
        
        # 4. Analyze order flow
        order_flow_summary = analyze_order_flow(order_items, orders)
        
        # 5. Analyze regional demand
        regional_summary = analyze_regional_demand(order_items, orders, customers)
        
        # 6. Create seller demand profile
        seller_profile = create_seller_demand_profile(seller_summary, order_flow_summary, regional_summary)
        
        # 7. visualizations
        create_visualizations(seller_profile)
        
        # 8. statistics
        print_summary_statistics(seller_profile)
        
        # 9. save
        save_outputs(seller_profile)
        
        print("\nSeller-level warehouse demand analysis completed!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 