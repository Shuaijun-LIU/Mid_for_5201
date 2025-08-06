"""
Week 5 - Task 3: Seller-Level Lifecycle and Product Strategy Mapping
Combine customer behavior and product lifecycle to assess operational risks and growth potential
Execution date: 2025-07-03
Update date: 2025-07-05
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    print("Loading data files...")
    
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # load customer segments data
    week3_path = os.path.join(script_dir, '../week3_customer_behavior/output')
    customer_segments = pd.read_csv(os.path.join(week3_path, 'final_customer_segments.csv'))
    print(f"Customer segments data: {customer_segments.shape}")
    
    # product warehouse summary data
    week4_path = os.path.join(script_dir, '../week4_product_warehouse_analysis/output')
    product_warehouse = pd.read_csv(os.path.join(week4_path, 'product_warehouse_summary.csv'))
    print(f"Product warehouse data: {product_warehouse.shape}")
    
    # order items data
    data_path = os.path.join(script_dir, '../data/processed_missing')
    order_items = pd.read_csv(os.path.join(data_path, 'olist_order_items_dataset.csv'))
    print(f"Order items data: {order_items.shape}")
    
    # orders data
    orders = pd.read_csv(os.path.join(data_path, 'olist_orders_dataset.csv'))
    print(f"Orders data: {orders.shape}")
    
    # customers data
    customers = pd.read_csv(os.path.join(data_path, 'olist_customers_dataset.csv'))
    print(f"Customers data: {customers.shape}")
    
    return customer_segments, product_warehouse, order_items, orders, customers

def clean_lifecycle_data(customer_segments):
    print("Cleaning lifecycle and churn risk data...")
    
    # clean lifecycle_stage
    customer_segments['lifecycle_stage'] = customer_segments['lifecycle_stage'].str.title()
    
    # clean churn_risk_level
    customer_segments['churn_risk_level'] = customer_segments['churn_risk_level'].str.title()
    
    # check unique values
    print(f"Unique lifecycle stages: {customer_segments['lifecycle_stage'].unique()}")
    print(f"Unique churn risk levels: {customer_segments['churn_risk_level'].unique()}")
    
    return customer_segments

def merge_product_lifecycle_info(order_items, product_warehouse):
    """merge product lifecycle information and calculate percentages by seller"""
    print("Merging product lifecycle information...")
    
    # merge order items with product warehouse data
    order_product_lifecycle = order_items.merge(
        product_warehouse[['product_id', 'lifecycle_stage']], 
        on='product_id', 
        how='left'
    )
    
    # calculate product lifecycle percentages by seller
    seller_product_lifecycle = order_product_lifecycle.groupby(['seller_id', 'lifecycle_stage']).size().reset_index(name='count')
    
    # pivot to get counts by lifecycle stage
    seller_lifecycle_pivot = seller_product_lifecycle.pivot(
        index='seller_id', 
        columns='lifecycle_stage', 
        values='count'
    ).fillna(0)
    
    # calculate total products per seller
    seller_lifecycle_pivot['total_products'] = seller_lifecycle_pivot.sum(axis=1)
    
    # calculate percentages
    lifecycle_stages = ['Introduction', 'Growth', 'Maturity', 'Decline']
    for stage in lifecycle_stages:
        if stage in seller_lifecycle_pivot.columns:
            seller_lifecycle_pivot[f'product_{stage.lower()}_pct'] = (
                seller_lifecycle_pivot[stage] / seller_lifecycle_pivot['total_products'] * 100
            ).round(1)
        else:
            seller_lifecycle_pivot[f'product_{stage.lower()}_pct'] = 0
    
    # reset index to get seller_id as column
    seller_lifecycle_pivot = seller_lifecycle_pivot.reset_index()
    
    print(f"Product lifecycle analysis completed: {seller_lifecycle_pivot.shape}")
    return seller_lifecycle_pivot

def merge_customer_lifecycle_info(orders, customers, customer_segments, order_items):
    """merge customer lifecycle information and calculate metrics by seller"""
    print("Merging customer lifecycle information...")
    
    # merge orders with customers to get customer_unique_id
    order_customer = orders.merge(
        customers[['customer_id', 'customer_unique_id']], 
        on='customer_id', 
        how='left'
    )
    
    # merge with customer segments to get lifecycle and churn risk info
    order_customer_lifecycle = order_customer.merge(
        customer_segments[['customer_unique_id', 'lifecycle_stage', 'churn_risk_level']], 
        on='customer_unique_id', 
        how='left'
    )
    
    # merge with order items to get seller_id
    order_items_with_customer = order_items.merge(
        orders[['order_id', 'customer_id']], on='order_id'
    )
    
    # merge with customers to get customer_unique_id
    order_items_with_customer_unique = order_items_with_customer.merge(
        customers[['customer_id', 'customer_unique_id']], on='customer_id'
    )
    
    # merge with customer segments
    order_items_with_lifecycle = order_items_with_customer_unique.merge(
        customer_segments[['customer_unique_id', 'lifecycle_stage', 'churn_risk_level']], 
        on='customer_unique_id'
    )
    
    # get unique customer-seller pairs
    customer_seller_lifecycle = order_items_with_lifecycle[['seller_id', 'customer_unique_id', 'lifecycle_stage', 'churn_risk_level']].drop_duplicates()
    
    # calculate customer lifecycle metrics by seller
    seller_customer_lifecycle = customer_seller_lifecycle.groupby(['seller_id', 'lifecycle_stage']).size().reset_index(name='count')
    
    # pivot to get counts by lifecycle stage
    seller_customer_pivot = seller_customer_lifecycle.pivot(
        index='seller_id', 
        columns='lifecycle_stage', 
        values='count'
    ).fillna(0)
    
    # calculate churn risk metrics
    seller_churn_risk = customer_seller_lifecycle.groupby(['seller_id', 'churn_risk_level']).size().reset_index(name='count')
    seller_churn_pivot = seller_churn_risk.pivot(
        index='seller_id', 
        columns='churn_risk_level', 
        values='count'
    ).fillna(0)
    
    # calculate total customers per seller
    seller_customer_pivot['total_customers'] = seller_customer_pivot.sum(axis=1)
    
    # calculate percentages for lifecycle stages
    lifecycle_stages = ['Active', 'At Risk', 'Churned', 'New']
    for stage in lifecycle_stages:
        if stage in seller_customer_pivot.columns:
            seller_customer_pivot[f'{stage.lower().replace(" ", "_")}_customers'] = seller_customer_pivot[stage]
        else:
            seller_customer_pivot[f'{stage.lower().replace(" ", "_")}_customers'] = 0
    
    # calculate churn risk percentage with adjusted thresholds
    if 'High' in seller_churn_pivot.columns:
        seller_customer_pivot['churn_risk_high_pct'] = (
            seller_churn_pivot['High'] / seller_customer_pivot['total_customers'] * 100
        ).round(1)
    else:
        seller_customer_pivot['churn_risk_high_pct'] = 0
    
    # add combined active and new customers as "engaged customers"
    seller_customer_pivot['engaged_customers'] = (
        seller_customer_pivot.get('active_customers', 0) + 
        seller_customer_pivot.get('new_customers', 0)
    )
    
    # calculate customer retention rate
    total_engaged = seller_customer_pivot['engaged_customers'] + seller_customer_pivot.get('churned_customers', 0)
    seller_customer_pivot['retention_rate'] = (
        seller_customer_pivot['engaged_customers'] / total_engaged * 100
    ).round(1)
    
    # replace NaN values with 0
    seller_customer_pivot['retention_rate'] = seller_customer_pivot['retention_rate'].fillna(0)
    
    # reset index
    seller_customer_pivot = seller_customer_pivot.reset_index()
    
    print(f"Customer lifecycle analysis completed: {seller_customer_pivot.shape}")
    return seller_customer_pivot

def create_seller_strategic_profile(seller_product_lifecycle, seller_customer_lifecycle):
    """create comprehensive seller strategic profile"""
    print("Creating seller strategic profile...")
    
    # merge product and customer lifecycle data
    seller_profile = seller_product_lifecycle.merge(
        seller_customer_lifecycle, on='seller_id', how='outer'
    )
    
    # fill missing values
    seller_profile = seller_profile.fillna(0)
    
    # determine seller type based on product and customer lifecycle
    seller_profile['seller_type'] = seller_profile.apply(
        lambda row: determine_seller_type(row), axis=1
    )
    
    # select and reorder columns with new metrics
    required_columns = [
        'seller_id', 'product_growth_pct', 'product_maturity_pct', 
        'churn_risk_high_pct', 'engaged_customers', 'churned_customers', 
        'retention_rate', 'seller_type'
    ]
    
    # ensure all columns exist
    for col in required_columns:
        if col not in seller_profile.columns:
            seller_profile[col] = 0
    
    seller_profile = seller_profile[required_columns]
    
    print(f"Seller strategic profile created: {seller_profile.shape}")
    return seller_profile

def determine_seller_type(row):
    """determine seller type based on product and customer lifecycle with optimized thresholds"""
    growth_pct = row.get('product_growth_pct', 0)
    maturity_pct = row.get('product_maturity_pct', 0)
    decline_pct = row.get('product_decline_pct', 0)
    churn_risk_high_pct = row.get('churn_risk_high_pct', 0)
    active_customers = row.get('active_customers', 0)
    churned_customers = row.get('churned_customers', 0)
    new_customers = row.get('new_customers', 0)
    engaged_customers = row.get('engaged_customers', 0)
    retention_rate = row.get('retention_rate', 0)
    
    # handle missing values
    if pd.isna(churn_risk_high_pct):
        churn_risk_high_pct = 0
    if pd.isna(active_customers):
        active_customers = 0
    if pd.isna(churned_customers):
        churned_customers = 0
    if pd.isna(new_customers):
        new_customers = 0
    if pd.isna(engaged_customers):
        engaged_customers = 0
    if pd.isna(retention_rate):
        retention_rate = 0
    
    # extremely risky: very high churn risk (>70%)
    if churn_risk_high_pct > 70:
        return 'Extremely Risky'
    
    # risky: high churn risk (>50%) or very low retention rate (<20%)
    elif churn_risk_high_pct > 50 or retention_rate < 20:
        return 'Risky'
    
    # growth-oriented: high growth product % + good customer engagement
    elif growth_pct > 60 and engaged_customers > 3:
        return 'Growth-Oriented'
    
    # stabilizing: high maturity product % + good retention
    elif maturity_pct > 50 and retention_rate > 60:
        return 'Stabilizing'
    
    # declining: high decline product % + poor retention
    elif decline_pct > 40 and retention_rate < 30:
        return 'Declining'
    
    # balanced: moderate metrics across all dimensions
    else:
        return 'Balanced'

def create_visualizations(seller_profile):
    """create visualizations for seller lifecycle and product strategy analysis"""
    print("Creating visualizations...")
    
    # set up the plotting area
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Seller Lifecycle and Product Strategy Analysis', fontsize=16, fontweight='bold')
    
    # 1. product growth percentage distribution
    axes[0, 0].hist(seller_profile['product_growth_pct'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Product Growth Percentage')
    axes[0, 0].set_ylabel('Number of Sellers')
    axes[0, 0].set_title('Distribution of Product Growth Percentages')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. churn risk high percentage distribution
    axes[0, 1].hist(seller_profile['churn_risk_high_pct'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('High Churn Risk Percentage')
    axes[0, 1].set_ylabel('Number of Sellers')
    axes[0, 1].set_title('Distribution of High Churn Risk Percentages')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. seller type distribution
    seller_type_counts = seller_profile['seller_type'].value_counts()
    axes[1, 0].pie(seller_type_counts.values, labels=seller_type_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Distribution of Seller Types')
    
    # 4. product growth vs churn risk scatter
    axes[1, 1].scatter(seller_profile['churn_risk_high_pct'], seller_profile['product_growth_pct'], alpha=0.6)
    axes[1, 1].set_xlabel('High Churn Risk Percentage')
    axes[1, 1].set_ylabel('Product Growth Percentage')
    axes[1, 1].set_title('Product Growth vs Churn Risk')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save the visualization
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'seller_lifecycle_strategy_visualizations.png'), dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to: {os.path.join(output_dir, 'seller_lifecycle_strategy_visualizations.png')}")
    
    # create additional detailed visualization
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    
    # engaged vs churned customers
    axes2[0].scatter(seller_profile['engaged_customers'], seller_profile['churned_customers'], alpha=0.6)
    axes2[0].set_xlabel('Engaged Customers')
    axes2[0].set_ylabel('Churned Customers')
    axes2[0].set_title('Engaged vs Churned Customers')
    axes2[0].grid(True, alpha=0.3)
    
    # seller type by customer metrics
    seller_type_metrics = seller_profile.groupby('seller_type').agg({
        'engaged_customers': 'mean',
        'churned_customers': 'mean',
        'retention_rate': 'mean'
    }).reset_index()
    
    x = np.arange(len(seller_type_metrics))
    width = 0.25
    
    axes2[1].bar(x - width, seller_type_metrics['engaged_customers'], width, label='Engaged Customers', alpha=0.7)
    axes2[1].bar(x, seller_type_metrics['churned_customers'], width, label='Churned Customers', alpha=0.7)
    axes2[1].bar(x + width, seller_type_metrics['retention_rate'], width, label='Retention Rate %', alpha=0.7)
    
    axes2[1].set_xlabel('Seller Type')
    axes2[1].set_ylabel('Average Count/Percentage')
    axes2[1].set_title('Customer Metrics by Seller Type')
    axes2[1].set_xticks(x)
    axes2[1].set_xticklabels(seller_type_metrics['seller_type'], rotation=45)
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seller_lifecycle_detailed_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Detailed analysis saved to: {os.path.join(output_dir, 'seller_lifecycle_detailed_analysis.png')}")

def print_summary_statistics(seller_profile):
    print("\n" + "="*60)
    print("SELLER LIFECYCLE AND PRODUCT STRATEGY ANALYSIS SUMMARY")
    print("="*60)
    
    # top 10 sellers with highest product growth percentage
    print("\nTOP 10 SELLERS WITH HIGHEST PRODUCT GROWTH PERCENTAGE:")
    top_growth_sellers = seller_profile.nlargest(10, 'product_growth_pct')
    for idx, row in top_growth_sellers.iterrows():
        print(f"  {row['seller_id']}: {row['product_growth_pct']:.1f}% growth products")
    
    # sellers with the largest churned customer base
    print("\nSELLERS WITH THE LARGEST CHURNED CUSTOMER BASE:")
    top_churned_sellers = seller_profile.nlargest(10, 'churned_customers')
    for idx, row in top_churned_sellers.iterrows():
        print(f"  {row['seller_id']}: {row['churned_customers']} churned customers")
    
    # summary counts of seller types
    print("\nSUMMARY COUNTS OF SELLER TYPES:")
    seller_type_counts = seller_profile['seller_type'].value_counts()
    for seller_type, count in seller_type_counts.items():
        print(f"  {seller_type}: {count} sellers")
    
    # overall statistics
    print("\nOVERALL STATISTICS:")
    print(f"  Total sellers analyzed: {len(seller_profile)}")
    print(f"  Average product growth percentage: {seller_profile['product_growth_pct'].mean():.1f}%")
    print(f"  Average churn risk high percentage: {seller_profile['churn_risk_high_pct'].mean():.1f}%")
    print(f"  Average engaged customers: {seller_profile['engaged_customers'].mean():.1f}")
    print(f"  Average churned customers: {seller_profile['churned_customers'].mean():.1f}")
    print(f"  Average retention rate: {seller_profile['retention_rate'].mean():.1f}%")
    
    # risk analysis with improved thresholds
    print("\nRISK ANALYSIS:")
    print(f"  Extremely risky sellers (>70% churn risk): {(seller_profile['churn_risk_high_pct'] > 70).sum()}")
    print(f"  High risk sellers (>50% churn risk): {(seller_profile['churn_risk_high_pct'] > 50).sum()}")
    print(f"  Medium risk sellers (30-50% churn risk): {((seller_profile['churn_risk_high_pct'] >= 30) & (seller_profile['churn_risk_high_pct'] <= 50)).sum()}")
    print(f"  Low risk sellers (<30% churn risk): {(seller_profile['churn_risk_high_pct'] < 30).sum()}")
    
    # retention analysis
    print("\nRETENTION ANALYSIS:")
    print(f"  High retention sellers (>60%): {(seller_profile['retention_rate'] > 60).sum()}")
    print(f"  Medium retention sellers (30-60%): {((seller_profile['retention_rate'] >= 30) & (seller_profile['retention_rate'] <= 60)).sum()}")
    print(f"  Low retention sellers (<30%): {(seller_profile['retention_rate'] < 30).sum()}")

def save_outputs(seller_profile):
    """save output files"""
    print("\nSaving output files...")
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # save seller lifecycle product profile
    output_file = os.path.join(output_dir, 'seller_lifecycle_product_profile.csv')
    seller_profile.to_csv(output_file, index=False)
    print(f"Seller lifecycle product profile saved to: {output_file}")

def main():
    print("Starting seller lifecycle and product strategy analysis...")
    
    try:
        # 1. data
        customer_segments, product_warehouse, order_items, orders, customers = load_data()
        
        # 2. clean lifecycle data
        customer_segments = clean_lifecycle_data(customer_segments)
        
        # 3. merge product lifecycle info
        seller_product_lifecycle = merge_product_lifecycle_info(order_items, product_warehouse)
        
        # 4. merge customer lifecycle info
        seller_customer_lifecycle = merge_customer_lifecycle_info(orders, customers, customer_segments, order_items)
        
        # 5. create seller strategic profile
        seller_profile = create_seller_strategic_profile(seller_product_lifecycle, seller_customer_lifecycle)
        
        # 6. visualizations
        create_visualizations(seller_profile)
        
        # 7. statistics
        print_summary_statistics(seller_profile)
        
        # 8. save
        save_outputs(seller_profile)
        
        print("\nSeller lifecycle and product strategy analysis completed!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 