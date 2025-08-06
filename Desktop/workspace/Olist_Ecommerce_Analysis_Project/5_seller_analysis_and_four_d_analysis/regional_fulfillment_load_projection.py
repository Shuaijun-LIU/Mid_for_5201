"""
Week 5 - Task 4: Regional Fulfillment Load Projection
Estimate current and near-future fulfillment workload per region/state
Execution date: 2025-07-04
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
    
    # load customer segments 
    customer_segments = pd.read_csv('../week3_customer_behavior/output/final_customer_segments.csv')
    print(f"Customer segments data: {customer_segments.shape}")
    
    # load product warehouse summary data
    product_warehouse = pd.read_csv('../week4_product_warehouse_analysis/output/product_warehouse_summary.csv')
    print(f"Product warehouse data: {product_warehouse.shape}")
    
    # load Task 3 output: seller lifecycle product profile
    try:
        seller_lifecycle_profile = pd.read_csv('output/seller_lifecycle_product_profile.csv')
        print(f"Task 3 output loaded: {seller_lifecycle_profile.shape}")
        use_task3_output = True
    except FileNotFoundError:
        print("Warning: Task 3 output file not found. Will calculate from raw data.")
        seller_lifecycle_profile = None
        use_task3_output = False
    
    # load orders data
    orders = pd.read_csv('../data/processed_missing/olist_orders_dataset.csv')
    print(f"Orders data: {orders.shape}")
    
    # load order items data
    order_items = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    print(f"Order items data: {order_items.shape}")
    
    # load customers data
    customers = pd.read_csv('../data/processed_missing/olist_customers_dataset.csv')
    print(f"Customers data: {customers.shape}")
    
    # load sellers data
    sellers = pd.read_csv('../data/processed_missing/olist_sellers_dataset.csv')
    print(f"Sellers data: {sellers.shape}")
    
    return customer_segments, product_warehouse, orders, order_items, customers, sellers, seller_lifecycle_profile, use_task3_output

def clean_lifecycle_data(customer_segments):
    """clean lifecycle and churn risk fields to consistent casing"""
    print("Cleaning lifecycle and churn risk data...")
    
    # clean lifecycle_stage
    customer_segments['lifecycle_stage'] = customer_segments['lifecycle_stage'].str.title()
    
    # clean churn_risk_level
    customer_segments['churn_risk_level'] = customer_segments['churn_risk_level'].str.title()
    
    # standardize state names
    customer_segments['customer_state'] = customer_segments['customer_state'].str.upper()
    
    print(f"Unique lifecycle stages: {customer_segments['lifecycle_stage'].unique()}")
    print(f"Unique churn risk levels: {customer_segments['churn_risk_level'].unique()}")
    print(f"Unique states: {customer_segments['customer_state'].unique()}")
    
    return customer_segments

def compute_regional_customer_demand(customer_segments):
    """compute regional active customer demand"""
    print("Computing regional customer demand...")
    
    # group by customer state and calculate customer counts by lifecycle stage
    regional_customer_demand = customer_segments.groupby('customer_state').agg({
        'lifecycle_stage': lambda x: (x == 'Active').sum(),  # active customer count
        'customer_unique_id': 'count'  # total customer count
    }).reset_index()
    
    # rename columns
    regional_customer_demand.columns = ['state', 'active_customer_count', 'total_customer_count']
    
    # calculate additional lifecycle counts
    at_risk_count = customer_segments[customer_segments['lifecycle_stage'] == 'At Risk'].groupby('customer_state').size()
    churned_count = customer_segments[customer_segments['lifecycle_stage'] == 'Churned'].size
    new_count = customer_segments[customer_segments['lifecycle_stage'] == 'New'].groupby('customer_state').size()
    
    # merge additional counts
    regional_customer_demand = regional_customer_demand.merge(
        at_risk_count.reset_index(name='at_risk_customer_count'), 
        left_on='state', right_on='customer_state', how='left'
    ).drop('customer_state', axis=1)
    
    regional_customer_demand['churned_customer_count'] = churned_count
    regional_customer_demand['new_customer_count'] = new_count
    
    # fill missing values
    regional_customer_demand = regional_customer_demand.fillna(0)
    
    # calculate weighted customer demand
    regional_customer_demand['weighted_customer_demand'] = (
        regional_customer_demand['active_customer_count'] * 1.2 + 
        regional_customer_demand['at_risk_customer_count'] * 0.8 + 
        regional_customer_demand['new_customer_count'] * 1.0
    )
    
    print(f"Regional customer demand computed: {regional_customer_demand.shape}")
    return regional_customer_demand

def estimate_seller_fulfillment_load(order_items, sellers, product_warehouse, seller_lifecycle_profile=None, use_task3_output=False):
    """estimate seller-side fulfillment load by state"""
    print("Estimating seller fulfillment load...")
    
    if use_task3_output and seller_lifecycle_profile is not None:
        print("Using Task 3 output data for seller fulfillment load calculation...")
        
        # merge seller lifecycle profile with sellers to get seller state
        seller_lifecycle_with_state = seller_lifecycle_profile.merge(
            sellers[['seller_id', 'seller_state']], 
            on='seller_id', 
            how='left'
        )
        
        # standardize seller state names
        seller_lifecycle_with_state['seller_state'] = seller_lifecycle_with_state['seller_state'].str.upper()
        
        # calculate seller activity by state using Task 3 data
        seller_activity_by_state = seller_lifecycle_with_state.groupby('seller_state').agg({
            'seller_id': 'nunique',  # unique active sellers
            'product_growth_pct': 'mean',  # average growth percentage
            'engaged_customers': 'sum',  # total engaged customers
            'churned_customers': 'sum'  # total churned customers
        }).reset_index()
        
        # rename columns
        seller_activity_by_state.columns = ['state', 'active_seller_count', 'avg_growth_pct', 'total_engaged_customers', 'total_churned_customers']
        
        # calculate proxy fulfillment load using Task 3 data
        seller_activity_by_state['seller_fulfillment_load'] = (
            seller_activity_by_state['active_seller_count'] * 
            seller_activity_by_state['avg_growth_pct'] * 10  # scaled factor
        )
        
        # add growth products count (estimated from growth percentage)
        seller_activity_by_state['growth_products'] = (
            seller_activity_by_state['active_seller_count'] * 
            seller_activity_by_state['avg_growth_pct'] / 100 * 10  # estimated count
        ).round(0)
        
    else:
        print("Calculating seller fulfillment load from raw data...")
        
        # merge order items with sellers to get seller state
        order_seller_state = order_items.merge(
            sellers[['seller_id', 'seller_state']], 
            on='seller_id', 
            how='left'
        )
        
        # merge with product warehouse to get lifecycle stage
        order_seller_product = order_seller_state.merge(
            product_warehouse[['product_id', 'lifecycle_stage']], 
            on='product_id', 
            how='left'
        )
        
        # standardize seller state names
        order_seller_product['seller_state'] = order_seller_product['seller_state'].str.upper()
        
        # calculate seller activity by state
        seller_activity_by_state = order_seller_product.groupby('seller_state').agg({
            'seller_id': 'nunique',  # unique active sellers
            'product_id': 'count',   # total products
            'lifecycle_stage': lambda x: (x == 'Growth').sum()  # growth products
        }).reset_index()
        
        # rename columns
        seller_activity_by_state.columns = ['state', 'active_seller_count', 'total_products', 'growth_products']
        
        # calculate total product growth percentage across sellers
        seller_growth_pct = order_seller_product.groupby('seller_state').apply(
            lambda x: (x['lifecycle_stage'] == 'Growth').sum() / len(x) * 100 if len(x) > 0 else 0
        ).reset_index(name='total_growth_pct')
        
        seller_activity_by_state = seller_activity_by_state.merge(
            seller_growth_pct, left_on='state', right_on='seller_state', how='left'
        ).drop('seller_state', axis=1)
        
        # calculate proxy fulfillment load
        seller_activity_by_state['seller_fulfillment_load'] = (
            seller_activity_by_state['active_seller_count'] * 
            seller_activity_by_state['total_growth_pct'] * 100
        )
    
    print(f"Seller fulfillment load estimated: {seller_activity_by_state.shape}")
    return seller_activity_by_state

def integrate_product_lifecycle_influence(order_items, sellers, product_warehouse):
    """integrate product lifecycle influence by state"""
    print("Integrating product lifecycle influence...")
    
    # merge order items with sellers and product warehouse
    order_seller_product = order_items.merge(
        sellers[['seller_id', 'seller_state']], 
        on='seller_id', 
        how='left'
    ).merge(
        product_warehouse[['product_id', 'lifecycle_stage']], 
        on='product_id', 
        how='left'
    )
    
    # standardize seller state names
    order_seller_product['seller_state'] = order_seller_product['seller_state'].str.upper()
    
    # aggregate product counts by lifecycle stage and state
    product_lifecycle_by_state = order_seller_product.groupby(['seller_state', 'lifecycle_stage']).size().reset_index(name='count')
    
    # pivot to get counts by lifecycle stage
    product_lifecycle_pivot = product_lifecycle_by_state.pivot(
        index='seller_state', 
        columns='lifecycle_stage', 
        values='count'
    ).fillna(0)
    
    # rename columns
    product_lifecycle_pivot.columns = [f"{col.lower()}_product_count" for col in product_lifecycle_pivot.columns]
    product_lifecycle_pivot = product_lifecycle_pivot.reset_index()
    product_lifecycle_pivot = product_lifecycle_pivot.rename(columns={'seller_state': 'state'})
    
    # calculate product growth contribution
    if 'growth_product_count' in product_lifecycle_pivot.columns:
        product_lifecycle_pivot['product_growth_contribution'] = product_lifecycle_pivot['growth_product_count'] * 2
    else:
        product_lifecycle_pivot['product_growth_contribution'] = 0
    
    print(f"Product lifecycle influence integrated: {product_lifecycle_pivot.shape}")
    return product_lifecycle_pivot

def construct_fulfillment_projection(regional_customer_demand, seller_fulfillment_load, product_lifecycle_influence):
    """construct fulfillment projection table"""
    print("Constructing fulfillment projection table...")
    
    # merge all components
    fulfillment_projection = regional_customer_demand.merge(
        seller_fulfillment_load, 
        on='state', 
        how='outer'
    ).merge(
        product_lifecycle_influence, 
        on='state', 
        how='outer'
    )
    
    # fill missing values
    fulfillment_projection = fulfillment_projection.fillna(0)
    
    # normalize each component
    fulfillment_projection['normalized_customer_demand'] = (
        fulfillment_projection['weighted_customer_demand'] / 
        fulfillment_projection['weighted_customer_demand'].max() * 1000
    )
    
    fulfillment_projection['normalized_seller_load'] = (
        fulfillment_projection['seller_fulfillment_load'] / 
        fulfillment_projection['seller_fulfillment_load'].max() * 1000
    ) if fulfillment_projection['seller_fulfillment_load'].max() > 0 else 0
    
    fulfillment_projection['normalized_product_contribution'] = (
        fulfillment_projection['product_growth_contribution'] / 
        fulfillment_projection['product_growth_contribution'].max() * 1000
    ) if fulfillment_projection['product_growth_contribution'].max() > 0 else 0
    
    # calculate projected warehouse load
    fulfillment_projection['projected_warehouse_load'] = (
        fulfillment_projection['normalized_customer_demand'] + 
        fulfillment_projection['normalized_seller_load'] + 
        fulfillment_projection['normalized_product_contribution']
    )
    
    # select and reorder columns
    required_columns = [
        'state', 'active_customer_count', 'growth_products', 
        'seller_fulfillment_load', 'projected_warehouse_load'
    ]
    
    # ensure all columns exist
    for col in required_columns:
        if col not in fulfillment_projection.columns:
            fulfillment_projection[col] = 0
    
    fulfillment_projection = fulfillment_projection[required_columns]
    
    # sort by projected warehouse load
    fulfillment_projection = fulfillment_projection.sort_values('projected_warehouse_load', ascending=False)
    
    print(f"Fulfillment projection constructed: {fulfillment_projection.shape}")
    return fulfillment_projection

def create_visualizations(fulfillment_projection):
    """create visualizations for regional fulfillment load projection"""
    print("Creating visualizations...")
    
    # set up the plotting area
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Regional Fulfillment Load Projection', fontsize=16, fontweight='bold')
    
    # 1. top 10 states by projected warehouse load
    top_states = fulfillment_projection.head(10)
    axes[0, 0].barh(range(len(top_states)), top_states['projected_warehouse_load'])
    axes[0, 0].set_yticks(range(len(top_states)))
    axes[0, 0].set_yticklabels(top_states['state'])
    axes[0, 0].set_xlabel('Projected Warehouse Load')
    axes[0, 0].set_title('Top 10 States by Projected Warehouse Load')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. active customers vs seller fulfillment load
    axes[0, 1].scatter(fulfillment_projection['active_customer_count'], 
                      fulfillment_projection['seller_fulfillment_load'], alpha=0.6)
    axes[0, 1].set_xlabel('Active Customer Count')
    axes[0, 1].set_ylabel('Seller Fulfillment Load')
    axes[0, 1].set_title('Active Customers vs Seller Fulfillment Load')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. growth products distribution
    axes[1, 0].hist(fulfillment_projection['growth_products'].dropna(), bins=15, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Growth Products Count')
    axes[1, 0].set_ylabel('Number of States')
    axes[1, 0].set_title('Distribution of Growth Products by State')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. projected warehouse load distribution
    axes[1, 1].hist(fulfillment_projection['projected_warehouse_load'].dropna(), bins=15, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Projected Warehouse Load')
    axes[1, 1].set_ylabel('Number of States')
    axes[1, 1].set_title('Distribution of Projected Warehouse Load')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save the visualization
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'regional_fulfillment_load_visualizations.png'), dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to: {os.path.join(output_dir, 'regional_fulfillment_load_visualizations.png')}")
    
    # create additional detailed visualization
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    
    # load components breakdown for top states
    top_5_states = fulfillment_projection.head(5)
    x = np.arange(len(top_5_states))
    width = 0.25
    
    axes2[0].bar(x - width, top_5_states['active_customer_count'], width, label='Active Customers', alpha=0.7)
    axes2[0].bar(x, top_5_states['seller_fulfillment_load'], width, label='Seller Load', alpha=0.7)
    axes2[0].bar(x + width, top_5_states['growth_products'], width, label='Growth Products', alpha=0.7)
    
    axes2[0].set_xlabel('State')
    axes2[0].set_ylabel('Count/Load')
    axes2[0].set_title('Load Components for Top 5 States')
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels(top_5_states['state'])
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # load vs capacity gap analysis
    axes2[1].scatter(fulfillment_projection['active_customer_count'], 
                    fulfillment_projection['projected_warehouse_load'], alpha=0.6)
    axes2[1].set_xlabel('Active Customer Count')
    axes2[1].set_ylabel('Projected Warehouse Load')
    axes2[1].set_title('Customer Demand vs Warehouse Load')
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regional_fulfillment_detailed_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Detailed analysis saved to: {os.path.join(output_dir, 'regional_fulfillment_detailed_analysis.png')}")

def print_summary_statistics(fulfillment_projection):
    """print summary statistics"""
    print("\n" + "="*60)
    print("REGIONAL FULFILLMENT LOAD PROJECTION SUMMARY")
    print("="*60)
    
    # top 5 states by projected warehouse load
    print("\nTOP 5 STATES BY PROJECTED WAREHOUSE LOAD:")
    top_states = fulfillment_projection.head(5)
    for idx, row in top_states.iterrows():
        print(f"  {row['state']}: {row['projected_warehouse_load']:.1f} projected load")
    
    # states with high demand but few active sellers
    print("\nSTATES WITH HIGH DEMAND BUT FEW ACTIVE SELLERS:")
    high_demand_low_sellers = fulfillment_projection[
        (fulfillment_projection['active_customer_count'] > fulfillment_projection['active_customer_count'].median()) &
        (fulfillment_projection['seller_fulfillment_load'] < fulfillment_projection['seller_fulfillment_load'].median())
    ].head(5)
    
    for idx, row in high_demand_low_sellers.iterrows():
        print(f"  {row['state']}: {row['active_customer_count']} customers, {row['seller_fulfillment_load']:.1f} seller load")
    
    # summary of growth vs maturity vs decline product contributions
    print("\nSUMMARY OF PRODUCT CONTRIBUTIONS BY STATE:")
    print(f"  Average growth products per state: {fulfillment_projection['growth_products'].mean():.1f}")
    print(f"  Total growth products across all states: {fulfillment_projection['growth_products'].sum():.0f}")
    
    # overall statistics
    print("\nOVERALL STATISTICS:")
    print(f"  Total states analyzed: {len(fulfillment_projection)}")
    print(f"  Average projected warehouse load: {fulfillment_projection['projected_warehouse_load'].mean():.1f}")
    print(f"  Average active customers per state: {fulfillment_projection['active_customer_count'].mean():.1f}")
    print(f"  Average seller fulfillment load: {fulfillment_projection['seller_fulfillment_load'].mean():.1f}")
    
    # load distribution analysis
    print("\nLOAD DISTRIBUTION ANALYSIS:")
    print(f"  High load states (>1000): {(fulfillment_projection['projected_warehouse_load'] > 1000).sum()}")
    print(f"  Medium load states (500-1000): {((fulfillment_projection['projected_warehouse_load'] >= 500) & (fulfillment_projection['projected_warehouse_load'] <= 1000)).sum()}")
    print(f"  Low load states (<500): {(fulfillment_projection['projected_warehouse_load'] < 500).sum()}")

def save_outputs(fulfillment_projection):
    print("\nSaving output files...")
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # save regional fulfillment projection
    output_file = os.path.join(output_dir, 'regional_fulfillment_projection.csv')
    fulfillment_projection.to_csv(output_file, index=False)
    print(f"Regional fulfillment projection saved to: {output_file}")

def main():
    print("Starting regional fulfillment load projection analysis...")
    
    try:
        # 1. 
        customer_segments, product_warehouse, orders, order_items, customers, sellers, seller_lifecycle_profile, use_task3_output = load_data()
        
        # display Task 3 output usage information
        if use_task3_output:
            print(f"\nTask 3 output file successfully loaded and will be used for analysis.")
            print(f"Task 3 data contains {len(seller_lifecycle_profile)} seller records.")
        else:
            print(f"\nTask 3 output file not found. Analysis will proceed using raw data calculations.")
        
        # 2. clean lifecycle data
        customer_segments = clean_lifecycle_data(customer_segments)
        
        # 3. compute regional customer demand
        regional_customer_demand = compute_regional_customer_demand(customer_segments)
        
        # 4. estimate seller fulfillment load
        seller_fulfillment_load = estimate_seller_fulfillment_load(order_items, sellers, product_warehouse, seller_lifecycle_profile, use_task3_output)
        
        # 5. integrate product lifecycle influence
        product_lifecycle_influence = integrate_product_lifecycle_influence(order_items, sellers, product_warehouse)
        
        # 6. construct fulfillment projection
        fulfillment_projection = construct_fulfillment_projection(
            regional_customer_demand, seller_fulfillment_load, product_lifecycle_influence
        )
        
        # 7.visualizations
        create_visualizations(fulfillment_projection)
        
        # 8. 
        print_summary_statistics(fulfillment_projection)
        
        # 9. save 
        save_outputs(fulfillment_projection)
        
        print("\nRegional fulfillment load projection analysis completed!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 