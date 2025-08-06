"""
Week 5 - Task 2: Seller Distance & Fulfillment Complexity Calculation
Calculate estimated average delivery distance and fulfillment complexity indicators
Execution date: 2025-07-02
Update date: 2025-07-05
"""

import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    calculate great-circle distance between two points using haversine formula
    returns distance in kilometers
    """
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # radius of earth in kilometers
    r = 6371
    
    return c * r

def load_data():
    print("Loading data files...")
    
    # load order items data
    order_items = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    print(f"Order items data: {order_items.shape}")
    
    # load orders data
    orders = pd.read_csv('../data/processed_missing/olist_orders_dataset.csv')
    print(f"Orders data: {orders.shape}")
    
    # load customers data
    customers = pd.read_csv('../data/processed_missing/olist_customers_dataset.csv')
    print(f"Customers data: {customers.shape}")
    
    # load sellers data
    sellers = pd.read_csv('../data/processed_missing/olist_sellers_dataset.csv')
    print(f"Sellers data: {sellers.shape}")
    
    # load geolocation data
    geolocation = pd.read_csv('../data/processed_missing/olist_geolocation_dataset.csv')
    print(f"Geolocation data: {geolocation.shape}")
    
    # load products data
    products = pd.read_csv('../data/processed_missing/olist_products_dataset.csv')
    print(f"Products data: {products.shape}")
    
    return order_items, orders, customers, sellers, geolocation, products

def prepare_geolocation_data(geolocation):
    """prepare geolocation data by averaging coordinates per zip code prefix"""
    print("Preparing geolocation data...")
    
    # group by zip code prefix and calculate mean coordinates
    geolocation_avg = geolocation.groupby('geolocation_zip_code_prefix').agg({
        'geolocation_lat': 'mean',
        'geolocation_lng': 'mean',
        'geolocation_city': 'first',
        'geolocation_state': 'first'
    }).reset_index()
    
    print(f"Geolocation data prepared: {geolocation_avg.shape}")
    return geolocation_avg

def merge_seller_customer_geolocation(order_items, orders, customers, sellers, geolocation_avg):
    """merge seller and customer geolocation data for each order"""
    print("Merging seller and customer geolocation data...")
    
    # merge order items with orders to get customer_id
    order_customer = order_items.merge(
        orders[['order_id', 'customer_id']], 
        on='order_id', 
        how='left'
    )
    
    # merge with customer data to get customer location
    order_customer_location = order_customer.merge(
        customers[['customer_id', 'customer_zip_code_prefix', 'customer_state', 'customer_city']], 
        on='customer_id', 
        how='left'
    )
    
    # merge with seller data to get seller location
    order_seller_customer = order_customer_location.merge(
        sellers[['seller_id', 'seller_zip_code_prefix', 'seller_state', 'seller_city']], 
        on='seller_id', 
        how='left'
    )
    
    # merge with geolocation data for customer coordinates
    order_with_coords = order_seller_customer.merge(
        geolocation_avg[['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']], 
        left_on='customer_zip_code_prefix', 
        right_on='geolocation_zip_code_prefix', 
        how='left',
        suffixes=('', '_customer')
    )
    
    # merge with geolocation data for seller coordinates
    order_with_all_coords = order_with_coords.merge(
        geolocation_avg[['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']], 
        left_on='seller_zip_code_prefix', 
        right_on='geolocation_zip_code_prefix', 
        how='left',
        suffixes=('_customer', '_seller')
    )
    
    # rename columns for clarity
    order_with_all_coords = order_with_all_coords.rename(columns={
        'geolocation_lat_customer': 'customer_lat',
        'geolocation_lng_customer': 'customer_lng',
        'geolocation_lat_seller': 'seller_lat',
        'geolocation_lng_seller': 'seller_lng'
    })
    
    print(f"Merged order data with coordinates: {order_with_all_coords.shape}")
    return order_with_all_coords

def calculate_distances(order_with_coords):
    """calculate delivery distances for each order"""
    print("Calculating delivery distances...")
    
    # filter out orders with missing coordinates
    valid_coords = order_with_coords.dropna(subset=['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng']).copy()
    
    print(f"Orders with valid coordinates: {len(valid_coords)} out of {len(order_with_coords)}")
    
    # calculate distances using vectorized operations
    distances = []
    for idx, row in valid_coords.iterrows():
        try:
            distance = haversine_distance(
                row['seller_lat'], row['seller_lng'],
                row['customer_lat'], row['customer_lng']
            )
            distances.append(distance)
        except:
            distances.append(np.nan)
    
    valid_coords['delivery_distance_km'] = distances
    
    # remove any orders with invalid distances
    valid_coords = valid_coords.dropna(subset=['delivery_distance_km'])
    
    print(f"Orders with valid distances: {len(valid_coords)}")
    return valid_coords

def calculate_seller_distance_metrics(orders_with_distances):
    """calculate distance metrics for each seller"""
    print("Calculating seller distance metrics...")
    
    # aggregate distance metrics by seller
    seller_distance_metrics = orders_with_distances.groupby('seller_id').agg({
        'delivery_distance_km': ['mean', 'max', 'std'],
        'order_id': 'count'
    }).reset_index()
    
    # flatten column names
    seller_distance_metrics.columns = [
        'seller_id', 'avg_delivery_distance', 'max_delivery_distance', 
        'std_delivery_distance', 'order_count'
    ]
    
    print(f"Seller distance metrics calculated: {seller_distance_metrics.shape}")
    return seller_distance_metrics

def calculate_fulfillment_complexity(orders_with_distances, products):
    """calculate fulfillment complexity indicators for each seller"""
    print("Calculating fulfillment complexity indicators...")
    
    # merge with products data to get product categories
    orders_with_products = orders_with_distances.merge(
        products[['product_id', 'product_category_name']], 
        on='product_id', 
        how='left'
    )
    
    # calculate complexity metrics by seller
    complexity_metrics = orders_with_products.groupby('seller_id').agg({
        'customer_state': 'nunique',  # unique customer states
        'product_category_name': 'nunique',  # unique product categories
        'order_id': 'count'  # total orders
    }).reset_index()
    
    # rename columns
    complexity_metrics.columns = [
        'seller_id', 'unique_customer_states', 'unique_product_categories', 'order_count'
    ]
    
    print(f"Fulfillment complexity metrics calculated: {complexity_metrics.shape}")
    return complexity_metrics

def create_seller_complexity_profile(seller_distance_metrics, complexity_metrics):
    """create comprehensive seller complexity profile"""
    print("Creating seller complexity profile...")
    
    # merge distance and complexity metrics
    seller_profile = seller_distance_metrics.merge(
        complexity_metrics[['seller_id', 'unique_customer_states', 'unique_product_categories']], 
        on='seller_id', 
        how='left'
    )
    
    # ensure all columns exist and handle missing values
    required_columns = [
        'seller_id', 'avg_delivery_distance', 'max_delivery_distance', 
        'std_delivery_distance', 'unique_customer_states', 
        'unique_product_categories', 'order_count'
    ]
    
    # create a copy to avoid SettingWithCopyWarning
    seller_profile = seller_profile.copy()
    
    for col in required_columns:
        if col not in seller_profile.columns:
            seller_profile[col] = 0
    
    seller_profile = seller_profile[required_columns]
    
    # fill missing values
    seller_profile = seller_profile.fillna(0)
    
    print(f"Seller complexity profile created: {seller_profile.shape}")
    return seller_profile

def create_visualizations(seller_profile):
    print("Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Seller Fulfillment Complexity Analysis', fontsize=16, fontweight='bold')
    
    # 1. average delivery distance distribution
    axes[0, 0].hist(seller_profile['avg_delivery_distance'].dropna(), bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Average Delivery Distance (km)')
    axes[0, 0].set_ylabel('Number of Sellers')
    axes[0, 0].set_title('Distribution of Average Delivery Distances')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. geographic dispersion (unique customer states)
    axes[0, 1].hist(seller_profile['unique_customer_states'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Number of Unique Customer States')
    axes[0, 1].set_ylabel('Number of Sellers')
    axes[0, 1].set_title('Geographic Dispersion of Sellers')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. product category variety
    axes[1, 0].hist(seller_profile['unique_product_categories'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Number of Unique Product Categories')
    axes[1, 0].set_ylabel('Number of Sellers')
    axes[1, 0].set_title('Product Category Variety')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. distance vs geographic dispersion scatter
    axes[1, 1].scatter(seller_profile['unique_customer_states'], seller_profile['avg_delivery_distance'], alpha=0.6)
    axes[1, 1].set_xlabel('Number of Unique Customer States')
    axes[1, 1].set_ylabel('Average Delivery Distance (km)')
    axes[1, 1].set_title('Distance vs Geographic Dispersion')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save the visualization
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'seller_fulfillment_complexity_visualizations.png'), dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to: {os.path.join(output_dir, 'seller_fulfillment_complexity_visualizations.png')}")
    
    # create additional detailed visualization
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    
    # complexity score (combination of distance and dispersion)
    seller_profile['complexity_score'] = (
        seller_profile['avg_delivery_distance'] * 0.4 + 
        seller_profile['unique_customer_states'] * 10 + 
        seller_profile['unique_product_categories'] * 5
    )
    axes2[0].hist(seller_profile['complexity_score'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    axes2[0].set_xlabel('Fulfillment Complexity Score')
    axes2[0].set_ylabel('Number of Sellers')
    axes2[0].set_title('Distribution of Fulfillment Complexity Scores')
    axes2[0].grid(True, alpha=0.3)
    
    # order count vs complexity
    axes2[1].scatter(seller_profile['order_count'], seller_profile['complexity_score'], alpha=0.6)
    axes2[1].set_xlabel('Order Count')
    axes2[1].set_ylabel('Fulfillment Complexity Score')
    axes2[1].set_title('Order Volume vs Complexity')
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seller_complexity_detailed_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Detailed analysis saved to: {os.path.join(output_dir, 'seller_complexity_detailed_analysis.png')}")

def print_summary_statistics(seller_profile):
    print("\n" + "="*60)
    print("SELLER FULFILLMENT COMPLEXITY ANALYSIS SUMMARY")
    print("="*60)
    
    # top 10 sellers with longest average delivery distances
    print("\nTOP 10 SELLERS WITH LONGEST AVERAGE DELIVERY DISTANCES:")
    top_distance_sellers = seller_profile.nlargest(10, 'avg_delivery_distance')
    for idx, row in top_distance_sellers.iterrows():
        print(f"  {row['seller_id']}: {row['avg_delivery_distance']:.1f} km average distance")
    
    # sellers serving the widest geographic regions
    print("\nSELLERS SERVING THE WIDEST GEOGRAPHIC REGIONS:")
    top_geographic_sellers = seller_profile.nlargest(10, 'unique_customer_states')
    for idx, row in top_geographic_sellers.iterrows():
        print(f"  {row['seller_id']}: {row['unique_customer_states']} unique states")
    
    # sellers with the broadest product category coverage
    print("\nSELLERS WITH THE BROADEST PRODUCT CATEGORY COVERAGE:")
    top_category_sellers = seller_profile.nlargest(10, 'unique_product_categories')
    for idx, row in top_category_sellers.iterrows():
        print(f"  {row['seller_id']}: {row['unique_product_categories']} unique categories")
    
    # overall statistics
    print("\nOVERALL STATISTICS:")
    print(f"  Total sellers analyzed: {len(seller_profile)}")
    print(f"  Average delivery distance: {seller_profile['avg_delivery_distance'].mean():.1f} km")
    print(f"  Average geographic dispersion: {seller_profile['unique_customer_states'].mean():.1f} states")
    print(f"  Average product variety: {seller_profile['unique_product_categories'].mean():.1f} categories")
    
    # distance analysis
    print("\nDISTANCE ANALYSIS:")
    print(f"  Sellers with long distance (>500km): {(seller_profile['avg_delivery_distance'] > 500).sum()}")
    print(f"  Sellers with medium distance (200-500km): {((seller_profile['avg_delivery_distance'] >= 200) & (seller_profile['avg_delivery_distance'] <= 500)).sum()}")
    print(f"  Sellers with short distance (<200km): {(seller_profile['avg_delivery_distance'] < 200).sum()}")
    
    # complexity analysis
    print("\nCOMPLEXITY ANALYSIS:")
    print(f"  High complexity sellers (>10 states): {(seller_profile['unique_customer_states'] > 10).sum()}")
    print(f"  Medium complexity sellers (5-10 states): {((seller_profile['unique_customer_states'] >= 5) & (seller_profile['unique_customer_states'] <= 10)).sum()}")
    print(f"  Low complexity sellers (<5 states): {(seller_profile['unique_customer_states'] < 5).sum()}")

def save_outputs(seller_profile):
    print("\nSaving output files...")
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # save seller fulfillment complexity profile
    output_file = os.path.join(output_dir, 'seller_fulfillment_complexity.csv')
    seller_profile.to_csv(output_file, index=False)
    print(f"Seller fulfillment complexity profile saved to: {output_file}")

def main():
    print("Starting seller distance and fulfillment complexity analysis...")
    
    try:
        # 1. data
        order_items, orders, customers, sellers, geolocation, products = load_data()
        
        # 2. prepare geolocation data
        geolocation_avg = prepare_geolocation_data(geolocation)
        
        # 3. Merge seller and customer geolocation
        order_with_coords = merge_seller_customer_geolocation(order_items, orders, customers, sellers, geolocation_avg)
        
        # 4. Calculate distances
        orders_with_distances = calculate_distances(order_with_coords)
        
        # 5. Calculate seller distance metrics
        seller_distance_metrics = calculate_seller_distance_metrics(orders_with_distances)
        
        # 6. Calculate fulfillment complexity
        complexity_metrics = calculate_fulfillment_complexity(orders_with_distances, products)
        
        # 7. Create seller complexity profile
        seller_profile = create_seller_complexity_profile(seller_distance_metrics, complexity_metrics)
        
        # 8. visualizations
        create_visualizations(seller_profile)
        
        # 9. statistics
        print_summary_statistics(seller_profile)
        
        # 10. save 
        save_outputs(seller_profile)
        
        print("\nSeller distance and fulfillment complexity analysis completed!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 