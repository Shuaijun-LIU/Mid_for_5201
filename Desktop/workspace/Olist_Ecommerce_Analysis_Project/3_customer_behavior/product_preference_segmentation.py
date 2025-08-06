"""
Step 7: Product Preference Segmentation
Execution date: 2025-06-23
"""

import pandas as pd
import numpy as np
from collections import Counter
from math import log2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    orders = pd.read_csv('../data/processed_missing/olist_orders_dataset.csv')
    order_items = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    products = pd.read_csv('../data/processed_missing/olist_products_dataset.csv')
    customers = pd.read_csv('../data/processed_missing/olist_customers_dataset.csv')
    
    # merge customer_unique_id to orders
    orders = orders.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id', how='left')
    print(f"Orders shape after merge: {orders.shape}")
    
    try:
        personas = pd.read_csv('output/final_customer_segments.csv')
    except Exception:
        personas = None
    return orders, order_items, products, personas

def merge_order_product_customer(orders, order_items, products):
    # filter only delivered orders
    delivered_orders = orders[orders['order_status'] == 'delivered']
    # merge order_items with products
    order_items = order_items.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
    # merge with delivered orders to get customer_unique_id
    merged = order_items.merge(delivered_orders[['order_id', 'customer_unique_id', 'order_purchase_timestamp']], on='order_id', how='inner')
    return merged

def compute_customer_product_metrics(merged):
    # group by customer and aggregate product category info
    customer_metrics = []
    for customer_id, group in merged.groupby('customer_unique_id'):
        cats = group['product_category_name'].dropna().tolist()
        cat_counts = Counter(cats)
        total = sum(cat_counts.values())
        if total == 0:
            continue
        # dominant category and loyalty
        dominant_category, dom_count = cat_counts.most_common(1)[0]
        loyalty_score = dom_count / total
        # diversity and entropy
        diversity = len(cat_counts)
        entropy = -sum((v/total) * log2(v/total) for v in cat_counts.values() if v > 0)
        # repeat category orders
        repeat_cats = [cat for cat, v in cat_counts.items() if v > 1]
        repeat_count = len(repeat_cats)
        # top 3 categories
        top3 = ', '.join([cat for cat, _ in cat_counts.most_common(3)])
        # avg price per category
        avg_price_per_cat = group.groupby('product_category_name')['price'].mean().to_dict()
        avg_price = np.mean(list(avg_price_per_cat.values())) if avg_price_per_cat else np.nan
        customer_metrics.append({
            'customer_unique_id': customer_id,
            'dominant_category': dominant_category,
            'category_loyalty_score': loyalty_score,
            'product_category_diversity': diversity,
            'category_entropy': entropy,
            'repeat_category_orders': repeat_count,
            'top_3_categories': top3,
            'avg_price_per_category': avg_price
        })
    df = pd.DataFrame(customer_metrics)
    print(f"Computed product metrics for {len(df)} customers")
    return df

def segment_customers(df):
    # use KMeans on loyalty, diversity, entropy for segmentation
    features = df[['category_loyalty_score', 'product_category_diversity', 'category_entropy']].fillna(0)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['segment'] = kmeans.fit_predict(features)
    # assign segment labels by cluster center characteristics
    centers = kmeans.cluster_centers_
    labels = []
    for i, row in df.iterrows():
        loyalty = row['category_loyalty_score']
        diversity = row['product_category_diversity']
        entropy = row['category_entropy']
        if loyalty > 0.7 and diversity <= 3:
            label = 'Specialist'
        elif entropy > 1.5 and diversity > 5:
            label = 'Generalist'
        elif row['avg_price_per_category'] < 50:
            label = 'Price-Focused'
        elif diversity <= 3 and row['avg_price_per_category'] > 100:
            label = 'Premium Niche'
        else:
            label = 'Other'
        labels.append(label)
    df['segment_label'] = labels
    print(df['segment_label'].value_counts())
    return df

def save_results(df):
    output_file = 'output/customer_product_preferences.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

def plot_segment_distribution(df):
    plt.figure(figsize=(8, 6))
    seg_counts = df['segment_label'].value_counts()
    sns.barplot(x=seg_counts.index, y=seg_counts.values, palette='husl')
    plt.title('Product Preference Segment Distribution')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    plt.savefig('output/product_segments_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: output/product_segments_summary.png")

def plot_segment_vs_category_heatmap(df):
    # create a heatmap of top categories by segment
    cross = pd.crosstab(df['segment_label'], df['dominant_category'])
    plt.figure(figsize=(12, 6))
    sns.heatmap(cross, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Segment vs Dominant Category Heatmap')
    plt.xlabel('Dominant Category')
    plt.ylabel('Segment')
    plt.tight_layout()
    plt.savefig('output/segment_vs_category_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: output/segment_vs_category_heatmap.png")

def main():
    print("=" * 80)
    print("PRODUCT PREFERENCE SEGMENTATION")
    print("=" * 80)

    orders, order_items, products, personas = load_data()
    # merge delivered order data with product categories and customers
    merged = merge_order_product_customer(orders, order_items, products)
    print(f"Merged data shape: {merged.shape}")
    print(f"Unique customers in merged data: {merged['customer_unique_id'].nunique()}")
    # compute customer-level product preference metrics
    df = compute_customer_product_metrics(merged)
    # segment customers by product preference
    df = segment_customers(df)

    print("\n" + "=" * 50)
    print("DETAILED STATISTICS")
    print("=" * 50)
    
    print("\nProduct Preference Segment Distribution:")
    print(df['segment_label'].value_counts())
    
    print("\nTop 10 Dominant Categories:")
    print(df['dominant_category'].value_counts().head(10))
    
    print("\nCategory Loyalty Score Summary:")
    print(f"Average loyalty score: {df['category_loyalty_score'].mean():.3f}")
    print(f"Loyalty score range: {df['category_loyalty_score'].min():.3f} to {df['category_loyalty_score'].max():.3f}")
    
    print("\nProduct Category Diversity Summary:")
    print(f"Average diversity: {df['product_category_diversity'].mean():.1f}")
    print(f"Diversity range: {df['product_category_diversity'].min()} to {df['product_category_diversity'].max()}")
    
    print("\nAverage Price per Category Summary:")
    print(f"Average price: {df['avg_price_per_category'].mean():.2f}")
    print(f"Price range: {df['avg_price_per_category'].min():.2f} to {df['avg_price_per_category'].max():.2f}")
    
    print("\nSegment Characteristics:")
    segment_stats = df.groupby('segment_label').agg({
        'category_loyalty_score': 'mean',
        'product_category_diversity': 'mean',
        'category_entropy': 'mean',
        'avg_price_per_category': 'mean'
    }).round(3)
    print(segment_stats)
    
    # data validation - show raw purchase behavior
    print("\n" + "=" * 50)
    print("DATA VALIDATION")
    print("=" * 50)
    
    print("\nRaw Purchase Behavior Analysis:")
    print(f"Total customers analyzed: {len(df)}")
    print(f"Customers with single category: {(df['product_category_diversity'] == 1).sum()} ({(df['product_category_diversity'] == 1).mean()*100:.1f}%)")
    print(f"Customers with 2+ categories: {(df['product_category_diversity'] > 1).sum()} ({(df['product_category_diversity'] > 1).mean()*100:.1f}%)")
    
    print("\nLoyalty Score Distribution:")
    loyalty_bins = [0.5, 0.7, 0.9, 1.0]
    loyalty_labels = ['0.5-0.7', '0.7-0.9', '0.9-1.0']
    df['loyalty_bin'] = pd.cut(df['category_loyalty_score'], bins=loyalty_bins, labels=loyalty_labels, include_lowest=True)
    print(df['loyalty_bin'].value_counts().sort_index())
    
    save_results(df)
    plot_segment_distribution(df)
    plot_segment_vs_category_heatmap(df)

    print("\n" + "=" * 80)
    print("PRODUCT PREFERENCE SEGMENTATION COMPLETE")
    print("=" * 80)
    print(f"Output files created:")
    print(f"  - output/customer_product_preferences.csv")
    print(f"  - output/product_segments_summary.png")
    print(f"  - output/segment_vs_category_heatmap.png")
    print("\nProduct preference segmentation completed successfully!")

if __name__ == "__main__":
    main() 