'''
Execution date: 2025-06-08
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# style
plt.style.use('default')
sns.set_theme()

# Create output directory if it doesn't exist
os.makedirs('output/sales_distribution', exist_ok=True)

def load_and_preprocess_data():

    orders_df = pd.read_csv('../data/processed_missing/olist_orders_dataset.csv')
    order_items_df = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    customers_df = pd.read_csv('../data/processed_missing/olist_customers_dataset.csv')
    products_df = pd.read_csv('../data/processed_missing/olist_products_dataset.csv')
    category_translation_df = pd.read_csv('../data/processed_missing/product_category_name_translation.csv')
    
    # Filter for delivered orders
    orders_df = orders_df[orders_df['order_status'] == 'delivered']
    
    # Merge all datasets
    merged_df = orders_df.merge(order_items_df, on='order_id', how='inner')
    merged_df = merged_df.merge(customers_df, on='customer_id', how='inner')
    merged_df = merged_df.merge(products_df, on='product_id', how='inner')
    merged_df = merged_df.merge(category_translation_df, on='product_category_name', how='left')
    
    return merged_df

def analyze_state_sales(df):

    # group by state and calculate metrics
    state_sales = df.groupby('customer_state').agg({
        'price': 'sum',
        'order_id': 'nunique'
    }).reset_index()
    
    # rename 
    state_sales.columns = ['state', 'total_revenue', 'order_count']
    
    # Calculate percentages
    total_revenue = state_sales['total_revenue'].sum()
    total_orders = state_sales['order_count'].sum()
    state_sales['revenue_pct'] = (state_sales['total_revenue'] / total_revenue * 100).round(2)
    state_sales['order_pct'] = (state_sales['order_count'] / total_orders * 100).round(2)
    
    # Sort by revenue
    state_sales = state_sales.sort_values('total_revenue', ascending=False)
    
    # Plot top 10 states
    plt.figure(figsize=(15, 6))
    sns.barplot(data=state_sales.head(10), x='state', y='total_revenue', hue='state', palette='husl', legend=False)
    plt.title('Top 10 States by Revenue')
    plt.xlabel('State')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/sales_distribution/top10_state_sales.png')
    plt.close()
    
    print("\nState-level Sales Analysis:")
    print(state_sales)
    
    return state_sales

def analyze_category_sales(df):
    """
    Analyze sales distribution by product category.
    """
    # Group by category and calculate metrics
    category_sales = df.groupby('product_category_name_english').agg({
        'price': 'sum',
        'order_id': 'nunique'
    }).reset_index()
    
    # rename
    category_sales.columns = ['category', 'total_revenue', 'order_count']
    
    # Calculate percentages
    total_revenue = category_sales['total_revenue'].sum()
    total_orders = category_sales['order_count'].sum()
    category_sales['revenue_pct'] = (category_sales['total_revenue'] / total_revenue * 100).round(2)
    category_sales['order_pct'] = (category_sales['order_count'] / total_orders * 100).round(2)
    
    # Sort by revenue
    category_sales = category_sales.sort_values('total_revenue', ascending=False)
    
    # create state-category heatmap for top 10 categories
    top_categories = category_sales.head(10)['category'].tolist()
    state_category = df[df['product_category_name_english'].isin(top_categories)].pivot_table(
        values='price',
        index='customer_state',
        columns='product_category_name_english',
        aggfunc='sum'
    ).fillna(0)
    
    plt.figure(figsize=(20, 10))
    sns.heatmap(state_category, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('Revenue Heatmap: State vs Top 10 Categories')
    plt.xlabel('Product Category')
    plt.ylabel('State')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/sales_distribution/top10_category_state_heatmap.png')
    plt.close()
    
    print("\nCategory-level Sales Analysis:")
    print(category_sales)
    
    return category_sales

def analyze_seller_performance(df):
    """
    Analyze sales distribution by seller.
    """
    # Group by seller and calculate metrics
    seller_sales = df.groupby('seller_id').agg({
        'price': 'sum',
        'order_id': 'nunique'
    }).reset_index()
    
    # rename
    seller_sales.columns = ['seller_id', 'total_revenue', 'order_count']
    
    # Calculate average order value and percentages
    total_revenue = seller_sales['total_revenue'].sum()
    total_orders = seller_sales['order_count'].sum()
    seller_sales['avg_order_value'] = seller_sales['total_revenue'] / seller_sales['order_count']
    seller_sales['revenue_pct'] = (seller_sales['total_revenue'] / total_revenue * 100).round(2)
    seller_sales['order_pct'] = (seller_sales['order_count'] / total_orders * 100).round(2)
    
    # Sort by revenue
    seller_sales = seller_sales.sort_values('total_revenue', ascending=False)
    
    # Plot top 10 sellers
    plt.figure(figsize=(15, 6))
    sns.barplot(data=seller_sales.head(10), x='seller_id', y='total_revenue', hue='seller_id', palette='Paired', legend=False)
    plt.title('Top 10 Sellers by Revenue')
    plt.xlabel('Seller ID')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/sales_distribution/top_sellers.png')
    plt.close()
    
    # Plot bottom 10 sellers
    plt.figure(figsize=(15, 6))
    sns.barplot(data=seller_sales.tail(10), x='seller_id', y='total_revenue', hue='seller_id', palette='Set2', legend=False)
    plt.title('Bottom 10 Sellers by Revenue')
    plt.xlabel('Seller ID')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/sales_distribution/bottom_sellers.png')
    plt.close()
    
    print("\nTop 10 Sellers:")
    print(seller_sales.head(10))
    print("\nBottom 10 Sellers:")
    print(seller_sales.tail(10))
    
    return seller_sales

def main():
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    print("\nAnalyzing state-level sales...")
    state_sales = analyze_state_sales(df)
    
    print("\nAnalyzing category-level sales...")
    category_sales = analyze_category_sales(df)
    
    print("\nAnalyzing seller performance...")
    seller_sales = analyze_seller_performance(df)

if __name__ == "__main__":
    main() 