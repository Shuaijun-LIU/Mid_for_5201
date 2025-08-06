'''
Execution date: 2025-06-09
'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# style
plt.style.use('default')
sns.set_theme()

# create output directory if it doesn't exist
os.makedirs('output/product_performance', exist_ok=True)

def load_and_preprocess_data():
    order_items_df = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    products_df = pd.read_csv('../data/processed_missing/olist_products_dataset.csv')
    category_translation_df = pd.read_csv('../data/processed_missing/product_category_name_translation.csv')
    
    # Merge datasets
    merged_df = order_items_df.merge(products_df, on='product_id', how='left')
    merged_df = merged_df.merge(category_translation_df, on='product_category_name', how='left')
    
    return merged_df

def calculate_product_stats(df):
    # Group by product and calculate metrics
    product_stats = df.groupby('product_id').agg({
        'order_id': 'count',  # order frequency
        'price': ['sum', 'mean', 'std'],  # revenue and price stats
        'freight_value': 'mean',  # average freight cost
        'product_category_name_english': 'first'  # category name
    }).reset_index()
    
    # Flatten 
    product_stats.columns = ['product_id', 'order_count', 'total_sales', 
                           'avg_price', 'price_std', 'avg_freight', 'category']
    
    # Fill NaN values
    product_stats['price_std'] = product_stats['price_std'].fillna(0)
    
    return product_stats

def identify_outliers(df, threshold_std=3):
    # Calculate z-scores for price and order count
    price_zscore = np.abs((df['avg_price'] - df['avg_price'].mean()) / df['avg_price'].std())
    order_zscore = np.abs((df['order_count'] - df['order_count'].mean()) / df['order_count'].std())
    
    # Identify outliers
    outliers = df[(price_zscore > threshold_std) | (order_zscore > threshold_std)]
    normal_products = df[~df['product_id'].isin(outliers['product_id'])]
    
    return normal_products, outliers

def classify_products(df):
    # Create a copy of the dataframe to avoid the warning
    df = df.copy()
    
    # Define thresholds
    high_order_threshold = df['order_count'].quantile(0.75)
    high_revenue_threshold = df['total_sales'].quantile(0.75)
    low_order_threshold = df['order_count'].quantile(0.25)
    low_revenue_threshold = df['total_sales'].quantile(0.25)
    
    # Classify products
    df['performance'] = 'average'
    df.loc[(df['order_count'] >= high_order_threshold) & 
           (df['total_sales'] >= high_revenue_threshold), 'performance'] = 'high'
    df.loc[(df['order_count'] <= low_order_threshold) & 
           (df['total_sales'] <= low_revenue_threshold), 'performance'] = 'low'
    
    return df

def generate_rankings(df):
    rankings = {
        'by_sales': df.sort_values('total_sales', ascending=False),
        'by_orders': df.sort_values('order_count', ascending=False),
        'by_price_variance': df.sort_values('price_std', ascending=False)
    }
    return rankings

def visualize_results(df, rankings):
    # plot top 10 products by sales
    plt.figure(figsize=(15, 6))
    sns.barplot(data=rankings['by_sales'].head(10), x='product_id', y='total_sales', hue='product_id', palette='husl', legend=False)
    plt.title('Top 10 Products by Sales')
    plt.xlabel('Product ID')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/product_performance/top10_sales.png')
    plt.close()
    
    # plot performance distribution
    plt.figure(figsize=(10, 6))
    performance_counts = df['performance'].value_counts().reset_index()
    performance_counts.columns = ['performance', 'count']
    sns.barplot(data=performance_counts, x='performance', y='count', hue='performance', palette='Set2', legend=False)
    plt.title('Product Performance Distribution')
    plt.xlabel('Performance Level')
    plt.ylabel('Number of Products')
    plt.tight_layout()
    plt.savefig('output/product_performance/performance_distribution.png')
    plt.close()
    
    # plot category analysis
    category_stats = df.groupby('category').agg({
        'total_sales': 'mean',
        'order_count': 'mean',
        'product_id': 'count'
    }).sort_values('total_sales', ascending=False).head(10).reset_index()
    
    plt.figure(figsize=(15, 6))
    sns.barplot(data=category_stats, x='category', y='total_sales', hue='category', palette='Paired', legend=False)
    plt.title('Top 10 Categories by Average Sales')
    plt.xlabel('Category')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/product_performance/top10_categories.png')
    plt.close()
    
    # plot price distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='avg_price', bins=50)
    plt.title('Product Price Distribution')
    plt.xlabel('Average Price')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('output/product_performance/price_distribution.png')
    plt.close()
    
    # plot price vs order count scatter
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='order_count', y='avg_price', alpha=0.5)
    plt.title('Price vs Order Count')
    plt.xlabel('Order Count')
    plt.ylabel('Average Price')
    plt.tight_layout()
    plt.savefig('output/product_performance/price_vs_orders.png')
    plt.close()

def analyze_outliers(outliers_df):
    # print outlier summary
    print("\nOutlier Analysis:")
    print("\nPrice Outliers (Top 10):")
    price_outliers = outliers_df.sort_values('avg_price', ascending=False).head(10)
    print(price_outliers[['product_id', 'category', 'avg_price', 'order_count', 'total_sales']])
    
    print("\nOrder Count Outliers (Top 10):")
    order_outliers = outliers_df.sort_values('order_count', ascending=False).head(10)
    print(order_outliers[['product_id', 'category', 'order_count', 'avg_price', 'total_sales']])
    
    # visualize outliers
    plt.figure(figsize=(15, 6))
    sns.scatterplot(data=outliers_df, x='order_count', y='avg_price', alpha=0.5)
    plt.title('Outlier Products: Price vs Order Count')
    plt.xlabel('Order Count')
    plt.ylabel('Average Price')
    plt.tight_layout()
    plt.savefig('output/product_performance/outlier_scatter.png')
    plt.close()
    
    # plot outlier categories
    plt.figure(figsize=(15, 6))
    category_counts = outliers_df['category'].value_counts().head(10).reset_index()
    category_counts.columns = ['category', 'count']
    sns.barplot(data=category_counts, x='category', y='count', hue='category', palette='Set3', legend=False)
    plt.title('Top 10 Categories with Outlier Products')
    plt.xlabel('Category')
    plt.ylabel('Number of Outlier Products')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/product_performance/outlier_categories.png')
    plt.close()
    
    # plot price distribution of outliers
    plt.figure(figsize=(12, 6))
    sns.histplot(data=outliers_df, x='avg_price', bins=50)
    plt.title('Price Distribution of Outlier Products')
    plt.xlabel('Average Price')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('output/product_performance/outlier_price_distribution.png')
    plt.close()
    
    # save outlier details to csv
    outliers_df.to_csv('output/product_performance/outlier_products.csv', index=False)

def main():
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    print("\nCalculating product statistics...")
    product_stats = calculate_product_stats(df)
    
    print("\nIdentifying outliers...")
    normal_products, outliers = identify_outliers(product_stats)
    print(f"Found {len(outliers)} outlier products")
    
    print("\nAnalyzing outliers...")
    analyze_outliers(outliers)
    
    print("\nClassifying products...")
    classified_products = classify_products(normal_products)
    
    print("\nGenerating rankings...")
    rankings = generate_rankings(classified_products)
    
    print("\nGenerating visualizations...")
    visualize_results(classified_products, rankings)
    
    # save detailed stats
    classified_products.to_csv('output/product_performance/product_performance_stats.csv', index=False)
    
    print("\nAnalysis Summary:")
    print("\nTop 10 Products by Sales:")
    print(rankings['by_sales'].head(10)[['product_id', 'category', 'total_sales', 'order_count', 'avg_price']])
    
    print("\nTop 10 Products by Order Count:")
    print(rankings['by_orders'].head(10)[['product_id', 'category', 'order_count', 'total_sales', 'avg_price']])
    
    print("\nProducts with Highest Price Variance:")
    print(rankings['by_price_variance'].head(10)[['product_id', 'category', 'price_std', 'avg_price', 'order_count']])
    
    print("\nPerformance Distribution:")
    print(classified_products['performance'].value_counts())
    
    print("\nCategory Analysis:")
    category_stats = classified_products.groupby('category').agg({
        'total_sales': 'mean',
        'order_count': 'mean',
        'product_id': 'count'
    }).sort_values('total_sales', ascending=False).head(5)
    print(category_stats)

if __name__ == "__main__":
    main() 