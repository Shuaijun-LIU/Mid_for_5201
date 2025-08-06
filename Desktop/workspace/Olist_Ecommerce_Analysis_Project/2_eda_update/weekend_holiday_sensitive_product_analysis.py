'''
Execution date: 2025-06-13
update date: 2025-06-20
Weekend and Holiday Sensitive Product Analysis for Olist E-commerce
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# style
plt.style.use('default')
sns.set_theme()

# create output directory if it doesn't exist
os.makedirs('output/holiday_sensitive', exist_ok=True)

def get_holiday_period(date, holiday_name):
    """
    Determine if a date falls within a holiday period (pre-holiday, holiday, or post-holiday)
    
    Args:
        date: The date to check
        holiday_name: Name of the holiday
    
    Returns:
        tuple: (is_holiday_period, period_type, holiday_name)
    """
    date = pd.to_datetime(date)
    
    # Define holiday periods
    if holiday_name in ['Black Friday', 'Cyber Monday', 'Green Monday']:
        # E-commerce holidays: 14 days before, 5 days after
        pre_holiday_days = 14
        post_holiday_days = 5
    else:
        # Traditional holidays: 7 days before, 3 days after
        pre_holiday_days = 7
        post_holiday_days = 3
    
    # Get holiday date from the date itself since it's already a holiday
    holiday_date = date
    
    # Check if date is within holiday period
    if date == holiday_date:
        return True, 'holiday', holiday_name
    elif holiday_date - pd.Timedelta(days=pre_holiday_days) <= date < holiday_date:
        return True, 'pre_holiday', holiday_name
    elif holiday_date < date <= holiday_date + pd.Timedelta(days=post_holiday_days):
        return True, 'post_holiday', holiday_name
    
    return False, None, None

def load_and_merge_data():

    orders_df = pd.read_csv('output/weekend_holiday_analysis/orders_with_weekend_holiday.csv')
    order_items_df = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    products_df = pd.read_csv('../data/processed_missing/olist_products_dataset.csv')
    category_translation_df = pd.read_csv('../data/processed_missing/product_category_name_translation.csv')
    
    # Convert date columns
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    
    # Create is_weekend field from weekday_name
    orders_df['is_weekend'] = orders_df['weekday_name'].isin(['Saturday', 'Sunday'])
    
    # Get unique holiday dates
    holiday_dates = orders_df[orders_df['is_holiday']].groupby('holiday_name')['order_purchase_timestamp'].min()
    
    # Add holiday period information
    holiday_periods = []
    for _, row in orders_df.iterrows():
        date = row['order_purchase_timestamp']
        is_period = False
        period_type = None
        holiday_name = None
        
        # Check if date is in any holiday period
        for h_name, h_date in holiday_dates.items():
            if h_name in ['Black Friday', 'Cyber Monday', 'Green Monday']:
                pre_days = 14
                post_days = 5
            else:
                pre_days = 7
                post_days = 3
            
            if date == h_date:
                is_period = True
                period_type = 'holiday'
                holiday_name = h_name
                break
            elif h_date - pd.Timedelta(days=pre_days) <= date < h_date:
                is_period = True
                period_type = 'pre_holiday'
                holiday_name = h_name
                break
            elif h_date < date <= h_date + pd.Timedelta(days=post_days):
                is_period = True
                period_type = 'post_holiday'
                holiday_name = h_name
                break
        
        holiday_periods.append((is_period, period_type, holiday_name))
    
    # Add new columns
    orders_df['is_holiday_period'] = [x[0] for x in holiday_periods]
    orders_df['holiday_period_type'] = [x[1] for x in holiday_periods]
    orders_df['holiday_period_name'] = [x[2] for x in holiday_periods]
    
    # Merge datasets
    merged_df = orders_df.merge(order_items_df, on='order_id', how='inner')
    merged_df = merged_df.merge(products_df, on='product_id', how='inner')
    merged_df = merged_df.merge(category_translation_df, on='product_category_name', how='inner')
    
    return merged_df

def preprocess_data(merged_df):

    merged_df = merged_df.dropna(subset=['product_id', 'product_category_name', 'product_category_name_english'])
    
    # Filter holidays with sufficient orders
    holiday_counts = merged_df[merged_df['is_holiday']].groupby('holiday_name')['order_id'].nunique()
    valid_holidays = holiday_counts[holiday_counts >= 10].index
    merged_df = merged_df[merged_df['holiday_name'].isin(valid_holidays) | ~merged_df['is_holiday']]
    
    # Filter products/categories with sufficient orders
    product_counts = merged_df.groupby('product_id')['order_id'].nunique()
    valid_products = product_counts[product_counts >= 30].index
    merged_df = merged_df[merged_df['product_id'].isin(valid_products)]
    
    category_counts = merged_df.groupby('product_category_name_english')['order_id'].nunique()
    valid_categories = category_counts[category_counts >= 30].index
    merged_df = merged_df[merged_df['product_category_name_english'].isin(valid_categories)]
    
    return merged_df

def calculate_weekend_metrics(merged_df):
    """
    Calculate weekend sensitivity metrics for products and categories
    """
    # Filter weekend and weekday data
    weekend_df = merged_df[merged_df['is_weekend']]
    weekday_df = merged_df[~merged_df['is_weekend']]
    
    # Calculate total orders for ratio computation
    total_weekend_orders = weekend_df['order_id'].nunique()
    total_weekday_orders = weekday_df['order_id'].nunique()
    
    # Category-level analysis
    category_metrics = merged_df.groupby('product_category_name_english').agg({
        'order_id': lambda x: {
            'weekend_count': x[merged_df.loc[x.index, 'is_weekend']].nunique(),
            'weekday_count': x[~merged_df.loc[x.index, 'is_weekend']].nunique()
        },
        'price': 'mean',
        'freight_value': 'mean'
    }).reset_index()
    
    # unpack the order_id dictionary
    category_metrics['weekend_count'] = category_metrics['order_id'].apply(lambda x: x['weekend_count'])
    category_metrics['weekday_count'] = category_metrics['order_id'].apply(lambda x: x['weekday_count'])
    category_metrics = category_metrics.drop('order_id', axis=1)
    
    # calculate ratios and lift
    category_metrics['weekend_ratio'] = category_metrics['weekend_count'] / total_weekend_orders
    category_metrics['weekday_ratio'] = category_metrics['weekday_count'] / total_weekday_orders
    category_metrics['weekend_lift'] = (category_metrics['weekend_ratio'] + 1e-6) / (category_metrics['weekday_ratio'] + 1e-6)
    
    # Product-level analysis
    product_metrics = merged_df.groupby(['product_id', 'product_category_name_english']).agg({
        'order_id': lambda x: {
            'weekend_count': x[merged_df.loc[x.index, 'is_weekend']].nunique(),
            'weekday_count': x[~merged_df.loc[x.index, 'is_weekend']].nunique()
        },
        'price': 'mean',
        'freight_value': 'mean'
    }).reset_index()
    
    # unpack the order_id dictionary
    product_metrics['weekend_count'] = product_metrics['order_id'].apply(lambda x: x['weekend_count'])
    product_metrics['weekday_count'] = product_metrics['order_id'].apply(lambda x: x['weekday_count'])
    product_metrics = product_metrics.drop('order_id', axis=1)
    
    # calculate ratios and lift
    product_metrics['weekend_ratio'] = product_metrics['weekend_count'] / total_weekend_orders
    product_metrics['weekday_ratio'] = product_metrics['weekday_count'] / total_weekday_orders
    product_metrics['weekend_lift'] = (product_metrics['weekend_ratio'] + 1e-6) / (product_metrics['weekday_ratio'] + 1e-6)
    
    return category_metrics, product_metrics

def calculate_metrics(merged_df, specific_holiday=None, period_type=None):
    """
    Calculate metrics for all holidays or a specific holiday
    
    Args:
        merged_df: Merged dataframe with all data
        specific_holiday: Name of specific holiday to analyze (optional)
        period_type: Type of holiday period to analyze ('holiday', 'pre_holiday', 'post_holiday', or None for all)
    """
    # Filter for specific holiday and period if provided
    if specific_holiday:
        if period_type:
            holiday_df = merged_df[
                (merged_df['holiday_period_name'] == specific_holiday) & 
                (merged_df['holiday_period_type'] == period_type)
            ]
        else:
            holiday_df = merged_df[merged_df['holiday_period_name'] == specific_holiday]
        
        if len(holiday_df) == 0:
            print(f"\nWarning: No orders found for {specific_holiday}" + 
                  (f" during {period_type}" if period_type else ""))
            return None, None
        
        print(f"\nAnalyzing {specific_holiday}" + 
              (f" during {period_type}" if period_type else ""))
        print(f"Total orders: {holiday_df['order_id'].nunique()}")
    else:
        holiday_df = merged_df[merged_df['is_holiday_period']]
    
    # calculate total orders for ratio computation
    total_holiday_orders = holiday_df['order_id'].nunique()
    total_regular_orders = merged_df[~merged_df['is_holiday_period']]['order_id'].nunique()
    
    # Category-level analysis
    category_metrics = merged_df.groupby('product_category_name_english').agg({
        'order_id': lambda x: {
            'holiday_count': len(set(x.index) & set(holiday_df.index)) if specific_holiday else x[merged_df.loc[x.index, 'is_holiday_period']].nunique(),
            'regular_count': x[~merged_df.loc[x.index, 'is_holiday_period']].nunique()
        },
        'price': 'mean',
        'freight_value': 'mean'
    }).reset_index()
    
    # unpack the order_id dictionary
    category_metrics['holiday_count'] = category_metrics['order_id'].apply(lambda x: x['holiday_count'])
    category_metrics['regular_count'] = category_metrics['order_id'].apply(lambda x: x['regular_count'])
    category_metrics = category_metrics.drop('order_id', axis=1)
    
    # calculate ratios and lift
    category_metrics['holiday_ratio'] = category_metrics['holiday_count'] / total_holiday_orders
    category_metrics['regular_ratio'] = category_metrics['regular_count'] / total_regular_orders
    category_metrics['lift'] = (category_metrics['holiday_ratio'] + 1e-6) / (category_metrics['regular_ratio'] + 1e-6)
    
    # Product-level analysis
    product_metrics = merged_df.groupby(['product_id', 'product_category_name_english']).agg({
        'order_id': lambda x: {
            'holiday_count': len(set(x.index) & set(holiday_df.index)) if specific_holiday else x[merged_df.loc[x.index, 'is_holiday_period']].nunique(),
            'regular_count': x[~merged_df.loc[x.index, 'is_holiday_period']].nunique()
        },
        'price': 'mean',
        'freight_value': 'mean'
    }).reset_index()
    
    # unpack the order_id dictionary
    product_metrics['holiday_count'] = product_metrics['order_id'].apply(lambda x: x['holiday_count'])
    product_metrics['regular_count'] = product_metrics['order_id'].apply(lambda x: x['regular_count'])
    product_metrics = product_metrics.drop('order_id', axis=1)
    
    # calculate ratios and lift
    product_metrics['holiday_ratio'] = product_metrics['holiday_count'] / total_holiday_orders
    product_metrics['regular_ratio'] = product_metrics['regular_count'] / total_regular_orders
    product_metrics['lift'] = (product_metrics['holiday_ratio'] + 1e-6) / (product_metrics['regular_ratio'] + 1e-6)
    
    return category_metrics, product_metrics

def generate_visualizations(category_metrics, product_metrics, weekend_category_metrics=None, weekend_product_metrics=None):
    """
    Generate visualizations for holiday and weekend sensitivity analysis
    """
    # 1. top 10 holiday-sensitive categories
    plt.figure(figsize=(12, 6))
    top_categories = category_metrics.nlargest(10, 'lift')
    sns.barplot(data=top_categories, y='product_category_name_english', x='lift', 
               hue='product_category_name_english', legend=False, palette='Set2')
    plt.title('Top 10 Holiday-Sensitive Product Categories')
    plt.xlabel('Lift')
    plt.ylabel('Product Category')
    plt.tight_layout()
    plt.savefig('output/weekend_holiday_sensitive/top10_holiday_sensitive_categories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. bubble chart: lift vs total orders
    plt.figure(figsize=(12, 6))
    category_metrics['total_orders'] = category_metrics['holiday_count'] + category_metrics['regular_count']
    sns.scatterplot(data=category_metrics, x='total_orders', y='lift', 
                   size='total_orders', hue='lift', palette='viridis', sizes=(50, 400))
    plt.title('Holiday Sensitivity vs Total Orders')
    plt.xlabel('Total Orders')
    plt.ylabel('Lift')
    plt.tight_layout()
    plt.savefig('output/weekend_holiday_sensitive/holiday_lift_bubble_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Weekend sensitivity visualizations (if data provided)
    if weekend_category_metrics is not None:
        # Top 10 weekend-sensitive categories
        plt.figure(figsize=(12, 6))
        top_weekend_categories = weekend_category_metrics.nlargest(10, 'weekend_lift')
        sns.barplot(data=top_weekend_categories, y='product_category_name_english', x='weekend_lift', 
                   hue='product_category_name_english', legend=False, palette='Set1')
        plt.title('Top 10 Weekend-Sensitive Product Categories')
        plt.xlabel('Weekend Lift')
        plt.ylabel('Product Category')
        plt.tight_layout()
        plt.savefig('output/weekend_holiday_sensitive/top10_weekend_sensitive_categories.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Weekend lift bubble chart
        plt.figure(figsize=(12, 6))
        weekend_category_metrics['total_orders'] = weekend_category_metrics['weekend_count'] + weekend_category_metrics['weekday_count']
        sns.scatterplot(data=weekend_category_metrics, x='total_orders', y='weekend_lift', 
                       size='total_orders', hue='weekend_lift', palette='viridis', sizes=(50, 400))
        plt.title('Weekend Sensitivity vs Total Orders')
        plt.xlabel('Total Orders')
        plt.ylabel('Weekend Lift')
        plt.tight_layout()
        plt.savefig('output/weekend_holiday_sensitive/weekend_lift_bubble_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Comparison chart: Holiday vs Weekend sensitivity
        plt.figure(figsize=(14, 8))
        comparison_data = pd.merge(
            category_metrics[['product_category_name_english', 'lift']].rename(columns={'lift': 'holiday_lift'}),
            weekend_category_metrics[['product_category_name_english', 'weekend_lift']],
            on='product_category_name_english',
            how='inner'
        )
        
        # Get top 15 categories by average lift
        comparison_data['avg_lift'] = (comparison_data['holiday_lift'] + comparison_data['weekend_lift']) / 2
        top_comparison = comparison_data.nlargest(15, 'avg_lift')
        
        # Create comparison plot
        x = np.arange(len(top_comparison))
        width = 0.35
        
        plt.bar(x - width/2, top_comparison['holiday_lift'], width, label='Holiday Lift', alpha=0.8)
        plt.bar(x + width/2, top_comparison['weekend_lift'], width, label='Weekend Lift', alpha=0.8)
        
        plt.xlabel('Product Categories')
        plt.ylabel('Lift')
        plt.title('Holiday vs Weekend Sensitivity Comparison (Top 15 Categories)')
        plt.xticks(x, top_comparison['product_category_name_english'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('output/weekend_holiday_sensitive/holiday_vs_weekend_sensitivity_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def print_analysis_results(category_metrics, product_metrics, specific_holiday=None, analysis_type="holiday"):
    if category_metrics is None or product_metrics is None:
        return
        
    if analysis_type == "holiday":
        print("\n=== Holiday Sensitivity Analysis Results ===\n")
        lift_column = 'lift'
        count_columns = ['holiday_count', 'regular_count']
        ratio_columns = ['holiday_ratio', 'regular_ratio']
    else:
        print("\n=== Weekend Sensitivity Analysis Results ===\n")
        lift_column = 'weekend_lift'
        count_columns = ['weekend_count', 'weekday_count']
        ratio_columns = ['weekend_ratio', 'weekday_ratio']
    
    if specific_holiday:
        print(f"Analysis for: {specific_holiday}")
    else:
        print(f"Analysis for all {analysis_type}s")
    
    print("\nMethodology:")
    if analysis_type == "holiday":
        print(" Holiday sensitivity is measured using the lift metric: (holiday_ratio + 1e-6) / (regular_ratio + 1e-6)")
        print(" Only holidays with 10+ orders are included")
    else:
        print(" Weekend sensitivity is measured using the lift metric: (weekend_ratio + 1e-6) / (weekday_ratio + 1e-6)")
    print(" Only products/categories with 30+ total orders are analyzed")
    print(" Lift > 1.5 indicates significant sensitivity\n")
    
    print(f"Top 10 {analysis_type.title()}-Sensitive Categories:")
    print("-" * 50)
    top_categories = category_metrics.nlargest(10, lift_column)
    display_columns = ['product_category_name_english'] + count_columns + [lift_column] + ratio_columns
    print(top_categories[display_columns].to_string())
    
    print(f"\nTop 10 {analysis_type.title()}-Sensitive Products:")
    print("-" * 45)
    top_products = product_metrics.nlargest(10, lift_column)
    display_columns = ['product_id', 'product_category_name_english'] + count_columns + [lift_column] + ratio_columns
    print(top_products[display_columns].to_string())
    
    print(f"\nKey Insights:")
    print(f"Categories with highest {analysis_type} sensitivity:")
    for _, row in top_categories.head(3).iterrows():
        print(f" {row['product_category_name_english']} (Lift: {row[lift_column]:.2f})")

def main():
    print("Loading and merging datasets...")
    merged_df = load_and_merge_data()
    
    print("\nPreprocessing data...")
    merged_df = preprocess_data(merged_df)
    
    # Analyze all holidays
    print("\nCalculating metrics for all holidays...")
    category_metrics, product_metrics = calculate_metrics(merged_df)
    
    # Analyze weekend sensitivity
    print("\nCalculating weekend sensitivity metrics...")
    weekend_category_metrics, weekend_product_metrics = calculate_weekend_metrics(merged_df)
    
    # save to CSV
    category_metrics.to_csv('output/weekend_holiday_sensitive/holiday_sensitive_categories.csv', index=False)
    product_metrics.to_csv('output/weekend_holiday_sensitive/holiday_sensitive_products.csv', index=False)
    weekend_category_metrics.to_csv('output/weekend_holiday_sensitive/weekend_sensitive_categories.csv', index=False)
    weekend_product_metrics.to_csv('output/weekend_holiday_sensitive/weekend_sensitive_products.csv', index=False)
    
    print("\nGenerating visualizations...")
    generate_visualizations(category_metrics, product_metrics, weekend_category_metrics, weekend_product_metrics)
    
    print("\nPrinting holiday sensitivity analysis results...")
    print_analysis_results(category_metrics, product_metrics, analysis_type="holiday")
    
    print("\nPrinting weekend sensitivity analysis results...")
    print_analysis_results(weekend_category_metrics, weekend_product_metrics, analysis_type="weekend")
    
    # Analyze specific holidays
    specific_holidays = ['Black Friday', 'Valentine\'s Day', 'Mother\'s Day', 'Father\'s Day']
    period_types = ['pre_holiday', 'holiday', 'post_holiday']
    
    for holiday in specific_holidays:
        print(f"\nAnalyzing {holiday}...")
        for period in period_types:
            holiday_category_metrics, holiday_product_metrics = calculate_metrics(
                merged_df, holiday, period
            )
            if holiday_category_metrics is not None:
                # save to CSV
                filename = f'{holiday.lower().replace(" ", "_")}_{period}_'
                holiday_category_metrics.to_csv(f'output/weekend_holiday_sensitive/{filename}categories.csv', index=False)
                holiday_product_metrics.to_csv(f'output/weekend_holiday_sensitive/{filename}products.csv', index=False)
                print_analysis_results(holiday_category_metrics, holiday_product_metrics, f"{holiday} ({period})", "holiday")
    
    print("\n====== Analysis Complete ======\n")

if __name__ == "__main__":
    main() 