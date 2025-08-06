'''
Execution date: 2025-06-07
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import numpy as np

# pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# style
plt.style.use('default')
sns.set_theme()

# Create output directory if it doesn't exist
os.makedirs('output/timeseries', exist_ok=True)

def load_and_preprocess_data():
    orders_df = pd.read_csv('../data/processed_missing/olist_orders_dataset.csv')
    order_items_df = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    order_payments_df = pd.read_csv('../data/processed_missing/olist_order_payments_dataset.csv')
    
    # Convert timestamp columns to datetime
    timestamp_columns = ['order_purchase_timestamp', 'order_approved_at', 
                        'order_delivered_carrier_date', 'order_delivered_customer_date']
    
    for col in timestamp_columns:
        # Convert to datetime, keeping '0000-00-00 00:00:00' as is
        orders_df[col] = pd.to_datetime(orders_df[col], errors='coerce')
    
    # filter for delivered orders only
    orders_df = orders_df[orders_df['order_status'] == 'delivered']
    
    # Drop rows where order_purchase_timestamp is NaT
    orders_df = orders_df.dropna(subset=['order_purchase_timestamp'])
    
    # merge orders with order items
    merged_df = orders_df.merge(order_items_df, on='order_id', how='inner')
    
    # merge with payments (optional, for additional payment information)
    merged_df = merged_df.merge(order_payments_df, on='order_id', how='left')
    
    return merged_df

def aggregate_time_series(df, freq='D'):
    """
    Aggregate data by time frequency (daily, weekly, or monthly).
    
    Args:
        df: DataFrame with order data
        freq: Frequency for aggregation ('D' for daily, 'W' for weekly, 'M' for monthly)
    
    Returns:
        Aggregated DataFrame with metrics
    """
    # Set the timestamp as index
    df = df.set_index('order_purchase_timestamp')
    
    # Perform aggregation
    agg_df = df.resample(freq).agg({
        'order_id': 'count',  # Order count
        'price': 'sum',       # Total revenue
        'customer_id': 'nunique'  # Unique customer count
    }).reset_index()
    
    # Calculate Average Order Value (AOV)
    # Only calculate AOV when there are orders
    agg_df['aov'] = np.where(agg_df['order_id'] > 0,
                            agg_df['price'] / agg_df['order_id'],
                            np.nan)
    
    # Rename columns for clarity
    agg_df.columns = ['date', 'order_count', 'total_revenue', 'unique_customers', 'aov']
    
    return agg_df

def plot_time_series(df, metric, freq, output_path):
    plt.figure(figsize=(15, 6))
    
    # the main metric
    plt.plot(df['date'], df[metric], label=f'{metric.replace("_", " ").title()}')
    
    # 7-day rolling average if daily data
    if freq == 'D':
        rolling_avg = df[metric].rolling(window=7).mean()
        plt.plot(df['date'], rolling_avg, '--', label='7-day Rolling Average')
    
    # Customize plot
    plt.title(f'{metric.replace("_", " ").title()} - {freq.title()} Trend')
    plt.xlabel('Date')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    frequencies = {
        'D': 'daily',
        'W': 'weekly',
        'ME': 'monthly'  
    }
    
    metrics = ['order_count', 'total_revenue', 'aov', 'unique_customers']
    
    for freq, freq_name in frequencies.items():
        print(f"\nAggregating {freq_name} data...")
        agg_df = aggregate_time_series(df, freq)
        
        # Calculate statistics for each metric separately
        print(f"\n{freq_name.title()} Summary Statistics:")
        for metric in metrics:
            print(f"\n{metric.replace('_', ' ').title()} Statistics:")
            stats = agg_df[metric].describe()
            print(f"Count: {stats['count']:.2f}")
            print(f"Mean: {stats['mean']:.2f}")
            print(f"Min: {stats['min']:.2f}")
            print(f"25%: {stats['25%']:.2f}")
            print(f"50%: {stats['50%']:.2f}")
            print(f"75%: {stats['75%']:.2f}")
            print(f"Max: {stats['max']:.2f}")
            print(f"Std: {stats['std']:.2f}")
            
            # Print the date when max value occurred
            max_date = agg_df.loc[agg_df[metric] == stats['max'], 'date'].iloc[0]
            print(f"Max value date: {max_date}")
        
        for metric in metrics:
            output_path = f'output/timeseries/{metric}_{freq_name}.png'
            plot_time_series(agg_df, metric, freq, output_path)
            print(f"Created plot: {output_path}")
            
            print(f"\n{freq_name.title()} {metric.replace('_', ' ').title()}:")
            print(agg_df[['date', metric]].head())

if __name__ == "__main__":
    main() 