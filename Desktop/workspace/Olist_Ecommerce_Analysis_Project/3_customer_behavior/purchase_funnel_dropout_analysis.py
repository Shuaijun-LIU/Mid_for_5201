"""
Step 5: Purchase Funnel Dropout Analysis
Execution date: 2025-06-21
Update date: 2025-06-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# define funnel stages
FUNNEL_STAGES = [
    'Order Created',
    'Payment Approved',
    'Order Delivered'
]

ORDER_STATUS_TO_STAGE = {
    'created': 'Order Created',
    'approved': 'Payment Approved',
    'processing': 'Payment Approved',
    'invoiced': 'Payment Approved',
    'shipped': 'Order Delivered',
    'delivered': 'Order Delivered',
    'canceled': 'Order Created',
    'unavailable': 'Order Created'
}

def load_data():
    print("Loading order, item and customer data...")
    orders = pd.read_csv('../data/processed_missing/olist_orders_dataset.csv')
    order_items = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    customers = pd.read_csv('../data/processed_missing/olist_customers_dataset.csv')
    print(f"Orders shape: {orders.shape}")
    print(f"Order items shape: {order_items.shape}")
    print(f"Customers shape: {customers.shape}")
    # merge customer_unique_id
    orders = orders.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id', how='left')
    print(f"Orders after merge shape: {orders.shape}")
    return orders, order_items

def map_order_status_to_funnel(orders):
    print("\nMapping order_status to funnel stages...")
    # map each order_status to a funnel stage
    orders['funnel_position'] = orders['order_status'].map(ORDER_STATUS_TO_STAGE)
    # fill missing with 'Order Created'
    orders['funnel_position'] = orders['funnel_position'].fillna('Order Created')
    return orders

def aggregate_funnel_stats(orders):
    print("\nAggregating funnel statistics...")
    # count unique orders at each stage
    stage_counts = orders.groupby('funnel_position')['order_id'].nunique().reindex(FUNNEL_STAGES, fill_value=0)
    # calculate dropouts and conversion
    stage_counts = stage_counts.to_frame('order_count')
    stage_counts['dropout'] = stage_counts['order_count'].shift(1, fill_value=stage_counts.iloc[0,0]) - stage_counts['order_count']
    stage_counts['conversion_rate'] = stage_counts['order_count'] / stage_counts.iloc[0,0]
    print(stage_counts)
    return stage_counts

def tag_orders_with_funnel(orders):
    print("\nTagging orders with funnel stage...")
    output_file = 'output/orders_with_funnel_tag.csv'
    orders[['order_id', 'customer_unique_id', 'order_status', 'funnel_position']].to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    return orders

def save_funnel_summary(stage_counts):
    output_file = 'output/funnel_stage_summary.csv'
    stage_counts.to_csv(output_file)
    print(f"Saved: {output_file}")

def plot_funnel(stage_counts):
    print("\nPlotting funnel chart...")
    plt.figure(figsize=(10, 6))
    # plot horizontal funnel
    bars = plt.barh(stage_counts.index[::-1], stage_counts['order_count'][::-1], color=sns.color_palette("husl", len(stage_counts)))
    for i, (count, dropout) in enumerate(zip(stage_counts['order_count'][::-1], stage_counts['dropout'][::-1])):
        plt.text(count + max(stage_counts['order_count']) * 0.01, i, f"{count:,}", va='center')
    plt.title('Order Conversion Funnel')
    plt.xlabel('Number of Orders')
    plt.ylabel('Funnel Stage')
    plt.tight_layout()
    plt.savefig('output/funnel_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: output/funnel_plot.png")

def plot_sankey(orders):
    print("\nPlotting order status flow diagram...")
    # build transition matrix for order_status
    status_flow = orders.groupby(['order_status', 'funnel_position']).size().reset_index(name='count')
    
    # print order status distribution
    status_counts = orders['order_status'].value_counts()
    print("\nOrder status distribution:")
    print(status_counts)
    
    # create a simple bar chart showing status distribution
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(status_counts)), status_counts.values, color=sns.color_palette("husl", len(status_counts)))
    plt.xticks(range(len(status_counts)), status_counts.index, rotation=45, ha='right')
    plt.title('Order Status Distribution')
    plt.xlabel('Order Status')
    plt.ylabel('Number of Orders')
    
    # add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, status_counts.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(status_counts.values)*0.01, 
                f'{count:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('output/order_status_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: output/order_status_distribution.png")

def main():
    print("=" * 80)
    print("PURCHASE FUNNEL DROPOUT ANALYSIS")
    print("=" * 80)
    
    # load order and item data
    orders, order_items = load_data()
    # map order_status to funnel stages
    orders = map_order_status_to_funnel(orders)
    # aggregate funnel statistics
    stage_counts = aggregate_funnel_stats(orders)
    # print funnel summary table
    print("\nFunnel summary table:")
    print(stage_counts)
    # tag orders with funnel stage
    orders = tag_orders_with_funnel(orders)

    save_funnel_summary(stage_counts)

    plot_funnel(stage_counts)
    plot_sankey(orders)
    
    print("\n" + "=" * 80)
    print("PURCHASE FUNNEL ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output files created:")
    print(f"  - output/funnel_stage_summary.csv")
    print(f"  - output/funnel_plot.png")
    print(f"  - output/order_status_distribution.png")
    print(f"  - output/orders_with_funnel_tag.csv")
    print("\nFunnel analysis completed successfully!")

if __name__ == "__main__":
    main() 