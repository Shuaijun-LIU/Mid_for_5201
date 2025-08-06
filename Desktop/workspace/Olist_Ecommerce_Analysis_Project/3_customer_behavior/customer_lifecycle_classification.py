"""
Step 4: Customer Lifecycle + Churn Classification
Execution date: 2025-06-20
Update date: 2025-06-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    print("Loading customer features and RFM segments...")
    logistics_df = pd.read_csv('output/customer_logistics_features.csv')
    rfm_df = pd.read_csv('output/rfm_segmented_customers.csv')
    print(f"Logistics features shape: {logistics_df.shape}")
    print(f"RFM segments shape: {rfm_df.shape}")
    return logistics_df, rfm_df

def merge_data(logistics_df, rfm_df):
    print("\nMerging data on customer_unique_id...")
    # inner join to ensure only customers present in both datasets
    merged = pd.merge(
        logistics_df,
        rfm_df,
        on='customer_unique_id',
        how='inner',
        suffixes=('_logistics', '_rfm')
    )
    print(f"Merged shape: {merged.shape}")
    print(f"Unique customers: {merged['customer_unique_id'].nunique()}")
    return merged

def compute_recency_days(merged):
    print("\nComputing recency_days...")
    merged['last_order_date'] = pd.to_datetime(merged['last_order_date'])
    reference_date = merged['last_order_date'].max()
    merged['recency_days'] = (reference_date - merged['last_order_date']).dt.days
    print(f"Reference date: {reference_date}")
    print(f"Recency_days range: {merged['recency_days'].min()} to {merged['recency_days'].max()}")
    return merged, reference_date

def assign_lifecycle_stage(merged):
    print("\nAssigning lifecycle stages...")
    # define lifecycle by recency and frequency
    conditions = [
        (merged['recency_days'] <= 30),
        (merged['recency_days'] <= 90) & (merged['order_count_logistics'] > 1),
        (merged['recency_days'] > 90) & (merged['recency_days'] <= 180),
        (merged['recency_days'] > 180)
    ]
    choices = ['New', 'Active', 'At-Risk', 'Churned']
    merged['lifecycle_stage'] = np.select(conditions, choices, default='Unknown')
    print(merged['lifecycle_stage'].value_counts())
    return merged

def assign_churn_risk(merged):
    print("\nAssigning churn risk level...")
    risk_map = {'Active': 'Low', 'At-Risk': 'Medium', 'Churned': 'High', 'New': 'Low'}
    merged['churn_risk_level'] = merged['lifecycle_stage'].map(risk_map).fillna('Unknown')
    print(merged['churn_risk_level'].value_counts())
    return merged

def assign_freight_priority(merged, freight_threshold=20):
    print("\nAssigning freight priority tag...")
    merged['freight_priority'] = (
        ((merged['lifecycle_stage'].isin(['Active', 'At-Risk'])) &
         (merged['avg_freight_value'] > freight_threshold))
    )
    print(f"Freight priority customers: {merged['freight_priority'].sum()} ({merged['freight_priority'].mean()*100:.1f}%)")
    return merged

def add_lifetime_features(merged):
    print("\nAdding lifetime features...")
    merged['first_order_date'] = pd.to_datetime(merged['first_order_date'])
    merged['days_since_first_purchase'] = (merged['last_order_date'] - merged['first_order_date']).dt.days
    return merged

def print_lifecycle_stats(merged):
    print("\nLifecycle stage distribution:")
    print(merged['lifecycle_stage'].value_counts())
    print("\nSummary statistics by lifecycle stage:")
    stats = merged.groupby('lifecycle_stage').agg({
        'customer_unique_id': 'count',
        'order_count_logistics': 'mean',
        'total_spending_logistics': 'mean',
        'avg_freight_value': 'mean',
        'recency_days': 'mean',
        'days_since_first_purchase': 'mean'
    }).rename(columns={
        'customer_unique_id': 'num_customers',
        'order_count_logistics': 'avg_order_count',
        'total_spending_logistics': 'avg_total_spending'
    })
    print(stats)
    return stats

def save_results(merged):
    output_file = 'output/customer_lifecycle.csv'
    merged.to_csv(output_file, index=False)
    print(f"Saved lifecycle data to: {output_file}")

def plot_lifecycle_distribution(merged):
    print("\nPlotting lifecycle distribution...")
    plt.figure(figsize=(8, 6))
    # set custom color palette for each stage
    stage_order = ['New', 'Active', 'At-Risk', 'Churned']
    stage_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']  # blue, orange, red, teal
    # use palette dict to ensure color mapping
    palette = dict(zip(stage_order, stage_colors))
    # countplot with custom colors
    sns.countplot(
        x='lifecycle_stage',
        data=merged,
        order=stage_order,
        palette=palette
    )
    plt.title('Customer Count per Lifecycle Stage')
    plt.xlabel('Lifecycle Stage')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    plt.savefig('output/lifecycle_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: output/lifecycle_distribution.png")

def plot_freight_boxplot(merged):
    print("\nPlotting freight value boxplot by lifecycle stage...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='lifecycle_stage', y='avg_freight_value', data=merged, order=['New', 'Active', 'At-Risk', 'Churned'])
    plt.title('Freight Value by Lifecycle Stage')
    plt.xlabel('Lifecycle Stage')
    plt.ylabel('Average Freight Value')
    plt.tight_layout()
    plt.savefig('output/lifecycle_freight_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: output/lifecycle_freight_boxplot.png")

def main():
    print("=" * 80)
    print("CUSTOMER LIFECYCLE + CHURN CLASSIFICATION")
    print("=" * 80)
    
    logistics_df, rfm_df = load_data()
    # merge logistics and RFM features on customer_unique_id
    merged = merge_data(logistics_df, rfm_df)
    # compute recency_days based on last_order_date
    merged, reference_date = compute_recency_days(merged)
    # assign lifecycle stage for each customer
    merged = assign_lifecycle_stage(merged)
    # assign churn risk level based on lifecycle stage
    merged = assign_churn_risk(merged)
    # tag customers with high freight value for fulfillment priority
    merged = assign_freight_priority(merged, freight_threshold=20)
    # add days_since_first_purchase and related features
    merged = add_lifetime_features(merged)

    print_lifecycle_stats(merged)
    save_results(merged)
    
    plot_lifecycle_distribution(merged)
    plot_freight_boxplot(merged)
    
    print("\n" + "=" * 80)
    print("CUSTOMER LIFECYCLE CLASSIFICATION COMPLETE")
    print("=" * 80)
    print(f"Output files created:")
    print(f"  - output/customer_lifecycle.csv")
    print(f"  - output/lifecycle_distribution.png")
    print(f"  - output/lifecycle_freight_boxplot.png")
    print("\nLifecycle classification completed successfully!")

if __name__ == "__main__":
    main() 