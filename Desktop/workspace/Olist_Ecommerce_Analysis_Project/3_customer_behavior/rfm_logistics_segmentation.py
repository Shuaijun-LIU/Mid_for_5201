"""
Step 2: RFM Scoring and Logistics-Aware Segmentation
Execution date: 2025-06-20
Update date: 2025-06-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_customer_features():
    """load customer features from step 1"""
    print("Loading customer features from Step 1...")
    
    customer_features = pd.read_csv('output/customer_logistics_features.csv')
    print(f"Customer features shape: {customer_features.shape}")
    print(f"Date range: {customer_features['first_order_date'].min()} to {customer_features['last_order_date'].max()}")
    
    return customer_features

def calculate_rfm_metrics(customer_features):
    """calculate RFM metrics for each customer"""
    print("\nCalculating RFM metrics...")
    
    # set reference date as the latest order date in the dataset
    reference_date = pd.to_datetime(customer_features['last_order_date'].max())
    print(f"Reference date: {reference_date}")
    
    # calculate recency (days since last order)
    customer_features['last_order_date'] = pd.to_datetime(customer_features['last_order_date'])
    customer_features['recency_days'] = (reference_date - customer_features['last_order_date']).dt.days
    
    # frequency is already calculated as order_count
    customer_features['frequency'] = customer_features['order_count']
    
    # monetary is total spending
    customer_features['monetary'] = customer_features['total_spending']
    
    print("RFM metrics calculated:")
    print(f"Recency range: {customer_features['recency_days'].min()} to {customer_features['recency_days'].max()} days")
    print(f"Frequency range: {customer_features['frequency'].min()} to {customer_features['frequency'].max()} orders")
    print(f"Monetary range: ${customer_features['monetary'].min():.2f} to ${customer_features['monetary'].max():.2f}")
    
    return customer_features

def score_rfm_metrics(customer_features):
    """score RFM metrics using quantiles (1-5 scale)"""
    print("\nScoring RFM metrics...")
    
    # create RFM scores (1-5 scale, higher is better)
    # for recency, lower days = higher score
    try:
        customer_features['r_score'] = pd.qcut(
            customer_features['recency_days'], 
            q=5, 
            labels=[5, 4, 3, 2, 1], 
            duplicates='drop'
        ).astype(int)
    except ValueError:
        # handle case with too many duplicates
        customer_features['r_score'] = pd.qcut(
            customer_features['recency_days'], 
            q=min(5, customer_features['recency_days'].nunique()), 
            labels=False, 
            duplicates='drop'
        ) + 1
        # reverse the scores for recency (lower days = higher score)
        max_score = customer_features['r_score'].max()
        customer_features['r_score'] = max_score - customer_features['r_score'] + 1
    
    # for frequency, use custom bins based on actual distribution
    frequency_counts = customer_features['frequency'].value_counts().sort_index()
    print(f"Frequency distribution: {frequency_counts.to_dict()}")
    
    # create frequency score based on actual order counts
    def score_frequency(freq):
        if freq == 1:
            return 1
        elif freq == 2:
            return 2
        elif freq == 3:
            return 3
        elif freq >= 4 and freq <= 6:
            return 4
        else:  # freq >= 5
            return 5
    
    customer_features['f_score'] = customer_features['frequency'].apply(score_frequency)
    
    # for monetary, higher spending = higher score
    try:
        customer_features['m_score'] = pd.qcut(
            customer_features['monetary'], 
            q=5, 
            labels=[1, 2, 3, 4, 5], 
            duplicates='drop'
        ).astype(int)
    except ValueError:
        # handle case with too many duplicates
        customer_features['m_score'] = pd.qcut(
            customer_features['monetary'], 
            q=min(5, customer_features['monetary'].nunique()), 
            labels=False, 
            duplicates='drop'
        ) + 1
    
    # calculate RFM score (sum of individual scores)
    customer_features['rfm_score'] = customer_features['r_score'] + customer_features['f_score'] + customer_features['m_score']
    
    print("RFM scoring completed:")
    print(f"R-score distribution: {customer_features['r_score'].value_counts().sort_index().to_dict()}")
    print(f"F-score distribution: {customer_features['f_score'].value_counts().sort_index().to_dict()}")
    print(f"M-score distribution: {customer_features['m_score'].value_counts().sort_index().to_dict()}")
    print(f"RFM score range: {customer_features['rfm_score'].min()} to {customer_features['rfm_score'].max()}")
    
    return customer_features

def add_logistics_tags(customer_features):
    """add logistics-focused behavioral tags"""
    print("\nAdding logistics tags...")
    
    # avg freight > 15% of order value
    customer_features['freight_sensitive'] = customer_features['avg_freight_percent'] > 15
    
    # avg days between orders < 30 days for multi-order customers
    customer_features['high_order_frequency'] = (
        (customer_features['avg_days_between_orders'] < 30) & 
        (customer_features['order_count'] > 1)
    )
    
    # multi-region tag (orders to multiple states)
    customer_features['multi_region'] = customer_features['num_distinct_states_ordered_to'] > 1
    
    # high value customer tag (total spending > 75th percentile)
    spending_threshold = customer_features['total_spending'].quantile(0.75)
    customer_features['high_value_customer'] = customer_features['total_spending'] > spending_threshold
    
    # recent customer tag (last order within 30 days)
    customer_features['recent_customer'] = customer_features['recency_days'] <= 30
    
    print("Logistics tags added:")
    print(f"Freight sensitive: {customer_features['freight_sensitive'].sum()} customers ({customer_features['freight_sensitive'].mean()*100:.1f}%)")
    print(f"High order frequency: {customer_features['high_order_frequency'].sum()} customers ({customer_features['high_order_frequency'].mean()*100:.1f}%)")
    print(f"Multi-region: {customer_features['multi_region'].sum()} customers ({customer_features['multi_region'].mean()*100:.1f}%)")
    print(f"High value: {customer_features['high_value_customer'].sum()} customers ({customer_features['high_value_customer'].mean()*100:.1f}%)")
    print(f"Recent customers: {customer_features['recent_customer'].sum()} customers ({customer_features['recent_customer'].mean()*100:.1f}%)")
    
    return customer_features

def determine_optimal_clusters(rfm_data):
    """determine optimal number of clusters using elbow method and silhouette score"""
    print("\nDetermining optimal number of clusters...")
    
    # prepare data for clustering (RFM scores)
    clustering_features = ['r_score', 'f_score', 'm_score']
    X = rfm_data[clustering_features].values
    
    # standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # test different numbers of clusters (limit to 6 for better business interpretation)
    n_clusters_range = range(2, 7)
    inertias = []
    silhouette_scores = []
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # find optimal number of clusters, but ensure at least 4 clusters for business value
    optimal_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    
    # if optimal is less than 4, use 4 clusters for better business segmentation
    if optimal_clusters < 4:
        print(f"Optimal clusters ({optimal_clusters}) too low for business value, using 4 clusters")
        optimal_clusters = 4
    
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Silhouette scores: {dict(zip(n_clusters_range, silhouette_scores))}")
    
    # plot elbow curve and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # elbow curve
    ax1.plot(n_clusters_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.axvline(x=optimal_clusters, color='red', linestyle='--', label=f'Optimal: {optimal_clusters}')
    ax1.legend()
    
    # silhouette scores
    ax2.plot(n_clusters_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.axvline(x=optimal_clusters, color='red', linestyle='--', label=f'Optimal: {optimal_clusters}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('output/rfm_cluster_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return optimal_clusters, scaler

def perform_clustering(rfm_data, n_clusters, scaler):
    """perform K-means clustering on RFM data"""
    print(f"\nPerforming K-means clustering with {n_clusters} clusters...")
    
    # prepare data
    clustering_features = ['r_score', 'f_score', 'm_score']
    X = rfm_data[clustering_features].values
    X_scaled = scaler.transform(X)
    
    # perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # add cluster labels to dataframe
    rfm_data['cluster'] = cluster_labels
    
    # analyze clusters
    cluster_analysis = rfm_data.groupby('cluster').agg({
        'recency_days': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'r_score': 'mean',
        'f_score': 'mean',
        'm_score': 'mean',
        'rfm_score': 'mean',
        'avg_freight_percent': 'mean',
        'order_count': 'mean',
        'customer_unique_id': 'count'
    }).round(2)
    
    cluster_analysis = cluster_analysis.rename(columns={'customer_unique_id': 'customer_count'})
    
    print("\nCluster Analysis:")
    print(cluster_analysis)
    
    return rfm_data, cluster_analysis

def create_cluster_labels(cluster_analysis):
    """create descriptive labels for clusters based on their characteristics"""
    print("\nCreating cluster labels...")
    
    cluster_labels = {}
    
    for cluster_id in cluster_analysis.index:
        r_avg = cluster_analysis.loc[cluster_id, 'r_score']
        f_avg = cluster_analysis.loc[cluster_id, 'f_score']
        m_avg = cluster_analysis.loc[cluster_id, 'm_score']
        size = cluster_analysis.loc[cluster_id, 'customer_count']
        recency_days = cluster_analysis.loc[cluster_id, 'recency_days']
        
        # determine cluster characteristics with more nuanced logic
        if r_avg >= 4 and f_avg >= 3 and m_avg >= 4:
            label = "Champions"
        elif r_avg >= 4 and f_avg >= 2 and m_avg >= 3:
            label = "Loyal Customers"
        elif r_avg >= 4 and f_avg >= 1 and m_avg >= 2:
            label = "Recent High-Value"
        elif r_avg >= 3 and f_avg >= 2 and m_avg >= 3:
            label = "At Risk"
        elif r_avg >= 3 and f_avg >= 1 and m_avg >= 2:
            label = "Recent Customers"
        elif r_avg >= 2 and f_avg >= 2 and m_avg >= 2:
            label = "Promising"
        elif r_avg >= 2 and f_avg >= 1 and m_avg >= 1:
            label = "About to Sleep"
        elif recency_days > 300:  # very old customers
            if m_avg >= 3:
                label = "Lost High-Value"
            else:
                label = "Lost"
        else:
            label = "Needs Attention"
        
        cluster_labels[cluster_id] = label
        print(f"Cluster {cluster_id}: {label} (R:{r_avg:.1f}, F:{f_avg:.1f}, M:{m_avg:.1f}, Size:{size})")
    
    return cluster_labels

def visualize_clusters(rfm_data, cluster_labels):
    """create visualizations for RFM clusters"""
    print("\nCreating cluster visualizations...")
    
    # add descriptive labels to data
    rfm_data['cluster_label'] = rfm_data['cluster'].map(cluster_labels)
    
    # create 3D scatter plot using RFM scores directly
    fig = plt.figure(figsize=(15, 10))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    for cluster_id in rfm_data['cluster'].unique():
        cluster_data = rfm_data[rfm_data['cluster'] == cluster_id]
        label = cluster_labels[cluster_id]
        ax1.scatter(cluster_data['r_score'], cluster_data['f_score'], cluster_data['m_score'], 
                   label=f"{label} (n={len(cluster_data)})", alpha=0.7, s=30)
    
    ax1.set_xlabel('Recency Score')
    ax1.set_ylabel('Frequency Score')
    ax1.set_zlabel('Monetary Score')
    ax1.set_title('Customer Segments in 3D RFM Space')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2D scatter plot: Recency vs Monetary
    ax2 = fig.add_subplot(222)
    for cluster_id in rfm_data['cluster'].unique():
        cluster_data = rfm_data[rfm_data['cluster'] == cluster_id]
        label = cluster_labels[cluster_id]
        ax2.scatter(cluster_data['r_score'], cluster_data['m_score'], 
                   label=f"{label} (n={len(cluster_data)})", alpha=0.7, s=50)
    
    ax2.set_xlabel('Recency Score')
    ax2.set_ylabel('Monetary Score')
    ax2.set_title('Recency vs Monetary')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2D scatter plot: Frequency vs Monetary
    ax3 = fig.add_subplot(223)
    for cluster_id in rfm_data['cluster'].unique():
        cluster_data = rfm_data[rfm_data['cluster'] == cluster_id]
        label = cluster_labels[cluster_id]
        ax3.scatter(cluster_data['f_score'], cluster_data['m_score'], 
                   label=f"{label} (n={len(cluster_data)})", alpha=0.7, s=50)
    
    ax3.set_xlabel('Frequency Score')
    ax3.set_ylabel('Monetary Score')
    ax3.set_title('Frequency vs Monetary')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2D scatter plot: Recency vs Frequency
    ax4 = fig.add_subplot(224)
    for cluster_id in rfm_data['cluster'].unique():
        cluster_data = rfm_data[rfm_data['cluster'] == cluster_id]
        label = cluster_labels[cluster_id]
        ax4.scatter(cluster_data['r_score'], cluster_data['f_score'], 
                   label=f"{label} (n={len(cluster_data)})", alpha=0.7, s=50)
    
    ax4.set_xlabel('Recency Score')
    ax4.set_ylabel('Frequency Score')
    ax4.set_title('Recency vs Frequency')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('output/rfm_cluster_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # create RFM score distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # R-score distribution
    sns.histplot(data=rfm_data, x='r_score', bins=5, ax=axes[0,0])
    axes[0,0].set_title('Recency Score Distribution')
    axes[0,0].set_xlabel('R-Score')
    
    # F-score distribution
    sns.histplot(data=rfm_data, x='f_score', bins=5, ax=axes[0,1])
    axes[0,1].set_title('Frequency Score Distribution')
    axes[0,1].set_xlabel('F-Score')
    
    # M-score distribution
    sns.histplot(data=rfm_data, x='m_score', bins=5, ax=axes[1,0])
    axes[1,0].set_title('Monetary Score Distribution')
    axes[1,0].set_xlabel('M-Score')
    
    # RFM score distribution
    sns.histplot(data=rfm_data, x='rfm_score', bins=15, ax=axes[1,1])
    axes[1,1].set_title('Total RFM Score Distribution')
    axes[1,1].set_xlabel('RFM Score')
    
    plt.tight_layout()
    plt.savefig('output/rfm_metrics_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # create segment comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Average RFM scores by segment
    segment_rfm = rfm_data.groupby('cluster_label')[['r_score', 'f_score', 'm_score']].mean()
    segment_rfm.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Average RFM Scores by Segment')
    axes[0,0].set_ylabel('Score')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Segment sizes
    segment_sizes = rfm_data['cluster_label'].value_counts()
    axes[0,1].pie(segment_sizes.values, labels=segment_sizes.index, autopct='%1.1f%%')
    axes[0,1].set_title('Segment Distribution')
    
    # Average monetary value by segment
    segment_monetary = rfm_data.groupby('cluster_label')['monetary'].mean().sort_values(ascending=False)
    segment_monetary.plot(kind='bar', ax=axes[1,0], color='green')
    axes[1,0].set_title('Average Monetary Value by Segment')
    axes[1,0].set_ylabel('Monetary Value ($)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Freight sensitivity by segment
    freight_sensitivity = rfm_data.groupby('cluster_label')['freight_sensitive'].mean().sort_values(ascending=False)
    freight_sensitivity.plot(kind='bar', ax=axes[1,1], color='orange')
    axes[1,1].set_title('Freight Sensitivity by Segment')
    axes[1,1].set_ylabel('Proportion of Freight Sensitive Customers')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('output/rfm_segment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return rfm_data

def print_segment_statistics(rfm_data, cluster_analysis, cluster_labels):
    """print detailed statistics for each customer segment"""
    print("\n" + "="*80)
    print("DETAILED SEGMENT STATISTICS")
    print("="*80)
    
    for cluster_id in sorted(rfm_data['cluster'].unique()):
        cluster_data = rfm_data[rfm_data['cluster'] == cluster_id]
        label = cluster_labels[cluster_id]
        
        print(f"\n{'='*60}")
        print(f"SEGMENT: {label} (Cluster {cluster_id})")
        print(f"{'='*60}")
        print(f"Size: {len(cluster_data)} customers ({len(cluster_data)/len(rfm_data)*100:.1f}%)")
        print(f"Average RFM Score: {cluster_data['rfm_score'].mean():.2f}")
        print(f"Average Recency: {cluster_data['recency_days'].mean():.1f} days")
        print(f"Average Frequency: {cluster_data['frequency'].mean():.1f} orders")
        print(f"Average Monetary: ${cluster_data['monetary'].mean():.2f}")
        print(f"Average Order Value: ${cluster_data['avg_order_value'].mean():.2f}")
        print(f"Average Freight %: {cluster_data['avg_freight_percent'].mean():.1f}%")
        
        # logistics behavior
        print(f"\nLogistics Behavior:")
        print(f"  Freight Sensitive: {cluster_data['freight_sensitive'].sum()} ({cluster_data['freight_sensitive'].mean()*100:.1f}%)")
        print(f"  High Order Frequency: {cluster_data['high_order_frequency'].sum()} ({cluster_data['high_order_frequency'].mean()*100:.1f}%)")
        print(f"  Multi-Region: {cluster_data['multi_region'].sum()} ({cluster_data['multi_region'].mean()*100:.1f}%)")
        print(f"  High Value: {cluster_data['high_value_customer'].sum()} ({cluster_data['high_value_customer'].mean()*100:.1f}%)")
        print(f"  Recent Customers: {cluster_data['recent_customer'].sum()} ({cluster_data['recent_customer'].mean()*100:.1f}%)")
        
        # top customers in this segment
        top_customers = cluster_data.nlargest(3, 'total_spending')[['customer_unique_id', 'total_spending', 'order_count', 'rfm_score']]
        print(f"\nTop 3 Customers by Spending:")
        print(top_customers.to_string(index=False))

def save_results(rfm_data):
    """save the final segmented customer data"""
    print("\nSaving results...")
    
    # select columns for output
    output_columns = [
        'customer_unique_id', 'cluster', 'cluster_label',
        'recency_days', 'frequency', 'monetary',
        'r_score', 'f_score', 'm_score', 'rfm_score',
        'freight_sensitive', 'high_order_frequency', 'multi_region', 
        'high_value_customer', 'recent_customer',
        'avg_freight_percent', 'order_count', 'total_spending', 'avg_order_value'
    ]
    
    output_data = rfm_data[output_columns].copy()
    output_file = 'output/rfm_segmented_customers.csv'
    output_data.to_csv(output_file, index=False)
    
    print(f"Results saved to: {output_file}")
    print(f"Output shape: {output_data.shape}")
    
    return output_data

def main():
    print("=" * 80)
    print("RFM SCORING AND LOGISTICS-AWARE SEGMENTATION")
    print("=" * 80)
    
    customer_features = load_customer_features()
    
    # calculate RFM metrics
    rfm_data = calculate_rfm_metrics(customer_features)
    
    # score RFM metrics
    rfm_data = score_rfm_metrics(rfm_data)
    
    # add logistics tags
    rfm_data = add_logistics_tags(rfm_data)
    
    # determine optimal clusters
    optimal_clusters, scaler = determine_optimal_clusters(rfm_data)
    # perform clustering
    rfm_data, cluster_analysis = perform_clustering(rfm_data, optimal_clusters, scaler)
    # create cluster labels
    cluster_labels = create_cluster_labels(cluster_analysis)

    rfm_data = visualize_clusters(rfm_data, cluster_labels)
    
    print_segment_statistics(rfm_data, cluster_analysis, cluster_labels)
    output_data = save_results(rfm_data)

    print("\n" + "=" * 80)
    print("RFM SEGMENTATION COMPLETE")
    print("=" * 80)
    print(f"Total customers segmented: {len(output_data)}")
    print(f"Number of segments: {len(cluster_labels)}")
    print(f"Output files created:")
    print(f"  - output/rfm_segmented_customers.csv")
    print(f"  - output/rfm_cluster_plot.png")
    print(f"  - output/rfm_metrics_distribution.png")
    print(f"  - output/rfm_cluster_evaluation.png")
    print(f"  - output/rfm_segment_comparison.png")
    
    print("\nRFM Segmentation completed successfully!")

if __name__ == "__main__":
    main() 