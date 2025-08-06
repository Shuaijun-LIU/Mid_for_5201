"""
Step 3: Geolocation Mapping of Customer Segments
Execution date: 2025-06-20
Update date: 2025-06-23
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from geobr import read_state
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    print("Loading RFM-segmented customers and geolocation data...")
    rfm_df = pd.read_csv('output/rfm_segmented_customers.csv')
    customers_df = pd.read_csv('../data/processed_missing/olist_customers_dataset.csv')
    print(f"RFM segments shape: {rfm_df.shape}")
    print(f"Customers shape: {customers_df.shape}")
    return rfm_df, customers_df

def merge_geolocation(rfm_df, customers_df):
    """Merge RFM segments with customer geolocation info"""
    print("\nMerging RFM segments with geolocation info...")
    merged = pd.merge(
        rfm_df,
        customers_df[['customer_unique_id', 'customer_city', 'customer_state']],
        on='customer_unique_id',
        how='left'
    )
    print(f"Merged shape: {merged.shape}")
    print(f"Unique customers: {merged['customer_unique_id'].nunique()}")
    print(f"Unique states: {merged['customer_state'].nunique()}")
    print(f"Unique cities: {merged['customer_city'].nunique()}")
    return merged

def save_merged_data(merged):
    output_file = 'output/customer_cluster_geolocation.csv'
    merged.to_csv(output_file, index=False)
    print(f"Merged data saved to: {output_file}")

def cluster_summary_tables(merged):
    """generate summary tables by state and city"""
    print("\nGenerating summary tables...")
    # count customers by state and cluster
    state_cluster = merged.groupby(['customer_state', 'cluster_label']).size().unstack(fill_value=0)
    print("\nCluster distribution by state:")
    print(state_cluster.head())
    
    # proportion of each cluster within each state
    state_totals = merged.groupby('customer_state').size()
    state_cluster_prop = state_cluster.div(state_totals, axis=0)
    print("\nCluster proportion by state:")
    print(state_cluster_prop.head())
    
    # Top cities by cluster
    top_cities = merged.groupby(['cluster_label', 'customer_city']).size().reset_index(name='count')
    top_cities = top_cities.sort_values(['cluster_label', 'count'], ascending=[True, False])
    print("\nTop cities by cluster:")
    print(top_cities.groupby('cluster_label').head(3))
    
    return state_cluster, state_cluster_prop, top_cities

def plot_cluster_distribution_by_state(state_cluster):
    print("\nPlotting cluster distribution by state...")
    plt.figure(figsize=(16, 8))
    state_cluster.plot(kind='bar', stacked=True, colormap='tab20', ax=plt.gca())
    plt.title('RFM Cluster Distribution by State')
    plt.xlabel('State')
    plt.ylabel('Number of Customers')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('output/cluster_distribution_by_state.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: output/cluster_distribution_by_state.png")

def plot_top_states_high_value(merged):
    print("\nPlotting top states with most high-value customers...")
    high_value_label = 'Recent Customers'
    top_states = merged[merged['cluster_label'] == high_value_label]['customer_state'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_states.values, y=top_states.index, palette='viridis')
    plt.title('Top 10 States with Most High-Value Customers')
    plt.xlabel('Number of High-Value Customers')
    plt.ylabel('State')
    plt.tight_layout()
    plt.savefig('output/top_states_high_value_customers.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: output/top_states_high_value_customers.png")

def plot_brazil_choropleth(merged):
    print("\nPlotting Brazil choropleth map (using geobr)...")
    try:
        br_states = read_state(year=2020)
        high_value_label = 'Recent Customers'
        state_counts = merged[merged['cluster_label'] == high_value_label]['customer_state'].value_counts().reset_index()
        state_counts.columns = ['state', 'high_value_count']
        # merge geobr state polygons with customer counts
        br_states = br_states.merge(state_counts, left_on='abbrev_state', right_on='state', how='left')
        br_states['high_value_count'] = br_states['high_value_count'].fillna(0)
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        br_states.plot(column='high_value_count', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
        ax.set_title('High-Value Customer Density by State (Brazil)')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('output/brazil_map_cluster_density.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: output/brazil_map_cluster_density.png")
    except Exception as e:
        print(f"Choropleth map not generated: {e}")
        print("Hint: Try running `pip install geobr geopandas` if not installed.")

def main():
    print("=" * 80)
    print("GEOLOCATION MAPPING OF CUSTOMER SEGMENTS")
    print("=" * 80)

    rfm_df, customers_df = load_data()
    
    # Merge geolocation
    merged = merge_geolocation(rfm_df, customers_df)
 
    save_merged_data(merged)
    
    # sum tables
    state_cluster, state_cluster_prop, top_cities = cluster_summary_tables(merged)

    plot_cluster_distribution_by_state(state_cluster)
    plot_top_states_high_value(merged)
    plot_brazil_choropleth(merged)
    
    print("\n" + "=" * 80)
    print("GEOLOCATION MAPPING COMPLETE")
    print("=" * 80)
    print(f"Output files created:")
    print(f"  - output/customer_cluster_geolocation.csv")
    print(f"  - output/cluster_distribution_by_state.png")
    print(f"  - output/top_states_high_value_customers.png")
    print(f"  - output/brazil_map_cluster_density.png")
    print("\nGeolocation mapping completed successfully!")

if __name__ == "__main__":
    main() 