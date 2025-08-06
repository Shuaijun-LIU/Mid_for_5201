"""
Week 5 - Task 5: Four-Dimensional Cross Analysis (Seller-Product-Geography-Time)
Perform comprehensive cross-dimensional analysis using existing Task 1-4 results
Focus on four-dimensional insights and opportunity identification
Execution date: 2025-07-11
Update date: 2025-07-12
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FourDimensionalAnalyzer:
    def __init__(self):
        self.data = {}
        self.analysis_results = {}
        self.output_dir = 'output'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        print("Loading data files for four-dimensional analysis...")
        
        # Load Week5 Task 1-4 outputs (avoiding re-calculation)
        self.data['seller_demand'] = pd.read_csv('output/seller_warehouse_demand.csv')
        self.data['seller_complexity'] = pd.read_csv('output/seller_fulfillment_complexity.csv')
        self.data['seller_lifecycle'] = pd.read_csv('output/seller_lifecycle_product_profile.csv')
        self.data['regional_projection'] = pd.read_csv('output/regional_fulfillment_projection.csv')
        
        # Load Week4 outputs
        print("Loading Week4 outputs...")
        self.data['product_warehouse'] = pd.read_csv('../week4_product_warehouse_analysis/output/product_warehouse_summary.csv')
        self.data['inventory_policy'] = pd.read_csv('../week4_product_warehouse_analysis/output/inventory_policy_matrix.csv')
        
        # Load Week3 outputs
        self.data['customer_geolocation'] = pd.read_csv('../week3_customer_behavior/output/customer_cluster_geolocation.csv')
        self.data['customer_lifecycle'] = pd.read_csv('../week3_customer_behavior/output/customer_lifecycle.csv')
        
        # Load Week2 outputs
        self.data['product_performance'] = pd.read_csv('../week2_eda_update/output/product_performance/product_performance_stats.csv')
        self.data['holiday_sensitive'] = pd.read_csv('../week2_eda_update/output/holiday_sensitive/holiday_sensitive_categories.csv')
        self.data['weekend_holiday'] = pd.read_csv('../week2_eda_update/output/weekend_holiday_analysis/orders_with_weekend_holiday.csv')
        
        # Load minimal raw data for four-dimensional connections only
        self.data['orders'] = pd.read_csv('../data/processed_missing/olist_orders_dataset.csv')
        self.data['order_items'] = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
        self.data['customers'] = pd.read_csv('../data/processed_missing/olist_customers_dataset.csv')
        self.data['sellers'] = pd.read_csv('../data/processed_missing/olist_sellers_dataset.csv')
        self.data['products'] = pd.read_csv('../data/processed_missing/olist_products_dataset.csv')
        
        print("Data loading completed!")
        self._print_data_summary()
        
    def _print_data_summary(self):
        print("\nData Summary:")
        for name, df in self.data.items():
            print(f"  {name}: {df.shape}")
            
    def prepare_four_dimensional_data(self):
        print("\nPreparing four-dimensional data using existing analysis results...")
        
        # Step 1: Create base order-seller-product-customer connection
        print("  Creating base connections...")
        
        # Merge order items with orders to get customer_id and timestamps
        order_base = self.data['order_items'].merge(
            self.data['orders'][['order_id', 'customer_id', 'order_purchase_timestamp']], 
            on='order_id', 
            how='left'
        )
        
        # Add customer and seller information
        order_base = order_base.merge(
            self.data['customers'][['customer_id', 'customer_state']], 
            on='customer_id', 
            how='left'
        ).merge(
            self.data['sellers'][['seller_id', 'seller_state']], 
            on='seller_id', 
            how='left'
        ).merge(
            self.data['products'][['product_id', 'product_category_name']], 
            on='product_id', 
            how='left'
        )
        
        # Add time features
        order_base['order_purchase_timestamp'] = pd.to_datetime(order_base['order_purchase_timestamp'])
        order_base['order_month'] = order_base['order_purchase_timestamp'].dt.to_period('M')
        
        # Step 2: Add existing analysis results (avoiding re-calculation)
        print("  Adding existing analysis results...")
        
        # Add seller analysis from Task 1-3
        order_base = order_base.merge(
            self.data['seller_demand'][['seller_id', 'total_sales_volume', 'avg_turnover_rate', 'total_skus']], 
            on='seller_id', 
            how='left'
        ).merge(
            self.data['seller_complexity'][['seller_id', 'avg_delivery_distance', 'complexity_score']], 
            on='seller_id', 
            how='left'
        ).merge(
            self.data['seller_lifecycle'][['seller_id', 'seller_type', 'retention_rate']], 
            on='seller_id', 
            how='left'
        )
        
        # Add product analysis from Week4
        order_base = order_base.merge(
            self.data['product_warehouse'][['product_id', 'lifecycle_stage', 'avg_monthly_sales']], 
            on='product_id', 
            how='left'
        ).merge(
            self.data['inventory_policy'][['product_id', 'policy_class', 'risk_flag']], 
            on='product_id', 
            how='left'
        )
        
        # Step 3: Add simplified geographic and time features
        print("  Adding geographic and time features...")
        order_base['delivery_distance_category'] = order_base.apply(
            lambda x: 'Same State' if x['customer_state'] == x['seller_state'] else 'Different State', 
            axis=1
        )
        
        # Add holiday information
        holiday_info = self.data['weekend_holiday'][['order_id', 'is_holiday', 'holiday_name', 'day_category']].drop_duplicates()
        order_base = order_base.merge(holiday_info, on='order_id', how='left')
        order_base['is_holiday'] = order_base['is_holiday'].fillna(False)
        order_base['holiday_name'] = order_base['holiday_name'].fillna('Regular Day')
        order_base['day_category'] = order_base['day_category'].fillna('Weekday')
        
        self.data['four_dimensional_base'] = order_base
        print(f"Four-dimensional base data prepared: {order_base.shape}")
        
    def perform_cross_dimensional_analysis(self):
        """Perform four-dimensional cross analysis focusing on unique insights"""
        print("\nPerforming four-dimensional cross analysis...")
        
        base_data = self.data['four_dimensional_base']
        
        # Focus on four-dimensional combinations (unique to this analysis)
        print("  Analyzing four-dimensional combinations...")
        
        # 1. Four-dimensional main analysis (Seller-Product-State-Month)
        four_d_main = base_data.groupby(['seller_id', 'product_category_name', 'customer_state', 'order_month']).agg({
            'order_id': 'count',
            'price': ['sum', 'mean'],
            'freight_value': 'mean',
            'delivery_distance_category': lambda x: (x == 'Same State').sum() / len(x),
            'lifecycle_stage': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'is_holiday': lambda x: x.sum() / len(x),
            'complexity_score': 'mean',
            'retention_rate': 'mean'
        }).reset_index()
        
        four_d_main.columns = [
            'seller_id', 'product_category', 'state', 'month', 'order_count', 
            'total_sales', 'avg_price', 'avg_freight', 'same_state_ratio',
            'dominant_lifecycle', 'holiday_ratio', 'complexity_score', 'retention_rate'
        ]
        
        # 2. Four-dimensional opportunity analysis
        print("  Analyzing four-dimensional opportunities...")
        
        # Calculate four-dimensional opportunity score
        four_d_main['four_d_opportunity_score'] = (
            four_d_main['order_count'] * 0.3 +
            four_d_main['total_sales'] * 0.25 +
            four_d_main['same_state_ratio'] * 0.2 +
            four_d_main['holiday_ratio'] * 0.15 +
            four_d_main['complexity_score'] * 0.1
        )
        
        # 3. Four-dimensional risk analysis
        print("  Analyzing four-dimensional risks...")
        
        # Calculate four-dimensional risk score
        four_d_main['four_d_risk_score'] = (
            (1 - four_d_main['same_state_ratio']) * 0.4 +  # Distance risk
            (1 - four_d_main['retention_rate'] / 100) * 0.3 +  # Retention risk
            four_d_main['complexity_score'] * 0.2 +  # Complexity risk
            (1 - four_d_main['holiday_ratio']) * 0.1  # Seasonal risk
        )
        
        # 4. Four-dimensional efficiency analysis
        print("  Analyzing four-dimensional efficiency...")
        
        # Calculate efficiency metrics
        # Handle division by zero and infinite values
        sales_freight_ratio = four_d_main['total_sales'] / four_d_main['avg_freight'].replace(0, np.nan)
        sales_freight_ratio = sales_freight_ratio.replace([np.inf, -np.inf], np.nan)
        sales_freight_ratio = sales_freight_ratio.fillna(sales_freight_ratio.median())
        
        four_d_main['efficiency_score'] = (
            sales_freight_ratio * 0.4 +
            four_d_main['same_state_ratio'] * 0.3 +
            four_d_main['retention_rate'] / 100 * 0.3
        )
        
        # Store four-dimensional analysis results
        self.analysis_results = {
            'four_d_main': four_d_main
        }
        
        print("Four-dimensional cross analysis completed!")
        
    def create_four_dimensional_summary(self):
        """Create four-dimensional specific summary tables"""
        print("\nCreating four-dimensional specific summaries...")
        
        four_d_main = self.analysis_results['four_d_main']
        
        # 1. Four-dimensional opportunity summary
        print("  Creating four-dimensional opportunity summary...")
        opportunity_summary = four_d_main.groupby(['seller_id', 'product_category', 'state']).agg({
            'four_d_opportunity_score': 'mean',
            'four_d_risk_score': 'mean',
            'efficiency_score': 'mean',
            'order_count': 'sum',
            'total_sales': 'sum',
            'same_state_ratio': 'mean',
            'retention_rate': 'mean'
        }).reset_index()
        
        # 2. Four-dimensional risk summary
        print("  Creating four-dimensional risk summary...")
        risk_summary = four_d_main.groupby(['product_category', 'state']).agg({
            'four_d_risk_score': 'mean',
            'efficiency_score': 'mean',
            'seller_id': 'nunique',
            'order_count': 'sum',
            'total_sales': 'sum'
        }).reset_index()
        
        # 3. Four-dimensional efficiency summary
        print("  Creating four-dimensional efficiency summary...")
        efficiency_summary = four_d_main.groupby(['seller_id', 'state']).agg({
            'efficiency_score': 'mean',
            'four_d_opportunity_score': 'mean',
            'four_d_risk_score': 'mean',
            'product_category': 'nunique',
            'order_count': 'sum',
            'total_sales': 'sum'
        }).reset_index()
        
        # Store four-dimensional specific summaries
        self.analysis_results.update({
            'four_d_opportunity_summary': opportunity_summary,
            'four_d_risk_summary': risk_summary,
            'four_d_efficiency_summary': efficiency_summary
        })
        
        print("Four-dimensional specific summaries created!")
        
    def calculate_complexity_metrics(self):
        """Calculate complexity metrics using existing analysis results"""
        print("\nCalculating complexity metrics using existing data...")
        
        # Use existing complexity scores from Task 2
        seller_complexity = self.data['seller_complexity'][['seller_id', 'complexity_score']].copy()
        seller_complexity = seller_complexity.rename(columns={'complexity_score': 'overall_complexity'})
        
        # Add additional complexity dimensions from four-dimensional data
        base_data = self.data['four_dimensional_base']
        
        # Calculate additional complexity metrics
        additional_complexity = base_data.groupby('seller_id').agg({
            'product_category_name': 'nunique',  # Product diversity
            'customer_state': 'nunique',         # Geographic coverage
            'order_month': 'nunique',            # Temporal coverage
            'delivery_distance_category': lambda x: (x == 'Different State').sum() / len(x),  # Distance complexity
            'is_holiday': lambda x: x.sum() / len(x)  # Holiday sensitivity
        }).reset_index()
        
        additional_complexity.columns = [
            'seller_id', 'product_complexity', 'geographic_complexity', 'temporal_complexity',
            'distance_complexity', 'holiday_sensitivity'
        ]
        
        # Merge with existing complexity scores
        seller_complexity_metrics = seller_complexity.merge(additional_complexity, on='seller_id', how='left')
        
        # Calculate product complexity
        product_complexity = base_data.groupby('product_category_name').agg({
            'seller_id': 'nunique',              # Seller diversity
            'customer_state': 'nunique',         # Geographic distribution
            'order_month': 'nunique',            # Temporal distribution
            'delivery_distance_category': lambda x: (x == 'Different State').sum() / len(x),  # Distance complexity
            'is_holiday': lambda x: x.sum() / len(x)  # Holiday sensitivity
        }).reset_index()
        
        product_complexity.columns = [
            'product_category', 'seller_complexity', 'geographic_complexity', 'temporal_complexity',
            'distance_complexity', 'holiday_sensitivity'
        ]
        
        # Calculate overall complexity score for products
        product_complexity['overall_complexity'] = (
            product_complexity['seller_complexity'] * 0.3 +
            product_complexity['geographic_complexity'] * 0.3 +
            product_complexity['temporal_complexity'] * 0.2 +
            product_complexity['distance_complexity'] * 0.1 +
            product_complexity['holiday_sensitivity'] * 0.1
        )
        
        self.analysis_results.update({
            'seller_complexity_metrics': seller_complexity_metrics,
            'product_complexity_metrics': product_complexity
        })
        
        print("Complexity metrics calculated using existing analysis results!")
        
    def identify_opportunities(self):
        """Identify four-d specific opportunities"""
        print("\nIdentifying four-dimensional opportunities...")
        
        four_d_main = self.analysis_results['four_d_main']
        
        # 1. High opportunity score combinations
        print("  Identifying high opportunity combinations...")
        top_opportunities = four_d_main.nlargest(50, 'four_d_opportunity_score')
        
        # 2. Low risk, high efficiency combinations
        print("  Identifying low risk, high efficiency combinations...")
        # Handle infinite values in efficiency score for quantile calculation
        valid_efficiency = four_d_main['efficiency_score'].replace([np.inf, -np.inf], np.nan).dropna()
        efficiency_threshold = valid_efficiency.quantile(0.7)
        
        low_risk_high_efficiency = four_d_main[
            (four_d_main['four_d_risk_score'] < four_d_main['four_d_risk_score'].quantile(0.3)) &
            (four_d_main['efficiency_score'] > efficiency_threshold) &
            (four_d_main['efficiency_score'] != np.inf) &
            (four_d_main['efficiency_score'] != -np.inf)
        ].nlargest(30, 'efficiency_score')
        
        # 3. Four-dimensional strategic combinations
        print("  Identifying strategic combinations...")
        strategic_combinations = four_d_main[
            (four_d_main['same_state_ratio'] > 0.8) &  # High local efficiency
            (four_d_main['retention_rate'] > 50) &     # High retention
            (four_d_main['holiday_ratio'] > 0.1)       # Holiday sensitive
        ].nlargest(20, 'four_d_opportunity_score')
        
        # 4. Four-dimensional expansion opportunities
        print("  Identifying expansion opportunities...")
        expansion_opportunities = four_d_main[
            (four_d_main['same_state_ratio'] < 0.3) &  # Cross-state potential
            (four_d_main['total_sales'] > four_d_main['total_sales'].quantile(0.7))  # High sales
        ].nlargest(20, 'total_sales')
        
        self.analysis_results.update({
            'top_opportunities': top_opportunities,
            'low_risk_high_efficiency': low_risk_high_efficiency,
            'strategic_combinations': strategic_combinations,
            'expansion_opportunities': expansion_opportunities
        })
        
        print("Four-dimensional opportunities identified!")
        
    def create_visualizations(self):
        """Create four-dimensional specific visualizations"""
        print("\nCreating four-dimensional visualizations...")

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Four-Dimensional Cross Analysis Results', fontsize=16, fontweight='bold')
        
        four_d_main = self.analysis_results['four_d_main']
        
        # 1. Four-dimensional opportunity score distribution
        print("  Creating opportunity score distribution...")
        axes[0,0].hist(four_d_main['four_d_opportunity_score'], bins=30, alpha=0.7, color='green')
        axes[0,0].set_title('Four-Dimensional Opportunity Score Distribution')
        axes[0,0].set_xlabel('Opportunity Score')
        axes[0,0].set_ylabel('Frequency')
        
        # 2. Risk vs Efficiency scatter plot
        print("  Creating risk vs efficiency scatter plot...")
        # Filter out infinite values for scatter plot
        valid_data = four_d_main[
            (four_d_main['four_d_risk_score'].notna()) & 
            (four_d_main['efficiency_score'].notna()) &
            (four_d_main['efficiency_score'] != np.inf) &
            (four_d_main['efficiency_score'] != -np.inf)
        ]
        sample_data = valid_data.sample(min(1000, len(valid_data)))
        if len(sample_data) > 0:
            axes[0,1].scatter(sample_data['four_d_risk_score'], sample_data['efficiency_score'], 
                             alpha=0.6, c='red', s=20)
            axes[0,1].set_title('Risk vs Efficiency Analysis')
            axes[0,1].set_xlabel('Risk Score')
            axes[0,1].set_ylabel('Efficiency Score')
        else:
            axes[0,1].text(0.5, 0.5, 'No valid data for scatter plot', ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Risk vs Efficiency Analysis')
        
        # 3. Four-dimensional opportunity heatmap (top combinations)
        print("  Creating four-dimensional opportunity heatmap...")
        top_opportunities = self.analysis_results['top_opportunities'].head(20)
        if len(top_opportunities) > 0:
            pivot_data = top_opportunities.pivot_table(
                values='four_d_opportunity_score', 
                index='seller_id', 
                columns='product_category', 
                fill_value=0
            )
            if not pivot_data.empty:
                sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0,2])
                axes[0,2].set_title('Top Four-Dimensional Opportunities')
                axes[0,2].set_xlabel('Product Category')
                axes[0,2].set_ylabel('Seller ID')
        
        # 4. Strategic combinations analysis
        print("  Creating strategic combinations analysis...")
        strategic = self.analysis_results['strategic_combinations']
        if len(strategic) > 0:
            axes[1,0].scatter(strategic['same_state_ratio'], strategic['retention_rate'], 
                             s=strategic['total_sales']/100, alpha=0.7, c='blue')
            axes[1,0].set_title('Strategic Combinations (Local + Retention)')
            axes[1,0].set_xlabel('Same State Ratio')
            axes[1,0].set_ylabel('Retention Rate (%)')
        
        # 5. Expansion opportunities by state
        print("  Creating expansion opportunities analysis...")
        expansion = self.analysis_results['expansion_opportunities']
        if len(expansion) > 0:
            state_expansion = expansion.groupby('state')['total_sales'].sum().sort_values(ascending=False).head(10)
            axes[1,1].barh(state_expansion.index, state_expansion.values, color='orange')
            axes[1,1].set_title('Expansion Opportunities by State')
            axes[1,1].set_xlabel('Total Sales')
            axes[1,1].set_ylabel('State')
        
        # 6. Four-dimensional efficiency distribution
        print("  Creating efficiency distribution...")
        # Filter out infinite values for histogram
        efficiency_data = four_d_main['efficiency_score'].replace([np.inf, -np.inf], np.nan).dropna()
        if len(efficiency_data) > 0:
            axes[1,2].hist(efficiency_data, bins=30, alpha=0.7, color='purple')
            axes[1,2].set_title('Four-Dimensional Efficiency Score Distribution')
            axes[1,2].set_xlabel('Efficiency Score')
            axes[1,2].set_ylabel('Frequency')
        else:
            axes[1,2].text(0.5, 0.5, 'No valid efficiency data', ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Four-Dimensional Efficiency Score Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/four_dimensional_analysis_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Four-dimensional visualizations created and saved!")
        
    def save_outputs(self):
        print("\nSaving four-dimensional analysis outputs...")
    
        for name, df in self.analysis_results.items():
            if isinstance(df, pd.DataFrame):
                output_file = f'{self.output_dir}/four_d_{name}.csv'
                df.to_csv(output_file, index=False)
                print(f"  Saved: {output_file}")
    
        self.print_analysis_summary()
        
        print("Four-dimensional analysis outputs saved successfully!")
        
    def print_analysis_summary(self):
        print("\n" + "=" * 60)
        print("FOUR-DIMENSIONAL CROSS ANALYSIS SUMMARY")
        print("=" * 60)

        four_d_main = self.analysis_results['four_d_main']
        
        print(f"\nFour-Dimensional Data Overview:")
        print(f"- Total four-dimensional combinations: {len(four_d_main)}")
        print(f"- Unique sellers: {four_d_main['seller_id'].nunique()}")
        print(f"- Unique product categories: {four_d_main['product_category'].nunique()}")
        print(f"- Unique states: {four_d_main['state'].nunique()}")
        print(f"- Time period: {four_d_main['month'].min()} to {four_d_main['month'].max()}")
        
        # Four-dimensional insights
        print(f"\nFour-Dimensional Insights:")
        
        # Opportunity analysis
        print(f"\nOpportunity Analysis:")
        avg_opportunity = four_d_main['four_d_opportunity_score'].mean()
        print(f"- Average opportunity score: {avg_opportunity:.2f}")
        
        high_opportunity_combinations = len(four_d_main[four_d_main['four_d_opportunity_score'] > avg_opportunity])
        print(f"- High opportunity combinations: {high_opportunity_combinations}")
        
        # Risk analysis
        print(f"\nRisk Analysis:")
        avg_risk = four_d_main['four_d_risk_score'].mean()
        print(f"- Average risk score: {avg_risk:.2f}")
        
        low_risk_combinations = len(four_d_main[four_d_main['four_d_risk_score'] < avg_risk])
        print(f"- Low risk combinations: {low_risk_combinations}")
        
        # Efficiency analysis
        print(f"\nEfficiency Analysis:")
        # Handle infinite values in efficiency score
        valid_efficiency = four_d_main['efficiency_score'].replace([np.inf, -np.inf], np.nan).dropna()
        avg_efficiency = valid_efficiency.mean()
        print(f"- Average efficiency score: {avg_efficiency:.2f}")
        
        high_efficiency_combinations = len(four_d_main[
            (four_d_main['efficiency_score'] > avg_efficiency) & 
            (four_d_main['efficiency_score'] != np.inf) & 
            (four_d_main['efficiency_score'] != -np.inf)
        ])
        print(f"- High efficiency combinations: {high_efficiency_combinations}")
        
        # Top opportunities
        print(f"\nTop Four-Dimensional Opportunities:")
        top_opp = self.analysis_results['top_opportunities'].head(3)
        for _, row in top_opp.iterrows():
            print(f"- {row['product_category']} in {row['state']} by Seller {row['seller_id']}: "
                  f"Opportunity Score {row['four_d_opportunity_score']:.2f}, "
                  f"Risk Score {row['four_d_risk_score']:.2f}, "
                  f"Efficiency Score {row['efficiency_score']:.2f}")
        
        # Strategic combinations
        print(f"\nStrategic Combinations Found:")
        strategic = self.analysis_results['strategic_combinations']
        if len(strategic) > 0:
            print(f"- {len(strategic)} high local efficiency + high retention combinations")
        
        # Expansion opportunities
        print(f"\nExpansion Opportunities Found:")
        expansion = self.analysis_results['expansion_opportunities']
        if len(expansion) > 0:
            print(f"- {len(expansion)} cross-state expansion opportunities")
        
        print(f"\n" + "=" * 60)
        
    def run_complete_analysis(self):
        print("=" * 60)
        print("FOUR-DIMENSIONAL CROSS ANALYSIS")
        print("=" * 60)
        
        self.load_data()
        self.prepare_four_dimensional_data()
        self.perform_cross_dimensional_analysis()
        self.create_four_dimensional_summary()
        self.calculate_complexity_metrics()
        self.identify_opportunities()
        self.create_visualizations()
        self.save_outputs()
        
        print("\n" + "=" * 60)
        print("FOUR-DIMENSIONAL ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return self.analysis_results

def main():
    analyzer = FourDimensionalAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print("\nAnalysis completed! Check the output directory for results.")
    print("Key output files:")
    print("  - four_dimensional_analysis_overview.png")
    print("  - four_dimensional_analysis_report.md")
    print("  - Various CSV files with detailed analysis results")

if __name__ == "__main__":
    main() 