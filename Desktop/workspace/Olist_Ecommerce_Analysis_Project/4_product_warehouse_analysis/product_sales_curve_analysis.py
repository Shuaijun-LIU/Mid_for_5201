"""
Step 1: Product Sales Curve Analysis
Execution date: 2025-06-25
Update date: 2025-06-30
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

class ProductSalesCurveAnalyzer:
    """Product sales curve analysis and visualization class"""
    
    def __init__(self, data_path="../data/processed_missing/"):
        """Initialize the analyzer with data paths"""
        self.data_path = data_path
        self.output_path = "./output/"
        self.create_output_directory()
        
    def create_output_directory(self):
        """create output directory if it doesn't exist"""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")
    
    def load_data(self):
        """load and merge the required datasets"""
        print("Loading datasets...")
        
        # load order items data
        order_items = pd.read_csv(f"{self.data_path}olist_order_items_dataset.csv")
        print(f"Loaded order items: {len(order_items):,} records")
        
        # load orders data
        orders = pd.read_csv(f"{self.data_path}olist_orders_dataset.csv")
        print(f"Loaded orders: {len(orders):,} records")
        
        # load products data
        products = pd.read_csv(f"{self.data_path}olist_products_dataset.csv")
        print(f"Loaded products: {len(products):,} records")
        
        return order_items, orders, products
    
    def preprocess_data(self, order_items, orders, products):
        """preprocess and merge datasets"""
        print("Preprocessing and merging data...")
        
        # merge order items with orders
        merged_data = order_items.merge(
            orders[['order_id', 'order_purchase_timestamp', 'order_status']], 
            on='order_id', 
            how='inner'
        )
        
        # filter for delivered orders only
        merged_data = merged_data[merged_data['order_status'] == 'delivered'].copy()
        print(f"After filtering delivered orders: {len(merged_data):,} records")
        
        # convert timestamp to datetime
        merged_data['order_purchase_timestamp'] = pd.to_datetime(
            merged_data['order_purchase_timestamp']
        )
        
        # extract order month in YYYY-MM format
        merged_data['order_month'] = merged_data['order_purchase_timestamp'].dt.strftime('%Y-%m')
        
        # merge with products data
        final_data = merged_data.merge(
            products[['product_id', 'product_category_name']], 
            on='product_id', 
            how='inner'
        )
        
        print(f"Final merged dataset: {len(final_data):,} records")
        return final_data
    
    def aggregate_sales(self, data):
        """aggregate sales by product and month"""
        print("Aggregating sales by product and month...")
        
        # group by product_id and order_month
        sales_agg = data.groupby(['product_id', 'order_month']).agg({
            'order_item_id': 'count',  # count of items sold
            'price': 'sum'  # total revenue
        }).reset_index()
        
        sales_agg.columns = ['product_id', 'order_month', 'monthly_sales_volume', 'monthly_sales_value']
        
        # convert order_month to datetime for proper sorting
        sales_agg['order_month_dt'] = pd.to_datetime(sales_agg['order_month'] + '-01')
        
        # create complete product-month matrix
        complete_matrix = self.create_complete_matrix(sales_agg)
        
        # filter products with at least 3 months of sales history
        product_months = complete_matrix.groupby('product_id')['monthly_sales_volume'].apply(
            lambda x: (x > 0).sum()
        )
        valid_products = product_months[product_months >= 3].index
        complete_matrix = complete_matrix[complete_matrix['product_id'].isin(valid_products)]
        
        print(f"Products with >=3 months of sales: {len(valid_products):,}")
        print(f"Total product-month combinations: {len(complete_matrix):,}")
        
        return complete_matrix
    
    def create_complete_matrix(self, sales_agg):
        """create complete product-month matrix with zero values for missing months"""
        # get all unique products and months
        all_products = sales_agg['product_id'].unique()
        all_months = sales_agg['order_month_dt'].unique()
        
        # create complete index
        complete_index = pd.MultiIndex.from_product(
            [all_products, all_months], 
            names=['product_id', 'order_month_dt']
        )
        
        # reindex and fill missing values with 0
        complete_matrix = sales_agg.set_index(['product_id', 'order_month_dt']).reindex(
            complete_index, fill_value=0
        ).reset_index()
        
        # convert back to YYYY-MM format
        complete_matrix['order_month'] = complete_matrix['order_month_dt'].dt.strftime('%Y-%m')
        
        return complete_matrix[['product_id', 'order_month', 'monthly_sales_volume', 'monthly_sales_value']]
    
    def apply_smoothing(self, data):
        """apply 3-month moving average smoothing"""
        print("Applying 3-month moving average smoothing...")
        
        # sort by product_id and order_month
        data = data.sort_values(['product_id', 'order_month'])
        
        # apply 3-month moving average
        data['smoothed_volume'] = data.groupby('product_id')['monthly_sales_volume'].transform(
            lambda x: x.rolling(window=3, min_periods=1, center=True).mean()
        )
        
        # calculate monthly growth rate
        data['monthly_growth_rate'] = data.groupby('product_id')['monthly_sales_volume'].pct_change() * 100
        
        # round numerical values to 2 decimal places
        data['monthly_sales_volume'] = data['monthly_sales_volume'].round(2)
        data['monthly_sales_value'] = data['monthly_sales_value'].round(2)
        data['smoothed_volume'] = data['smoothed_volume'].round(2)
        data['monthly_growth_rate'] = data['monthly_growth_rate'].round(2)
        
        # calculate average growth rate excluding infinite values
        valid_growth_rates = data['monthly_growth_rate'].replace([np.inf, -np.inf], np.nan).dropna()
        avg_growth_rate = valid_growth_rates.mean() if len(valid_growth_rates) > 0 else 0
        
        print(f"Applied smoothing to {len(data):,} records")
        print(f"Average smoothed volume: {data['smoothed_volume'].mean():.2f}")
        print(f"Average growth rate: {avg_growth_rate:.2f}%")
        print(f"Valid growth rate records: {len(valid_growth_rates):,} out of {len(data):,}")
        
        return data
    
    def generate_summary_stats(self, data):
        """generate product-level summary statistics"""
        print("Generating product summary statistics...")
        
        summary_stats = data.groupby('product_id').agg({
            'order_month': 'count',  # total months active
            'monthly_sales_volume': ['sum', 'mean', 'max', 'std'],  # sales statistics
            'monthly_sales_value': 'sum'  # total revenue
        }).reset_index()
        
        # flatten column names
        summary_stats.columns = [
            'product_id', 'total_months_active', 'total_units_sold', 
            'average_monthly_sales', 'max_monthly_sales', 'std_dev_sales_volume', 'total_revenue'
        ]
        
        # find month of maximum sales for each product
        max_sales_months = data.loc[data.groupby('product_id')['monthly_sales_volume'].idxmax()]
        max_sales_months = max_sales_months[['product_id', 'order_month']].rename(
            columns={'order_month': 'month_of_max_sales'}
        )
        
        # merge with summary stats
        summary_stats = summary_stats.merge(max_sales_months, on='product_id', how='left')
        
        # round numerical values
        numeric_cols = ['total_units_sold', 'average_monthly_sales', 'max_monthly_sales', 
                       'std_dev_sales_volume', 'total_revenue']
        summary_stats[numeric_cols] = summary_stats[numeric_cols].round(2)
        
        print(f"Generated summary stats for {len(summary_stats):,} products")
        print(f"Average total units sold per product: {summary_stats['total_units_sold'].mean():.2f}")
        print(f"Average monthly sales per product: {summary_stats['average_monthly_sales'].mean():.2f}")
        print(f"Total units sold across all products: {summary_stats['total_units_sold'].sum():,}")
        print(f"Products with >10 units sold: {len(summary_stats[summary_stats['total_units_sold'] > 10]):,}")
        print(f"Products with >50 units sold: {len(summary_stats[summary_stats['total_units_sold'] > 50]):,}")
        print(f"Products with >100 units sold: {len(summary_stats[summary_stats['total_units_sold'] > 100]):,}")
        
        return summary_stats
    
    def identify_top_products(self, summary_stats, n=10):
        """identify top N products by total sales volume"""
        top_products = summary_stats.nlargest(n, 'total_units_sold')['product_id'].tolist()
        print(f"Identified top {len(top_products)} products by sales volume")
        return top_products
    
    def plot_sales_curves(self, data, top_products):
        """generate sales curve plots for top products"""
        print(f"Generating sales curve plots for top {len(top_products)} products...")
        
        plots_created = 0
        for i, product_id in enumerate(top_products, 1):
            product_data = data[data['product_id'] == product_id].sort_values('order_month')
            
            if len(product_data) == 0:
                continue
                
            plt.figure(figsize=(12, 6))
            
            # plot raw sales volume
            plt.plot(range(len(product_data)), product_data['monthly_sales_volume'], 
                    'o-', alpha=0.7, label='Raw Sales Volume', linewidth=2, markersize=4)
            
            # plot smoothed sales volume
            plt.plot(range(len(product_data)), product_data['smoothed_volume'], 
                    'r-', alpha=0.8, label='Smoothed Volume (3-month MA)', linewidth=3)
            
            # annotate significant growth/decline points
            growth_rates = product_data['monthly_growth_rate'].dropna()
            if len(growth_rates) > 0:
                # find points with significant growth (>20%) or decline (<-20%)
                significant_points = product_data[
                    (product_data['monthly_growth_rate'] > 20) | 
                    (product_data['monthly_growth_rate'] < -20)
                ]
                
                for idx, row in significant_points.iterrows():
                    point_idx = product_data[product_data['order_month'] == row['order_month']].index[0]
                    point_idx_in_plot = list(product_data.index).index(point_idx)
                    
                    color = 'green' if row['monthly_growth_rate'] > 20 else 'red'
                    plt.annotate(f"{row['monthly_growth_rate']:.1f}%", 
                               xy=(point_idx_in_plot, row['monthly_sales_volume']),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=8, color=color, weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # customize plot
            plt.title(f'Product Sales Curve - {product_id}\n(Top {i} by Total Sales Volume)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Time Period (Months)', fontsize=12)
            plt.ylabel('Sales Volume (Units)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # x-axis labels
            plt.xticks(range(len(product_data)), product_data['order_month'], 
                      rotation=45, ha='right')
            
            # add statistics text
            total_sales = product_data['monthly_sales_volume'].sum()
            avg_sales = product_data['monthly_sales_volume'].mean()
            max_sales = product_data['monthly_sales_volume'].max()
            
            stats_text = f'Total Sales: {total_sales:.0f} units\n'
            stats_text += f'Avg Monthly: {avg_sales:.1f} units\n'
            stats_text += f'Peak Sales: {max_sales:.0f} units'
            
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # save the plot
            plot_filename = f"{self.output_path}product_sales_plot_{product_id}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots_created += 1
            print(f"Created plot {plots_created}: {plot_filename}")
        
        print(f"Total plots created: {plots_created}")
    
    def save_outputs(self, monthly_data, summary_stats):
         
        print("Saving output files...")
        
        # save monthly product sales data
        monthly_data.to_csv(f"{self.output_path}monthly_product_sales.csv", index=False)
        print(f"Saved: {self.output_path}monthly_product_sales.csv")
        
        # save product summary statistics
        summary_stats.to_csv(f"{self.output_path}product_sales_summary.csv", index=False)
        print(f"Saved: {self.output_path}product_sales_summary.csv")
        
        # print summary statistics
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total products analyzed: {summary_stats['product_id'].nunique():,}")
        print(f"Total product-month combinations: {len(monthly_data):,}")
        print(f"Average months active per product: {summary_stats['total_months_active'].mean():.1f}")
        print(f"Average total units sold per product: {summary_stats['total_units_sold'].mean():.1f}")
        print(f"Top 10 products by sales volume:")
        
        top_10 = summary_stats.nlargest(10, 'total_units_sold')
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            print(f"  {i:2d}. {row['product_id']}: {row['total_units_sold']:,.0f} units")
        
        print(f"\nOutput files created:")
        print(f"  - {self.output_path}monthly_product_sales.csv")
        print(f"  - {self.output_path}product_sales_summary.csv")
        print(f"  - {len(top_10)} sales curve plots")
    
    def run_analysis(self):
        """run the complete sales curve analysis"""
        print("="*60)
        print("PRODUCT SALES CURVE ANALYSIS")
        print("="*60)

        order_items, orders, products = self.load_data()

        processed_data = self.preprocess_data(order_items, orders, products)
        
        # aggregate sales
        sales_data = self.aggregate_sales(processed_data)
        
        # apply smoothing and calculate growth rates
        final_data = self.apply_smoothing(sales_data)
        
        # generate summary statistics
        summary_stats = self.generate_summary_stats(final_data)
        
        # identify top products
        top_products = self.identify_top_products(summary_stats, n=10)
        
        # generate plots
        self.plot_sales_curves(final_data, top_products)

        self.save_outputs(final_data, summary_stats)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return final_data, summary_stats

def main():
    """main execution function"""
    analyzer = ProductSalesCurveAnalyzer()
    monthly_data, summary_stats = analyzer.run_analysis()
    
    return monthly_data, summary_stats

if __name__ == "__main__":
    main() 