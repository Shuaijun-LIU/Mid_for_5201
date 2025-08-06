
"""
Step 3: Inventory Efficiency Analysis
Execution date: 2025-06-26
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

class InventoryEfficiencyAnalyzer:
    """inventory efficiency analysis across product lifecycle stages"""
    
    def __init__(self, data_path="../data/processed_missing/", output_path="./output/"):
        """initialize the analyzer with data paths"""
        self.data_path = data_path
        self.output_path = output_path
        self.create_output_directory()
        
    def create_output_directory(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")
    
    def load_data(self):
         
        print("Loading datasets...")
        
        # load lifecycle labels from task 2
        lifecycle_labels = pd.read_csv(f"{self.output_path}product_lifecycle_labels.csv")
        print(f"Loaded lifecycle labels: {len(lifecycle_labels):,} products")
        
        # load order items data
        order_items = pd.read_csv(f"{self.data_path}olist_order_items_dataset.csv")
        print(f"Loaded order items: {len(order_items):,} records")
        
        # load orders data
        orders = pd.read_csv(f"{self.data_path}olist_orders_dataset.csv")
        print(f"Loaded orders: {len(orders):,} records")
        
        # load products data for category information
        products = pd.read_csv(f"{self.data_path}olist_products_dataset.csv")
        products = products[['product_id', 'product_category_name']]
        print(f"Loaded product categories: {len(products):,} records")
        
        return lifecycle_labels, order_items, orders, products
    
    def merge_order_data(self, order_items, orders):
        """merge order items with orders to compute delivery time"""
        print("Merging order data...")
        
        # merge order items with orders
        merged_data = order_items.merge(
            orders[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date', 'order_status']], 
            on='order_id', 
            how='inner'
        )
        
        # filter for delivered orders only
        merged_data = merged_data[merged_data['order_status'] == 'delivered'].copy()
        print(f"After filtering delivered orders: {len(merged_data):,} records")
        
        # clean invalid date values
        merged_data = merged_data[
            (merged_data['order_delivered_customer_date'] != '0000-00-00 00:00:00') &
            (merged_data['order_delivered_customer_date'].notna())
        ]
        print(f"After filtering invalid delivery dates: {len(merged_data):,} records")
        
        # convert timestamps to datetime with error handling
        merged_data['order_purchase_timestamp'] = pd.to_datetime(merged_data['order_purchase_timestamp'], errors='coerce')
        merged_data['order_delivered_customer_date'] = pd.to_datetime(merged_data['order_delivered_customer_date'], errors='coerce')
        
        # remove rows with invalid dates
        merged_data = merged_data.dropna(subset=['order_purchase_timestamp', 'order_delivered_customer_date'])
        print(f"After removing invalid dates: {len(merged_data):,} records")
        
        # calculate delivery time in days
        merged_data['delivery_days'] = (
            merged_data['order_delivered_customer_date'] - merged_data['order_purchase_timestamp']
        ).dt.days
        
        # filter out invalid delivery times (negative or too long)
        merged_data = merged_data[
            (merged_data['delivery_days'] >= 0) & 
            (merged_data['delivery_days'] <= 100)  # reasonable delivery time limit
        ]
        
        print(f"After filtering valid delivery times: {len(merged_data):,} records")
        print(f"Average delivery time: {merged_data['delivery_days'].mean():.1f} days")
        
        return merged_data
    
    def calculate_inventory_metrics(self, merged_data):
        """calculate inventory efficiency metrics per product"""
        print("Calculating inventory efficiency metrics...")
        
        inventory_metrics = []
        
        for product_id in merged_data['product_id'].unique():
            product_data = merged_data[merged_data['product_id'] == product_id]
            
            # basic metrics
            total_units_sold = len(product_data)
            total_revenue = product_data['price'].sum()
            avg_price = product_data['price'].mean()
            avg_delivery_time = product_data['delivery_days'].mean()
            
            # date range
            first_sale_date = product_data['order_purchase_timestamp'].min()
            last_sale_date = product_data['order_purchase_timestamp'].max()
            
            # inventory duration in months
            inventory_duration_days = (last_sale_date - first_sale_date).days
            inventory_duration_months = max(1, inventory_duration_days / 30.44)  # avoid division by zero
            
            # turnover rate
            turnover_rate = total_units_sold / inventory_duration_months
            
            # estimated holding cost (2% per day)
            holding_cost_rate = 0.02  # 2% per day
            estimated_holding_cost = avg_delivery_time * avg_price * holding_cost_rate
            
            inventory_metrics.append({
                'product_id': product_id,
                'total_units_sold': total_units_sold,
                'total_revenue': round(total_revenue, 2),
                'avg_price': round(avg_price, 2),
                'avg_delivery_time': round(avg_delivery_time, 2),
                'inventory_duration_days': inventory_duration_days,
                'inventory_duration_months': round(inventory_duration_months, 2),
                'turnover_rate': round(turnover_rate, 2),
                'estimated_holding_cost': round(estimated_holding_cost, 2)
            })
        
        metrics_df = pd.DataFrame(inventory_metrics)
        print(f"Calculated metrics for {len(metrics_df):,} products")
        print(f"Average turnover rate: {metrics_df['turnover_rate'].mean():.2f}")
        print(f"Average delivery time: {metrics_df['avg_delivery_time'].mean():.2f} days")
        print(f"Average holding cost: {metrics_df['estimated_holding_cost'].mean():.2f}")
        
        return metrics_df
    
    def join_lifecycle_stages(self, inventory_metrics, lifecycle_labels, products):
        """join inventory metrics with lifecycle stages and product categories"""
        print("Joining with lifecycle stages...")
        
        # merge with lifecycle labels
        final_data = inventory_metrics.merge(
            lifecycle_labels[['product_id', 'lifecycle_stage']], 
            on='product_id', 
            how='inner'
        )
        
        # merge with product categories
        final_data = final_data.merge(products, on='product_id', how='left')
        
        # fill missing categories
        final_data['product_category_name'] = final_data['product_category_name'].fillna('Unknown')
        
        print(f"Final dataset: {len(final_data):,} products")
        print(f"Lifecycle stage distribution:")
        print(final_data['lifecycle_stage'].value_counts())
        
        return final_data
    
    def generate_stage_summaries(self, final_data):
        """generate summary statistics by lifecycle stage"""
        print("Generating stage summaries...")
        
        # summary by lifecycle stage
        stage_summary = final_data.groupby('lifecycle_stage').agg({
            'total_units_sold': 'sum',
            'total_revenue': 'sum',
            'turnover_rate': 'mean',
            'avg_delivery_time': 'mean',
            'estimated_holding_cost': 'sum',
            'product_id': 'count'
        }).round(2)
        
        stage_summary.columns = [
            'total_units_sold', 'total_revenue', 'avg_turnover_rate', 
            'avg_delivery_time', 'total_holding_cost', 'product_count'
        ]
        
        print(f"\nSummary by lifecycle stage:")
        for stage in stage_summary.index:
            print(f"\n  {stage}:")
            print(f"    Product count: {stage_summary.loc[stage, 'product_count']:,}")
            print(f"    Total units sold: {stage_summary.loc[stage, 'total_units_sold']:,}")
            print(f"    Total revenue: ${stage_summary.loc[stage, 'total_revenue']:,.2f}")
            print(f"    Avg turnover rate: {stage_summary.loc[stage, 'avg_turnover_rate']:.2f}")
            print(f"    Avg delivery time: {stage_summary.loc[stage, 'avg_delivery_time']:.1f} days")
            print(f"    Total holding cost: ${stage_summary.loc[stage, 'total_holding_cost']:,.2f}")
        
        return stage_summary
    
    def create_visualizations(self, final_data, stage_summary):
        """create visualizations for inventory efficiency analysis"""
        print("Creating inventory efficiency visualizations...")
        
        # 1. bar chart: average turnover rate per lifecycle stage
        plt.figure(figsize=(10, 6))
        # use actual stages from data instead of hardcoded order
        stage_order = stage_summary.index.tolist()
        colors = ['#2E8B57', '#FFD700', '#4169E1', '#DC143C'][:len(stage_order)]
        
        turnover_data = stage_summary.loc[stage_order, 'avg_turnover_rate']
        bars = plt.bar(turnover_data.index, turnover_data.values, color=colors, alpha=0.8)
        
        plt.title('Average Inventory Turnover Rate by Lifecycle Stage', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Lifecycle Stage', fontsize=12)
        plt.ylabel('Average Turnover Rate (units/month)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # add value labels
        for bar, value in zip(bars, turnover_data.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}inventory_turnover_by_stage.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}inventory_turnover_by_stage.png")
        
        # 2. heatmap: holding cost by stage and category
        plt.figure(figsize=(14, 8))
        
        # prepare data for heatmap
        heatmap_data = final_data.groupby(['lifecycle_stage', 'product_category_name']).agg({
            'estimated_holding_cost': 'mean',
            'product_id': 'count'
        }).reset_index()
        
        # pivot for heatmap
        heatmap_pivot = heatmap_data.pivot(
            index='product_category_name', 
            columns='lifecycle_stage', 
            values='estimated_holding_cost'
        ).fillna(0)
        
        # create heatmap
        sns.heatmap(heatmap_pivot, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Average Holding Cost ($)'})
        
        plt.title('Average Holding Cost by Product Category and Lifecycle Stage', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Lifecycle Stage', fontsize=12)
        plt.ylabel('Product Category', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}holding_cost_by_category_and_stage.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}holding_cost_by_category_and_stage.png")
        
        # 3. boxplot: delivery time distribution per lifecycle stage
        plt.figure(figsize=(12, 8))
        
        # filter out extreme outliers for better visualization
        delivery_data = final_data[final_data['avg_delivery_time'] <= 30]
        
        # create boxplot
        # use actual stages from data
        stage_order = final_data['lifecycle_stage'].unique()
        delivery_data['lifecycle_stage'] = pd.Categorical(
            delivery_data['lifecycle_stage'], 
            categories=stage_order, 
            ordered=True
        )
        
        # create color palette based on number of stages
        colors = ['#2E8B57', '#FFD700', '#4169E1', '#DC143C'][:len(stage_order)]
        
        sns.boxplot(data=delivery_data, x='lifecycle_stage', y='avg_delivery_time', 
                   palette=colors)
        
        plt.title('Delivery Time Distribution by Lifecycle Stage', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Lifecycle Stage', fontsize=12)
        plt.ylabel('Average Delivery Time (Days)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # add statistics text
        stats_text = f"Total products: {len(delivery_data):,}\n"
        stats_text += f"Avg delivery time: {delivery_data['avg_delivery_time'].mean():.1f} days"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}delivery_time_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}delivery_time_boxplot.png")
    
    def save_outputs(self, final_data, stage_summary):
         
        print("Saving output files...")
        
        # save inventory efficiency metrics
        final_data.to_csv(f"{self.output_path}inventory_efficiency_metrics.csv", index=False)
        print(f"Saved: {self.output_path}inventory_efficiency_metrics.csv")
        
        # print detailed summary
        print("\n" + "="*60)
        print("INVENTORY EFFICIENCY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total products analyzed: {len(final_data):,}")
        
        # overall statistics
        print(f"\nOverall statistics:")
        print(f"  Average turnover rate: {final_data['turnover_rate'].mean():.2f}")
        print(f"  Average delivery time: {final_data['avg_delivery_time'].mean():.1f} days")
        print(f"  Total holding cost: ${final_data['estimated_holding_cost'].sum():,.2f}")
        print(f"  Total revenue: ${final_data['total_revenue'].sum():,.2f}")
        
        # efficiency insights
        print(f"\nEfficiency insights:")
        best_turnover_stage = stage_summary['avg_turnover_rate'].idxmax()
        worst_turnover_stage = stage_summary['avg_turnover_rate'].idxmin()
        print(f"  Best turnover: {best_turnover_stage} ({stage_summary.loc[best_turnover_stage, 'avg_turnover_rate']:.2f})")
        print(f"  Worst turnover: {worst_turnover_stage} ({stage_summary.loc[worst_turnover_stage, 'avg_turnover_rate']:.2f})")
        
        best_delivery_stage = stage_summary['avg_delivery_time'].idxmin()
        worst_delivery_stage = stage_summary['avg_delivery_time'].idxmax()
        print(f"  Fastest delivery: {best_delivery_stage} ({stage_summary.loc[best_delivery_stage, 'avg_delivery_time']:.1f} days)")
        print(f"  Slowest delivery: {worst_delivery_stage} ({stage_summary.loc[worst_delivery_stage, 'avg_delivery_time']:.1f} days)")
        
        print(f"\nOutput files created:")
        print(f"  - {self.output_path}inventory_efficiency_metrics.csv")
        print(f"  - {self.output_path}inventory_turnover_by_stage.png")
        print(f"  - {self.output_path}holding_cost_by_category_and_stage.png")
        print(f"  - {self.output_path}delivery_time_boxplot.png")
    
    def run_analysis(self):
        """run the complete inventory efficiency analysis"""
        print("="*60)
        print("INVENTORY EFFICIENCY ANALYSIS")
        print("="*60)
        
        lifecycle_labels, order_items, orders, products = self.load_data()
        
        # merge order data
        merged_data = self.merge_order_data(order_items, orders)
        
        # calculate inventory metrics
        inventory_metrics = self.calculate_inventory_metrics(merged_data)
        
        # join with lifecycle stages
        final_data = self.join_lifecycle_stages(inventory_metrics, lifecycle_labels, products)
        
        # stage sum
        stage_summary = self.generate_stage_summaries(final_data)
        
        # visual
        self.create_visualizations(final_data, stage_summary)
        
        self.save_outputs(final_data, stage_summary)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return final_data, stage_summary

def main():
    """main execution function"""
    analyzer = InventoryEfficiencyAnalyzer()
    final_data, stage_summary = analyzer.run_analysis()
    
    return final_data, stage_summary

if __name__ == "__main__":
    main() 