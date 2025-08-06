
"""
Step 4: Stock Risk Detection
Execution date: 2025-06-27
Update date: 2025-07-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StockRiskDetector:
    """stock risk detection based on sales velocity, turnover, and lifecycle trends"""
    
    def __init__(self, data_path="./output/", output_path="./output/"):
        """initialize the detector with data paths"""
        self.data_path = data_path
        self.output_path = output_path
        self.create_output_directory()
        
    def create_output_directory(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")
    
    def load_data(self):
         
        print("Loading datasets...")
        
        # load inventory efficiency metrics from task 3
        inventory_metrics = pd.read_csv(f"{self.data_path}inventory_efficiency_metrics.csv")
        print(f"Loaded inventory metrics: {len(inventory_metrics):,} products")
        
        # load lifecycle labels from task 2
        lifecycle_labels = pd.read_csv(f"{self.data_path}product_lifecycle_labels.csv")
        print(f"Loaded lifecycle labels: {len(lifecycle_labels):,} products")
        
        # load order items for cross-checking
        order_items = pd.read_csv("../data/processed_missing/olist_order_items_dataset.csv")
        print(f"Loaded order items: {len(order_items):,} records")
        
        return inventory_metrics, lifecycle_labels, order_items
    
    def calculate_risk_thresholds(self, inventory_metrics):
        """calculate risk thresholds using quantile-based cutoffs"""
        print("Calculating risk thresholds...")
        
        # calculate percentiles for risk thresholds
        turnover_90th = inventory_metrics['turnover_rate'].quantile(0.90)
        holding_cost_90th = inventory_metrics['estimated_holding_cost'].quantile(0.90)
        turnover_median = inventory_metrics['turnover_rate'].quantile(0.50)
        delivery_time_median = inventory_metrics['avg_delivery_time'].quantile(0.50)
        
        # define risk thresholds
        risk_thresholds = {
            'overstock_turnover_threshold': 2.0,  # low turnover threshold
            'overstock_holding_cost_threshold': holding_cost_90th,
            'stockout_turnover_threshold': turnover_90th,
            'stockout_delivery_time_threshold': delivery_time_median,
            'turnover_90th_percentile': turnover_90th,
            'holding_cost_90th_percentile': holding_cost_90th,
            'turnover_median': turnover_median,
            'delivery_time_median': delivery_time_median
        }
        
        print(f"Risk thresholds calculated:")
        print(f"  Overstock turnover threshold: {risk_thresholds['overstock_turnover_threshold']:.2f}")
        print(f"  Overstock holding cost threshold: {risk_thresholds['overstock_holding_cost_threshold']:.2f}")
        print(f"  Stockout turnover threshold: {risk_thresholds['stockout_turnover_threshold']:.2f}")
        print(f"  Stockout delivery time threshold: {risk_thresholds['stockout_delivery_time_threshold']:.2f}")
        print(f"  Turnover 90th percentile: {risk_thresholds['turnover_90th_percentile']:.2f}")
        print(f"  Holding cost 90th percentile: {risk_thresholds['holding_cost_90th_percentile']:.2f}")
        
        return risk_thresholds
    
    def apply_risk_tagging(self, risk_data, risk_thresholds):
        """apply risk tagging based on defined criteria"""
        print("Applying risk tagging...")
        
        def assign_risk_flag(row):
            """assign risk flag based on business rules"""
            
            # overstock risk: maturity/decline stage with low turnover and high holding cost
            if (row['lifecycle_stage'] in ['Maturity', 'Decline'] and
                row['turnover_rate'] < risk_thresholds['overstock_turnover_threshold'] and
                row['estimated_holding_cost'] > risk_thresholds['overstock_holding_cost_threshold']):
                return 'Overstock Risk'
            
            # stockout risk: growth/introduction stage with high turnover and short delivery time
            elif (row['lifecycle_stage'] in ['Growth', 'Introduction'] and
                  row['turnover_rate'] > risk_thresholds['stockout_turnover_threshold'] and
                  row['avg_delivery_time'] < risk_thresholds['stockout_delivery_time_threshold']):
                return 'Stockout Risk'
            
            # stable: neither condition met
            else:
                return 'Stable'
        
        # apply risk classification
        risk_data['risk_flag'] = risk_data.apply(assign_risk_flag, axis=1)
        
        # print risk distribution
        risk_counts = risk_data['risk_flag'].value_counts()
        print(f"\nRisk distribution:")
        for risk_type, count in risk_counts.items():
            percentage = (count / len(risk_data)) * 100
            print(f"  {risk_type}: {count:,} products ({percentage:.1f}%)")
        
        return risk_data
    
    def generate_risk_summary(self, risk_data):
        """generate comprehensive risk summary"""
        print("Generating risk summary...")
        
        # risk distribution by lifecycle stage
        stage_risk_summary = risk_data.groupby(['lifecycle_stage', 'risk_flag']).size().unstack(fill_value=0)
        
        # risk distribution by product category
        category_risk_summary = risk_data.groupby(['product_category_name', 'risk_flag']).size().unstack(fill_value=0)
        
        # top high-risk products
        overstock_risk_products = risk_data[risk_data['risk_flag'] == 'Overstock Risk'].nlargest(10, 'estimated_holding_cost')
        stockout_risk_products = risk_data[risk_data['risk_flag'] == 'Stockout Risk'].nlargest(10, 'turnover_rate')
        
        print(f"\nRisk summary by lifecycle stage:")
        print(stage_risk_summary)
        
        print(f"\nTop 10 overstock risk products (by holding cost):")
        for i, (_, row) in enumerate(overstock_risk_products.iterrows(), 1):
            print(f"  {i:2d}. {row['product_id']}: ${row['estimated_holding_cost']:.2f} holding cost, {row['turnover_rate']:.2f} turnover")
        
        print(f"\nTop 10 stockout risk products (by turnover rate):")
        for i, (_, row) in enumerate(stockout_risk_products.iterrows(), 1):
            print(f"  {i:2d}. {row['product_id']}: {row['turnover_rate']:.2f} turnover, {row['avg_delivery_time']:.1f} days delivery")
        
        return {
            'stage_risk_summary': stage_risk_summary,
            'category_risk_summary': category_risk_summary,
            'top_overstock_risk': overstock_risk_products,
            'top_stockout_risk': stockout_risk_products
        }
    
    def create_visualizations(self, risk_data, risk_summary):
        """create visualizations for risk analysis"""
        print("Creating risk analysis visualizations...")
        
        # 1. bar chart: count of products in each risk category
        plt.figure(figsize=(10, 6))
        risk_counts = risk_data['risk_flag'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # red, teal, blue
        
        bars = plt.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.8)
        plt.title('Product Distribution by Stock Risk Category', fontsize=16, fontweight='bold')
        plt.xlabel('Risk Category', fontsize=12)
        plt.ylabel('Number of Products', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # add value labels on bars
        for bar, count in zip(bars, risk_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}risk_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}risk_distribution.png")
        
        # 2. heatmap: lifecycle stage vs. risk type
        plt.figure(figsize=(12, 8))
        
        # prepare data for heatmap
        heatmap_data = risk_data.groupby(['lifecycle_stage', 'risk_flag']).size().unstack(fill_value=0)
        
        # create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Number of Products'})
        
        plt.title('Risk Distribution by Lifecycle Stage', fontsize=16, fontweight='bold')
        plt.xlabel('Risk Category', fontsize=12)
        plt.ylabel('Lifecycle Stage', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}risk_summary_by_lifecycle.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}risk_summary_by_lifecycle.png")
        
        # 3. scatterplot: turnover rate vs. holding cost with color-coded risk flags
        plt.figure(figsize=(14, 10))
        
        # create scatter plot with different colors for each risk type
        risk_colors = {'Overstock Risk': '#FF6B6B', 'Stockout Risk': '#4ECDC4', 'Stable': '#45B7D1'}
        
        for risk_type in risk_data['risk_flag'].unique():
            subset = risk_data[risk_data['risk_flag'] == risk_type]
            plt.scatter(subset['turnover_rate'], subset['estimated_holding_cost'], 
                       c=risk_colors[risk_type], label=risk_type, alpha=0.7, s=50)
        
        # add threshold lines
        turnover_90th = risk_data['turnover_rate'].quantile(0.90)
        holding_cost_90th = risk_data['estimated_holding_cost'].quantile(0.90)
        
        plt.axhline(y=holding_cost_90th, color='red', linestyle='--', alpha=0.5, 
                   label=f'Holding Cost 90th Percentile ({holding_cost_90th:.2f})')
        plt.axvline(x=turnover_90th, color='blue', linestyle='--', alpha=0.5, 
                   label=f'Turnover 90th Percentile ({turnover_90th:.2f})')
        plt.axvline(x=2.0, color='orange', linestyle='--', alpha=0.5, 
                   label='Low Turnover Threshold (2.0)')
        
        plt.title('Stock Risk Analysis: Turnover Rate vs. Holding Cost', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Turnover Rate (units/month)', fontsize=12)
        plt.ylabel('Estimated Holding Cost ($)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # add statistics text
        stats_text = f"Total products: {len(risk_data):,}\n"
        stats_text += f"Overstock risk: {len(risk_data[risk_data['risk_flag'] == 'Overstock Risk']):,}\n"
        stats_text += f"Stockout risk: {len(risk_data[risk_data['risk_flag'] == 'Stockout Risk']):,}\n"
        stats_text += f"Stable: {len(risk_data[risk_data['risk_flag'] == 'Stable']):,}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}stock_risk_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}stock_risk_scatter.png")
    
    def save_outputs(self, risk_data, risk_summary):
         
        print("Saving output files...")
        
        # save risk flags
        risk_data.to_csv(f"{self.output_path}product_stock_risk_flags.csv", index=False)
        print(f"Saved: {self.output_path}product_stock_risk_flags.csv")
        
        # print detailed summary
        print("\n" + "="*60)
        print("STOCK RISK DETECTION SUMMARY")
        print("="*60)
        print(f"Total products analyzed: {len(risk_data):,}")
        
        # risk distribution
        risk_counts = risk_data['risk_flag'].value_counts()
        print(f"\nRisk distribution:")
        for risk_type, count in risk_counts.items():
            percentage = (count / len(risk_data)) * 100
            print(f"  {risk_type}: {count:,} products ({percentage:.1f}%)")
        
        # risk by lifecycle stage
        print(f"\nRisk by lifecycle stage:")
        stage_risk = risk_data.groupby(['lifecycle_stage', 'risk_flag']).size().unstack(fill_value=0)
        for stage in stage_risk.index:
            print(f"\n  {stage}:")
            for risk_type in ['Overstock Risk', 'Stockout Risk', 'Stable']:
                if risk_type in stage_risk.columns:
                    count = stage_risk.loc[stage, risk_type]
                    percentage = (count / stage_risk.loc[stage].sum()) * 100
                    print(f"    {risk_type}: {count:,} ({percentage:.1f}%)")
        
        # business insights
        print(f"\nBusiness insights:")
        overstock_count = len(risk_data[risk_data['risk_flag'] == 'Overstock Risk'])
        stockout_count = len(risk_data[risk_data['risk_flag'] == 'Stockout Risk'])
        
        if overstock_count > 0:
            avg_overstock_cost = risk_data[risk_data['risk_flag'] == 'Overstock Risk']['estimated_holding_cost'].mean()
            print(f"  Overstock risk products: {overstock_count:,} (avg holding cost: ${avg_overstock_cost:.2f})")
        
        if stockout_count > 0:
            avg_stockout_turnover = risk_data[risk_data['risk_flag'] == 'Stockout Risk']['turnover_rate'].mean()
            print(f"  Stockout risk products: {stockout_count:,} (avg turnover: {avg_stockout_turnover:.2f})")
        
        print(f"\nOutput files created:")
        print(f"  - {self.output_path}product_stock_risk_flags.csv")
        print(f"  - {self.output_path}risk_distribution.png")
        print(f"  - {self.output_path}risk_summary_by_lifecycle.png")
        print(f"  - {self.output_path}stock_risk_scatter.png")
    
    def run_detection(self):
        """run the complete stock risk detection analysis"""
        print("="*60)
        print("STOCK RISK DETECTION")
        print("="*60)

        inventory_metrics, lifecycle_labels, order_items = self.load_data()
        
        # merge with lifecycle stages
        risk_data = inventory_metrics.merge(
            lifecycle_labels[['product_id', 'lifecycle_stage']], 
            on='product_id', 
            how='inner'
        )
        
        # handle duplicate lifecycle_stage columns if they exist
        if 'lifecycle_stage_x' in risk_data.columns and 'lifecycle_stage_y' in risk_data.columns:
            # use the one from lifecycle_labels (y)
            risk_data['lifecycle_stage'] = risk_data['lifecycle_stage_y']
            risk_data = risk_data.drop(['lifecycle_stage_x', 'lifecycle_stage_y'], axis=1)
        elif 'lifecycle_stage_x' in risk_data.columns:
            risk_data['lifecycle_stage'] = risk_data['lifecycle_stage_x']
            risk_data = risk_data.drop('lifecycle_stage_x', axis=1)
        elif 'lifecycle_stage_y' in risk_data.columns:
            risk_data['lifecycle_stage'] = risk_data['lifecycle_stage_y']
            risk_data = risk_data.drop('lifecycle_stage_y', axis=1)
        
        print(f"Merged data: {len(risk_data):,} products")
        print(f"Columns in risk_data: {list(risk_data.columns)}")
        print(f"Sample lifecycle stages: {risk_data['lifecycle_stage'].value_counts().head()}")
        
        # calculate risk thresholds
        risk_thresholds = self.calculate_risk_thresholds(risk_data)
        
        # apply risk tagging
        risk_data = self.apply_risk_tagging(risk_data, risk_thresholds)
        
        # risk summary
        risk_summary = self.generate_risk_summary(risk_data)
        
        # visual
        self.create_visualizations(risk_data, risk_summary)

        self.save_outputs(risk_data, risk_summary)
        
        print("\n" + "="*60)
        print("RISK DETECTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return risk_data, risk_summary

def main():
    """main execution function"""
    detector = StockRiskDetector()
    risk_data, risk_summary = detector.run_detection()
    
    return risk_data, risk_summary

if __name__ == "__main__":
    main() 