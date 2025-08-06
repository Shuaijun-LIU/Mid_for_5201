"""
Step 5: Inventory Policy Recommendation
Execution date: 2025-06-28
Update date: 2025-06-30
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

class InventoryPolicyRecommender:
    """inventory policy recommendation based on lifecycle stage and sales volatility"""
    
    def __init__(self, data_path="./output/", output_path="./output/"):
        """initialize the recommender with data paths"""
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
        lifecycle_labels = pd.read_csv(f"{self.data_path}product_lifecycle_labels.csv")
        print(f"Loaded lifecycle labels: {len(lifecycle_labels):,} products")
        
        # load stock risk flags from task 4
        risk_flags = pd.read_csv(f"{self.data_path}product_stock_risk_flags.csv")
        print(f"Loaded risk flags: {len(risk_flags):,} products")
        
        # load inventory efficiency metrics from task 3
        inventory_metrics = pd.read_csv(f"{self.data_path}inventory_efficiency_metrics.csv")
        print(f"Loaded inventory metrics: {len(inventory_metrics):,} products")
        
        # load monthly sales data from task 1 for volatility calculation
        monthly_sales = pd.read_csv(f"{self.data_path}monthly_product_sales.csv")
        print(f"Loaded monthly sales: {len(monthly_sales):,} records")
        
        return lifecycle_labels, risk_flags, inventory_metrics, monthly_sales
    
    def calculate_sales_volatility(self, monthly_sales):
        """calculate sales volatility using coefficient of variation"""
        print("Calculating sales volatility...")
        
        volatility_data = []
        
        for product_id in monthly_sales['product_id'].unique():
            product_data = monthly_sales[monthly_sales['product_id'] == product_id]
            
            # filter out zero sales months for volatility calculation
            sales_volume = product_data[product_data['monthly_sales_volume'] > 0]['monthly_sales_volume']
            
            if len(sales_volume) >= 2:
                # calculate coefficient of variation
                mean_sales = sales_volume.mean()
                std_sales = sales_volume.std()
                cv = std_sales / mean_sales if mean_sales > 0 else 0
            else:
                cv = 0
            
            volatility_data.append({
                'product_id': product_id,
                'cv': round(cv, 3),
                'sales_months': len(sales_volume)
            })
        
        volatility_df = pd.DataFrame(volatility_data)
        print(f"Calculated volatility for {len(volatility_df):,} products")
        print(f"Average CV: {volatility_df['cv'].mean():.3f}")
        print(f"CV range: {volatility_df['cv'].min():.3f} - {volatility_df['cv'].max():.3f}")
        
        return volatility_df
    
    def categorize_volatility_levels(self, volatility_df):
        """categorize volatility into low, medium, high levels"""
        print("Categorizing volatility levels...")
        
        def assign_volatility_level(cv):
            """assign volatility level based on CV thresholds"""
            if cv < 0.3:
                return 'Low'
            elif cv < 0.6:
                return 'Medium'
            else:
                return 'High'
        
        volatility_df['volatility_level'] = volatility_df['cv'].apply(assign_volatility_level)
        
        # print distribution
        level_counts = volatility_df['volatility_level'].value_counts()
        print(f"\nVolatility level distribution:")
        for level, count in level_counts.items():
            percentage = (count / len(volatility_df)) * 100
            print(f"  {level}: {count:,} products ({percentage:.1f}%)")
        
        return volatility_df
    
    def define_policy_matrix(self):
        """define the inventory policy matrix"""
        print("Defining inventory policy matrix...")
        
        # policy matrix based on lifecycle stage and volatility level
        policy_matrix = {
            'Introduction': {
                'Low': {
                    'reorder_frequency': 'Monthly',
                    'safety_stock_percent': 10,
                    'review_strategy': 'Fixed Interval'
                },
                'Medium': {
                    'reorder_frequency': 'Bi-weekly',
                    'safety_stock_percent': 20,
                    'review_strategy': 'Dynamic Recalibration'
                },
                'High': {
                    'reorder_frequency': 'Weekly',
                    'safety_stock_percent': 30,
                    'review_strategy': 'Just-in-Time'
                }
            },
            'Growth': {
                'Low': {
                    'reorder_frequency': 'Bi-weekly',
                    'safety_stock_percent': 15,
                    'review_strategy': 'Dynamic Recalibration'
                },
                'Medium': {
                    'reorder_frequency': 'Weekly',
                    'safety_stock_percent': 25,
                    'review_strategy': 'Just-in-Time'
                },
                'High': {
                    'reorder_frequency': 'Weekly',
                    'safety_stock_percent': 40,
                    'review_strategy': 'Just-in-Time + Review'
                }
            },
            'Maturity': {
                'Low': {
                    'reorder_frequency': 'Monthly',
                    'safety_stock_percent': 10,
                    'review_strategy': 'Fixed Interval'
                },
                'Medium': {
                    'reorder_frequency': 'Bi-weekly',
                    'safety_stock_percent': 15,
                    'review_strategy': 'Dynamic Recalibration'
                },
                'High': {
                    'reorder_frequency': 'Weekly',
                    'safety_stock_percent': 25,
                    'review_strategy': 'Dynamic Recalibration'
                }
            },
            'Decline': {
                'Low': {
                    'reorder_frequency': 'Monthly',
                    'safety_stock_percent': 5,
                    'review_strategy': 'Fixed Interval'
                },
                'Medium': {
                    'reorder_frequency': 'Monthly',
                    'safety_stock_percent': 10,
                    'review_strategy': 'Fixed Interval'
                },
                'High': {
                    'reorder_frequency': 'Bi-weekly',
                    'safety_stock_percent': 15,
                    'review_strategy': 'Dynamic Recalibration'
                }
            }
        }
        
        print(f"Policy matrix defined with {len(policy_matrix)} lifecycle stages and 3 volatility levels")
        
        return policy_matrix
    
    def assign_policies(self, merged_data, policy_matrix):
        """assign inventory policies to each product"""
        print("Assigning inventory policies...")
        
        def get_policy(row):
            """get policy based on lifecycle stage and volatility level"""
            lifecycle = row['lifecycle_stage']
            volatility = row['volatility_level']
            
            if lifecycle in policy_matrix and volatility in policy_matrix[lifecycle]:
                return policy_matrix[lifecycle][volatility]
            else:
                # default policy for unknown combinations
                return {
                    'reorder_frequency': 'Monthly',
                    'safety_stock_percent': 15,
                    'review_strategy': 'Fixed Interval'
                }
        
        # apply policies
        policies = merged_data.apply(get_policy, axis=1)
        
        # extract policy components
        merged_data['reorder_frequency'] = [p['reorder_frequency'] for p in policies]
        merged_data['safety_stock_percent'] = [p['safety_stock_percent'] for p in policies]
        merged_data['review_strategy'] = [p['review_strategy'] for p in policies]
        
        # adjust policies based on risk flags
        merged_data = self.adjust_policies_for_risk(merged_data)
        
        # print policy distribution
        print(f"\nPolicy distribution:")
        print(f"  Reorder frequency:")
        freq_counts = merged_data['reorder_frequency'].value_counts()
        for freq, count in freq_counts.items():
            percentage = (count / len(merged_data)) * 100
            print(f"    {freq}: {count:,} products ({percentage:.1f}%)")
        
        print(f"  Review strategy:")
        strategy_counts = merged_data['review_strategy'].value_counts()
        for strategy, count in strategy_counts.items():
            percentage = (count / len(merged_data)) * 100
            print(f"    {strategy}: {count:,} products ({percentage:.1f}%)")
        
        return merged_data
    
    def adjust_policies_for_risk(self, data):
        """adjust policies based on risk flags"""
        print("Adjusting policies for risk flags...")
        
        # for stockout risk products, make policies more aggressive
        stockout_mask = data['risk_flag'] == 'Stockout Risk'
        data.loc[stockout_mask, 'safety_stock_percent'] = data.loc[stockout_mask, 'safety_stock_percent'] * 1.5
        data.loc[stockout_mask, 'reorder_frequency'] = 'Weekly'
        data.loc[stockout_mask, 'review_strategy'] = 'Just-in-Time'
        
        # for overstock risk products, make policies more conservative
        overstock_mask = data['risk_flag'] == 'Overstock Risk'
        data.loc[overstock_mask, 'safety_stock_percent'] = data.loc[overstock_mask, 'safety_stock_percent'] * 0.5
        data.loc[overstock_mask, 'reorder_frequency'] = 'Monthly'
        data.loc[overstock_mask, 'review_strategy'] = 'Fixed Interval'
        
        # round safety stock percentages
        data['safety_stock_percent'] = data['safety_stock_percent'].round(0).astype(int)
        
        adjustments_made = stockout_mask.sum() + overstock_mask.sum()
        print(f"Adjusted policies for {adjustments_made:,} products based on risk flags")
        
        return data
    
    def generate_policy_summary(self, policy_data):
        """generate comprehensive policy summary"""
        print("Generating policy summary...")
        
        # summary by lifecycle stage and volatility
        stage_volatility_summary = policy_data.groupby(['lifecycle_stage', 'volatility_level']).agg({
            'product_id': 'count',
            'safety_stock_percent': 'mean',
            'turnover_rate': 'mean'
        }).round(2)
        
        stage_volatility_summary.columns = ['product_count', 'avg_safety_stock', 'avg_turnover_rate']
        
        # summary by policy combination
        policy_combination_summary = policy_data.groupby(['reorder_frequency', 'review_strategy']).agg({
            'product_id': 'count',
            'safety_stock_percent': 'mean',
            'risk_flag': lambda x: (x == 'Stockout Risk').sum()
        }).round(2)
        
        policy_combination_summary.columns = ['product_count', 'avg_safety_stock', 'stockout_risk_count']
        
        print(f"\nPolicy summary by lifecycle stage and volatility:")
        print(stage_volatility_summary)
        
        print(f"\nPolicy combination summary:")
        print(policy_combination_summary)
        
        return {
            'stage_volatility_summary': stage_volatility_summary,
            'policy_combination_summary': policy_combination_summary
        }
    
    def create_visualizations(self, policy_data, policy_summary):
        """create visualizations for policy analysis"""
        print("Creating policy analysis visualizations...")
        
        # 1. bar chart: count of products per policy class
        plt.figure(figsize=(14, 8))
        
        # create policy class by combining frequency and strategy
        policy_data['policy_class'] = policy_data['reorder_frequency'] + ' - ' + policy_data['review_strategy']
        
        policy_counts = policy_data['policy_class'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(policy_counts)))
        
        bars = plt.bar(range(len(policy_counts)), policy_counts.values, color=colors, alpha=0.8)
        plt.title('Product Distribution by Inventory Policy Class', fontsize=16, fontweight='bold')
        plt.xlabel('Policy Class', fontsize=12)
        plt.ylabel('Number of Products', fontsize=12)
        plt.xticks(range(len(policy_counts)), policy_counts.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # add value labels
        for bar, count in zip(bars, policy_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}inventory_policy_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}inventory_policy_summary.png")
        
        # 2. heatmap: lifecycle stage vs volatility level with safety stock
        plt.figure(figsize=(12, 8))
        
        # prepare data for heatmap
        heatmap_data = policy_data.groupby(['lifecycle_stage', 'volatility_level'])['safety_stock_percent'].mean().unstack(fill_value=0)
        
        # create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Average Safety Stock (%)'})
        
        plt.title('Average Safety Stock Percentage by Lifecycle Stage and Volatility', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Volatility Level', fontsize=12)
        plt.ylabel('Lifecycle Stage', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}safety_stock_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}safety_stock_heatmap.png")
        
        # 3. scatter plot: turnover rate vs safety stock with policy colors
        plt.figure(figsize=(12, 8))
        
        # create color mapping for reorder frequency
        freq_colors = {'Weekly': '#FF6B6B', 'Bi-weekly': '#4ECDC4', 'Monthly': '#45B7D1'}
        
        for freq in policy_data['reorder_frequency'].unique():
            subset = policy_data[policy_data['reorder_frequency'] == freq]
            plt.scatter(subset['turnover_rate'], subset['safety_stock_percent'], 
                       c=freq_colors[freq], label=freq, alpha=0.7, s=50)
        
        plt.title('Inventory Policy Analysis: Turnover Rate vs Safety Stock', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Turnover Rate (units/month)', fontsize=12)
        plt.ylabel('Safety Stock Percentage (%)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # add statistics text
        stats_text = f"Total products: {len(policy_data):,}\n"
        stats_text += f"Avg safety stock: {policy_data['safety_stock_percent'].mean():.1f}%\n"
        stats_text += f"Avg turnover rate: {policy_data['turnover_rate'].mean():.2f}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}policy_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}policy_scatter.png")
    
    def save_outputs(self, policy_data, policy_summary, policy_matrix):
         
        print("Saving output files...")
        
        # save policy matrix
        policy_data.to_csv(f"{self.output_path}inventory_policy_matrix.csv", index=False)
        print(f"Saved: {self.output_path}inventory_policy_matrix.csv")
        
        # save policy matrix reference
        self.save_policy_matrix_reference(policy_matrix)
        
        # print detailed summary
        print("\n" + "="*60)
        print("INVENTORY POLICY RECOMMENDATION SUMMARY")
        print("="*60)
        print(f"Total products with policies: {len(policy_data):,}")
        
        # policy distribution
        print(f"\nPolicy distribution:")
        freq_counts = policy_data['reorder_frequency'].value_counts()
        for freq, count in freq_counts.items():
            percentage = (count / len(policy_data)) * 100
            print(f"  {freq}: {count:,} products ({percentage:.1f}%)")
        
        # safety stock distribution
        print(f"\nSafety stock distribution:")
        safety_ranges = [(0, 10), (11, 20), (21, 30), (31, 50)]
        for low, high in safety_ranges:
            count = len(policy_data[(policy_data['safety_stock_percent'] >= low) & 
                                   (policy_data['safety_stock_percent'] <= high)])
            percentage = (count / len(policy_data)) * 100
            print(f"  {low}-{high}%: {count:,} products ({percentage:.1f}%)")
        
        # risk-adjusted policies
        print(f"\nRisk-adjusted policies:")
        risk_adjustments = policy_data.groupby('risk_flag').agg({
            'safety_stock_percent': 'mean',
            'product_id': 'count'
        })
        for risk_type in risk_adjustments.index:
            avg_safety = risk_adjustments.loc[risk_type, 'safety_stock_percent']
            count = risk_adjustments.loc[risk_type, 'product_id']
            print(f"  {risk_type}: {count:,} products, avg safety stock {avg_safety:.1f}%")
        
        print(f"\nOutput files created:")
        print(f"  - {self.output_path}inventory_policy_matrix.csv")
        print(f"  - {self.output_path}policy_matrix_reference.csv")
        print(f"  - {self.output_path}inventory_policy_summary.png")
        print(f"  - {self.output_path}safety_stock_heatmap.png")
        print(f"  - {self.output_path}policy_scatter.png")
    
    def save_policy_matrix_reference(self, policy_matrix):
        """save policy matrix as reference table"""
        reference_data = []
        
        for lifecycle, volatility_policies in policy_matrix.items():
            for volatility, policy in volatility_policies.items():
                reference_data.append({
                    'lifecycle_stage': lifecycle,
                    'volatility_level': volatility,
                    'reorder_frequency': policy['reorder_frequency'],
                    'safety_stock_percent': policy['safety_stock_percent'],
                    'review_strategy': policy['review_strategy']
                })
        
        reference_df = pd.DataFrame(reference_data)
        reference_df.to_csv(f"{self.output_path}policy_matrix_reference.csv", index=False)
        print(f"Saved: {self.output_path}policy_matrix_reference.csv")
    
    def run_recommendation(self):
        """run the complete inventory policy recommendation analysis"""
        print("="*60)
        print("INVENTORY POLICY RECOMMENDATION")
        print("="*60)
        
        lifecycle_labels, risk_flags, inventory_metrics, monthly_sales = self.load_data()
        
        # calculate volatility
        volatility_df = self.calculate_sales_volatility(monthly_sales)
        volatility_df = self.categorize_volatility_levels(volatility_df)
        
        # merge all data
        print("Merging all datasets...")
        merged_data = lifecycle_labels.merge(risk_flags[['product_id', 'risk_flag']], on='product_id', how='inner')
        merged_data = merged_data.merge(inventory_metrics[['product_id', 'turnover_rate']], on='product_id', how='inner')
        merged_data = merged_data.merge(volatility_df[['product_id', 'volatility_level', 'cv']], on='product_id', how='inner')
        
        print(f"Final merged dataset: {len(merged_data):,} products")
        
        # define policy matrix
        policy_matrix = self.define_policy_matrix()
        
        # assign policies
        policy_data = self.assign_policies(merged_data, policy_matrix)
        
        # sum
        policy_summary = self.generate_policy_summary(policy_data)
        
        # visualizations
        self.create_visualizations(policy_data, policy_summary)
        
        self.save_outputs(policy_data, policy_summary, policy_matrix)
        
        print("\n" + "="*60)
        print("POLICY RECOMMENDATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return policy_data, policy_summary

def main():
    """main execution function"""
    recommender = InventoryPolicyRecommender()
    policy_data, policy_summary = recommender.run_recommendation()
    
    return policy_data, policy_summary

if __name__ == "__main__":
    main() 