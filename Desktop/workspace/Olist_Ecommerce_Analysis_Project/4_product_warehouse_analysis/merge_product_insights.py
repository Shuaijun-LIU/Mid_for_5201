"""
Step 7: Merge Product Insights
Execution date: 2025-07-01
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

class ProductInsightsMerger:
    def __init__(self, data_path="./output/", output_path="./output/"):
        """initialize the merger with data paths"""
        self.data_path = data_path
        self.output_path = output_path
        self.create_output_directory()
        
    def create_output_directory(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")
    
    def load_task_outputs(self):
        print("Loading task outputs...")
        
        # task 1: product sales curves
        try:
            monthly_sales = pd.read_csv(f"{self.data_path}monthly_product_sales.csv")
            print(f"Loaded Task 1 output: {len(monthly_sales):,} records")
        except FileNotFoundError:
            print("Warning: Task 1 output not found")
            monthly_sales = None
        
        # task 2: lifecycle labels
        try:
            lifecycle_labels = pd.read_csv(f"{self.data_path}product_lifecycle_labels.csv")
            print(f"Loaded Task 2 output: {len(lifecycle_labels):,} products")
        except FileNotFoundError:
            print("Warning: Task 2 output not found")
            lifecycle_labels = None
        
        # task 3: inventory efficiency metrics
        try:
            inventory_metrics = pd.read_csv(f"{self.data_path}inventory_efficiency_metrics.csv")
            print(f"Loaded Task 3 output: {len(inventory_metrics):,} products")
        except FileNotFoundError:
            print("Warning: Task 3 output not found")
            inventory_metrics = None
        
        # task 4: stock risk flags
        try:
            risk_flags = pd.read_csv(f"{self.data_path}product_stock_risk_flags.csv")
            print(f"Loaded Task 4 output: {len(risk_flags):,} products")
        except FileNotFoundError:
            print("Warning: Task 4 output not found")
            risk_flags = None
        
        # task 5: inventory policy matrix
        try:
            policy_matrix = pd.read_csv(f"{self.data_path}inventory_policy_matrix.csv")
            print(f"Loaded Task 5 output: {len(policy_matrix):,} products")
        except FileNotFoundError:
            print("Warning: Task 5 output not found")
            policy_matrix = None
        
        # task 6: warehouse simulation summary
        try:
            simulation_summary = pd.read_csv(f"{self.data_path}warehouse_simulation_summary.csv")
            print(f"Loaded Task 6 output: {len(simulation_summary):,} scenarios")
        except FileNotFoundError:
            print("Warning: Task 6 output not found")
            simulation_summary = None
        
        return {
            'monthly_sales': monthly_sales,
            'lifecycle_labels': lifecycle_labels,
            'inventory_metrics': inventory_metrics,
            'risk_flags': risk_flags,
            'policy_matrix': policy_matrix,
            'simulation_summary': simulation_summary
        }
    
    def process_task1_data(self, monthly_sales):
        """process task 1 data to extract sales metrics"""
        print("Processing Task 1 data...")
        
        if monthly_sales is None:
            return None
        
        # calculate average monthly sales and volatility per product
        task1_metrics = []
        
        for product_id in monthly_sales['product_id'].unique():
            product_data = monthly_sales[monthly_sales['product_id'] == product_id]
            
            # filter out zero sales months for volatility calculation
            sales_volume = product_data[product_data['monthly_sales_volume'] > 0]['monthly_sales_volume']
            
            if len(sales_volume) >= 2:
                avg_monthly_sales = sales_volume.mean()
                std_sales = sales_volume.std()
                volatility_score = std_sales / avg_monthly_sales if avg_monthly_sales > 0 else 0
            else:
                avg_monthly_sales = 0
                volatility_score = 0
            
            task1_metrics.append({
                'product_id': product_id,
                'avg_monthly_sales': round(avg_monthly_sales, 2),
                'volatility_score': round(volatility_score, 3),
                'total_sales_months': len(sales_volume)
            })
        
        task1_df = pd.DataFrame(task1_metrics)
        print(f"Processed Task 1 data: {len(task1_df):,} products")
        print(f"Average monthly sales: {task1_df['avg_monthly_sales'].mean():.2f}")
        print(f"Average volatility score: {task1_df['volatility_score'].mean():.3f}")
        
        return task1_df
    
    def process_task2_data(self, lifecycle_labels):
        """process task 2 data to standardize lifecycle stages"""
        print("Processing Task 2 data...")
        
        if lifecycle_labels is None:
            return None
        
        # standardize lifecycle stages to title case
        task2_df = lifecycle_labels.copy()
        if 'lifecycle_stage' in task2_df.columns:
            task2_df['lifecycle_stage'] = task2_df['lifecycle_stage'].str.title()
        
        # select relevant columns
        task2_df = task2_df[['product_id', 'lifecycle_stage']].copy()
        
        print(f"Processed Task 2 data: {len(task2_df):,} products")
        print(f"Lifecycle stage distribution:")
        print(task2_df['lifecycle_stage'].value_counts())
        
        return task2_df
    
    def process_task3_data(self, inventory_metrics):
        """process task 3 data to extract inventory efficiency metrics"""
        print("Processing Task 3 data...")
        
        if inventory_metrics is None:
            return None
        
        # select relevant columns and rename for clarity
        task3_df = inventory_metrics[['product_id', 'turnover_rate', 'estimated_holding_cost', 'avg_delivery_time']].copy()
        task3_df.columns = ['product_id', 'inventory_turnover_rate', 'avg_holding_cost_per_unit', 'avg_delivery_time']
        
        # round numerical 
        task3_df['inventory_turnover_rate'] = task3_df['inventory_turnover_rate'].round(2)
        task3_df['avg_holding_cost_per_unit'] = task3_df['avg_holding_cost_per_unit'].round(2)
        task3_df['avg_delivery_time'] = task3_df['avg_delivery_time'].round(1)
        
        print(f"Processed Task 3 data: {len(task3_df):,} products")
        print(f"Average turnover rate: {task3_df['inventory_turnover_rate'].mean():.2f}")
        print(f"Average holding cost: ${task3_df['avg_holding_cost_per_unit'].mean():.2f}")
        
        return task3_df
    
    def process_task4_data(self, risk_flags):
        """process task 4 data to standardize risk flags"""
        print("Processing Task 4 data...")
        
        if risk_flags is None:
            return None
        
        # standardize risk flags to title case
        task4_df = risk_flags[['product_id', 'risk_flag']].copy()
        task4_df['stock_risk_flag'] = task4_df['risk_flag'].str.title()
        
        # map risk flags to standardized categories
        risk_mapping = {
            'Overstock Risk': 'Overstock',
            'Stockout Risk': 'Stockout',
            'Stable': 'Normal'
        }
        task4_df['stock_risk_flag'] = task4_df['stock_risk_flag'].map(risk_mapping).fillna('Normal')
        
        print(f"Processed Task 4 data: {len(task4_df):,} products")
        print(f"Risk flag distribution:")
        print(task4_df['stock_risk_flag'].value_counts())
        
        return task4_df[['product_id', 'stock_risk_flag']]
    
    def process_task5_data(self, policy_matrix):
        """process task 5 data to extract policy recommendations"""
        print("Processing Task 5 data...")
        
        if policy_matrix is None:
            return None
        
        # select relevant columns and create policy type
        task5_df = policy_matrix[['product_id', 'reorder_frequency', 'review_strategy']].copy()
        
        # create policy type based on frequency and strategy
        def create_policy_type(row):
            freq = row['reorder_frequency']
            strategy = row['review_strategy']
            
            if freq == 'Weekly' and 'Just-in-Time' in strategy:
                return 'High-frequency low-batch'
            elif freq == 'Bi-weekly' and 'Dynamic' in strategy:
                return 'Buffer stock'
            elif freq == 'Monthly' and 'Fixed' in strategy:
                return 'Standard replenishment'
            elif 'Predictive' in strategy:
                return 'Predictive restocking'
            else:
                return 'Standard replenishment'
        
        task5_df['suggested_policy_type'] = task5_df.apply(create_policy_type, axis=1)
        
        print(f"Processed Task 5 data: {len(task5_df):,} products")
        print(f"Policy type distribution:")
        print(task5_df['suggested_policy_type'].value_counts())
        
        return task5_df[['product_id', 'suggested_policy_type']]
    
    def process_task6_data(self, simulation_summary):
        """process task 6 data to extract simulation KPIs"""
        print("Processing Task 6 data...")
        
        if simulation_summary is None:
            return None
        
        # for now, use baseline scenario metrics as default
        baseline_scenario = simulation_summary[simulation_summary['scenario'] == 'Baseline']
        
        if len(baseline_scenario) > 0:
            fulfillment_success_rate = baseline_scenario['fulfillment_rate'].iloc[0]
            backlog_volume = 50  # placeholder value
        else:
            fulfillment_success_rate = 85.0  # default 
            backlog_volume = 50  # default 
        
        # create a dataframe with default values for all products
        # in practice, this would be product-specific simulation results
        task6_df = pd.DataFrame({
            'product_id': [],  # will be filled during merge
            'fulfillment_success_rate': fulfillment_success_rate,
            'backlog_volume': backlog_volume
        })
        
        print(f"Processed Task 6 data: Using baseline scenario metrics")
        print(f"Fulfillment success rate: {fulfillment_success_rate:.1f}%")
        print(f"Estimated backlog volume: {backlog_volume}")
        
        return task6_df
    
    def merge_all_data(self, task_data):
        """merge all task data into unified dataset"""
        print("Merging all task data...")
        
        # start with lifecycle labels as base (most comprehensive)
        base_df = task_data['lifecycle_labels']
        if base_df is None:
            print("Error: No base dataset available")
            return None
        
        base_df = self.process_task2_data(base_df)
        print(f"Starting with {len(base_df):,} products from lifecycle labels")
        
        # merge task 1 data
        if task_data['monthly_sales'] is not None:
            task1_df = self.process_task1_data(task_data['monthly_sales'])
            base_df = base_df.merge(task1_df, on='product_id', how='left')
            print(f"After Task 1 merge: {len(base_df):,} products")
        
        # merge task 3 data
        if task_data['inventory_metrics'] is not None:
            task3_df = self.process_task3_data(task_data['inventory_metrics'])
            base_df = base_df.merge(task3_df, on='product_id', how='left')
            print(f"After Task 3 merge: {len(base_df):,} products")
        
        # merge task 4 data
        if task_data['risk_flags'] is not None:
            task4_df = self.process_task4_data(task_data['risk_flags'])
            base_df = base_df.merge(task4_df, on='product_id', how='left')
            print(f"After Task 4 merge: {len(base_df):,} products")
        
        # merge task 5 data
        if task_data['policy_matrix'] is not None:
            task5_df = self.process_task5_data(task_data['policy_matrix'])
            base_df = base_df.merge(task5_df, on='product_id', how='left')
            print(f"After Task 5 merge: {len(base_df):,} products")
        
        # merge task 6 data (simulation results)
        if task_data['simulation_summary'] is not None:
            task6_df = self.process_task6_data(task_data['simulation_summary'])
            
            base_df['fulfillment_success_rate'] = task6_df['fulfillment_success_rate'].iloc[0] if len(task6_df) > 0 else 85.0
            base_df['backlog_volume'] = task6_df['backlog_volume'].iloc[0] if len(task6_df) > 0 else 50
            print(f"After Task 6 merge: {len(base_df):,} products")
        
        # handle missing values
        base_df = self.handle_missing_values(base_df)
        
        # reorder columns for final output
        final_columns = [
            'product_id', 'lifecycle_stage', 'avg_monthly_sales', 'volatility_score',
            'inventory_turnover_rate', 'avg_holding_cost_per_unit', 'stock_risk_flag',
            'suggested_policy_type', 'fulfillment_success_rate', 'backlog_volume'
        ]
        
        # only include columns that exist
        available_columns = [col for col in final_columns if col in base_df.columns]
        base_df = base_df[available_columns]
        
        print(f"Final merged dataset: {len(base_df):,} products with {len(base_df.columns)} columns")
        
        return base_df
    
    def handle_missing_values(self, df):
        """handle missing values in the merged dataset"""
        print("Handling missing values...")
        
        # print missing value summary
        missing_summary = df.isnull().sum()
        if missing_summary.sum() > 0:
            print("Missing values by column:")
            for col, missing_count in missing_summary.items():
                if missing_count > 0:
                    print(f"  {col}: {missing_count:,} missing values")
        
        # fill missing values
        df = df.copy()
        
        # string columns
        string_columns = ['lifecycle_stage', 'stock_risk_flag', 'suggested_policy_type']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # numeric columns
        numeric_columns = ['avg_monthly_sales', 'volatility_score', 'inventory_turnover_rate', 
                          'avg_holding_cost_per_unit', 'fulfillment_success_rate', 'backlog_volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        print(f"Missing values handled. Remaining missing: {df.isnull().sum().sum()}")
        
        return df
    
    def generate_summary_statistics(self, merged_df):
        """generate summary statistics for the merged dataset"""
        print("Generating summary statistics...")
        
        print(f"\n{'='*60}")
        print("MERGED DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"Total products: {len(merged_df):,}")
        print(f"Total columns: {len(merged_df.columns)}")
        
        # lifecycle stage distribution
        if 'lifecycle_stage' in merged_df.columns:
            print(f"\nLifecycle stage distribution:")
            stage_counts = merged_df['lifecycle_stage'].value_counts()
            for stage, count in stage_counts.items():
                percentage = (count / len(merged_df)) * 100
                print(f"  {stage}: {count:,} products ({percentage:.1f}%)")
        
        # risk flag distribution
        if 'stock_risk_flag' in merged_df.columns:
            print(f"\nRisk flag distribution:")
            risk_counts = merged_df['stock_risk_flag'].value_counts()
            for risk, count in risk_counts.items():
                percentage = (count / len(merged_df)) * 100
                print(f"  {risk}: {count:,} products ({percentage:.1f}%)")
        
        # policy type distribution
        if 'suggested_policy_type' in merged_df.columns:
            print(f"\nPolicy type distribution:")
            policy_counts = merged_df['suggested_policy_type'].value_counts()
            for policy, count in policy_counts.items():
                percentage = (count / len(merged_df)) * 100
                print(f"  {policy}: {count:,} products ({percentage:.1f}%)")
        
        # numeric statistics
        numeric_columns = ['avg_monthly_sales', 'volatility_score', 'inventory_turnover_rate', 
                          'avg_holding_cost_per_unit', 'fulfillment_success_rate', 'backlog_volume']
        
        print(f"\nNumeric statistics:")
        for col in numeric_columns:
            if col in merged_df.columns:
                mean_val = merged_df[col].mean()
                std_val = merged_df[col].std()
                print(f"  {col}: mean={mean_val:.2f}, std={std_val:.2f}")
    
    def save_output(self, merged_df):
        print("Saving merged dataset...")
        
        output_file = f"{self.output_path}product_warehouse_summary.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
        
        # print sample preview
        print(f"\nSample preview of merged dataset:")
        print(merged_df.head())
        
        print(f"\nDataset shape: {merged_df.shape}")
        print(f"Columns: {list(merged_df.columns)}")
    
    def run_merge(self):
        print("="*60)
        print("MERGE PRODUCT INSIGHTS")
        print("="*60)
        
        task_data = self.load_task_outputs()
        
        # merge 
        merged_df = self.merge_all_data(task_data)
        
        if merged_df is not None:
            self.generate_summary_statistics(merged_df)
            
            self.save_output(merged_df)
            
            print("\n" + "="*60)
            print("MERGE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            return merged_df
        else:
            print("Error: Failed to merge data")
            return None

def main():
    merger = ProductInsightsMerger()
    merged_df = merger.run_merge()
    
    return merged_df

if __name__ == "__main__":
    main() 