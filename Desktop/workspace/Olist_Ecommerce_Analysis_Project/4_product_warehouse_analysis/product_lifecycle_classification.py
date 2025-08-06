
"""
Step 2: Product Lifecycle Classification
Execution date: 2025-06-26
Update date: 2025-06-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ProductLifecycleClassifier:
    """product lifecycle classification based on sales performance metrics"""
    
    def __init__(self, data_path="./output/", output_path="./output/"):
        """initialize the classifier with data paths"""
        self.data_path = data_path
        self.output_path = output_path
        self.create_output_directory()
        
    def create_output_directory(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")
    
    def load_data(self):
        """load data from task 1 outputs"""
        print("Loading data from task 1 outputs...")
        
        # load monthly product sales data
        monthly_sales = pd.read_csv(f"{self.data_path}monthly_product_sales.csv")
        print(f"Loaded monthly sales: {len(monthly_sales):,} records")
        
        # load product summary data
        product_summary = pd.read_csv(f"{self.data_path}product_sales_summary.csv")
        print(f"Loaded product summary: {len(product_summary):,} records")
        
        # load product category data for additional context
        products_data = pd.read_csv("../data/processed_missing/olist_products_dataset.csv")
        products_data = products_data[['product_id', 'product_category_name']]
        print(f"Loaded product categories: {len(products_data):,} records")
        
        return monthly_sales, product_summary, products_data
    
    def calculate_initial_growth_rate(self, data):
        """calculate average monthly growth rate in the first 3 months of sales"""
        print("Calculating initial growth rates...")
        
        initial_growth = []
        
        for product_id in data['product_id'].unique():
            product_data = data[data['product_id'] == product_id].sort_values('order_month')
            
            # get first 3 months with sales > 0
            sales_months = product_data[product_data['monthly_sales_volume'] > 0].head(3)
            
            if len(sales_months) >= 2:
                # calculate growth rates between consecutive months
                growth_rates = sales_months['monthly_growth_rate'].dropna()
                if len(growth_rates) > 0:
                    # handle infinite values
                    valid_rates = growth_rates[np.isfinite(growth_rates)]
                    if len(valid_rates) > 0:
                        avg_growth = valid_rates.mean()
                    else:
                        avg_growth = 0
                else:
                    avg_growth = 0
            else:
                avg_growth = 0
            
            initial_growth.append({
                'product_id': product_id,
                'initial_growth_rate': round(avg_growth, 2)
            })
        
        initial_growth_df = pd.DataFrame(initial_growth)
        print(f"Calculated initial growth rates for {len(initial_growth_df):,} products")
        print(f"Average initial growth rate: {initial_growth_df['initial_growth_rate'].mean():.2f}%")
        
        return initial_growth_df
    
    def calculate_saturation_duration(self, data):
        """calculate number of months with less than 5% month-over-month change"""
        print("Calculating saturation duration...")
        
        saturation_data = []
        
        for product_id in data['product_id'].unique():
            product_data = data[data['product_id'] == product_id].sort_values('order_month')
            
            # count months with stable growth (between -5% and +5%)
            stable_months = product_data[
                (product_data['monthly_growth_rate'] >= -5) & 
                (product_data['monthly_growth_rate'] <= 5) &
                (product_data['monthly_growth_rate'].notna())
            ]
            
            saturation_duration = len(stable_months)
            
            saturation_data.append({
                'product_id': product_id,
                'saturation_duration': saturation_duration
            })
        
        saturation_df = pd.DataFrame(saturation_data)
        print(f"Calculated saturation duration for {len(saturation_df):,} products")
        print(f"Average saturation duration: {saturation_df['saturation_duration'].mean():.2f} months")
        
        return saturation_df
    
    def calculate_decline_months(self, data):
        """calculate number of recent months with consistent negative growth"""
        print("Calculating decline months...")
        
        decline_data = []
        
        for product_id in data['product_id'].unique():
            product_data = data[data['product_id'] == product_id].sort_values('order_month')
            
            # find consecutive negative growth months
            growth_rates = product_data['monthly_growth_rate'].dropna()
            
            if len(growth_rates) > 0:
                # count consecutive negative months from the end
                consecutive_declines = 0
                # convert to list to avoid pandas indexing issues
                rates_list = growth_rates.tolist()
                for rate in reversed(rates_list):
                    if rate < 0:
                        consecutive_declines += 1
                    else:
                        break
            else:
                consecutive_declines = 0
            
            decline_data.append({
                'product_id': product_id,
                'decline_months': consecutive_declines
            })
        
        decline_df = pd.DataFrame(decline_data)
        print(f"Calculated decline months for {len(decline_df):,} products")
        print(f"Average decline months: {decline_df['decline_months'].mean():.2f}")
        
        return decline_df
    
    def calculate_sales_peak_position(self, data):
        """calculate position of peak sales (early, mid, or late in timeline)"""
        print("Calculating sales peak positions...")
        
        peak_data = []
        
        for product_id in data['product_id'].unique():
            product_data = data[data['product_id'] == product_id].sort_values('order_month')
            
            if len(product_data) == 0:
                peak_position = 'unknown'
            else:
                # find month with maximum sales
                max_sales_idx = product_data['monthly_sales_volume'].idxmax()
                max_sales_position = product_data.index.get_loc(max_sales_idx)
                total_months = len(product_data)
                
                # determine position
                if max_sales_position < total_months * 0.33:
                    peak_position = 'early'
                elif max_sales_position < total_months * 0.67:
                    peak_position = 'mid'
                else:
                    peak_position = 'late'
            
            peak_data.append({
                'product_id': product_id,
                'sales_peak_position': peak_position
            })
        
        peak_df = pd.DataFrame(peak_data)
        print(f"Calculated peak positions for {len(peak_df):,} products")
        print(f"Peak position distribution:")
        print(peak_df['sales_peak_position'].value_counts())
        
        return peak_df
    
    def calculate_volatility(self, data):
        """calculate standard deviation of monthly growth rates"""
        print("Calculating volatility metrics...")
        
        volatility_data = []
        
        for product_id in data['product_id'].unique():
            product_data = data[data['product_id'] == product_id].sort_values('order_month')
            
            # calculate standard deviation of growth rates
            growth_rates = product_data['monthly_growth_rate'].dropna()
            
            if len(growth_rates) > 1:
                # handle infinite values
                valid_rates = growth_rates[np.isfinite(growth_rates)]
                if len(valid_rates) > 1:
                    volatility = valid_rates.std()
                else:
                    volatility = 0
            else:
                volatility = 0
            
            volatility_data.append({
                'product_id': product_id,
                'volatility': round(volatility, 2)
            })
        
        volatility_df = pd.DataFrame(volatility_data)
        print(f"Calculated volatility for {len(volatility_df):,} products")
        print(f"Average volatility: {volatility_df['volatility'].mean():.2f}")
        
        return volatility_df
    
    def classify_lifecycle_stage(self, features_df):
        """classify products into lifecycle stages based on calculated features"""
        print("Classifying products into lifecycle stages...")
        
        def assign_stage(row):
            """assign lifecycle stage based on business rules"""
            
            # introduction stage: low initial sales, positive growth, short history
            if (row['total_months_active'] <= 6 and 
                row['initial_growth_rate'] > 10 and 
                row['average_monthly_sales'] < 5):
                return 'Introduction'
            
            # growth stage: rapid increase, high volatility, recent peak
            elif (row['initial_growth_rate'] > 20 or 
                  row['volatility'] > 30 or 
                  row['sales_peak_position'] == 'mid'):
                return 'Growth'
            
            # decline stage: sustained negative growth, recent drops
            elif (row['decline_months'] >= 3 or 
                  row['initial_growth_rate'] < -10):
                return 'Decline'
            
            # maturity stage: stable volume, long saturation, peak passed
            elif (row['saturation_duration'] >= 4 or 
                  row['sales_peak_position'] == 'late' or
                  row['volatility'] < 15):
                return 'Maturity'
            
            # default to maturity if unclear
            else:
                return 'Maturity'
        
        # apply classification
        features_df['lifecycle_stage'] = features_df.apply(assign_stage, axis=1)
        
        # print classification results
        stage_counts = features_df['lifecycle_stage'].value_counts()
        print(f"\nLifecycle stage classification results:")
        for stage, count in stage_counts.items():
            percentage = (count / len(features_df)) * 100
            print(f"  {stage}: {count:,} products ({percentage:.1f}%)")
        
        return features_df
    
    def create_visualizations(self, lifecycle_df):
        """create visualizations for lifecycle analysis"""
        print("Creating lifecycle visualizations...")
        
        # 1. bar chart showing count of products per lifecycle stage
        plt.figure(figsize=(10, 6))
        stage_counts = lifecycle_df['lifecycle_stage'].value_counts()
        colors = ['#2E8B57', '#FFD700', '#4169E1', '#DC143C']  # green, gold, blue, red
        
        bars = plt.bar(stage_counts.index, stage_counts.values, color=colors, alpha=0.8)
        plt.title('Product Distribution by Lifecycle Stage', fontsize=16, fontweight='bold')
        plt.xlabel('Lifecycle Stage', fontsize=12)
        plt.ylabel('Number of Products', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # add value labels on bars
        for bar, count in zip(bars, stage_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}lifecycle_stage_counts.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}lifecycle_stage_counts.png")
        
        # 2. boxplot comparing monthly sales across stages
        plt.figure(figsize=(12, 8))
        
        # prepare data for boxplot
        monthly_sales = pd.read_csv(f"{self.data_path}monthly_product_sales.csv")
        sales_with_stage = monthly_sales.merge(
            lifecycle_df[['product_id', 'lifecycle_stage']], 
            on='product_id', 
            how='inner'
        )
        
        # filter out zero sales for better visualization
        sales_with_stage = sales_with_stage[sales_with_stage['monthly_sales_volume'] > 0]
        
        # create boxplot
        stage_order = ['Introduction', 'Growth', 'Maturity', 'Decline']
        sales_with_stage['lifecycle_stage'] = pd.Categorical(
            sales_with_stage['lifecycle_stage'], 
            categories=stage_order, 
            ordered=True
        )
        
        sns.boxplot(data=sales_with_stage, x='lifecycle_stage', y='monthly_sales_volume', 
                   palette=['#2E8B57', '#FFD700', '#4169E1', '#DC143C'])
        
        plt.title('Monthly Sales Volume Distribution by Lifecycle Stage', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Lifecycle Stage', fontsize=12)
        plt.ylabel('Monthly Sales Volume (Units)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # add statistics text
        stats_text = f"Total products: {len(lifecycle_df):,}\n"
        stats_text += f"Total sales records: {len(sales_with_stage):,}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}lifecycle_sales_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}lifecycle_sales_boxplot.png")
    
    def save_outputs(self, lifecycle_df):
         
        print("Saving output files...")
        
        # save lifecycle labels
        lifecycle_df.to_csv(f"{self.output_path}product_lifecycle_labels.csv", index=False)
        print(f"Saved: {self.output_path}product_lifecycle_labels.csv")
        
        # print detailed summary
        print("\n" + "="*60)
        print("LIFECYCLE CLASSIFICATION SUMMARY")
        print("="*60)
        print(f"Total products classified: {len(lifecycle_df):,}")
        
        # stage distribution
        stage_counts = lifecycle_df['lifecycle_stage'].value_counts()
        print(f"\nStage distribution:")
        for stage, count in stage_counts.items():
            percentage = (count / len(lifecycle_df)) * 100
            print(f"  {stage}: {count:,} products ({percentage:.1f}%)")
        
        # average metrics by stage
        print(f"\nAverage metrics by stage:")
        stage_metrics = lifecycle_df.groupby('lifecycle_stage').agg({
            'total_months_active': 'mean',
            'average_monthly_sales': 'mean',
            'volatility': 'mean',
            'saturation_duration': 'mean'
        }).round(2)
        
        for stage in stage_metrics.index:
            print(f"\n  {stage}:")
            print(f"    Avg months active: {stage_metrics.loc[stage, 'total_months_active']:.1f}")
            print(f"    Avg monthly sales: {stage_metrics.loc[stage, 'average_monthly_sales']:.1f}")
            print(f"    Avg volatility: {stage_metrics.loc[stage, 'volatility']:.1f}")
            print(f"    Avg saturation duration: {stage_metrics.loc[stage, 'saturation_duration']:.1f}")
        
        print(f"\nOutput files created:")
        print(f"  - {self.output_path}product_lifecycle_labels.csv")
        print(f"  - {self.output_path}lifecycle_stage_counts.png")
        print(f"  - {self.output_path}lifecycle_sales_boxplot.png")
    
    def run_classification(self):
        """run the complete lifecycle classification analysis"""
        print("="*60)
        print("PRODUCT LIFECYCLE CLASSIFICATION")
        print("="*60)
        
        # load data
        monthly_sales, product_summary, products_data = self.load_data()
        
        # calculate lifecycle features
        initial_growth = self.calculate_initial_growth_rate(monthly_sales)
        saturation_duration = self.calculate_saturation_duration(monthly_sales)
        decline_months = self.calculate_decline_months(monthly_sales)
        peak_position = self.calculate_sales_peak_position(monthly_sales)
        volatility = self.calculate_volatility(monthly_sales)
        
        # merge all features
        print("Merging all features...")
        features_df = product_summary.merge(initial_growth, on='product_id', how='left')
        features_df = features_df.merge(saturation_duration, on='product_id', how='left')
        features_df = features_df.merge(decline_months, on='product_id', how='left')
        features_df = features_df.merge(peak_position, on='product_id', how='left')
        features_df = features_df.merge(volatility, on='product_id', how='left')
        features_df = features_df.merge(products_data, on='product_id', how='left')
        
        # fill missing values
        features_df = features_df.fillna(0)
        print(f"Merged features for {len(features_df):,} products")
        
        # classify lifecycle stages
        lifecycle_df = self.classify_lifecycle_stage(features_df)
        
        # create visualizations
        self.create_visualizations(lifecycle_df)
        
        # save outputs
        self.save_outputs(lifecycle_df)
        
        print("\n" + "="*60)
        print("CLASSIFICATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return lifecycle_df

def main():
    """main execution function"""
    classifier = ProductLifecycleClassifier()
    lifecycle_df = classifier.run_classification()
    
    return lifecycle_df

if __name__ == "__main__":
    main() 