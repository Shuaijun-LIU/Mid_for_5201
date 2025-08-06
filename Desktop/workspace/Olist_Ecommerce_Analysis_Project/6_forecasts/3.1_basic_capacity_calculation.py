"""
Task 3.1: Basic Capacity Calculation
Calculate warehouse capacity, labor requirements, and equipment needs
Based on demand forecasts and historical data
Execution date: 2025-07-17
Update date: 2025-07-18
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# Import shared components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.shared_components import DataLoader, FeatureProcessor, ModelEvaluator, OutputManager, BaseForecaster
from config.model_config import OUTPUT_CONFIG

# Suppress warnings
warnings.filterwarnings('ignore')

class BasicCapacityCalculator(BaseForecaster):
    """Basic capacity calculation for warehouse, labor, and equipment"""
    
    def __init__(self, output_dir: str = 'output'):
        super().__init__(output_dir)
        self.output_config = OUTPUT_CONFIG
        # Capacity calculation parameters
        self.config = {
            'safety_margin': 0.15,
            'peak_multiplier': 1.3,
            'target_utilization': 0.85,
            'warehouse_rent_per_sqm': 15.5,
            'labor_cost_per_hour': 25.0,
            'equipment_maintenance_ratio': 0.05
        }
        
        # Data storage
        self.capacity_results = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Task 2, Week4, and Week3 outputs"""
        print("Loading data from Task 2, Week4, and Week3 outputs...")
        
        data = {}
        
        # Load Task 2 outputs
        task2_files = [
            'ensemble_forecasts.csv',
            'model_performance_metrics.csv',
            'forecast_confidence_intervals.csv'
        ]
        
        for file in task2_files:
            file_path = os.path.join(self.output_dir, file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        # Load Week4 outputs
        week4_files = [
            'inventory_efficiency_metrics.csv',
            'inventory_policy_matrix.csv',
            'product_warehouse_summary.csv',
            'warehouse_simulation_summary.csv'
        ]
        
        for file in week4_files:
            file_path = os.path.join('../week4_product_warehouse_analysis/output', file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        # Load Week3 outputs
        week3_files = [
            'customer_lifecycle.csv',
            'customer_logistics_features.csv',
            'final_customer_segments.csv'
        ]
        
        for file in week3_files:
            file_path = os.path.join('../week3_customer_behavior/output', file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        # Load original data
        original_files = [
            'olist_order_items_dataset.csv',
            'olist_products_dataset.csv'
        ]
        
        for file in original_files:
            file_path = os.path.join('../data/processed_missing', file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        return data
    
    def prepare_features(self) -> Dict[str, pd.DataFrame]:
        """Prepare data for capacity calculations"""
        print("Preparing capacity calculation data...")
        
        capacity_data = {}
        
        # Process demand forecasts
        if 'ensemble_forecasts' in self.data:
            demand_df = self.data['ensemble_forecasts'].copy()
            capacity_data['demand_forecasts'] = demand_df
            print(f"Processed demand forecasts: {demand_df.shape}")
        elif 'demand_forecasts' in self.data:
            demand_df = self.data['demand_forecasts'].copy()
            capacity_data['demand_forecasts'] = demand_df
            print(f"Processed demand forecasts: {demand_df.shape}")
        
        # Process inventory efficiency data
        if 'inventory_efficiency_metrics' in self.data:
            inventory_df = self.data['inventory_efficiency_metrics'].copy()
            capacity_data['inventory_efficiency'] = inventory_df
            print(f"Processed inventory efficiency: {inventory_df.shape}")
        
        # Process customer behavior data
        if 'final_customer_segments' in self.data:
            customer_df = self.data['final_customer_segments'].copy()
            capacity_data['customer_segments'] = customer_df
            print(f"Processed customer segments: {customer_df.shape}")
        
        # Process product data
        if 'olist_products_dataset' in self.data:
            product_df = self.data['olist_products_dataset'].copy()
            capacity_data['products'] = product_df
            print(f"Processed products: {product_df.shape}")
        
        return capacity_data
    
    def calculate_warehouse_capacity(self, demand_data: pd.DataFrame, product_data: pd.DataFrame) -> Dict:
        """Calculate warehouse capacity requirements"""
        print("Calculating warehouse capacity requirements...")
        warehouse_capacity = {}
        # Compatible with global forecast
        if 'ensemble_value' in demand_data.columns:
            # Use order_count instead of total_sales for capacity calculation
            if demand_data.columns.str.contains('order_count').any():
                # If we have order_count data, use it
                total_orders = demand_data['ensemble_value'].sum()
            else:
                # Fallback to total_sales but convert to reasonable order count
                total_sales = demand_data['ensemble_value'].sum()
                avg_order_value = 100  # Assume average order value of $100
                total_orders = total_sales / avg_order_value
            
            storage_sqm = total_orders * 0.1  # Assume 0.1 sqm per order
            processing_sqm = storage_sqm * 0.3
            auxiliary_sqm = storage_sqm * 0.15
            warehouse_capacity = {
                'total_needed_sqm': storage_sqm + processing_sqm + auxiliary_sqm,
                'storage_sqm': storage_sqm,
                'processing_sqm': processing_sqm,
                'auxiliary_sqm': auxiliary_sqm,
                'peak_multiplier': self.config['peak_multiplier'],
                'safety_margin': self.config['safety_margin']
            }
            print(f"Calculated warehouse capacity for global forecast: {total_orders:.2f} orders")
        return warehouse_capacity

    def calculate_labor_requirements(self, demand_data: pd.DataFrame, customer_data: pd.DataFrame) -> Dict:
        """Calculate labor requirements"""
        print("Calculating labor requirements...")
        labor_requirements = {}
        if 'ensemble_value' in demand_data.columns:
            # Use order_count instead of total_sales for labor calculation
            if demand_data.columns.str.contains('order_count').any():
                # If we have order_count data, use it
                total_orders = demand_data['ensemble_value'].sum()
            else:
                # Fallback to total_sales but convert to reasonable order count
                total_sales = demand_data['ensemble_value'].sum()
                avg_order_value = 100  # Assume average order value of $100
                total_orders = total_sales / avg_order_value
            
            base_hours_per_order = 0.5  # 30 minutes per order
            total_hours = total_orders * base_hours_per_order
            fte = total_hours / 160  # 160 hours per month
            peak_hours_multiplier = self.config['peak_multiplier']
            peak_hours = total_hours * peak_hours_multiplier
            overtime_hours = (peak_hours - total_hours) * 0.7
            temporary_hours = (peak_hours - total_hours) * 0.3
            labor_requirements = {
                'total_hours_monthly': total_hours,
                'full_time_equivalents': fte,
                'peak_hours_multiplier': peak_hours_multiplier,
                'overtime_hours': overtime_hours,
                'temporary_hours': temporary_hours,
                'skill_mix': {
                    'pickers': 0.6,
                    'packers': 0.3,
                    'supervisors': 0.1
                },
                'training_hours': fte * 5
            }
            print(f"Calculated labor requirements for global forecast: {total_orders:.2f} orders")
        return labor_requirements

    def calculate_equipment_needs(self, demand_data: pd.DataFrame, inventory_data: pd.DataFrame) -> Dict:
        """Calculate equipment requirements"""
        print("Calculating equipment requirements...")
        equipment_needs = {}
        if 'ensemble_value' in demand_data.columns:
            total_orders = demand_data['ensemble_value'].sum()
            forklifts = int(np.ceil(total_orders / 1000))
            conveyors = int(np.ceil(total_orders / 2000))
            packing_stations = int(np.ceil(total_orders / 500))
            storage_racks = int(np.ceil(total_orders / 100))
            computers = int(np.ceil(total_orders / 300))
            maintenance_hours = (forklifts + conveyors + packing_stations) * 2
            equipment_needs = {
                'forklifts': forklifts,
                'conveyors': conveyors,
                'packing_stations': packing_stations,
                'storage_racks': storage_racks,
                'computers': computers,
                'maintenance_hours': maintenance_hours
            }
            print(f"Calculated equipment needs for global forecast: {total_orders} orders")
        return equipment_needs
    
    def train_models(self) -> Dict:
        """Calculate basic capacity requirements"""
        print("Calculating basic capacity requirements...")
        
        capacity_results = {}
        
        if 'demand_forecasts' in self.features:
            demand_df = self.features['demand_forecasts']
            # Try to find the prediction column
            prediction_col = None
            for col in ['ensemble_value', 'predicted_value', 'prediction', 'yhat', 'forecast', 'total_orders', 'total_sales']:
                if col in demand_df.columns:
                    prediction_col = col
                    break
            if prediction_col is None:
                print("No prediction column found in demand_df!")
                return {}
            # If both seller_id and timeframe exist, do group calculation
            if 'seller_id' in demand_df.columns and 'timeframe' in demand_df.columns:
                for (seller_id, timeframe), group in demand_df.groupby(['seller_id', 'timeframe']):
                    print(f"Processing basic capacity for seller {seller_id}, timeframe {timeframe}")
                    base_capacity = group[prediction_col].mean()
                    warehouse_capacity = self.calculate_warehouse_capacity(
                        group, self.features.get('products', pd.DataFrame())
                    )
                    labor_requirements = self.calculate_labor_requirements(
                        group, self.features.get('customer_segments', pd.DataFrame())
                    )
                    equipment_needs = self.calculate_equipment_needs(
                        group, self.features.get('inventory_efficiency', pd.DataFrame())
                    )
                    capacity_results[f"{seller_id}_{timeframe}"] = {
                        'seller_id': seller_id,
                        'timeframe': timeframe,
                        'base_capacity': base_capacity,
                        'warehouse_capacity': warehouse_capacity,
                        'labor_requirements': labor_requirements,
                        'equipment_needs': equipment_needs
                    }
            else:
                # Global calculation if no seller_id/timeframe columns
                print("Processing basic capacity for overall forecast")
                group = demand_df
                base_capacity = group[prediction_col].mean()
                warehouse_capacity = self.calculate_warehouse_capacity(
                    group, self.features.get('products', pd.DataFrame())
                )
                labor_requirements = self.calculate_labor_requirements(
                    group, self.features.get('customer_segments', pd.DataFrame())
                )
                equipment_needs = self.calculate_equipment_needs(
                    group, self.features.get('inventory_efficiency', pd.DataFrame())
                )
                capacity_results['overall'] = {
                    'seller_id': 'overall',
                    'timeframe': group['timeframe'].iloc[0] if 'timeframe' in group.columns else 'overall',
                    'base_capacity': base_capacity,
                    'warehouse_capacity': warehouse_capacity,
                    'labor_requirements': labor_requirements,
                    'equipment_needs': equipment_needs
                }
        
        self.capacity_results = capacity_results
        return capacity_results
    
    def generate_forecasts(self) -> Dict:
        """Generate basic capacity forecasts"""
        print("Generating basic capacity forecasts...")
        
        forecasts = {}
        
        for key, capacity_data in self.capacity_results.items():
            forecasts[key] = {
                'seller_id': capacity_data['seller_id'],
                'timeframe': capacity_data['timeframe'],
                'base_capacity': capacity_data['base_capacity'],
                'warehouse_sqm': capacity_data['warehouse_capacity'].get('total_needed_sqm', 0),
                'labor_hours': capacity_data['labor_requirements'].get('total_hours_monthly', 0),
                'fte_count': capacity_data['labor_requirements'].get('full_time_equivalents', 0),
                'equipment_count': sum([
                    capacity_data['equipment_needs'].get('forklifts', 0),
                    capacity_data['equipment_needs'].get('conveyors', 0),
                    capacity_data['equipment_needs'].get('packing_stations', 0)
                ])
            }
        
        self.forecast_results = forecasts
        return forecasts
    
    def evaluate_models(self) -> Dict:
        """Evaluate basic capacity calculations"""
        print("Evaluating basic capacity calculations...")
        
        evaluation_results = {}
        
        for key, capacity_data in self.capacity_results.items():
            # Calculate basic efficiency metrics
            base_capacity = capacity_data['base_capacity']
            warehouse_sqm = capacity_data['warehouse_capacity'].get('total_needed_sqm', 0)
            labor_hours = capacity_data['labor_requirements'].get('total_hours_monthly', 0)
            
            # Calculate utilization metrics
            warehouse_utilization = min(1.0, base_capacity / (warehouse_sqm * 10)) if warehouse_sqm > 0 else 0
            labor_utilization = min(1.0, base_capacity / (labor_hours * 2)) if labor_hours > 0 else 0
            
            evaluation_results[key] = {
                'warehouse_utilization': warehouse_utilization,
                'labor_utilization': labor_utilization,
                'overall_efficiency': (warehouse_utilization + labor_utilization) / 2
            }
        
        return evaluation_results
    
    def save_results(self) -> None:
        """Save basic capacity calculation results"""
        print("Saving basic capacity calculation results...")
        
        # Save capacity forecasts
        if self.capacity_results:
            capacity_data = []
            for key, capacity_data_dict in self.capacity_results.items():
                capacity_data.append({
                    'seller_id': capacity_data_dict['seller_id'],
                    'timeframe': capacity_data_dict['timeframe'],
                    'base_capacity': capacity_data_dict['base_capacity'],
                    'warehouse_sqm': capacity_data_dict['warehouse_capacity'].get('total_needed_sqm', 0),
                    'storage_sqm': capacity_data_dict['warehouse_capacity'].get('storage_sqm', 0),
                    'processing_sqm': capacity_data_dict['warehouse_capacity'].get('processing_sqm', 0),
                    'auxiliary_sqm': capacity_data_dict['warehouse_capacity'].get('auxiliary_sqm', 0),
                    'labor_hours': capacity_data_dict['labor_requirements'].get('total_hours_monthly', 0),
                    'fte_count': capacity_data_dict['labor_requirements'].get('full_time_equivalents', 0),
                    'overtime_hours': capacity_data_dict['labor_requirements'].get('overtime_hours', 0),
                    'temporary_hours': capacity_data_dict['labor_requirements'].get('temporary_hours', 0),
                    'forklifts': capacity_data_dict['equipment_needs'].get('forklifts', 0),
                    'conveyors': capacity_data_dict['equipment_needs'].get('conveyors', 0),
                    'packing_stations': capacity_data_dict['equipment_needs'].get('packing_stations', 0),
                    'storage_racks': capacity_data_dict['equipment_needs'].get('storage_racks', 0),
                    'computers': capacity_data_dict['equipment_needs'].get('computers', 0),
                    'maintenance_hours': capacity_data_dict['equipment_needs'].get('maintenance_hours', 0)
                })
            
            if capacity_data:
                capacity_df = pd.DataFrame(capacity_data)
                self.output_manager.save_dataframe(capacity_df, 'basic_capacity_calculations.csv')
        
        # Create visualizations
        if self.output_config['save_visualizations']:
            self.create_visualizations()
    
    def create_visualizations(self):
        """Create basic capacity visualizations"""
        print("Creating basic capacity visualizations...")
        
        if not self.capacity_results:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create capacity comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Basic Capacity Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for visualization
        sellers = []
        base_capacities = []
        warehouse_sqm = []
        labor_hours = []
        equipment_counts = []
        
        for key, capacity_data in self.capacity_results.items():
            sellers.append(capacity_data['seller_id'])
            base_capacities.append(capacity_data['base_capacity'])
            warehouse_sqm.append(capacity_data['warehouse_capacity'].get('total_needed_sqm', 0))
            labor_hours.append(capacity_data['labor_requirements'].get('total_hours_monthly', 0))
            equipment_counts.append(sum([
                capacity_data['equipment_needs'].get('forklifts', 0),
                capacity_data['equipment_needs'].get('conveyors', 0),
                capacity_data['equipment_needs'].get('packing_stations', 0)
            ]))
        
        # Plot 1: Base capacity by seller
        axes[0, 0].bar(sellers, base_capacities, alpha=0.8, color='blue')
        axes[0, 0].set_title('Base Capacity by Seller')
        axes[0, 0].set_xlabel('Seller ID')
        axes[0, 0].set_ylabel('Base Capacity')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Warehouse space requirements
        axes[0, 1].bar(sellers, warehouse_sqm, alpha=0.8, color='green')
        axes[0, 1].set_title('Warehouse Space Requirements')
        axes[0, 1].set_xlabel('Seller ID')
        axes[0, 1].set_ylabel('Warehouse Space (sqm)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Labor hours requirements
        axes[1, 0].bar(sellers, labor_hours, alpha=0.8, color='orange')
        axes[1, 0].set_title('Labor Hours Requirements')
        axes[1, 0].set_xlabel('Seller ID')
        axes[1, 0].set_ylabel('Labor Hours (monthly)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Equipment counts
        axes[1, 1].bar(sellers, equipment_counts, alpha=0.8, color='red')
        axes[1, 1].set_title('Equipment Requirements')
        axes[1, 1].set_xlabel('Seller ID')
        axes[1, 1].set_ylabel('Equipment Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.output_manager.save_visualization(fig, 'basic_capacity_analysis.png')

def main():
    """Main execution function"""
    print("Starting Basic Capacity Calculation Analysis")
    
    # Initialize calculator
    calculator = BasicCapacityCalculator()
    
    try:
        # Run complete pipeline
        results = calculator.run_complete_pipeline()
        
        print("\nBasic Capacity Calculation Analysis completed successfully")
        print(f"Capacity calculations completed for {len(calculator.capacity_results)} seller-timeframe combinations")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 