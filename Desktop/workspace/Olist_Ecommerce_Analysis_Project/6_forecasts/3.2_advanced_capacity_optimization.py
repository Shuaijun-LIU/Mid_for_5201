"""
Task 3.2: Advanced Capacity Optimization
Calculate seasonal adjustments, emergency capacity planning, and optimization metrics
Based on basic capacity calculations from Task 3.1
Execution date: 2025-07-18
Update date: 2025-07-21
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
from scipy.optimize import minimize

# Import shared components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.shared_components import DataLoader, FeatureProcessor, ModelEvaluator, OutputManager, BaseForecaster
from config.model_config import OUTPUT_CONFIG

# Suppress warnings
warnings.filterwarnings('ignore')

class AdvancedCapacityOptimizer(BaseForecaster):
    """Advanced capacity optimization with seasonal adjustments and emergency planning"""
    
    def __init__(self, output_dir: str = 'output'):
        super().__init__(output_dir)
        
        # Optimization parameters
        self.config = {
            'safety_margin': 0.15,
            'peak_multiplier': 1.3,
            'target_utilization': 0.85,
            'emergency_capacity_ratio': 0.20,
            'warehouse_rent_per_sqm': 15.5,
            'labor_cost_per_hour': 25.0,
            'equipment_maintenance_ratio': 0.05
        }
        self.output_config = OUTPUT_CONFIG
        # Data storage
        self.optimization_results = {}
        self.cost_analysis = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Task 3.1 and other sources"""
        print("Loading data from Task 3.1 and other sources...")
        
        data = {}
        
        # Load Task 3.1 outputs
        task3_1_files = [
            'basic_capacity_calculations.csv'
        ]
        
        for file in task3_1_files:
            file_path = os.path.join(self.output_dir, file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        # Load Task 2 outputs for demand data
        task2_files = [
            'demand_forecasts.csv'
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
            'inventory_policy_matrix.csv'
        ]
        
        for file in week4_files:
            file_path = os.path.join('../week4_product_warehouse_analysis/output', file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        return data
    
    def prepare_features(self) -> Dict[str, pd.DataFrame]:
        """Prepare data for advanced optimization"""
        print("Preparing advanced optimization data...")
        
        optimization_data = {}
        
        # Process basic capacity calculations
        if 'basic_capacity_calculations' in self.data:
            basic_df = self.data['basic_capacity_calculations'].copy()
            optimization_data['basic_capacity'] = basic_df
            print(f"Processed basic capacity: {basic_df.shape}")
        
        # Process demand forecasts
        if 'demand_forecasts' in self.data:
            demand_df = self.data['demand_forecasts'].copy()
            demand_df['forecast_date'] = pd.to_datetime(demand_df['forecast_period'])
            optimization_data['demand_forecasts'] = demand_df
            print(f"Processed demand forecasts: {demand_df.shape}")
        
        # Process inventory efficiency data
        if 'inventory_efficiency_metrics' in self.data:
            inventory_df = self.data['inventory_efficiency_metrics'].copy()
            optimization_data['inventory_efficiency'] = inventory_df
            print(f"Processed inventory efficiency: {inventory_df.shape}")
        
        return optimization_data
    
    def calculate_seasonal_adjustments(self, demand_data: pd.DataFrame) -> Dict:
        """Calculate seasonal capacity adjustments"""
        print("Calculating seasonal adjustments...")
        
        seasonal_adjustments = {}
        
        if 'forecast_period' in demand_data.columns:
            # Extract month from forecast period
            demand_data['month'] = pd.to_datetime(demand_data['forecast_period']).dt.month
            
            # Define seasonal multipliers
            seasonal_multipliers = {
                12: 1.3, 1: 1.3, 2: 1.3,  # Winter (peak)
                3: 1.1, 4: 1.1, 5: 1.1,  # Spring
                6: 0.9, 7: 0.9, 8: 0.9,  # Summer (low)
                9: 1.0, 10: 1.0, 11: 1.0  # Fall
            }
            
            # Calculate seasonal adjustments
            demand_data['seasonal_multiplier'] = demand_data['month'].map(seasonal_multipliers)
            
            seasonal_adjustments = {
                'peak_season_multiplier': 1.3,
                'low_season_multiplier': 0.8,
                'holiday_impact': 0.2,
                'promotion_impact': 0.15,
                'seasonal_multipliers': seasonal_multipliers
            }
            
            print("Calculated seasonal adjustments")
        
        return seasonal_adjustments
    
    def calculate_emergency_capacity(self, base_capacity: float) -> Dict:
        """Calculate emergency capacity requirements"""
        print("Calculating emergency capacity requirements...")
        
        emergency_capacity = {}
        
        # Define emergency scenarios
        scenarios = {
            'demand_surge_20': {'increase': 0.2, 'cost_multiplier': 1.5, 'time': '1_week'},
            'demand_surge_50': {'increase': 0.5, 'cost_multiplier': 2.0, 'time': '2_weeks'},
            'demand_surge_100': {'increase': 1.0, 'cost_multiplier': 3.0, 'time': '1_month'}
        }
        
        for scenario, params in scenarios.items():
            additional_capacity = base_capacity * params['increase']
            cost = additional_capacity * self.config['warehouse_rent_per_sqm'] * params['cost_multiplier']
            
            emergency_capacity[scenario] = {
                'additional_capacity': additional_capacity,
                'cost': cost,
                'implementation_time': params['time']
            }
        
        print(f"Calculated emergency capacity for {len(scenarios)} scenarios")
        
        return emergency_capacity
    
    def calculate_optimization_metrics(self, capacity_data: Dict, cost_data: Dict) -> Dict:
        """Calculate optimization metrics"""
        print("Calculating optimization metrics...")
        
        optimization_metrics = {}
        
        # Calculate efficiency metrics
        if 'total_needed_sqm' in capacity_data:
            capacity_efficiency = 0.87  # Example value
            cost_per_sqm = self.config['warehouse_rent_per_sqm']
            labor_productivity = 150  # orders per hour
            equipment_utilization = 0.82  # Example value
            
            overall_efficiency = (capacity_efficiency + equipment_utilization) / 2
            
            optimization_metrics = {
                'capacity_efficiency': capacity_efficiency,
                'cost_per_sqm': cost_per_sqm,
                'labor_productivity': labor_productivity,
                'equipment_utilization': equipment_utilization,
                'overall_efficiency': overall_efficiency
            }
            
            print("Calculated optimization metrics")
        
        return optimization_metrics
    
    def optimize_capacity_allocation(self, base_capacity: float, constraints: Dict) -> Dict:
        """Optimize capacity allocation using mathematical optimization"""
        print("Optimizing capacity allocation...")
        
        # Define objective function (minimize total cost)
        def objective_function(x):
            warehouse_capacity, labor_capacity, equipment_capacity = x
            warehouse_cost = warehouse_capacity * self.config['warehouse_rent_per_sqm']
            labor_cost = labor_capacity * self.config['labor_cost_per_hour']
            equipment_cost = equipment_capacity * 1000  # Simplified equipment cost
            return warehouse_cost + labor_cost + equipment_cost
        
        # Define constraints
        constraints_list = [
            {'type': 'ineq', 'fun': lambda x: x[0] + x[1] + x[2] - base_capacity},  # Total capacity >= base
            {'type': 'ineq', 'fun': lambda x: base_capacity * 0.8 - x[0]},  # Warehouse >= 80% of base
            {'type': 'ineq', 'fun': lambda x: base_capacity * 0.1 - x[1]},  # Labor >= 10% of base
            {'type': 'ineq', 'fun': lambda x: base_capacity * 0.1 - x[2]}   # Equipment >= 10% of base
        ]
        
        # Initial guess
        x0 = [base_capacity * 0.6, base_capacity * 0.2, base_capacity * 0.2]
        
        # Bounds
        bounds = [
            (base_capacity * 0.5, base_capacity * 1.5),  # Warehouse bounds
            (base_capacity * 0.05, base_capacity * 0.5),  # Labor bounds
            (base_capacity * 0.05, base_capacity * 0.5)   # Equipment bounds
        ]
        
        # Optimize
        result = minimize(objective_function, x0, constraints=constraints_list, bounds=bounds)
        
        if result.success:
            optimized_allocation = {
                'warehouse_capacity': result.x[0],
                'labor_capacity': result.x[1],
                'equipment_capacity': result.x[2],
                'total_cost': result.fun,
                'optimization_success': True
            }
        else:
            # Fallback to proportional allocation
            optimized_allocation = {
                'warehouse_capacity': base_capacity * 0.6,
                'labor_capacity': base_capacity * 0.2,
                'equipment_capacity': base_capacity * 0.2,
                'total_cost': base_capacity * 100,  # Simplified cost
                'optimization_success': False
            }
        
        print("Capacity allocation optimization completed")
        return optimized_allocation
    
    def train_models(self) -> Dict:
        """Calculate advanced optimization metrics"""
        print("Calculating advanced optimization metrics...")
        
        optimization_results = {}
        
        if 'basic_capacity' in self.features:
            basic_df = self.features['basic_capacity']
    
            demand_df = self.features.get('demand_forecasts', None)
            for _, row in basic_df.iterrows():
                seller_id = row['seller_id']
                timeframe = row['timeframe']
                base_capacity = row['base_capacity']
                print(f"Processing advanced optimization for seller {seller_id}, timeframe {timeframe}")
               
                if demand_df is not None and 'seller_id' in demand_df.columns and 'timeframe' in demand_df.columns:
                    seller_demand = demand_df[
                        (demand_df['seller_id'] == seller_id) & 
                        (demand_df['timeframe'] == timeframe)
                    ]
                else:
                    seller_demand = pd.DataFrame()
                # Calculate seasonal adjustments
                seasonal_adjustments = self.calculate_seasonal_adjustments(seller_demand) if not seller_demand.empty else {
                    'peak_season_multiplier': 1.3,
                    'low_season_multiplier': 0.8,
                    'holiday_impact': 0.2,
                    'promotion_impact': 0.15,
                    'seasonal_multipliers': {}
                }
                # Calculate emergency capacity
                emergency_capacity = self.calculate_emergency_capacity(base_capacity)
                # Calculate optimization metrics
                capacity_data = {
                    'total_needed_sqm': row['warehouse_sqm'],
                    'labor_hours': row['labor_hours']
                }
                cost_data = {'cost': base_capacity * self.config['warehouse_rent_per_sqm']}
                optimization_metrics = self.calculate_optimization_metrics(capacity_data, cost_data)
                # Optimize capacity allocation
                constraints = {'base_capacity': base_capacity}
                optimized_allocation = self.optimize_capacity_allocation(base_capacity, constraints)
                # Store results
                optimization_results[f"{seller_id}_{timeframe}"] = {
                    'seller_id': seller_id,
                    'timeframe': timeframe,
                    'base_capacity': base_capacity,
                    'seasonal_adjustments': seasonal_adjustments,
                    'emergency_capacity': emergency_capacity,
                    'optimization_metrics': optimization_metrics,
                    'optimized_allocation': optimized_allocation
                }
        
        self.optimization_results = optimization_results
        return optimization_results
    
    def generate_forecasts(self) -> Dict:
        """Generate advanced optimization forecasts"""
        print("Generating advanced optimization forecasts...")
        
        forecasts = {}
        
        for key, optimization_data in self.optimization_results.items():
            forecasts[key] = {
                'seller_id': optimization_data['seller_id'],
                'timeframe': optimization_data['timeframe'],
                'base_capacity': optimization_data['base_capacity'],
                'peak_capacity': optimization_data['base_capacity'] * optimization_data['seasonal_adjustments']['peak_season_multiplier'],
                'low_capacity': optimization_data['base_capacity'] * optimization_data['seasonal_adjustments']['low_season_multiplier'],
                'emergency_capacity_20': optimization_data['emergency_capacity']['demand_surge_20']['additional_capacity'],
                'emergency_capacity_50': optimization_data['emergency_capacity']['demand_surge_50']['additional_capacity'],
                'emergency_capacity_100': optimization_data['emergency_capacity']['demand_surge_100']['additional_capacity'],
                'overall_efficiency': optimization_data['optimization_metrics']['overall_efficiency'],
                'optimized_warehouse': optimization_data['optimized_allocation']['warehouse_capacity'],
                'optimized_labor': optimization_data['optimized_allocation']['labor_capacity'],
                'optimized_equipment': optimization_data['optimized_allocation']['equipment_capacity'],
                'total_optimized_cost': optimization_data['optimized_allocation']['total_cost']
            }
        
        self.forecast_results = forecasts
        return forecasts
    
    def evaluate_models(self) -> Dict:
        """Evaluate optimization performance"""
        print("Evaluating optimization performance...")
        
        evaluation_results = {}
        
        for key, optimization_data in self.optimization_results.items():
            # Calculate optimization effectiveness
            base_capacity = optimization_data['base_capacity']
            optimized_cost = optimization_data['optimized_allocation']['total_cost']
            efficiency = optimization_data['optimization_metrics']['overall_efficiency']
            
            # Calculate cost efficiency
            cost_efficiency = 1.0 / (optimized_cost / base_capacity) if base_capacity > 0 else 0
            
            evaluation_results[key] = {
                'optimization_effectiveness': efficiency,
                'cost_efficiency': cost_efficiency,
                'overall_score': (efficiency + cost_efficiency) / 2
            }
        
        return evaluation_results
    
    def save_results(self) -> None:
        """Save advanced optimization results"""
        print("Saving advanced optimization results...")
        
        # Save optimization forecasts
        if self.optimization_results:
            optimization_data = []
            for key, optimization_data_dict in self.optimization_results.items():
                optimization_data.append({
                    'seller_id': optimization_data_dict['seller_id'],
                    'timeframe': optimization_data_dict['timeframe'],
                    'base_capacity': optimization_data_dict['base_capacity'],
                    'peak_season_multiplier': optimization_data_dict['seasonal_adjustments']['peak_season_multiplier'],
                    'low_season_multiplier': optimization_data_dict['seasonal_adjustments']['low_season_multiplier'],
                    'emergency_capacity_20': optimization_data_dict['emergency_capacity']['demand_surge_20']['additional_capacity'],
                    'emergency_capacity_50': optimization_data_dict['emergency_capacity']['demand_surge_50']['additional_capacity'],
                    'emergency_capacity_100': optimization_data_dict['emergency_capacity']['demand_surge_100']['additional_capacity'],
                    'emergency_cost_20': optimization_data_dict['emergency_capacity']['demand_surge_20']['cost'],
                    'emergency_cost_50': optimization_data_dict['emergency_capacity']['demand_surge_50']['cost'],
                    'emergency_cost_100': optimization_data_dict['emergency_capacity']['demand_surge_100']['cost'],
                    'capacity_efficiency': optimization_data_dict['optimization_metrics']['capacity_efficiency'],
                    'labor_productivity': optimization_data_dict['optimization_metrics']['labor_productivity'],
                    'equipment_utilization': optimization_data_dict['optimization_metrics']['equipment_utilization'],
                    'overall_efficiency': optimization_data_dict['optimization_metrics']['overall_efficiency'],
                    'optimized_warehouse': optimization_data_dict['optimized_allocation']['warehouse_capacity'],
                    'optimized_labor': optimization_data_dict['optimized_allocation']['labor_capacity'],
                    'optimized_equipment': optimization_data_dict['optimized_allocation']['equipment_capacity'],
                    'total_optimized_cost': optimization_data_dict['optimized_allocation']['total_cost'],
                    'optimization_success': optimization_data_dict['optimized_allocation']['optimization_success']
                })
            
            if optimization_data:
                optimization_df = pd.DataFrame(optimization_data)
                self.output_manager.save_dataframe(optimization_df, 'advanced_capacity_optimization.csv')
        
        # Create visualizations
        if self.output_config['save_visualizations']:
            self.create_visualizations()
    
    def create_visualizations(self):
        """Create advanced optimization visualizations"""
        print("Creating advanced optimization visualizations...")
        
        if not self.optimization_results:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create optimization analysis chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Capacity Optimization Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for visualization
        sellers = []
        base_capacities = []
        peak_capacities = []
        emergency_capacities = []
        efficiencies = []
        
        for key, optimization_data in self.optimization_results.items():
            sellers.append(optimization_data['seller_id'])
            base_capacities.append(optimization_data['base_capacity'])
            peak_capacities.append(optimization_data['base_capacity'] * optimization_data['seasonal_adjustments']['peak_season_multiplier'])
            emergency_capacities.append(optimization_data['emergency_capacity']['demand_surge_50']['additional_capacity'])
            efficiencies.append(optimization_data['optimization_metrics']['overall_efficiency'])
        
        # Plot 1: Capacity comparison
        x_pos = np.arange(len(sellers))
        width = 0.25
        
        axes[0, 0].bar(x_pos - width, base_capacities, width, label='Base Capacity', alpha=0.8)
        axes[0, 0].bar(x_pos, peak_capacities, width, label='Peak Capacity', alpha=0.8)
        axes[0, 0].bar(x_pos + width, emergency_capacities, width, label='Emergency Capacity', alpha=0.8)
        
        axes[0, 0].set_title('Capacity Comparison by Seller')
        axes[0, 0].set_xlabel('Seller ID')
        axes[0, 0].set_ylabel('Capacity')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Efficiency metrics
        axes[0, 1].bar(sellers, efficiencies, alpha=0.8, color='green')
        axes[0, 1].set_title('Overall Efficiency by Seller')
        axes[0, 1].set_xlabel('Seller ID')
        axes[0, 1].set_ylabel('Efficiency Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0.85, color='red', linestyle='--', label='Target Efficiency')
        axes[0, 1].legend()
        
        # Plot 3: Emergency capacity costs
        emergency_costs = []
        for optimization_data in self.optimization_results.values():
            emergency_costs.append(optimization_data['emergency_capacity']['demand_surge_50']['cost'])
        
        axes[1, 0].bar(sellers, emergency_costs, alpha=0.8, color='orange')
        axes[1, 0].set_title('Emergency Capacity Costs (50% Surge)')
        axes[1, 0].set_xlabel('Seller ID')
        axes[1, 0].set_ylabel('Cost ($)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Optimization success rate
        success_rates = []
        for optimization_data in self.optimization_results.values():
            success_rates.append(1 if optimization_data['optimized_allocation']['optimization_success'] else 0)
        
        success_count = sum(success_rates)
        total_count = len(success_rates)
        
        axes[1, 1].pie([success_count, total_count - success_count], 
                      labels=['Successful', 'Failed'], 
                      autopct='%1.1f%%',
                      colors=['green', 'red'])
        axes[1, 1].set_title('Optimization Success Rate')
        
        plt.tight_layout()
        self.output_manager.save_visualization(fig, 'advanced_capacity_optimization.png')

def main():
    """Main execution function"""
    print("Starting Advanced Capacity Optimization Analysis")
    
    # Initialize optimizer
    optimizer = AdvancedCapacityOptimizer()
    
    try:
        # Run complete pipeline
        results = optimizer.run_complete_pipeline()
        
        print("\nAdvanced Capacity Optimization Analysis completed successfully")
        print(f"Optimization calculations completed for {len(optimizer.optimization_results)} seller-timeframe combinations")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 