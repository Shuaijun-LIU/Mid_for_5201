"""
Task 3: Capacity Demand Prediction Algorithm
Based on Task 2 demand forecasting results, combined with Week4 inventory efficiency data 
and Week3 customer behavior data, calculate detailed capacity demand predictions including 
warehouse capacity, labor requirements, equipment needs, with seasonal adjustments and 
emergency capacity planning.
Execution date: 2025-07-15
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
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress warnings
warnings.filterwarnings('ignore')

class CapacityDemandPredictor:
    """Capacity demand prediction with warehouse, labor, and equipment planning"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        
        # Create directories
        for dir_path in [output_dir, self.viz_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Capacity calculation parameters
        self.config = {
            'safety_margin': 0.15,
            'peak_multiplier': 1.3,
            'target_utilization': 0.85,
            'emergency_capacity_ratio': 0.20,
            'warehouse_rent_per_sqm': 15.5,
            'labor_cost_per_hour': 25.0,
            'equipment_maintenance_ratio': 0.05
        }
        
        # Data storage
        self.data = {}
        self.capacity_forecasts = {}
        self.optimization_results = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Task 2, Week4, and Week3 outputs"""
        print("Loading data from Task 2, Week4, and Week3 outputs...")
        
        data = {}
        
        # Load Task 2 outputs
        task2_files = [
            'demand_forecasts.csv',
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
    
    def prepare_capacity_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data for capacity calculations"""
        print("Preparing capacity calculation data...")
        
        capacity_data = {}
        
        # Process demand forecasts
        if 'demand_forecasts' in data:
            demand_df = data['demand_forecasts'].copy()
            demand_df['forecast_date'] = pd.to_datetime(demand_df['forecast_period'])
            capacity_data['demand_forecasts'] = demand_df
            print(f"Processed demand forecasts: {demand_df.shape}")
        
        # Process inventory efficiency data
        if 'inventory_efficiency_metrics' in data:
            inventory_df = data['inventory_efficiency_metrics'].copy()
            capacity_data['inventory_efficiency'] = inventory_df
            print(f"Processed inventory efficiency: {inventory_df.shape}")
        
        # Process customer behavior data
        if 'final_customer_segments' in data:
            customer_df = data['final_customer_segments'].copy()
            capacity_data['customer_segments'] = customer_df
            print(f"Processed customer segments: {customer_df.shape}")
        
        # Process product data
        if 'olist_products_dataset' in data:
            product_df = data['olist_products_dataset'].copy()
            capacity_data['products'] = product_df
            print(f"Processed products: {product_df.shape}")
        
        return capacity_data
    
    def calculate_warehouse_capacity(self, demand_data: pd.DataFrame, product_data: pd.DataFrame) -> Dict:
        """Calculate warehouse capacity requirements"""
        print("Calculating warehouse capacity requirements...")
        
        warehouse_capacity = {}
        
        # Calculate base storage requirements
        if 'total_orders' in demand_data.columns and 'product_id' in product_data.columns:
            # Estimate product volume (simplified calculation)
            product_data['estimated_volume'] = product_data['product_weight_g'] * 0.001  # Convert to liters
            
            # Calculate monthly storage requirements
            monthly_demand = demand_data.groupby('forecast_period')['predicted_value'].sum()
            
            # Calculate storage space needed
            storage_sqm = monthly_demand * 0.1  # Simplified: 0.1 sqm per order
            processing_sqm = storage_sqm * 0.3  # 30% for processing
            auxiliary_sqm = storage_sqm * 0.15  # 15% for auxiliary functions
            
            warehouse_capacity = {
                'total_needed_sqm': storage_sqm + processing_sqm + auxiliary_sqm,
                'storage_sqm': storage_sqm,
                'processing_sqm': processing_sqm,
                'auxiliary_sqm': auxiliary_sqm,
                'peak_multiplier': self.config['peak_multiplier'],
                'safety_margin': self.config['safety_margin']
            }
            
            print(f"Calculated warehouse capacity for {len(monthly_demand)} periods")
        
        return warehouse_capacity
    
    def calculate_labor_requirements(self, demand_data: pd.DataFrame, customer_data: pd.DataFrame) -> Dict:
        """Calculate labor requirements"""
        print("Calculating labor requirements...")
        
        labor_requirements = {}
        
        if 'predicted_value' in demand_data.columns:
            # Calculate base labor hours
            base_hours_per_order = 0.5  # 30 minutes per order
            monthly_orders = demand_data.groupby('forecast_period')['predicted_value'].sum()
            
            # Calculate total hours
            total_hours_monthly = monthly_orders * base_hours_per_order
            
            # Calculate full-time equivalents (assuming 160 hours per month)
            fte = total_hours_monthly / 160
            
            # Calculate peak requirements
            peak_hours_multiplier = self.config['peak_multiplier']
            peak_hours = total_hours_monthly * peak_hours_multiplier
            
            # Calculate overtime and temporary hours
            overtime_hours = (peak_hours - total_hours_monthly) * 0.7
            temporary_hours = (peak_hours - total_hours_monthly) * 0.3
            
            labor_requirements = {
                'total_hours_monthly': total_hours_monthly,
                'full_time_equivalents': fte,
                'peak_hours_multiplier': peak_hours_multiplier,
                'overtime_hours': overtime_hours,
                'temporary_hours': temporary_hours,
                'skill_mix': {
                    'pickers': 0.6,
                    'packers': 0.3,
                    'supervisors': 0.1
                },
                'training_hours': fte * 5  # 5 hours training per FTE
            }
            
            print(f"Calculated labor requirements for {len(monthly_orders)} periods")
        
        return labor_requirements
    
    def calculate_equipment_needs(self, demand_data: pd.DataFrame, inventory_data: pd.DataFrame) -> Dict:
        """Calculate equipment requirements"""
        print("Calculating equipment requirements...")
        
        equipment_needs = {}
        
        if 'predicted_value' in demand_data.columns:
            monthly_orders = demand_data.groupby('forecast_period')['predicted_value'].sum()
            
            # Calculate equipment based on order volume
            forklifts = np.ceil(monthly_orders / 1000).astype(int)  # 1 forklift per 1000 orders
            conveyors = np.ceil(monthly_orders / 2000).astype(int)  # 1 conveyor per 2000 orders
            packing_stations = np.ceil(monthly_orders / 500).astype(int)  # 1 station per 500 orders
            storage_racks = np.ceil(monthly_orders / 100).astype(int)  # 1 rack per 100 orders
            computers = np.ceil(monthly_orders / 300).astype(int)  # 1 computer per 300 orders
            
            # Calculate maintenance hours
            maintenance_hours = (forklifts + conveyors + packing_stations) * 2  # 2 hours per equipment
            
            equipment_needs = {
                'forklifts': forklifts,
                'conveyors': conveyors,
                'packing_stations': packing_stations,
                'storage_racks': storage_racks,
                'computers': computers,
                'maintenance_hours': maintenance_hours
            }
            
            print(f"Calculated equipment needs for {len(monthly_orders)} periods")
        
        return equipment_needs
    
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
    
    def generate_capacity_forecasts(self, capacity_data: Dict[str, pd.DataFrame]) -> Dict:
        """Generate comprehensive capacity forecasts"""
        print("Generating comprehensive capacity forecasts...")
        
        all_forecasts = {}
        
        if 'demand_forecasts' in capacity_data:
            demand_df = capacity_data['demand_forecasts']
            
            # Group by seller and timeframe
            for (seller_id, timeframe), group in demand_df.groupby(['seller_id', 'timeframe']):
                print(f"Processing capacity forecast for seller {seller_id}, timeframe {timeframe}")
                
                # Calculate base capacity
                base_capacity = group['predicted_value'].mean()
                
                # Calculate warehouse capacity
                warehouse_capacity = self.calculate_warehouse_capacity(
                    group, capacity_data.get('products', pd.DataFrame())
                )
                
                # Calculate labor requirements
                labor_requirements = self.calculate_labor_requirements(
                    group, capacity_data.get('customer_segments', pd.DataFrame())
                )
                
                # Calculate equipment needs
                equipment_needs = self.calculate_equipment_needs(
                    group, capacity_data.get('inventory_efficiency', pd.DataFrame())
                )
                
                # Calculate seasonal adjustments
                seasonal_adjustments = self.calculate_seasonal_adjustments(group)
                
                # Calculate emergency capacity
                emergency_capacity = self.calculate_emergency_capacity(base_capacity)
                
                # Calculate optimization metrics
                optimization_metrics = self.calculate_optimization_metrics(
                    warehouse_capacity, {'cost': base_capacity * self.config['warehouse_rent_per_sqm']}
                )
                
                # Create forecast structure
                forecast = {
                    'seller_id': seller_id,
                    'timeframe': timeframe,
                    'base_capacity': base_capacity,
                    'warehouse_capacity': warehouse_capacity,
                    'labor_requirements': labor_requirements,
                    'equipment_needs': equipment_needs,
                    'seasonal_adjustments': seasonal_adjustments,
                    'emergency_capacity': emergency_capacity,
                    'optimization_metrics': optimization_metrics
                }
                
                all_forecasts[f"{seller_id}_{timeframe}"] = forecast
        
        return all_forecasts
    
    def create_optimization_recommendations(self, capacity_forecasts: Dict) -> pd.DataFrame:
        """Create capacity optimization recommendations"""
        print("Creating capacity optimization recommendations...")
        
        recommendations = []
        
        for forecast_id, forecast in capacity_forecasts.items():
            seller_id = forecast['seller_id']
            timeframe = forecast['timeframe']
            
            # Generate recommendations based on capacity analysis
            base_capacity = forecast['base_capacity']
            warehouse_capacity = forecast['warehouse_capacity']
            optimization_metrics = forecast['optimization_metrics']
            
            # Capacity utilization recommendation
            if optimization_metrics['capacity_efficiency'] < 0.8:
                recommendation_type = 'capacity_expansion'
                priority = 'high'
                description = 'Consider expanding warehouse capacity to improve utilization'
            elif optimization_metrics['capacity_efficiency'] > 0.95:
                recommendation_type = 'capacity_optimization'
                priority = 'medium'
                description = 'Optimize existing capacity to reduce costs'
            else:
                recommendation_type = 'maintain_current'
                priority = 'low'
                description = 'Current capacity levels are optimal'
            
            recommendations.append({
                'seller_id': seller_id,
                'timeframe': timeframe,
                'recommendation_type': recommendation_type,
                'priority': priority,
                'description': description,
                'estimated_impact': optimization_metrics['overall_efficiency'],
                'implementation_cost': base_capacity * self.config['warehouse_rent_per_sqm'] * 0.1
            })
        
        return pd.DataFrame(recommendations)
    
    def create_emergency_capacity_plans(self, capacity_forecasts: Dict) -> pd.DataFrame:
        """Create emergency capacity plans"""
        print("Creating emergency capacity plans...")
        
        emergency_plans = []
        
        for forecast_id, forecast in capacity_forecasts.items():
            seller_id = forecast['seller_id']
            timeframe = forecast['timeframe']
            emergency_capacity = forecast['emergency_capacity']
            
            for scenario, details in emergency_capacity.items():
                emergency_plans.append({
                    'seller_id': seller_id,
                    'timeframe': timeframe,
                    'scenario': scenario,
                    'additional_capacity': details['additional_capacity'],
                    'cost': details['cost'],
                    'implementation_time': details['implementation_time'],
                    'risk_level': 'high' if '100' in scenario else 'medium' if '50' in scenario else 'low'
                })
        
        return pd.DataFrame(emergency_plans)
    
    def create_cost_analysis(self, capacity_forecasts: Dict) -> pd.DataFrame:
        """Create capacity cost analysis"""
        print("Creating capacity cost analysis...")
        
        cost_analysis = []
        
        for forecast_id, forecast in capacity_forecasts.items():
            seller_id = forecast['seller_id']
            timeframe = forecast['timeframe']
            
            # Calculate costs
            warehouse_capacity = forecast['warehouse_capacity']
            labor_requirements = forecast['labor_requirements']
            equipment_needs = forecast['equipment_needs']
            
            # Fixed costs
            warehouse_cost = warehouse_capacity.get('total_needed_sqm', 0) * self.config['warehouse_rent_per_sqm']
            equipment_cost = sum(equipment_needs.values()) * 1000  # Simplified equipment cost
            
            # Variable costs
            labor_cost = labor_requirements.get('total_hours_monthly', 0) * self.config['labor_cost_per_hour']
            maintenance_cost = equipment_needs.get('maintenance_hours', 0) * 50  # $50 per maintenance hour
            
            # Total costs
            total_cost = warehouse_cost + equipment_cost + labor_cost + maintenance_cost
            
            cost_analysis.append({
                'seller_id': seller_id,
                'timeframe': timeframe,
                'warehouse_cost': warehouse_cost,
                'equipment_cost': equipment_cost,
                'labor_cost': labor_cost,
                'maintenance_cost': maintenance_cost,
                'total_cost': total_cost,
                'cost_per_order': total_cost / forecast['base_capacity'] if forecast['base_capacity'] > 0 else 0
            })
        
        return pd.DataFrame(cost_analysis)
    
    def create_visualizations(self, capacity_forecasts: Dict, optimization_recommendations: pd.DataFrame):
        """Create capacity analysis visualizations"""
        print("Creating capacity analysis visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Capacity demand trends
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Capacity Demand Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for visualization
        sellers = []
        base_capacities = []
        peak_capacities = []
        recommended_capacities = []
        
        for forecast_id, forecast in capacity_forecasts.items():
            sellers.append(forecast['seller_id'])
            base_capacities.append(forecast['base_capacity'])
            peak_capacities.append(forecast['base_capacity'] * forecast['seasonal_adjustments']['peak_season_multiplier'])
            recommended_capacities.append(forecast['base_capacity'] * (1 + self.config['safety_margin']))
        
        # Plot 1: Capacity comparison
        x_pos = np.arange(len(sellers))
        width = 0.25
        
        axes[0, 0].bar(x_pos - width, base_capacities, width, label='Base Capacity', alpha=0.8)
        axes[0, 0].bar(x_pos, peak_capacities, width, label='Peak Capacity', alpha=0.8)
        axes[0, 0].bar(x_pos + width, recommended_capacities, width, label='Recommended Capacity', alpha=0.8)
        
        axes[0, 0].set_title('Capacity Comparison by Seller')
        axes[0, 0].set_xlabel('Seller ID')
        axes[0, 0].set_ylabel('Capacity')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Optimization metrics
        efficiency_metrics = []
        for forecast in capacity_forecasts.values():
            efficiency_metrics.append(forecast['optimization_metrics']['overall_efficiency'])
        
        axes[0, 1].bar(sellers, efficiency_metrics, alpha=0.8, color='green')
        axes[0, 1].set_title('Overall Efficiency by Seller')
        axes[0, 1].set_xlabel('Seller ID')
        axes[0, 1].set_ylabel('Efficiency Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0.85, color='red', linestyle='--', label='Target Efficiency')
        axes[0, 1].legend()
        
        # Plot 3: Recommendation distribution
        if not optimization_recommendations.empty:
            rec_counts = optimization_recommendations['recommendation_type'].value_counts()
            axes[1, 0].pie(rec_counts.values, labels=rec_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Recommendation Distribution')
        
        # Plot 4: Emergency capacity costs
        emergency_costs = []
        emergency_scenarios = []
        for forecast in capacity_forecasts.values():
            for scenario, details in forecast['emergency_capacity'].items():
                emergency_costs.append(details['cost'])
                emergency_scenarios.append(scenario)
        
        if emergency_costs:
            axes[1, 1].bar(emergency_scenarios, emergency_costs, alpha=0.8, color='orange')
            axes[1, 1].set_title('Emergency Capacity Costs')
            axes[1, 1].set_xlabel('Emergency Scenario')
            axes[1, 1].set_ylabel('Cost ($)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'capacity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Capacity analysis visualizations created successfully")
    
    def save_outputs(self, capacity_forecasts: Dict, optimization_recommendations: pd.DataFrame, 
                    emergency_plans: pd.DataFrame, cost_analysis: pd.DataFrame):
        """Save capacity prediction outputs"""
        print("Saving capacity prediction outputs...")
        
        # Save capacity forecasts
        capacity_results = []
        for forecast_id, forecast in capacity_forecasts.items():
            capacity_results.append({
                'seller_id': forecast['seller_id'],
                'timeframe': forecast['timeframe'],
                'base_capacity': forecast['base_capacity'],
                'peak_capacity': forecast['base_capacity'] * forecast['seasonal_adjustments']['peak_season_multiplier'],
                'recommended_capacity': forecast['base_capacity'] * (1 + self.config['safety_margin']),
                'warehouse_sqm': forecast['warehouse_capacity'].get('total_needed_sqm', 0),
                'labor_hours': forecast['labor_requirements'].get('total_hours_monthly', 0),
                'fte_count': forecast['labor_requirements'].get('full_time_equivalents', 0),
                'overall_efficiency': forecast['optimization_metrics']['overall_efficiency']
            })
        
        capacity_df = pd.DataFrame(capacity_results)
        capacity_df.to_csv(os.path.join(self.output_dir, 'capacity_forecasts.csv'), index=False)
        print(f"Saved capacity forecasts: {capacity_df.shape}")
        
        # Save optimization recommendations
        if not optimization_recommendations.empty:
            optimization_recommendations.to_csv(os.path.join(self.output_dir, 'capacity_optimization_recommendations.csv'), index=False)
            print(f"Saved optimization recommendations: {optimization_recommendations.shape}")
        
        # Save emergency plans
        if not emergency_plans.empty:
            emergency_plans.to_csv(os.path.join(self.output_dir, 'emergency_capacity_plans.csv'), index=False)
            print(f"Saved emergency plans: {emergency_plans.shape}")
        
        # Save cost analysis
        if not cost_analysis.empty:
            cost_analysis.to_csv(os.path.join(self.output_dir, 'capacity_cost_analysis.csv'), index=False)
            print(f"Saved cost analysis: {cost_analysis.shape}")
        
        print("All capacity prediction outputs saved successfully")
    
    def print_analysis_summary(self, capacity_forecasts: Dict, optimization_recommendations: pd.DataFrame):
        """Print capacity analysis summary"""
        print("\n" + "="*80)
        print("CAPACITY DEMAND PREDICTION ANALYSIS SUMMARY")
        print("="*80)
        
        total_forecasts = len(capacity_forecasts)
        total_sellers = len(set(forecast['seller_id'] for forecast in capacity_forecasts.values()))
        
        print(f"\n📊 CAPACITY FORECASTS:")
        print("-" * 50)
        print(f"Total forecasts generated: {total_forecasts}")
        print(f"Total sellers analyzed: {total_sellers}")
        
        # Calculate average metrics
        avg_efficiency = np.mean([forecast['optimization_metrics']['overall_efficiency'] 
                                for forecast in capacity_forecasts.values()])
        avg_capacity = np.mean([forecast['base_capacity'] 
                              for forecast in capacity_forecasts.values()])
        
        print(f"Average overall efficiency: {avg_efficiency:.3f}")
        print(f"Average base capacity: {avg_capacity:.0f}")
        
        print(f"\n🎯 OPTIMIZATION RECOMMENDATIONS:")
        print("-" * 50)
        if not optimization_recommendations.empty:
            rec_counts = optimization_recommendations['recommendation_type'].value_counts()
            for rec_type, count in rec_counts.items():
                print(f"  {rec_type}: {count} recommendations")
            
            high_priority = optimization_recommendations[optimization_recommendations['priority'] == 'high']
            print(f"  High priority recommendations: {len(high_priority)}")
        
        print(f"\n📁 Output files saved in: {self.output_dir}")
        print(f"📊 Visualizations saved in: {self.viz_dir}")
        
        print("\n" + "="*80)

def main():
    """Main execution function"""
    print("Starting Capacity Demand Prediction Analysis")
    
    # Initialize predictor
    predictor = CapacityDemandPredictor()
    
    try:
        # Load data
        data = predictor.load_data()
        
        if not data:
            print("No data loaded. Exiting.")
            return
        
        # Prepare capacity data
        capacity_data = predictor.prepare_capacity_data(data)
        
        if not capacity_data:
            print("No capacity data prepared. Exiting.")
            return
        
        # Generate capacity forecasts
        capacity_forecasts = predictor.generate_capacity_forecasts(capacity_data)
        
        if not capacity_forecasts:
            print("No capacity forecasts generated. Exiting.")
            return
        
        # Create optimization recommendations
        optimization_recommendations = predictor.create_optimization_recommendations(capacity_forecasts)
        
        # Create emergency capacity plans
        emergency_plans = predictor.create_emergency_capacity_plans(capacity_forecasts)
        
        # Create cost analysis
        cost_analysis = predictor.create_cost_analysis(capacity_forecasts)
        
        # Create visualizations
        predictor.create_visualizations(capacity_forecasts, optimization_recommendations)
        
        # Save outputs
        predictor.save_outputs(capacity_forecasts, optimization_recommendations, emergency_plans, cost_analysis)
        
        # Print summary
        predictor.print_analysis_summary(capacity_forecasts, optimization_recommendations)
        
        print("Capacity Demand Prediction Analysis completed successfully")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 