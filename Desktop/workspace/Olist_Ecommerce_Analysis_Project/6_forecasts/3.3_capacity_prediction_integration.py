"""
Task 3.3: Capacity Prediction Integration
Integrate results from Tasks 3.1 and 3.2, create recommendations, and generate final outputs
Based on basic capacity calculations and advanced optimization results
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

# Import shared components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.shared_components import DataLoader, FeatureProcessor, ModelEvaluator, OutputManager, BaseForecaster
from config.model_config import OUTPUT_CONFIG

# Suppress warnings
warnings.filterwarnings('ignore')

class CapacityPredictionIntegrator(BaseForecaster):
    """Integrate capacity prediction results and create final recommendations"""
    
    def __init__(self, output_dir: str = 'output'):
        super().__init__(output_dir)
        
        # Integration parameters
        self.config = {
            'safety_margin': 0.15,
            'peak_multiplier': 1.3,
            'target_utilization': 0.85,
            'warehouse_rent_per_sqm': 15.5,
            'labor_cost_per_hour': 25.0,
            'equipment_maintenance_ratio': 0.05
        }
        self.output_config = OUTPUT_CONFIG
        # Data storage
        self.integrated_results = {}
        self.recommendations = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Tasks 3.1 and 3.2"""
        print("Loading data from Tasks 3.1 and 3.2...")
        
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
        
        # Load Task 3.2 outputs
        task3_2_files = [
            'advanced_capacity_optimization.csv'
        ]
        
        for file in task3_2_files:
            file_path = os.path.join(self.output_dir, file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        # Load Task 2 outputs for reference
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
        
        return data
    
    def prepare_features(self) -> Dict[str, pd.DataFrame]:
        """Prepare data for integration"""
        print("Preparing integration data...")
        
        integration_data = {}
        
        # Process basic capacity calculations
        if 'basic_capacity_calculations' in self.data:
            basic_df = self.data['basic_capacity_calculations'].copy()
            integration_data['basic_capacity'] = basic_df
            print(f"Processed basic capacity: {basic_df.shape}")
        
        # Process advanced optimization results
        if 'advanced_capacity_optimization' in self.data:
            advanced_df = self.data['advanced_capacity_optimization'].copy()
            integration_data['advanced_optimization'] = advanced_df
            print(f"Processed advanced optimization: {advanced_df.shape}")
        
        # Process demand forecasts
        if 'demand_forecasts' in self.data:
            demand_df = self.data['demand_forecasts'].copy()
            integration_data['demand_forecasts'] = demand_df
            print(f"Processed demand forecasts: {demand_df.shape}")
        
        return integration_data
    
    def integrate_capacity_results(self, basic_data: pd.DataFrame, advanced_data: pd.DataFrame) -> Dict:
        """Integrate basic capacity and advanced optimization results"""
        print("Integrating capacity results...")
        
        integrated_results = {}
        
        # Merge basic and advanced data
        if not basic_data.empty and not advanced_data.empty:
            merged_data = pd.merge(
                basic_data, 
                advanced_data, 
                on=['seller_id', 'timeframe'], 
                how='inner',
                suffixes=('_basic', '_advanced')
            )
            
            for _, row in merged_data.iterrows():
                seller_id = row['seller_id']
                timeframe = row['timeframe']
                
                # Create integrated result
                integrated_result = {
                    'seller_id': seller_id,
                    'timeframe': timeframe,
                    'base_capacity': row['base_capacity_basic'],
                    'warehouse_sqm': row['warehouse_sqm'],
                    'labor_hours': row['labor_hours'],
                    'fte_count': row['fte_count'],
                    'equipment_count': row['equipment_count'] if 'equipment_count' in row else (
                        (row['forklifts'] if 'forklifts' in row else 0) +
                        (row['conveyors'] if 'conveyors' in row else 0) +
                        (row['packing_stations'] if 'packing_stations' in row else 0)
                    ),
                    'peak_capacity': row['peak_capacity'] if 'peak_capacity' in row else (
                        row['base_capacity_basic'] * 1.3 if 'base_capacity_basic' in row else 0),
                    'low_capacity': row['low_capacity'] if 'low_capacity' in row else (
                        row['base_capacity_basic'] * 0.8 if 'base_capacity_basic' in row else 0),
                    'emergency_capacity_20': row['emergency_capacity_20'],
                    'emergency_capacity_50': row['emergency_capacity_50'],
                    'emergency_capacity_100': row['emergency_capacity_100'],
                    'overall_efficiency': row['overall_efficiency'],
                    'optimized_warehouse': row['optimized_warehouse'],
                    'optimized_labor': row['optimized_labor'],
                    'optimized_equipment': row['optimized_equipment'],
                    'total_optimized_cost': row['total_optimized_cost'],
                    'optimization_success': row['optimization_success']
                }
                
                integrated_results[f"{seller_id}_{timeframe}"] = integrated_result
            
            print(f"Integrated {len(integrated_results)} capacity results")
        
        return integrated_results
    
    def create_optimization_recommendations(self, integrated_results: Dict) -> pd.DataFrame:
        """Create capacity optimization recommendations"""
        print("Creating capacity optimization recommendations...")
        
        recommendations = []
        
        for forecast_id, forecast in integrated_results.items():
            seller_id = forecast['seller_id']
            timeframe = forecast['timeframe']
            
            # Generate recommendations based on integrated analysis
            base_capacity = forecast['base_capacity']
            overall_efficiency = forecast['overall_efficiency']
            optimization_success = forecast['optimization_success']
            
            # Capacity utilization recommendation
            if overall_efficiency < 0.8:
                recommendation_type = 'capacity_expansion'
                priority = 'high'
                description = 'Consider expanding warehouse capacity to improve utilization'
            elif overall_efficiency > 0.95:
                recommendation_type = 'capacity_optimization'
                priority = 'medium'
                description = 'Optimize existing capacity to reduce costs'
            else:
                recommendation_type = 'maintain_current'
                priority = 'low'
                description = 'Current capacity levels are optimal'
            
            # Add optimization success factor
            if not optimization_success:
                recommendation_type = 'optimization_review'
                priority = 'high'
                description = 'Review capacity optimization parameters for better results'
            
            recommendations.append({
                'seller_id': seller_id,
                'timeframe': timeframe,
                'recommendation_type': recommendation_type,
                'priority': priority,
                'description': description,
                'estimated_impact': overall_efficiency,
                'implementation_cost': base_capacity * self.config['warehouse_rent_per_sqm'] * 0.1,
                'current_efficiency': overall_efficiency,
                'optimization_success': optimization_success
            })
        
        return pd.DataFrame(recommendations)
    
    def create_emergency_capacity_plans(self, integrated_results: Dict) -> pd.DataFrame:
        """Create emergency capacity plans"""
        print("Creating emergency capacity plans...")
        
        emergency_plans = []
        
        for forecast_id, forecast in integrated_results.items():
            seller_id = forecast['seller_id']
            timeframe = forecast['timeframe']
            
            # Create emergency plans for different scenarios
            scenarios = [
                ('demand_surge_20', forecast['emergency_capacity_20'], 1.5, 'low'),
                ('demand_surge_50', forecast['emergency_capacity_50'], 2.0, 'medium'),
                ('demand_surge_100', forecast['emergency_capacity_100'], 3.0, 'high')
            ]
            
            for scenario, additional_capacity, cost_multiplier, risk_level in scenarios:
                cost = additional_capacity * self.config['warehouse_rent_per_sqm'] * cost_multiplier
                
                emergency_plans.append({
                    'seller_id': seller_id,
                    'timeframe': timeframe,
                    'scenario': scenario,
                    'additional_capacity': additional_capacity,
                    'cost': cost,
                    'implementation_time': '1_week' if '20' in scenario else '2_weeks' if '50' in scenario else '1_month',
                    'risk_level': risk_level,
                    'cost_multiplier': cost_multiplier
                })
        
        return pd.DataFrame(emergency_plans)
    
    def create_cost_analysis(self, integrated_results: Dict) -> pd.DataFrame:
        """Create comprehensive capacity cost analysis"""
        print("Creating comprehensive capacity cost analysis...")
        
        cost_analysis = []
        
        for forecast_id, forecast in integrated_results.items():
            seller_id = forecast['seller_id']
            timeframe = forecast['timeframe']
            
            # Calculate costs from integrated data
            warehouse_sqm = forecast['warehouse_sqm']
            labor_hours = forecast['labor_hours']
            equipment_count = forecast['equipment_count']
            optimized_warehouse = forecast['optimized_warehouse']
            optimized_labor = forecast['optimized_labor']
            optimized_equipment = forecast['optimized_equipment']
            
            # Current costs
            warehouse_cost = warehouse_sqm * self.config['warehouse_rent_per_sqm']
            labor_cost = labor_hours * self.config['labor_cost_per_hour']
            equipment_cost = equipment_count * 1000  # Simplified equipment cost
            maintenance_cost = equipment_count * 50  # $50 per equipment maintenance
            
            # Optimized costs
            optimized_warehouse_cost = optimized_warehouse * self.config['warehouse_rent_per_sqm']
            optimized_labor_cost = optimized_labor * self.config['labor_cost_per_hour']
            optimized_equipment_cost = optimized_equipment * 1000
            optimized_maintenance_cost = optimized_equipment * 50
            
            # Output pre- and post-optimization allocation for debugging
            print(f"[DEBUG] seller_id={seller_id}, timeframe={timeframe}")
            print(f"  Current allocation: warehouse={warehouse_sqm}, labor={labor_hours}, equipment={equipment_count}")
            print(f"  Optimized allocation: warehouse={optimized_warehouse}, labor={optimized_labor}, equipment={optimized_equipment}")
            
            # Total costs
            current_total_cost = warehouse_cost + labor_cost + equipment_cost + maintenance_cost
            optimized_total_cost = optimized_warehouse_cost + optimized_labor_cost + optimized_equipment_cost + optimized_maintenance_cost
            
            # Cost savings
            cost_savings = current_total_cost - optimized_total_cost
            savings_percentage = (cost_savings / current_total_cost * 100) if current_total_cost > 0 else 0
            
            cost_analysis.append({
                'seller_id': seller_id,
                'timeframe': timeframe,
                'current_warehouse_cost': warehouse_cost,
                'current_labor_cost': labor_cost,
                'current_equipment_cost': equipment_cost,
                'current_maintenance_cost': maintenance_cost,
                'current_total_cost': current_total_cost,
                'optimized_warehouse_cost': optimized_warehouse_cost,
                'optimized_labor_cost': optimized_labor_cost,
                'optimized_equipment_cost': optimized_equipment_cost,
                'optimized_total_cost': optimized_total_cost,
                'cost_savings': cost_savings,
                'savings_percentage': savings_percentage,
                'cost_per_order': optimized_total_cost / forecast['base_capacity'] if forecast['base_capacity'] > 0 else 0
            })
        
        return pd.DataFrame(cost_analysis)
    
    def train_models(self) -> Dict:
        """Integrate all capacity prediction results"""
        print("Integrating capacity prediction results...")
        
        if 'basic_capacity' in self.features and 'advanced_optimization' in self.features:
            basic_df = self.features['basic_capacity']
            advanced_df = self.features['advanced_optimization']
            
            # Integrate results
            integrated_results = self.integrate_capacity_results(basic_df, advanced_df)
            
            # Create recommendations
            recommendations = self.create_optimization_recommendations(integrated_results)
            
            # Create emergency plans
            emergency_plans = self.create_emergency_capacity_plans(integrated_results)
            
            # Create cost analysis
            cost_analysis = self.create_cost_analysis(integrated_results)
            
            # Store all results
            self.integrated_results = integrated_results
            self.recommendations = {
                'optimization_recommendations': recommendations,
                'emergency_plans': emergency_plans,
                'cost_analysis': cost_analysis
            }
            
            return integrated_results
        
        return {}
    
    def generate_forecasts(self) -> Dict:
        """Generate final integrated forecasts"""
        print("Generating final integrated forecasts...")
        
        forecasts = {}
        
        for key, integrated_data in self.integrated_results.items():
            forecasts[key] = {
                'seller_id': integrated_data['seller_id'],
                'timeframe': integrated_data['timeframe'],
                'base_capacity': integrated_data['base_capacity'],
                'peak_capacity': integrated_data['peak_capacity'],
                'recommended_capacity': integrated_data['base_capacity'] * (1 + self.config['safety_margin']),
                'warehouse_sqm': integrated_data['warehouse_sqm'],
                'labor_hours': integrated_data['labor_hours'],
                'fte_count': integrated_data['fte_count'],
                'equipment_count': integrated_data['equipment_count'],
                'overall_efficiency': integrated_data['overall_efficiency'],
                'optimized_total_cost': integrated_data['total_optimized_cost'],
                'optimization_success': integrated_data['optimization_success']
            }
        
        self.forecast_results = forecasts
        return forecasts
    
    def evaluate_models(self) -> Dict:
        """Evaluate integrated capacity prediction performance"""
        print("Evaluating integrated capacity prediction performance...")
        
        evaluation_results = {}
        
        for key, integrated_data in self.integrated_results.items():
            # Calculate comprehensive performance metrics
            base_capacity = integrated_data['base_capacity']
            overall_efficiency = integrated_data['overall_efficiency']
            optimization_success = integrated_data['optimization_success']
            optimized_cost = integrated_data['total_optimized_cost']
            
            # Calculate cost efficiency
            cost_efficiency = 1.0 / (optimized_cost / base_capacity) if base_capacity > 0 else 0
            
            # Calculate overall performance score
            performance_score = (overall_efficiency + cost_efficiency) / 2
            
            # Add optimization success bonus
            if optimization_success:
                performance_score *= 1.1
            
            evaluation_results[key] = {
                'capacity_efficiency': overall_efficiency,
                'cost_efficiency': cost_efficiency,
                'optimization_success': optimization_success,
                'overall_performance_score': performance_score
            }
        
        return evaluation_results
    
    def save_results(self) -> None:
        """Save integrated capacity prediction outputs"""
        print("Saving integrated capacity prediction outputs...")
        
        # Save integrated capacity forecasts
        if self.integrated_results:
            capacity_results = []
            for forecast_id, forecast in self.integrated_results.items():
                capacity_results.append({
                    'seller_id': forecast['seller_id'],
                    'timeframe': forecast['timeframe'],
                    'base_capacity': forecast['base_capacity'],
                    'peak_capacity': forecast['peak_capacity'],
                    'recommended_capacity': forecast['base_capacity'] * (1 + self.config['safety_margin']),
                    'warehouse_sqm': forecast['warehouse_sqm'],
                    'labor_hours': forecast['labor_hours'],
                    'fte_count': forecast['fte_count'],
                    'equipment_count': forecast['equipment_count'],
                    'overall_efficiency': forecast['overall_efficiency'],
                    'optimized_total_cost': forecast['total_optimized_cost'],
                    'optimization_success': forecast['optimization_success']
                })
            
            if capacity_results:
                capacity_df = pd.DataFrame(capacity_results)
                self.output_manager.save_dataframe(capacity_df, 'capacity_forecasts.csv')
        
        # Save optimization recommendations
        if 'optimization_recommendations' in self.recommendations:
            recommendations_df = self.recommendations['optimization_recommendations']
            if not recommendations_df.empty:
                self.output_manager.save_dataframe(recommendations_df, 'capacity_optimization_recommendations.csv')
        
        # Save emergency plans
        if 'emergency_plans' in self.recommendations:
            emergency_df = self.recommendations['emergency_plans']
            if not emergency_df.empty:
                self.output_manager.save_dataframe(emergency_df, 'emergency_capacity_plans.csv')
        
        # Save cost analysis
        if 'cost_analysis' in self.recommendations:
            cost_df = self.recommendations['cost_analysis']
            if not cost_df.empty:
                self.output_manager.save_dataframe(cost_df, 'capacity_cost_analysis.csv')
        
        # Create visualizations
        if self.output_config['save_visualizations']:
            self.create_visualizations()
    
    def create_visualizations(self):
        """Create comprehensive capacity analysis visualizations"""
        print("Creating comprehensive capacity analysis visualizations...")
        
        if not self.integrated_results:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create comprehensive analysis chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Capacity Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for visualization
        sellers = []
        base_capacities = []
        peak_capacities = []
        recommended_capacities = []
        efficiencies = []
        
        for key, integrated_data in self.integrated_results.items():
            sellers.append(integrated_data['seller_id'])
            base_capacities.append(integrated_data['base_capacity'])
            peak_capacities.append(integrated_data['peak_capacity'])
            recommended_capacities.append(integrated_data['base_capacity'] * (1 + self.config['safety_margin']))
            efficiencies.append(integrated_data['overall_efficiency'])
        
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
        
        # Plot 2: Efficiency metrics
        axes[0, 1].bar(sellers, efficiencies, alpha=0.8, color='green')
        axes[0, 1].set_title('Overall Efficiency by Seller')
        axes[0, 1].set_xlabel('Seller ID')
        axes[0, 1].set_ylabel('Efficiency Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0.85, color='red', linestyle='--', label='Target Efficiency')
        axes[0, 1].legend()
        
        # Plot 3: Recommendation distribution
        if 'optimization_recommendations' in self.recommendations:
            rec_df = self.recommendations['optimization_recommendations']
            if not rec_df.empty:
                rec_counts = rec_df['recommendation_type'].value_counts()
                axes[1, 0].pie(rec_counts.values, labels=rec_counts.index, autopct='%1.1f%%')
                axes[1, 0].set_title('Recommendation Distribution')
        
        # Plot 4: Cost savings
        if 'cost_analysis' in self.recommendations:
            cost_df = self.recommendations['cost_analysis']
            if not cost_df.empty:
                savings_percentages = cost_df['savings_percentage'].values
                axes[1, 1].bar(range(len(savings_percentages)), savings_percentages, alpha=0.8, color='blue')
                axes[1, 1].set_title('Cost Savings Percentage by Seller')
                axes[1, 1].set_xlabel('Seller Index')
                axes[1, 1].set_ylabel('Savings (%)')
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        self.output_manager.save_visualization(fig, 'comprehensive_capacity_analysis.png')
    
    def print_analysis_summary(self):
        """Print comprehensive capacity analysis summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE CAPACITY PREDICTION ANALYSIS SUMMARY")
        print("="*80)
        
        total_forecasts = len(self.integrated_results)
        total_sellers = len(set(forecast['seller_id'] for forecast in self.integrated_results.values()))
        
        print(f"\nINTEGRATED CAPACITY FORECASTS:")
        print("-" * 50)
        print(f"Total forecasts generated: {total_forecasts}")
        print(f"Total sellers analyzed: {total_sellers}")
        
        # Calculate average metrics
        avg_efficiency = np.mean([forecast['overall_efficiency'] 
                                for forecast in self.integrated_results.values()])
        avg_capacity = np.mean([forecast['base_capacity'] 
                              for forecast in self.integrated_results.values()])
        
        print(f"Average overall efficiency: {avg_efficiency:.3f}")
        print(f"Average base capacity: {avg_capacity:.0f}")
        
        print(f"\nOPTIMIZATION RECOMMENDATIONS:")
        print("-" * 50)
        if 'optimization_recommendations' in self.recommendations:
            rec_df = self.recommendations['optimization_recommendations']
            if not rec_df.empty:
                rec_counts = rec_df['recommendation_type'].value_counts()
                for rec_type, count in rec_counts.items():
                    print(f"  {rec_type}: {count} recommendations")
                
                high_priority = rec_df[rec_df['priority'] == 'high']
                print(f"  High priority recommendations: {len(high_priority)}")
        
        print(f"\nCOST ANALYSIS:")
        print("-" * 50)
        if 'cost_analysis' in self.recommendations:
            cost_df = self.recommendations['cost_analysis']
            if not cost_df.empty:
                avg_savings = cost_df['savings_percentage'].mean()
                total_savings = cost_df['cost_savings'].sum()
                print(f"  Average cost savings: {avg_savings:.2f}%")
                print(f"  Total potential savings: ${total_savings:,.2f}")
        
        print(f"\nOutput files saved in: {self.output_dir}")
        print(f"Visualizations saved in: {self.output_dir}/visualizations")
        
        print("\n" + "="*80)

def main():
    """Main execution function"""
    print("Starting Capacity Prediction Integration Analysis")
    
    # Initialize integrator
    integrator = CapacityPredictionIntegrator()
    
    try:
        # Run complete pipeline
        results = integrator.run_complete_pipeline()
        
        # Print comprehensive summary
        integrator.print_analysis_summary()
        
        print("\nCapacity Prediction Integration Analysis completed successfully")
        print(f"Integration completed for {len(integrator.integrated_results)} seller-timeframe combinations")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 