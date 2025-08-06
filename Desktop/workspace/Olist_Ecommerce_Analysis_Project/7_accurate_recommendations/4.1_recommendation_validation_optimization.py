"""
Task 4: Recommendation Validation and Optimization
Based on Week6-Task 3 capacity prediction results, Week4 cost data, and Week5 seller analysis,
conduct comprehensive validation of recommendations including historical backtesting, Monte Carlo
simulation, expert review, A/B testing design, and optimization recommendations to ensure
recommendation accuracy and effectiveness.
Execution date: 2025-07-19
Update date: 2025-07-27
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

import importlib.util

# Suppress warnings
warnings.filterwarnings('ignore')

class RecommendationValidationOptimization:
    """Recommendation validation and optimization with comprehensive testing"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        
        # Create directories
        for dir_path in [output_dir, self.viz_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Validation parameters
        self.config = {
            'analysis_period': 3,  # years
            'confidence_level': 0.95,
            'validation_methods': [
                'historical_backtest',
                'monte_carlo_simulation',
                'expert_review',
                'ab_testing'
            ],
            'optimization_criteria': {
                'accuracy': 0.4,
                'robustness': 0.3,
                'feasibility': 0.2,
                'cost_effectiveness': 0.1
            }
        }
        
        # Initialize calculation and analysis modules
        # Dynamic import for modules with numeric prefixes
        spec_calc = importlib.util.spec_from_file_location("validation_calculations", "4.2_validation_calculations.py")
        module_calc = importlib.util.module_from_spec(spec_calc)
        spec_calc.loader.exec_module(module_calc)
        
        spec_analysis = importlib.util.spec_from_file_location("validation_analysis", "4.3_validation_analysis.py")
        module_analysis = importlib.util.module_from_spec(spec_analysis)
        spec_analysis.loader.exec_module(module_analysis)
        
        self.calculations = module_calc.ValidationCalculations(self.config)
        self.analysis = module_analysis.ValidationAnalysis(self.config)
        
        # Data storage
        self.data = {}
        self.validation_results = {}
        self.optimization_recommendations = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Task 1-3 outputs"""
        print("Loading data from Task 1-3 outputs...")
        
        data = {}
        
        # Load Task 1-3 outputs (Week6)
        task_outputs = [
            'week6_time_series_features_matrix.csv',  # 使用实际存在的文件
            'statistical_forecasts.csv',  # 使用实际存在的文件
            'capacity_forecasts.csv',
            'capacity_cost_analysis.csv',  # 使用实际存在的文件
            'ml_forecasts.csv'  # 使用实际存在的文件
        ]
        
        for file in task_outputs:
            file_path = os.path.join('../week6_forecasts/output', file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        # Load Week4 outputs
        week4_files = [
            'inventory_efficiency_metrics.csv',
            'warehouse_simulation_summary.csv',
            'holding_cost_by_category_and_stage.csv'
        ]
        
        for file in week4_files:
            file_path = os.path.join('../week4_product_warehouse_analysis/output', file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        # Load Week5 outputs
        week5_files = [
            'seller_summary_analysis.csv',
            'product_summary_analysis.csv',
            'four_d_main_analysis.csv'
        ]
        
        for file in week5_files:
            file_path = os.path.join('../week5_seller_analysis_and_four_d_analysis/output', file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        # Load original data
        original_files = [
            'olist_order_payments_dataset.csv',
            'olist_order_items_dataset.csv'
        ]
        
        for file in original_files:
            file_path = os.path.join('../data/processed_missing', file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        return data
    
    def prepare_validation_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data for validation analysis"""
        print("Preparing validation data...")
        
        validation_data = {}
        
        # Process capacity forecasts
        if 'capacity_forecasts' in data:
            capacity_df = data['capacity_forecasts'].copy()
            validation_data['capacity_forecasts'] = capacity_df
            print(f"Processed capacity forecasts: {capacity_df.shape}")
        
        # Process cost analysis
        if 'capacity_cost_analysis' in data:
            cost_df = data['capacity_cost_analysis'].copy()
            validation_data['cost_analysis'] = cost_df
            print(f"Processed cost analysis: {cost_df.shape}")
        
        # Process seller analysis
        if 'seller_summary_analysis' in data:
            seller_df = data['seller_summary_analysis'].copy()
            validation_data['seller_analysis'] = seller_df
            print(f"Processed seller analysis: {seller_df.shape}")
        
        # Process inventory efficiency
        if 'inventory_efficiency_metrics' in data:
            inventory_df = data['inventory_efficiency_metrics'].copy()
            validation_data['inventory_efficiency'] = inventory_df
            print(f"Processed inventory efficiency: {inventory_df.shape}")
        
        # Process payment data
        if 'olist_order_payments_dataset' in data:
            payment_df = data['olist_order_payments_dataset'].copy()
            validation_data['payments'] = payment_df
            print(f"Processed payments: {payment_df.shape}")
        
        return validation_data
    
    def create_comprehensive_validation_results(self, validation_data: Dict[str, pd.DataFrame]) -> Dict:
        """Create comprehensive validation results"""
        print("Creating comprehensive validation results...")
        
        comprehensive_results = {}
        
        if 'capacity_forecasts' in validation_data:
            capacity_df = validation_data['capacity_forecasts']
            
            # Group by seller for analysis
            for seller_id, group in capacity_df.groupby('seller_id'):
                print(f"Validating recommendations for seller {seller_id}")
                
                # Historical backtest validation using calculation module
                historical_results = self.calculations.historical_backtest_validation(validation_data)
                
                # Monte Carlo simulation using calculation module
                monte_carlo_results = self.calculations.monte_carlo_simulation(validation_data)
                
                # Expert review system using analysis module
                expert_review_results = self.analysis.expert_review_system(validation_data)
                
                # A/B test design using analysis module
                ab_test_results = self.analysis.design_ab_tests(validation_data)
                
                # Generate optimization recommendations using analysis module
                optimization_recommendations = self.analysis.generate_optimization_recommendations(
                    validation_data, historical_results, monte_carlo_results, expert_review_results
                )
                
                # Create comprehensive validation structure
                validation_summary = {
                    'seller_id': seller_id,
                    'validation_id': f'validation_{seller_id}',
                    'validation_date': datetime.now().strftime('%Y-%m-%d'),
                    'validation_methods': {
                        'historical_backtest': historical_results,
                        'monte_carlo_simulation': monte_carlo_results,
                        'expert_review': expert_review_results,
                        'ab_testing': ab_test_results
                    },
                    'optimization_recommendations': optimization_recommendations,
                    'overall_validation_score': np.mean([
                        historical_results['validation_score'],
                        monte_carlo_results['validation_score'],
                        expert_review_results['validation_score'],
                        ab_test_results['validation_score']
                    ])
                }
                
                comprehensive_results[seller_id] = validation_summary
        
        return comprehensive_results
    
    def create_visualizations(self, comprehensive_results: Dict):
        """Create validation and optimization visualizations"""
        print("Creating validation and optimization visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Recommendation Validation and Optimization Results', fontsize=16, fontweight='bold')
        
        # Extract data for visualization
        sellers = list(comprehensive_results.keys())
        
        # 1. Validation scores by method
        validation_methods = ['historical_backtest', 'monte_carlo_simulation', 'expert_review', 'ab_testing']
        method_scores = {}
        
        for method in validation_methods:
            scores = []
            for seller in sellers:
                scores.append(comprehensive_results[seller]['validation_methods'][method]['validation_score'])
            method_scores[method] = np.mean(scores)
        
        method_names = [method.replace('_', ' ').title() for method in validation_methods]
        method_values = list(method_scores.values())
        
        axes[0, 0].bar(method_names, method_values, alpha=0.8, color='blue')
        axes[0, 0].set_title('Average Validation Scores by Method')
        axes[0, 0].set_ylabel('Validation Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Overall validation scores by seller
        overall_scores = []
        for seller in sellers:
            overall_scores.append(comprehensive_results[seller]['overall_validation_score'])
        
        axes[0, 1].bar(sellers, overall_scores, alpha=0.8, color='green')
        axes[0, 1].set_title('Overall Validation Scores by Seller')
        axes[0, 1].set_xlabel('Seller ID')
        axes[0, 1].set_ylabel('Overall Validation Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Historical backtest results
        if sellers:
            historical_data = comprehensive_results[sellers[0]]['validation_methods']['historical_backtest']
            accuracy_metrics = ['mape', 'rmse', 'mae']
            accuracy_values = [historical_data[metric] for metric in accuracy_metrics]
            
            axes[0, 2].bar(accuracy_metrics, accuracy_values, alpha=0.8, color='orange')
            axes[0, 2].set_title('Historical Backtest Accuracy Metrics')
            axes[0, 2].set_ylabel('Error Rate')
        
        # 4. Monte Carlo simulation results
        if sellers:
            mc_data = comprehensive_results[sellers[0]]['validation_methods']['monte_carlo_simulation']
            confidence_levels = ['90%', '95%', '99%']
            confidence_values = [mc_data['confidence_90'], mc_data['confidence_95'], mc_data['confidence_99']]
            
            axes[1, 0].bar(confidence_levels, confidence_values, alpha=0.8, color='red')
            axes[1, 0].set_title('Monte Carlo Confidence Intervals')
            axes[1, 0].set_ylabel('Confidence Value')
        
        # 5. Expert review scores
        if sellers:
            expert_data = comprehensive_results[sellers[0]]['validation_methods']['expert_review']
            if 'expert_scores' in expert_data:
                expert_scores = expert_data['expert_scores']
                expert_criteria = list(expert_scores.keys())
                expert_values = list(expert_scores.values())
                
                axes[1, 1].bar(expert_criteria, expert_values, alpha=0.8, color='purple')
                axes[1, 1].set_title('Expert Review Scores by Criteria')
                axes[1, 1].set_ylabel('Expert Score')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. A/B test design metrics
        if sellers:
            ab_data = comprehensive_results[sellers[0]]['validation_methods']['ab_testing']
            test_metrics = ['sample_size', 'test_duration', 'statistical_power', 'significance_level']
            test_values = [ab_data[metric] for metric in test_metrics]
            
            axes[1, 2].bar(test_metrics, test_values, alpha=0.8, color='brown')
            axes[1, 2].set_title('A/B Test Design Metrics')
            axes[1, 2].set_ylabel('Metric Value')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'validation_optimization_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Validation and optimization visualizations created successfully")
    
    def save_outputs(self, comprehensive_results: Dict):
        """Save validation and optimization outputs"""
        print("Saving validation and optimization outputs...")
        
        # Save comprehensive validation results
        validation_results = []
        for seller_id, validation in comprehensive_results.items():
            result = {
                'seller_id': seller_id,
                'validation_id': validation['validation_id'],
                'validation_date': validation['validation_date'],
                'overall_validation_score': validation['overall_validation_score'],
                'historical_backtest_score': validation['validation_methods']['historical_backtest']['validation_score'],
                'monte_carlo_score': validation['validation_methods']['monte_carlo_simulation']['validation_score'],
                'expert_review_score': validation['validation_methods']['expert_review']['validation_score'],
                'ab_test_score': validation['validation_methods']['ab_testing']['validation_score']
            }
            validation_results.append(result)
        
        validation_df = pd.DataFrame(validation_results)
        validation_df.to_csv(os.path.join(self.output_dir, 'validation_results.csv'), index=False)
        print(f"Saved validation results: {validation_df.shape}")
        
        # Save historical backtest results
        historical_results = []
        for seller_id, validation in comprehensive_results.items():
            historical = validation['validation_methods']['historical_backtest']
            result = {
                'seller_id': seller_id,
                'validation_score': historical['validation_score'],
                'mape': historical['mape'],
                'rmse': historical['rmse'],
                'mae': historical['mae'],
                'accuracy_rate': historical['accuracy_rate']
            }
            historical_results.append(result)
        
        historical_df = pd.DataFrame(historical_results)
        historical_df.to_csv(os.path.join(self.output_dir, 'historical_backtest_results.csv'), index=False)
        print(f"Saved historical backtest results: {historical_df.shape}")
        
        # Save Monte Carlo simulation results
        monte_carlo_results = []
        for seller_id, validation in comprehensive_results.items():
            mc = validation['validation_methods']['monte_carlo_simulation']
            result = {
                'seller_id': seller_id,
                'validation_score': mc['validation_score'],
                'confidence_90': mc['confidence_90'],
                'confidence_95': mc['confidence_95'],
                'confidence_99': mc['confidence_99'],
                'mean_performance': mc['mean_performance'],
                'std_performance': mc['std_performance']
            }
            monte_carlo_results.append(result)
        
        mc_df = pd.DataFrame(monte_carlo_results)
        mc_df.to_csv(os.path.join(self.output_dir, 'monte_carlo_validation_results.csv'), index=False)
        print(f"Saved Monte Carlo validation results: {mc_df.shape}")
        
        # Save expert review results
        expert_results = []
        for seller_id, validation in comprehensive_results.items():
            expert = validation['validation_methods']['expert_review']
            result = {
                'seller_id': seller_id,
                'validation_score': expert['validation_score']
            }
            # Add expert scores if available
            if 'expert_scores' in expert:
                for criterion, score in expert['expert_scores'].items():
                    result[criterion] = score
            
            expert_results.append(result)
        
        expert_df = pd.DataFrame(expert_results)
        expert_df.to_csv(os.path.join(self.output_dir, 'expert_review_results.csv'), index=False)
        print(f"Saved expert review results: {expert_df.shape}")
        
        # Save optimization recommendations
        optimization_results = []
        for seller_id, validation in comprehensive_results.items():
            optimizations = validation['optimization_recommendations']
            for opt_type, opt_data in optimizations.items():
                result = {
                    'seller_id': seller_id,
                    'optimization_type': opt_type
                }
                # Add optimization data if available
                if isinstance(opt_data, dict):
                    for key, value in opt_data.items():
                        if key in ['priority_score', 'expected_improvement', 'implementation_complexity', 
                                  'time_to_implement', 'cost_estimate']:
                            result[key] = value
                else:
                    # If opt_data is a single value, store it as overall_score
                    result['overall_score'] = opt_data
                
                optimization_results.append(result)
        
        optimization_df = pd.DataFrame(optimization_results)
        optimization_df.to_csv(os.path.join(self.output_dir, 'optimization_recommendations.csv'), index=False)
        print(f"Saved optimization recommendations: {optimization_df.shape}")
        
        print("All validation and optimization outputs saved successfully")
    
    def print_analysis_summary(self, comprehensive_results: Dict):
        """Print validation and optimization summary"""
        print("\n" + "="*80)
        print("RECOMMENDATION VALIDATION AND OPTIMIZATION SUMMARY")
        print("="*80)
        
        total_validations = len(comprehensive_results)
        
        print(f"\nVALIDATION OVERVIEW:")
        print("-" * 50)
        print(f"Total validations completed: {total_validations}")
        
        if comprehensive_results:
            # Calculate average validation scores
            all_overall_scores = []
            method_scores = {
                'historical_backtest': [],
                'monte_carlo_simulation': [],
                'expert_review': [],
                'ab_testing': []
            }
            
            for seller_id, validation in comprehensive_results.items():
                all_overall_scores.append(validation['overall_validation_score'])
                for method in method_scores.keys():
                    method_scores[method].append(validation['validation_methods'][method]['validation_score'])
            
            avg_overall_score = np.mean(all_overall_scores)
            avg_method_scores = {method: np.mean(scores) for method, scores in method_scores.items()}
            
            print(f"Average overall validation score: {avg_overall_score:.3f}")
            print(f"\nAVERAGE VALIDATION SCORES BY METHOD:")
            print("-" * 50)
            for method, score in avg_method_scores.items():
                print(f"  {method.replace('_', ' ').title()}: {score:.3f}")
            
            # Validation score distribution
            high_validation = sum(1 for score in all_overall_scores if score > 0.8)
            medium_validation = sum(1 for score in all_overall_scores if 0.6 <= score <= 0.8)
            low_validation = sum(1 for score in all_overall_scores if score < 0.6)
            
            print(f"\nVALIDATION SCORE DISTRIBUTION:")
            print("-" * 50)
            print(f"  High validation (>0.8): {high_validation} sellers")
            print(f"  Medium validation (0.6-0.8): {medium_validation} sellers")
            print(f"  Low validation (<0.6): {low_validation} sellers")
            
            # Optimization recommendations
            total_optimizations = sum(len(validation['optimization_recommendations']) 
                                    for validation in comprehensive_results.values())
            
            print(f"\nOPTIMIZATION RECOMMENDATIONS:")
            print("-" * 50)
            print(f"Total optimization recommendations generated: {total_optimizations}")
        
        print(f"\nOutput files saved in: {self.output_dir}")
        print(f"Visualizations saved in: {self.viz_dir}")
        
        print("\n" + "="*80)

def main():
    """Main execution function"""
    print("Starting Recommendation Validation and Optimization")
    
    # Initialize validator
    validator = RecommendationValidationOptimization()
    
    try:
        # Load data
        data = validator.load_data()
        
        if not data:
            print("No data loaded. Exiting.")
            return
        
        # Prepare validation data
        validation_data = validator.prepare_validation_data(data)
        
        if not validation_data:
            print("No validation data prepared. Exiting.")
            return
        
        # Create comprehensive validation results
        comprehensive_results = validator.create_comprehensive_validation_results(validation_data)
        
        if not comprehensive_results:
            print("No validation results generated. Exiting.")
            return
        
        # Create visualizations
        validator.create_visualizations(comprehensive_results)
        
        # Save outputs
        validator.save_outputs(comprehensive_results)
        
        # Print summary
        validator.print_analysis_summary(comprehensive_results)
        
        print("Recommendation Validation and Optimization completed successfully")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 