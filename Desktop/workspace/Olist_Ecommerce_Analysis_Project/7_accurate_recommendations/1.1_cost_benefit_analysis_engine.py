"""
Task 1: Cost-Benefit Analysis Engine
1.1 main function
Based on Week6-Task 3 capacity prediction results, combined with Week4 cost data and historical
financial data, conduct comprehensive and systematic cost-benefit analysis including fixed 
costs, variable costs, revenue prediction, ROI calculation and sensitivity analysis to 
provide financial basis for warehouse strategy decisions.
Execution date: 2025-07-19
Update date: 2025-07-24
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

class CostBenefitAnalyzer:
    """Cost-benefit analysis engine with financial metrics and sensitivity analysis"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        
        # Create directories
        for dir_path in [output_dir, self.viz_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Financial analysis parameters
        self.config = {
            'analysis_period': 10,  
            'discount_rates': [0.05, 0.10, 0.15],
            'recommended_discount_rate': 0.10,
            'inflation_rate': 0.03,
            'risk_premium': 0.05,
            'tax_rate': 0.25,
            'sensitivity_range': 0.30,
            'monte_carlo_iterations': 10000,
            'confidence_level': 0.95
        }
        
        # Initialize calculation and analysis modules
        # Dynamic import for modules with numeric prefixes
        spec_calc = importlib.util.spec_from_file_location("cost_benefit_calculations", "1.2_cost_benefit_calculations.py")
        module_calc = importlib.util.module_from_spec(spec_calc)
        spec_calc.loader.exec_module(module_calc)
        
        spec_analysis = importlib.util.spec_from_file_location("cost_benefit_analysis", "1.3_cost_benefit_analysis.py")
        module_analysis = importlib.util.module_from_spec(spec_analysis)
        spec_analysis.loader.exec_module(module_analysis)
        
        self.calculations = module_calc.CostBenefitCalculations(self.config)
        self.analysis = module_analysis.CostBenefitAnalysis(self.config)
        
        # Data storage
        self.data = {}
        self.cost_benefit_results = {}
        self.financial_metrics = {}
        self.sensitivity_results = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Task 3, Week4, and Week5 outputs"""
        print("Loading data from Task 3, Week4, and Week5 outputs...")
        
        data = {}
        
        # Load Task 3 outputs (Week6)
        task3_files = [
            'capacity_forecasts.csv',
            'capacity_cost_analysis.csv',
            'emergency_capacity_plans.csv'
        ]
        
        for file in task3_files:
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
    
    def prepare_analysis_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data for cost-benefit analysis"""
        print("Preparing cost-benefit analysis data...")
        
        analysis_data = {}
        
        # Process capacity forecasts
        if 'capacity_forecasts' in data:
            capacity_df = data['capacity_forecasts'].copy()
            analysis_data['capacity_forecasts'] = capacity_df
            print(f"Processed capacity forecasts: {capacity_df.shape}")
        
        # Process cost analysis
        if 'capacity_cost_analysis' in data:
            cost_df = data['capacity_cost_analysis'].copy()
            analysis_data['cost_analysis'] = cost_df
            print(f"Processed cost analysis: {cost_df.shape}")
        
        # Process inventory efficiency
        if 'inventory_efficiency_metrics' in data:
            inventory_df = data['inventory_efficiency_metrics'].copy()
            analysis_data['inventory_efficiency'] = inventory_df
            print(f"Processed inventory efficiency: {inventory_df.shape}")
        
        # Process payment data
        if 'olist_order_payments_dataset' in data:
            payment_df = data['olist_order_payments_dataset'].copy()
            analysis_data['payments'] = payment_df
            print(f"Processed payments: {payment_df.shape}")
        
        return analysis_data
    
    def create_comprehensive_analysis(self, analysis_data: Dict[str, pd.DataFrame]) -> Dict:
        """Create comprehensive cost-benefit analysis"""
        print("Creating comprehensive cost-benefit analysis...")
        
        all_analyses = {}
        
        if 'capacity_forecasts' in analysis_data:
            capacity_df = analysis_data['capacity_forecasts']
            
            # Group by seller for analysis
            for seller_id, group in capacity_df.groupby('seller_id'):
                print(f"Analyzing cost-benefit for seller {seller_id}")
                
                # Calculate costs using calculation module
                fixed_costs = self.calculations.calculate_fixed_costs(group, pd.DataFrame())
                variable_costs = self.calculations.calculate_variable_costs(group, pd.DataFrame())
                
                # Combine costs
                total_costs = {}
                for year in range(1, self.config['analysis_period'] + 1):
                    total_costs[f'year_{year}'] = (fixed_costs['total_fixed'][f'year_{year}'] + 
                                                 variable_costs['total_variable'][f'year_{year}'])
                
                total_costs['total'] = sum(total_costs.values())
                
                costs = {
                    'fixed_costs': fixed_costs,
                    'variable_costs': variable_costs,
                    'total_costs': total_costs
                }
                
                # Calculate benefits using calculation module
                benefits = self.calculations.calculate_benefits(group, analysis_data.get('payments', pd.DataFrame()))
                
                # Calculate financial metrics using calculation module
                financial_metrics = self.calculations.calculate_financial_metrics(costs, benefits)
                
                # Perform sensitivity analysis using analysis module
                sensitivity_results = self.analysis.perform_sensitivity_analysis(costs, benefits, financial_metrics)
                
                # Run Monte Carlo simulation using analysis module
                monte_carlo_results = self.analysis.run_monte_carlo_simulation(costs, benefits, financial_metrics)
                
                # Add Monte Carlo results to sensitivity analysis
                sensitivity_results['monte_carlo_results'] = monte_carlo_results
                
                # Calculate risk-adjusted metrics using analysis module
                risk_adjusted_metrics = self.analysis.calculate_risk_adjusted_metrics(financial_metrics, sensitivity_results)
                
                # Generate optimization recommendations using analysis module
                optimization_recommendations = self.analysis.generate_optimization_recommendations(
                    costs, benefits, financial_metrics, sensitivity_results
                )
                
                # Create comprehensive analysis structure
                analysis = {
                    'seller_id': seller_id,
                    'recommendation_id': f'rec_{seller_id}',
                    'analysis_period': f'{self.config["analysis_period"]}_years',
                    'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                    'costs': costs,
                    'benefits': benefits,
                    'financial_metrics': financial_metrics,
                    'sensitivity_analysis': sensitivity_results,
                    'risk_adjusted_metrics': risk_adjusted_metrics,
                    'optimization_recommendations': optimization_recommendations
                }
                
                all_analyses[seller_id] = analysis
        
        return all_analyses
    
    def create_visualizations(self, cost_benefit_results: Dict):
        """Create cost-benefit analysis visualizations"""
        print("Creating cost-benefit analysis visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cost-Benefit Analysis Results', fontsize=16, fontweight='bold')
        
        # Extract data for visualization
        sellers = list(cost_benefit_results.keys())
        
        # 1. Cost structure pie chart
        if sellers:
            seller_data = cost_benefit_results[sellers[0]]
            fixed_costs = seller_data['costs']['fixed_costs']['total_fixed']['total']
            variable_costs = seller_data['costs']['variable_costs']['total_variable']['total']
            
            cost_labels = ['Fixed Costs', 'Variable Costs']
            cost_values = [fixed_costs, variable_costs]
            
            axes[0, 0].pie(cost_values, labels=cost_labels, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Cost Structure')
        
        # 2. Benefits composition
        if sellers:
            direct_benefits = sum(seller_data['benefits']['direct_benefits'][key]['total'] 
                                for key in seller_data['benefits']['direct_benefits'])
            indirect_benefits = sum(seller_data['benefits']['indirect_benefits'][key]['total'] 
                                  for key in seller_data['benefits']['indirect_benefits'])
            
            benefit_labels = ['Direct Benefits', 'Indirect Benefits']
            benefit_values = [direct_benefits, indirect_benefits]
            
            axes[0, 1].pie(benefit_values, labels=benefit_labels, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Benefits Composition')
        
        # 3. ROI comparison
        roi_values = []
        for seller in sellers:
            roi_values.append(cost_benefit_results[seller]['financial_metrics']['roi']['overall'])
        
        axes[0, 2].bar(sellers, roi_values, alpha=0.8, color='green')
        axes[0, 2].set_title('ROI by Seller')
        axes[0, 2].set_xlabel('Seller ID')
        axes[0, 2].set_ylabel('ROI')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].axhline(y=0, color='red', linestyle='--')
        
        # 4. NPV comparison
        npv_values = []
        for seller in sellers:
            npv_values.append(cost_benefit_results[seller]['financial_metrics']['npv']['recommended_rate'])
        
        axes[1, 0].bar(sellers, npv_values, alpha=0.8, color='blue')
        axes[1, 0].set_title('NPV by Seller')
        axes[1, 0].set_xlabel('Seller ID')
        axes[1, 0].set_ylabel('NPV ($)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        
        # 5. Risk score distribution
        risk_scores = []
        for seller in sellers:
            risk_scores.append(cost_benefit_results[seller]['risk_adjusted_metrics']['risk_score'])
        
        risk_counts = pd.Series(risk_scores).value_counts()
        axes[1, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Risk Score Distribution')
        
        # 6. Monte Carlo simulation results
        if sellers:
            try:
                monte_carlo_data = cost_benefit_results[sellers[0]]['sensitivity_analysis']['monte_carlo_results']
                mean_roi = monte_carlo_data['mean_roi']
                std_roi = monte_carlo_data['std_roi']
                
                # Generate sample data for histogram
                sample_roi = np.random.normal(mean_roi, std_roi, 1000)
                axes[1, 2].hist(sample_roi, bins=30, alpha=0.7, color='orange')
                axes[1, 2].axvline(mean_roi, color='red', linestyle='--', label=f'Mean: {mean_roi:.3f}')
                axes[1, 2].set_title('Monte Carlo ROI Distribution')
                axes[1, 2].set_xlabel('ROI')
                axes[1, 2].set_ylabel('Frequency')
                axes[1, 2].legend()
            except KeyError:
                # If Monte Carlo results not available, create a placeholder
                axes[1, 2].text(0.5, 0.5, 'Monte Carlo data\nnot available', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Monte Carlo ROI Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'cost_benefit_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Cost-benefit analysis visualizations created successfully")
    
    def save_outputs(self, cost_benefit_results: Dict):
        """Save cost-benefit analysis outputs"""
        print("Saving cost-benefit analysis outputs...")
        
        # Save comprehensive analysis results
        analysis_results = []
        for seller_id, analysis in cost_benefit_results.items():
            result = {
                'seller_id': seller_id,
                'recommendation_id': analysis['recommendation_id'],
                'analysis_period': analysis['analysis_period'],
                'total_costs': analysis['costs']['total_costs']['total'],
                'total_benefits': analysis['benefits']['total_benefits']['total'],
                'overall_roi': analysis['financial_metrics']['roi']['overall'],
                'npv_recommended': analysis['financial_metrics']['npv']['recommended_rate'],
                'irr_value': analysis['financial_metrics']['irr']['value'],
                'payback_period': analysis['financial_metrics']['payback_period']['simple'],
                'risk_score': analysis['risk_adjusted_metrics']['risk_score'],
                'probability_positive_npv': analysis['sensitivity_analysis'].get('monte_carlo_results', {}).get('probability_positive_npv', 0.0)
            }
            analysis_results.append(result)
        
        analysis_df = pd.DataFrame(analysis_results)
        analysis_df.to_csv(os.path.join(self.output_dir, 'cost_benefit_analysis.csv'), index=False)
        print(f"Saved cost-benefit analysis: {analysis_df.shape}")
        
        # Save financial metrics summary
        financial_summary = []
        for seller_id, analysis in cost_benefit_results.items():
            metrics = analysis['financial_metrics']
            summary = {
                'seller_id': seller_id,
                'roi_year_1': metrics['roi']['year_1'],
                'roi_year_2': metrics['roi']['year_2'],
                'roi_year_3': metrics['roi']['year_3'],
                'overall_roi': metrics['roi']['overall'],
                'npv_5_percent': metrics['npv']['discount_rate_5'],
                'npv_10_percent': metrics['npv']['discount_rate_10'],
                'npv_15_percent': metrics['npv']['discount_rate_15'],
                'irr_value': metrics['irr']['value'],
                'payback_period': metrics['payback_period']['simple']
            }
            financial_summary.append(summary)
        
        financial_df = pd.DataFrame(financial_summary)
        financial_df.to_csv(os.path.join(self.output_dir, 'financial_metrics_summary.csv'), index=False)
        print(f"Saved financial metrics summary: {financial_df.shape}")
        
        # Save sensitivity analysis results
        sensitivity_results = []
        for seller_id, analysis in cost_benefit_results.items():
            sensitivity = analysis['sensitivity_analysis']
            roi_impacts = sensitivity['roi_impacts']
            for variable, data in sensitivity['key_variables'].items():
                for i, scenario in enumerate(['pessimistic', 'base_case', 'optimistic']):
                    result = {
                        'seller_id': seller_id,
                        'variable': variable,
                        'scenario': scenario,
                        'value': data[scenario],
                        'roi_impact': roi_impacts[variable][i]
                    }
                    sensitivity_results.append(result)
        
        sensitivity_df = pd.DataFrame(sensitivity_results)
        sensitivity_df.to_csv(os.path.join(self.output_dir, 'sensitivity_analysis_results.csv'), index=False)
        print(f"Saved sensitivity analysis results: {sensitivity_df.shape}")
        
        # Save optimization recommendations
        optimization_results = []
        for seller_id, analysis in cost_benefit_results.items():
            recommendations = analysis['optimization_recommendations']
            for category, recs in recommendations.items():
                for rec in recs:
                    result = {
                        'seller_id': seller_id,
                        'category': category,
                        'recommendation': rec
                    }
                    optimization_results.append(result)
        
        optimization_df = pd.DataFrame(optimization_results)
        optimization_df.to_csv(os.path.join(self.output_dir, 'optimization_recommendations.csv'), index=False)
        print(f"Saved optimization recommendations: {optimization_df.shape}")
        
        print("All cost-benefit analysis outputs saved successfully")
    
    def print_analysis_summary(self, cost_benefit_results: Dict):
        """Print cost-benefit analysis summary"""
        print("\n" + "="*80)
        print("COST-BENEFIT ANALYSIS SUMMARY")
        print("="*80)
        
        total_analyses = len(cost_benefit_results)
        
        print(f"\nANALYSIS OVERVIEW:")
        print("-" * 50)
        print(f"Total analyses completed: {total_analyses}")
        print(f"Analysis period: {self.config['analysis_period']} years")
        print(f"Monte Carlo iterations: {self.config['monte_carlo_iterations']}")
        
        if cost_benefit_results:
            # Calculate average metrics
            avg_roi = np.mean([analysis['financial_metrics']['roi']['overall'] 
                             for analysis in cost_benefit_results.values()])
            avg_npv = np.mean([analysis['financial_metrics']['npv']['recommended_rate'] 
                             for analysis in cost_benefit_results.values()])
            avg_irr = np.mean([analysis['financial_metrics']['irr']['value'] 
                             for analysis in cost_benefit_results.values()])
            
            print(f"\nFINANCIAL METRICS:")
            print("-" * 50)
            print(f"Average ROI: {avg_roi:.3f}")
            print(f"Average NPV: ${avg_npv:,.0f}")
            print(f"Average IRR: {avg_irr:.3f}")
            
            # Risk analysis
            risk_scores = [analysis['risk_adjusted_metrics']['risk_score'] 
                          for analysis in cost_benefit_results.values()]
            risk_distribution = pd.Series(risk_scores).value_counts()
            
            print(f"\nRISK ANALYSIS:")
            print("-" * 50)
            for risk_level, count in risk_distribution.items():
                print(f"  {risk_level.capitalize()} risk: {count} sellers")
            
            # Optimization recommendations
            total_recommendations = sum(len(analysis['optimization_recommendations']['cost_optimization']) +
                                      len(analysis['optimization_recommendations']['benefit_maximization']) +
                                      len(analysis['optimization_recommendations']['risk_mitigation'])
                                      for analysis in cost_benefit_results.values())
            
            print(f"\nOPTIMIZATION RECOMMENDATIONS:")
            print("-" * 50)
            print(f"Total recommendations generated: {total_recommendations}")
        
        print(f"\nOutput files saved in: {self.output_dir}")
        print(f"Visualizations saved in: {self.viz_dir}")
        
        print("\n" + "="*80)

def main():
    """Main execution function"""
    print("Starting Cost-Benefit Analysis Engine")
    
    # Initialize analyzer
    analyzer = CostBenefitAnalyzer()
    
    try:
        # Load data
        data = analyzer.load_data()
        
        if not data:
            print("No data loaded. Exiting.")
            return
        
        # Prepare analysis data
        analysis_data = analyzer.prepare_analysis_data(data)
        
        if not analysis_data:
            print("No analysis data prepared. Exiting.")
            return
        
        # Create comprehensive analysis
        cost_benefit_results = analyzer.create_comprehensive_analysis(analysis_data)
        
        if not cost_benefit_results:
            print("No cost-benefit analysis results generated. Exiting.")
            return
        
        # Create visualizations
        analyzer.create_visualizations(cost_benefit_results)
        
        # Save outputs
        analyzer.save_outputs(cost_benefit_results)
        
        # Print summary
        analyzer.print_analysis_summary(cost_benefit_results)
        
        print("Cost-Benefit Analysis Engine completed successfully")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 