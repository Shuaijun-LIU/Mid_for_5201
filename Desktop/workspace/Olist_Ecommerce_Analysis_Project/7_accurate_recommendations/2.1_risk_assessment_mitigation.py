"""
Task 2: Risk Assessment and Mitigation Model
Based on Week6-Task 3 capacity prediction results and Week4 cost data, conduct 
comprehensive risk assessment including demand risk, operational risk, market risk, 
financial risk, scenario analysis and Monte Carlo simulation to provide risk 
mitigation strategies for warehouse strategy decisions.
Execution date: 2025-07-19
Update date: 2025-07-25
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

warnings.filterwarnings('ignore')

class RiskAssessmentMitigation:
    """Risk assessment and mitigation model with comprehensive risk analysis"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        
        # Create directories
        for dir_path in [output_dir, self.viz_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Risk assessment parameters
        self.config = {
            'analysis_period': 3,  # years
            'confidence_level': 0.95,
            'risk_thresholds': {
                'low': 0.3,
                'medium': 0.6,
                'high': 1.0
            },
            'monte_carlo_iterations': 10000,
            'scenario_probabilities': {
                'optimistic': 0.2,
                'base_case': 0.6,
                'pessimistic': 0.2
            }
        }
        
        # Initialize calculation and analysis modules
        # Dynamic import for modules with numeric prefixes
        spec_calc = importlib.util.spec_from_file_location("risk_assessment_calculations", "2.2_risk_assessment_calculations.py")
        module_calc = importlib.util.module_from_spec(spec_calc)
        spec_calc.loader.exec_module(module_calc)
        
        spec_analysis = importlib.util.spec_from_file_location("risk_analysis_mitigation", "2.3_risk_analysis_mitigation.py")
        module_analysis = importlib.util.module_from_spec(spec_analysis)
        spec_analysis.loader.exec_module(module_analysis)
        
        self.calculations = module_calc.RiskAssessmentCalculations(self.config)
        self.analysis = module_analysis.RiskAnalysisMitigation(self.config)
        
        # Data storage
        self.data = {}
        self.risk_assessments = {}
        self.scenario_results = {}
        self.mitigation_strategies = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Task 3, Week4, and Week5 outputs"""
        print("Loading data from Task 3, Week4, and Week5 outputs...")
        
        data = {}
        
        # Load Task 2-4 outputs (Week6)
        task_outputs = [
            'statistical_forecasts.csv',  
            'capacity_forecasts.csv',
            'capacity_cost_analysis.csv', 
            'ml_forecasts.csv'  
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
    
    def prepare_risk_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data for risk assessment"""
        print("Preparing risk assessment data...")
        
        risk_data = {}
        
        # Process demand forecasts
        if 'statistical_forecasts' in data:
            demand_df = data['statistical_forecasts'].copy()
            risk_data['demand_forecasts'] = demand_df
            print(f"Processed demand forecasts: {demand_df.shape}")
        
        # Process capacity forecasts
        if 'capacity_forecasts' in data:
            capacity_df = data['capacity_forecasts'].copy()
            risk_data['capacity_forecasts'] = capacity_df
            print(f"Processed capacity forecasts: {capacity_df.shape}")
        
        # Process cost analysis
        if 'capacity_cost_analysis' in data:
            cost_df = data['capacity_cost_analysis'].copy()
            risk_data['cost_analysis'] = cost_df
            print(f"Processed cost analysis: {cost_df.shape}")
        
        # Process inventory efficiency
        if 'inventory_efficiency_metrics' in data:
            inventory_df = data['inventory_efficiency_metrics'].copy()
            risk_data['inventory_efficiency'] = inventory_df
            print(f"Processed inventory efficiency: {inventory_df.shape}")
        
        # Process payment data
        if 'olist_order_payments_dataset' in data:
            payment_df = data['olist_order_payments_dataset'].copy()
            risk_data['payments'] = payment_df
            print(f"Processed payments: {payment_df.shape}")
        
        return risk_data
    
    def create_comprehensive_risk_assessment(self, risk_data: Dict[str, pd.DataFrame]) -> Dict:
        """Create comprehensive risk assessment"""
        print("Creating comprehensive risk assessment...")
        
        all_assessments = {}
        
        if 'demand_forecasts' in risk_data:
            demand_df = risk_data['demand_forecasts']
            
            # Check if seller_id exists, if not use overall analysis
            if 'seller_id' in demand_df.columns:
                # Group by seller for analysis
                for seller_id, group in demand_df.groupby('seller_id'):
                    print(f"Assessing risks for seller {seller_id}")
                    
                    # Assess demand risk using calculation module
                    demand_risk = self.calculations.assess_demand_risk(group, risk_data.get('payments', pd.DataFrame()))
                    
                    # Assess operational risk using calculation module
                    operational_risk = self.calculations.assess_operational_risk(
                        risk_data.get('capacity_forecasts', pd.DataFrame()),
                        risk_data.get('inventory_efficiency', pd.DataFrame())
                    )
                    
                    # Assess market risk using calculation module
                    market_risk = self.calculations.assess_market_risk(
                        risk_data.get('cost_analysis', pd.DataFrame()),
                        risk_data.get('payments', pd.DataFrame())
                    )
                    
                    # Assess financial risk using calculation module
                    financial_risk = self.calculations.assess_financial_risk(
                        risk_data.get('cost_analysis', pd.DataFrame()),
                        risk_data.get('capacity_forecasts', pd.DataFrame())
                    )
                    
                    # Perform scenario analysis using analysis module
                    scenario_analysis = self.analysis.perform_scenario_analysis({
                        'demand_risk': demand_risk,
                        'operational_risk': operational_risk,
                        'market_risk': market_risk,
                        'financial_risk': financial_risk
                    })
                    
                    # Run Monte Carlo simulation using analysis module
                    monte_carlo_results = self.analysis.run_monte_carlo_simulation({
                        'demand_risk': demand_risk,
                        'operational_risk': operational_risk,
                        'market_risk': market_risk,
                        'financial_risk': financial_risk
                    })
                    
                    # Create risk monitoring metrics using analysis module
                    monitoring_metrics = self.analysis.create_risk_monitoring_metrics({
                        'demand_risk': demand_risk,
                        'operational_risk': operational_risk,
                        'market_risk': market_risk,
                        'financial_risk': financial_risk
                    })
                    
                    # Generate recommended actions using analysis module
                    recommended_actions = self.analysis.generate_recommended_actions({
                        'demand_risk': demand_risk,
                        'operational_risk': operational_risk,
                        'market_risk': market_risk,
                        'financial_risk': financial_risk
                    }, scenario_analysis)
                    
                    # Create comprehensive assessment structure
                    assessment = {
                        'seller_id': seller_id,
                        'assessment_id': f'risk_{seller_id}',
                        'assessment_date': datetime.now().strftime('%Y-%m-%d'),
                        'risk_categories': {
                            'demand_risk': demand_risk,
                            'operational_risk': operational_risk,
                            'market_risk': market_risk,
                            'financial_risk': financial_risk
                        },
                        'scenario_analysis': scenario_analysis,
                        'monte_carlo_simulation': monte_carlo_results,
                        'risk_monitoring_metrics': monitoring_metrics,
                        'recommended_actions': recommended_actions
                    }
                    
                    all_assessments[seller_id] = assessment
            else:
                # Use overall analysis for global forecasts
                print("Assessing risks for overall forecast")
                group = demand_df
                
                # Assess demand risk using calculation module
                demand_risk = self.calculations.assess_demand_risk(group, risk_data.get('payments', pd.DataFrame()))
                
                # Assess operational risk using calculation module
                operational_risk = self.calculations.assess_operational_risk(
                    risk_data.get('capacity_forecasts', pd.DataFrame()),
                    risk_data.get('inventory_efficiency', pd.DataFrame())
                )
                
                # Assess market risk using calculation module
                market_risk = self.calculations.assess_market_risk(
                    risk_data.get('cost_analysis', pd.DataFrame()),
                    risk_data.get('payments', pd.DataFrame())
                )
                
                # Assess financial risk using calculation module
                financial_risk = self.calculations.assess_financial_risk(
                    risk_data.get('cost_analysis', pd.DataFrame()),
                    risk_data.get('capacity_forecasts', pd.DataFrame())
                )
                
                # Perform scenario analysis using analysis module
                scenario_analysis = self.analysis.perform_scenario_analysis({
                    'demand_risk': demand_risk,
                    'operational_risk': operational_risk,
                    'market_risk': market_risk,
                    'financial_risk': financial_risk
                })
                
                # Run Monte Carlo simulation using analysis module
                monte_carlo_results = self.analysis.run_monte_carlo_simulation({
                    'demand_risk': demand_risk,
                    'operational_risk': operational_risk,
                    'market_risk': market_risk,
                    'financial_risk': financial_risk
                })
                
                # Create risk monitoring metrics using analysis module
                monitoring_metrics = self.analysis.create_risk_monitoring_metrics({
                    'demand_risk': demand_risk,
                    'operational_risk': operational_risk,
                    'market_risk': market_risk,
                    'financial_risk': financial_risk
                })
                
                # Generate recommended actions using analysis module
                recommended_actions = self.analysis.generate_recommended_actions({
                    'demand_risk': demand_risk,
                    'operational_risk': operational_risk,
                    'market_risk': market_risk,
                    'financial_risk': financial_risk
                }, scenario_analysis)
                
                # Create comprehensive assessment structure
                assessment = {
                    'seller_id': 'overall',
                    'assessment_id': 'risk_overall',
                    'assessment_date': datetime.now().strftime('%Y-%m-%d'),
                    'risk_categories': {
                        'demand_risk': demand_risk,
                        'operational_risk': operational_risk,
                        'market_risk': market_risk,
                        'financial_risk': financial_risk
                    },
                    'scenario_analysis': scenario_analysis,
                    'monte_carlo_simulation': monte_carlo_results,
                    'risk_monitoring_metrics': monitoring_metrics,
                    'recommended_actions': recommended_actions
                }
                
                all_assessments['overall'] = assessment
        
        return all_assessments
    
    def create_visualizations(self, risk_assessments: Dict):
        """Create risk assessment visualizations"""
        print("Creating risk assessment visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Risk Assessment Results', fontsize=16, fontweight='bold')
        
        # Extract data for visualization
        sellers = list(risk_assessments.keys())
        
        # 1. Risk level distribution
        risk_levels = []
        for seller in sellers:
            risk_levels.append(risk_assessments[seller]['risk_categories']['demand_risk']['risk_level'])
        
        risk_counts = pd.Series(risk_levels).value_counts()
        axes[0, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Demand Risk Distribution')
        
        # 2. Risk scores by category
        categories = ['demand_risk', 'operational_risk', 'market_risk', 'financial_risk']
        category_scores = {}
        for category in categories:
            scores = []
            for seller in sellers:
                scores.append(risk_assessments[seller]['risk_categories'][category]['risk_score'])
            category_scores[category] = np.mean(scores)
        
        category_names = [cat.replace('_', ' ').title() for cat in categories]
        category_values = list(category_scores.values())
        
        axes[0, 1].bar(category_names, category_values, alpha=0.8, color='red')
        axes[0, 1].set_title('Average Risk Scores by Category')
        axes[0, 1].set_ylabel('Risk Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Overall risk scores
        overall_scores = []
        for seller in sellers:
            scores = []
            for category in categories:
                scores.append(risk_assessments[seller]['risk_categories'][category]['risk_score'])
            overall_scores.append(np.mean(scores))
        
        axes[0, 2].bar(sellers, overall_scores, alpha=0.8, color='orange')
        axes[0, 2].set_title('Overall Risk Scores by Seller')
        axes[0, 2].set_xlabel('Seller ID')
        axes[0, 2].set_ylabel('Risk Score')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Scenario analysis results
        if sellers:
            scenario_data = risk_assessments[sellers[0]]['scenario_analysis']
            scenarios = list(scenario_data.keys())
            scenario_scores = [scenario_data[scenario]['risk_score'] for scenario in scenarios]
            
            axes[1, 0].bar(scenarios, scenario_scores, alpha=0.8, color='blue')
            axes[1, 0].set_title('Scenario Analysis Results')
            axes[1, 0].set_ylabel('Risk Score')
        
        # 5. Monte Carlo simulation results
        if sellers:
            monte_carlo_data = risk_assessments[sellers[0]]['monte_carlo_simulation']
            mean_risk = monte_carlo_data['mean_risk_score']
            std_risk = monte_carlo_data['std_risk_score']
            
            # Generate sample data for histogram
            sample_risk = np.random.normal(mean_risk, std_risk, 1000)
            axes[1, 1].hist(sample_risk, bins=30, alpha=0.7, color='green')
            axes[1, 1].axvline(mean_risk, color='red', linestyle='--', label=f'Mean: {mean_risk:.3f}')
            axes[1, 1].set_title('Monte Carlo Risk Distribution')
            axes[1, 1].set_xlabel('Risk Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        # 6. Mitigation strategies count
        strategy_counts = []
        for seller in sellers:
            actions = risk_assessments[seller]['recommended_actions']
            total_strategies = (len(actions.get('high_priority', [])) +
                              len(actions.get('medium_priority', [])) +
                              len(actions.get('low_priority', [])))
            strategy_counts.append(total_strategies)
        
        axes[1, 2].bar(sellers, strategy_counts, alpha=0.8, color='purple')
        axes[1, 2].set_title('Mitigation Strategies by Seller')
        axes[1, 2].set_xlabel('Seller ID')
        axes[1, 2].set_ylabel('Number of Strategies')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'risk_assessment_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Risk assessment visualizations created successfully")
    
    def save_outputs(self, risk_assessments: Dict):
        """Save risk assessment outputs"""
        print("Saving risk assessment outputs...")
        
        # Save comprehensive risk assessment results
        assessment_results = []
        for seller_id, assessment in risk_assessments.items():
            result = {
                'seller_id': seller_id,
                'assessment_id': assessment['assessment_id'],
                'assessment_date': assessment['assessment_date'],
                'demand_risk_score': assessment['risk_categories']['demand_risk']['risk_score'],
                'operational_risk_score': assessment['risk_categories']['operational_risk']['risk_score'],
                'market_risk_score': assessment['risk_categories']['market_risk']['risk_score'],
                'financial_risk_score': assessment['risk_categories']['financial_risk']['risk_score'],
                'overall_risk_score': np.mean([
                    assessment['risk_categories']['demand_risk']['risk_score'],
                    assessment['risk_categories']['operational_risk']['risk_score'],
                    assessment['risk_categories']['market_risk']['risk_score'],
                    assessment['risk_categories']['financial_risk']['risk_score']
                ]),
                'risk_level': assessment['risk_categories']['demand_risk']['risk_level']
            }
            assessment_results.append(result)
        
        assessment_df = pd.DataFrame(assessment_results)
        assessment_df.to_csv(os.path.join(self.output_dir, 'risk_assessment_results.csv'), index=False)
        print(f"Saved risk assessment results: {assessment_df.shape}")
        
        # scenario analysis results
        scenario_results = []
        for seller_id, assessment in risk_assessments.items():
            scenarios = assessment['scenario_analysis']
            for scenario, data in scenarios.items():
                result = {
                    'seller_id': seller_id,
                    'scenario': scenario,
                    'risk_score': data['risk_score'],
                    'probability': data['probability'],
                    'impact_level': data['impact_level']
                }
                scenario_results.append(result)
        
        scenario_df = pd.DataFrame(scenario_results)
        scenario_df.to_csv(os.path.join(self.output_dir, 'scenario_analysis_results.csv'), index=False)
        print(f"Saved scenario analysis results: {scenario_df.shape}")
        
        # Monte Carlo simulation results
        monte_carlo_results = []
        for seller_id, assessment in risk_assessments.items():
            mc_data = assessment['monte_carlo_simulation']
            result = {
                'seller_id': seller_id,
                'mean_risk_score': mc_data['mean_risk_score'],
                'std_risk_score': mc_data['std_risk_score'],
                'min_risk_score': mc_data['min_risk_score'],
                'max_risk_score': mc_data['max_risk_score'],
                'percentile_5': mc_data['percentile_5'],
                'percentile_95': mc_data['percentile_95'],
                'var_95': mc_data['var_95']
            }
            monte_carlo_results.append(result)
        
        mc_df = pd.DataFrame(monte_carlo_results)
        mc_df.to_csv(os.path.join(self.output_dir, 'monte_carlo_simulation_results.csv'), index=False)
        print(f"Saved Monte Carlo simulation results: {mc_df.shape}")
        
        # Save recommended actions
        action_results = []
        for seller_id, assessment in risk_assessments.items():
            actions = assessment['recommended_actions']
            for priority, action_list in actions.items():
                for action in action_list:
                    result = {
                        'seller_id': seller_id,
                        'priority': priority,
                        'action': action
                    }
                    action_results.append(result)
        
        action_df = pd.DataFrame(action_results)
        action_df.to_csv(os.path.join(self.output_dir, 'recommended_actions.csv'), index=False)
        print(f"Saved recommended actions: {action_df.shape}")
        
        print("All risk assessment outputs saved successfully")
    
    def print_analysis_summary(self, risk_assessments: Dict):
        """Print risk assessment summary"""
        print("\n" + "="*80)
        print("RISK ASSESSMENT AND MITIGATION SUMMARY")
        print("="*80)
        
        total_assessments = len(risk_assessments)
        
        print(f"\nASSESSMENT OVERVIEW:")
        print("-" * 50)
        print(f"Total risk assessments completed: {total_assessments}")
        print(f"Monte Carlo simulation iterations: {self.config['monte_carlo_iterations']}")
        
        if risk_assessments:
            # Calculate average metrics
            avg_risk_score = np.mean([assessment['risk_categories']['demand_risk']['risk_score'] 
                                    for assessment in risk_assessments.values()])
            
            risk_levels = [assessment['risk_categories']['demand_risk']['risk_level'] 
                          for assessment in risk_assessments.values()]
            risk_level_distribution = pd.Series(risk_levels).value_counts()
            
            print(f"\nRISK ANALYSIS:")
            print("-" * 50)
            print(f"Average overall risk score: {avg_risk_score:.3f}")
            print("Risk level distribution:")
            for level, count in risk_level_distribution.items():
                print(f"  {level.capitalize()} risk: {count} sellers")
            
            # Risk category analysis
            risk_categories = ['demand_risk', 'operational_risk', 'market_risk', 'financial_risk']
            category_scores = {}
            for category in risk_categories:
                category_scores[category] = np.mean([assessment['risk_categories'][category]['risk_score'] 
                                                   for assessment in risk_assessments.values()])
            
            print(f"\nRISK CATEGORY ANALYSIS:")
            print("-" * 50)
            for category, score in category_scores.items():
                print(f"  {category.replace('_', ' ').title()}: {score:.3f}")
            
            # Mitigation strategies
            total_strategies = sum(len(assessment['recommended_actions']['high_priority']) +
                                 len(assessment['recommended_actions']['medium_priority']) +
                                 len(assessment['recommended_actions']['low_priority'])
                                 for assessment in risk_assessments.values())
            
            print(f"\nMITIGATION STRATEGIES:")
            print("-" * 50)
            print(f"Total mitigation strategies generated: {total_strategies}")
        
        print(f"\nOutput files saved in: {self.output_dir}")
        print(f"Visualizations saved in: {self.viz_dir}")
        
        print("\n" + "="*80)

def main():
    print("Starting Risk Assessment and Mitigation Model")
    
    # Initialize risk assessor
    risk_assessor = RiskAssessmentMitigation()
    
    try:
        data = risk_assessor.load_data()
        
        if not data:
            print("No data loaded. Exiting.")
            return
        
        # Prepare risk data
        risk_data = risk_assessor.prepare_risk_data(data)
        
        if not risk_data:
            print("No risk data prepared. Exiting.")
            return
        
        # Create comprehensive risk assessment
        risk_assessments = risk_assessor.create_comprehensive_risk_assessment(risk_data)
        
        if not risk_assessments:
            print("No risk assessments generated. Exiting.")
            return
        
        # Create visualizations
        risk_assessor.create_visualizations(risk_assessments)
        
        # Save outputs
        risk_assessor.save_outputs(risk_assessments)
        
        # Print summary
        risk_assessor.print_analysis_summary(risk_assessments)
        
        print("Risk Assessment and Mitigation Model completed successfully")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 