"""
Task 2.3: Risk Analysis and Mitigation Module
Contains analysis methods including scenario analysis, Monte Carlo simulation,
risk monitoring metrics, and mitigation strategy generation.
Execution date: 2025-07-19
Update date: 2025-07-25
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class RiskAnalysisMitigation:
    """Risk analysis and mitigation module with comprehensive risk management"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def perform_scenario_analysis(self, risk_assessments: Dict) -> Dict:
        """Perform scenario analysis for different risk scenarios"""
        print("Performing scenario analysis...")
        
        scenario_analysis = {}
        
        # Define scenarios
        scenarios = {
            'optimistic': {
                'demand_multiplier': 1.2,
                'cost_multiplier': 0.9,
                'market_multiplier': 1.1,
                'financial_multiplier': 1.15
            },
            'base_case': {
                'demand_multiplier': 1.0,
                'cost_multiplier': 1.0,
                'market_multiplier': 1.0,
                'financial_multiplier': 1.0
            },
            'pessimistic': {
                'demand_multiplier': 0.8,
                'cost_multiplier': 1.2,
                'market_multiplier': 0.9,
                'financial_multiplier': 0.85
            }
        }
        
        for scenario_name, multipliers in scenarios.items():
            # Calculate scenario-adjusted risk scores
            demand_risk = risk_assessments['demand_risk']['risk_score'] * multipliers['demand_multiplier']
            operational_risk = risk_assessments['operational_risk']['risk_score'] * multipliers['cost_multiplier']
            market_risk = risk_assessments['market_risk']['risk_score'] * multipliers['market_multiplier']
            financial_risk = risk_assessments['financial_risk']['risk_score'] * multipliers['financial_multiplier']
            
            # Calculate overall scenario risk score
            scenario_risk_score = np.mean([demand_risk, operational_risk, market_risk, financial_risk])
            
            # Determine impact level
            if scenario_risk_score < self.config['risk_thresholds']['low']:
                impact_level = 'low'
            elif scenario_risk_score < self.config['risk_thresholds']['medium']:
                impact_level = 'medium'
            else:
                impact_level = 'high'
            
            scenario_analysis[scenario_name] = {
                'risk_score': scenario_risk_score,
                'probability': self.config['scenario_probabilities'][scenario_name],
                'impact_level': impact_level,
                'category_scores': {
                    'demand_risk': demand_risk,
                    'operational_risk': operational_risk,
                    'market_risk': market_risk,
                    'financial_risk': financial_risk
                }
            }
        
        print("Scenario analysis completed")
        
        return scenario_analysis
    
    def run_monte_carlo_simulation(self, risk_assessments: Dict) -> Dict:
        """Run Monte Carlo simulation for risk assessment"""
        print("Running Monte Carlo simulation...")
        
        monte_carlo_results = {}
        
        # Define probability distributions for risk factors
        np.random.seed(42)  # For reproducibility
        
        n_iterations = self.config['monte_carlo_iterations']
        
        # Generate random samples for each risk category
        demand_risk_samples = np.random.normal(
            risk_assessments['demand_risk']['risk_score'], 
            risk_assessments['demand_risk']['risk_score'] * 0.2, 
            n_iterations
        )
        
        operational_risk_samples = np.random.normal(
            risk_assessments['operational_risk']['risk_score'],
            risk_assessments['operational_risk']['risk_score'] * 0.2,
            n_iterations
        )
        
        market_risk_samples = np.random.normal(
            risk_assessments['market_risk']['risk_score'],
            risk_assessments['market_risk']['risk_score'] * 0.2,
            n_iterations
        )
        
        financial_risk_samples = np.random.normal(
            risk_assessments['financial_risk']['risk_score'],
            risk_assessments['financial_risk']['risk_score'] * 0.2,
            n_iterations
        )
        
        # Calculate overall risk scores for each iteration
        overall_risk_scores = np.mean([demand_risk_samples, operational_risk_samples, 
                                     market_risk_samples, financial_risk_samples], axis=0)
        
        # Calculate statistics
        monte_carlo_results = {
            'mean_risk_score': np.mean(overall_risk_scores),
            'std_risk_score': np.std(overall_risk_scores),
            'min_risk_score': np.min(overall_risk_scores),
            'max_risk_score': np.max(overall_risk_scores),
            'percentile_5': np.percentile(overall_risk_scores, 5),
            'percentile_25': np.percentile(overall_risk_scores, 25),
            'percentile_75': np.percentile(overall_risk_scores, 75),
            'percentile_95': np.percentile(overall_risk_scores, 95),
            'var_95': np.percentile(overall_risk_scores, 5),  # Value at Risk
            'probability_high_risk': np.mean(overall_risk_scores > self.config['risk_thresholds']['medium']),
            'probability_extreme_risk': np.mean(overall_risk_scores > self.config['risk_thresholds']['high'])
        }
        
        print(f"Monte Carlo simulation completed with {n_iterations} iterations")
        
        return monte_carlo_results
    
    def create_risk_monitoring_metrics(self, risk_assessments: Dict) -> Dict:
        """Create risk monitoring metrics and early warning signals"""
        print("Creating risk monitoring metrics...")
        
        monitoring_metrics = {}
        
        # Calculate key risk indicators (KRIs)
        kris = {}
        
        # Demand risk KRIs
        demand_risk = risk_assessments['demand_risk']
        kris['demand_forecast_accuracy'] = demand_risk['forecast_accuracy']['mape']
        kris['demand_volatility'] = demand_risk['demand_volatility']
        kris['customer_concentration'] = demand_risk['customer_concentration_risk']
        
        # Operational risk KRIs
        operational_risk = risk_assessments['operational_risk']
        kris['capacity_utilization'] = operational_risk['capacity_utilization']['average_utilization']
        kris['inventory_turnover'] = operational_risk['inventory_efficiency']['average_turnover']
        kris['delivery_performance'] = operational_risk['delivery_performance']['average_delivery_time']
        
        # Market risk KRIs
        market_risk = risk_assessments['market_risk']
        kris['price_volatility'] = market_risk['price_volatility']['volatility']
        kris['market_concentration'] = market_risk['market_concentration']['market_share']
        kris['competitive_intensity'] = market_risk['competitive_intensity']['competitor_count']
        
        # Financial risk KRIs
        financial_risk = risk_assessments['financial_risk']
        kris['cost_structure'] = financial_risk['cost_structure']['fixed_cost_ratio']
        kris['profit_margin'] = financial_risk['profitability']['profit_margin']
        kris['cash_flow_volatility'] = financial_risk['cash_flow']['cash_flow_volatility']
        
        # Create early warning signals
        warning_signals = {}
        
        # Demand warning signals
        if kris['demand_forecast_accuracy'] > 20:
            warning_signals['demand_forecast'] = 'High forecast error detected'
        if kris['demand_volatility'] > 0.4:
            warning_signals['demand_volatility'] = 'High demand volatility detected'
        if kris['customer_concentration'] > 0.6:
            warning_signals['customer_concentration'] = 'High customer concentration risk'
        
        # Operational warning signals
        if kris['capacity_utilization'] < 0.7:
            warning_signals['capacity_utilization'] = 'Low capacity utilization'
        if kris['inventory_turnover'] < 8:
            warning_signals['inventory_turnover'] = 'Low inventory turnover'
        if kris['delivery_performance'] > 5:
            warning_signals['delivery_performance'] = 'Poor delivery performance'
        
        # Market warning signals
        if kris['price_volatility'] > 0.25:
            warning_signals['price_volatility'] = 'High price volatility'
        if kris['market_concentration'] < 0.1:
            warning_signals['market_concentration'] = 'Low market share'
        if kris['competitive_intensity'] > 6:
            warning_signals['competitive_intensity'] = 'High competitive intensity'
        
        # Financial warning signals
        if kris['cost_structure'] > 0.6:
            warning_signals['cost_structure'] = 'High fixed cost ratio'
        if kris['profit_margin'] < 0.15:
            warning_signals['profit_margin'] = 'Low profit margin'
        if kris['cash_flow_volatility'] > 0.4:
            warning_signals['cash_flow'] = 'High cash flow volatility'
        
        monitoring_metrics = {
            'key_risk_indicators': kris,
            'early_warning_signals': warning_signals,
            'risk_thresholds': self.config['risk_thresholds'],
            'monitoring_frequency': 'daily',
            'alert_threshold': 0.7
        }
        
        print("Risk monitoring metrics created")
        
        return monitoring_metrics
    
    def generate_recommended_actions(self, risk_assessments: Dict, scenario_analysis: Dict) -> Dict:
        """Generate recommended actions for risk mitigation"""
        print("Generating recommended actions...")
        
        recommended_actions = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        # Demand risk mitigation actions
        demand_risk = risk_assessments['demand_risk']
        if demand_risk['risk_score'] > 0.6:
            recommended_actions['high_priority'].append(
                "Implement advanced demand forecasting models with machine learning"
            )
            recommended_actions['high_priority'].append(
                "Diversify customer base to reduce concentration risk"
            )
        elif demand_risk['risk_score'] > 0.4:
            recommended_actions['medium_priority'].append(
                "Improve demand forecasting accuracy through data quality enhancement"
            )
            recommended_actions['medium_priority'].append(
                "Develop customer retention strategies"
            )
        else:
            recommended_actions['low_priority'].append(
                "Monitor demand patterns for early warning signals"
            )
        
        # Operational risk mitigation actions
        operational_risk = risk_assessments['operational_risk']
        if operational_risk['risk_score'] > 0.6:
            recommended_actions['high_priority'].append(
                "Optimize capacity utilization through flexible scheduling"
            )
            recommended_actions['high_priority'].append(
                "Implement inventory optimization systems"
            )
            recommended_actions['high_priority'].append(
                "Improve delivery performance through route optimization"
            )
        elif operational_risk['risk_score'] > 0.4:
            recommended_actions['medium_priority'].append(
                "Review and optimize operational processes"
            )
            recommended_actions['medium_priority'].append(
                "Implement performance monitoring dashboards"
            )
        else:
            recommended_actions['low_priority'].append(
                "Maintain current operational efficiency levels"
            )
        
        # Market risk mitigation actions
        market_risk = risk_assessments['market_risk']
        if market_risk['risk_score'] > 0.6:
            recommended_actions['high_priority'].append(
                "Develop competitive pricing strategies"
            )
            recommended_actions['high_priority'].append(
                "Enhance market positioning and differentiation"
            )
            recommended_actions['high_priority'].append(
                "Improve customer satisfaction through service enhancement"
            )
        elif market_risk['risk_score'] > 0.4:
            recommended_actions['medium_priority'].append(
                "Monitor market trends and competitor activities"
            )
            recommended_actions['medium_priority'].append(
                "Develop customer feedback mechanisms"
            )
        else:
            recommended_actions['low_priority'].append(
                "Continue current market strategies"
            )
        
        # Financial risk mitigation actions
        financial_risk = risk_assessments['financial_risk']
        if financial_risk['risk_score'] > 0.6:
            recommended_actions['high_priority'].append(
                "Optimize cost structure through process improvement"
            )
            recommended_actions['high_priority'].append(
                "Implement cash flow management strategies"
            )
            recommended_actions['high_priority'].append(
                "Review and optimize pricing strategies"
            )
        elif financial_risk['risk_score'] > 0.4:
            recommended_actions['medium_priority'].append(
                "Monitor financial performance indicators"
            )
            recommended_actions['medium_priority'].append(
                "Develop cost control measures"
            )
        else:
            recommended_actions['low_priority'].append(
                "Maintain current financial management practices"
            )
        
        # Scenario-based actions
        pessimistic_scenario = scenario_analysis.get('pessimistic', {})
        if pessimistic_scenario.get('risk_score', 0) > 0.7:
            recommended_actions['high_priority'].append(
                "Develop contingency plans for worst-case scenarios"
            )
            recommended_actions['high_priority'].append(
                "Establish emergency response procedures"
            )
        
        print(f"Generated {sum(len(actions) for actions in recommended_actions.values())} recommended actions")
        
        return recommended_actions
    
    def create_scenario_visualization(self, scenario_analysis: Dict, output_path: str):
        """Create scenario analysis visualization"""
        print("Creating scenario analysis visualization...")
        
        # Prepare data for visualization
        scenarios = list(scenario_analysis.keys())
        risk_scores = [scenario_analysis[scenario]['risk_score'] for scenario in scenarios]
        probabilities = [scenario_analysis[scenario]['probability'] for scenario in scenarios]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scenario risk scores
        colors = ['green', 'blue', 'red']
        bars = ax1.bar(scenarios, risk_scores, color=colors, alpha=0.7)
        ax1.set_title('Risk Scores by Scenario')
        ax1.set_ylabel('Risk Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, risk_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Scenario probabilities
        ax2.pie(probabilities, labels=scenarios, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Scenario Probabilities')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Scenario analysis visualization created")
    
    def create_monte_carlo_visualization(self, monte_carlo_results: Dict, output_path: str):
        """Create Monte Carlo simulation visualization"""
        print("Creating Monte Carlo simulation visualization...")
        
        # Generate sample data for histogram
        np.random.seed(42)
        sample_risk = np.random.normal(monte_carlo_results['mean_risk_score'], 
                                     monte_carlo_results['std_risk_score'], 1000)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Risk distribution
        ax1.hist(sample_risk, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(monte_carlo_results['mean_risk_score'], color='red', linestyle='--', 
                   label=f"Mean: {monte_carlo_results['mean_risk_score']:.3f}")
        ax1.axvline(monte_carlo_results['percentile_5'], color='orange', linestyle='--',
                   label=f"5th percentile: {monte_carlo_results['percentile_5']:.3f}")
        ax1.axvline(monte_carlo_results['percentile_95'], color='orange', linestyle='--',
                   label=f"95th percentile: {monte_carlo_results['percentile_95']:.3f}")
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Monte Carlo Risk Distribution')
        ax1.legend()
        
        # Statistics summary
        stats_text = f"""
        Monte Carlo Simulation Results:
        
        Mean Risk Score: {monte_carlo_results['mean_risk_score']:.3f}
        Std Risk Score: {monte_carlo_results['std_risk_score']:.3f}
        Min Risk Score: {monte_carlo_results['min_risk_score']:.3f}
        Max Risk Score: {monte_carlo_results['max_risk_score']:.3f}
        
        P(High Risk): {monte_carlo_results['probability_high_risk']:.1%}
        P(Extreme Risk): {monte_carlo_results['probability_extreme_risk']:.1%}
        
        VaR (95%): {monte_carlo_results['var_95']:.3f}
        """
        
        ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Simulation Statistics')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Monte Carlo simulation visualization created") 