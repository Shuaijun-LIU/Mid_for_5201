"""
Task 1.3: Cost-Benefit Analysis Module
Contains analysis methods including sensitivity analysis, Monte Carlo simulation,
risk-adjusted metrics calculation, and optimization recommendations.
Execution date: 2025-07-19
Update date: 2025-07-24
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class CostBenefitAnalysis:
    """Cost-benefit analysis module with sensitivity analysis and optimization"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def perform_sensitivity_analysis(self, costs: Dict, benefits: Dict, financial_metrics: Dict) -> Dict:
        """Perform sensitivity analysis on key variables"""
        print("Performing sensitivity analysis...")
        
        sensitivity_results = {}
        
        # Define key variables for sensitivity analysis
        key_variables = {
            'revenue_growth': {
                'base_case': 0.5,
                'pessimistic': 0.3,
                'optimistic': 0.7
            },
            'cost_inflation': {
                'base_case': self.config['inflation_rate'],
                'pessimistic': self.config['inflation_rate'] * 1.5,
                'optimistic': self.config['inflation_rate'] * 0.5
            },
            'discount_rate': {
                'base_case': self.config['recommended_discount_rate'],
                'pessimistic': self.config['recommended_discount_rate'] * 1.3,
                'optimistic': self.config['recommended_discount_rate'] * 0.7
            },
            'capacity_utilization': {
                'base_case': 0.8,
                'pessimistic': 0.6,
                'optimistic': 0.95
            }
        }
        
        # Calculate ROI impact for each variable
        roi_impacts = {}
        base_roi = financial_metrics['roi']['overall']
        
        for variable, scenarios in key_variables.items():
            roi_impact = []
            for scenario in ['pessimistic', 'base_case', 'optimistic']:
                # Simplified impact calculation
                if variable == 'revenue_growth':
                    impact_factor = scenarios[scenario] / scenarios['base_case']
                    adjusted_roi = base_roi * impact_factor
                elif variable == 'cost_inflation':
                    impact_factor = 1 - (scenarios[scenario] - scenarios['base_case'])
                    adjusted_roi = base_roi * impact_factor
                elif variable == 'discount_rate':
                    # NPV impact on ROI
                    impact_factor = 1 - (scenarios[scenario] - scenarios['base_case']) * 0.5
                    adjusted_roi = base_roi * impact_factor
                else:  # capacity_utilization
                    impact_factor = scenarios[scenario] / scenarios['base_case']
                    adjusted_roi = base_roi * impact_factor
                
                roi_impact.append(adjusted_roi - base_roi)
            
            roi_impacts[variable] = roi_impact
        
        # Create sensitivity analysis results
        sensitivity_results = {
            'key_variables': key_variables,
            'roi_impacts': roi_impacts,
            'base_roi': base_roi,
            'most_sensitive_variable': max(roi_impacts.keys(), 
                                         key=lambda x: max(abs(roi_impacts[x][0]), abs(roi_impacts[x][2])))
        }
        
        print("Sensitivity analysis completed")
        
        return sensitivity_results
    
    def run_monte_carlo_simulation(self, costs: Dict, benefits: Dict, financial_metrics: Dict) -> Dict:
        """Run Monte Carlo simulation for risk assessment"""
        print("Running Monte Carlo simulation...")
        
        monte_carlo_results = {}
        
        # Define probability distributions for key variables
        np.random.seed(42)  # For reproducibility
        
        n_iterations = self.config['monte_carlo_iterations']
        
        # Generate random samples for key variables
        revenue_growth_samples = np.random.normal(0.5, 0.15, n_iterations)
        cost_inflation_samples = np.random.normal(self.config['inflation_rate'], 0.01, n_iterations)
        capacity_utilization_samples = np.random.normal(0.8, 0.1, n_iterations)
        
        # Calculate ROI for each iteration
        roi_samples = []
        npv_samples = []
        
        for i in range(n_iterations):
            # Adjust benefits based on revenue growth
            adjusted_benefits = benefits['total_benefits']['total'] * (1 + revenue_growth_samples[i])
            
            # Adjust costs based on inflation
            adjusted_costs = costs['total_costs']['total'] * (1 + cost_inflation_samples[i])
            
            # Adjust for capacity utilization
            utilization_factor = capacity_utilization_samples[i] / 0.8
            adjusted_benefits *= utilization_factor
            adjusted_costs *= utilization_factor
            
            # Calculate ROI
            roi = (adjusted_benefits - adjusted_costs) / adjusted_costs if adjusted_costs > 0 else 0
            roi_samples.append(roi)
            
            # Calculate NPV
            npv = 0
            for year in range(1, self.config['analysis_period'] + 1):
                cash_flow = (benefits['total_benefits'][f'year_{year}'] - costs['total_costs'][f'year_{year}'])
                cash_flow *= (1 + revenue_growth_samples[i]) * utilization_factor
                npv += cash_flow / ((1 + self.config['recommended_discount_rate']) ** year)
            npv_samples.append(npv)
        
        # Calculate statistics
        roi_samples = np.array(roi_samples)
        npv_samples = np.array(npv_samples)
        
        monte_carlo_results = {
            'mean_roi': np.mean(roi_samples),
            'std_roi': np.std(roi_samples),
            'min_roi': np.min(roi_samples),
            'max_roi': np.max(roi_samples),
            'percentile_5_roi': np.percentile(roi_samples, 5),
            'percentile_95_roi': np.percentile(roi_samples, 95),
            'mean_npv': np.mean(npv_samples),
            'std_npv': np.std(npv_samples),
            'probability_positive_npv': np.mean(npv_samples > 0),
            'probability_positive_roi': np.mean(roi_samples > 0),
            'var_95_roi': np.percentile(roi_samples, 5),  # Value at Risk
            'var_95_npv': np.percentile(npv_samples, 5)
        }
        
        print(f"Monte Carlo simulation completed with {n_iterations} iterations")
        
        return monte_carlo_results
    
    def calculate_risk_adjusted_metrics(self, financial_metrics: Dict, sensitivity_results: Dict) -> Dict:
        """Calculate risk-adjusted financial metrics"""
        print("Calculating risk-adjusted metrics...")
        
        risk_adjusted_metrics = {}
        
        # Calculate risk score based on sensitivity analysis
        roi_impacts = sensitivity_results['roi_impacts']
        max_impact = max(max(abs(impacts[0]), abs(impacts[2])) for impacts in roi_impacts.values())
        
        risk_score = 'low'
        if max_impact > 0.3:
            risk_score = 'high'
        elif max_impact > 0.15:
            risk_score = 'medium'
        
        # Calculate risk-adjusted ROI
        base_roi = financial_metrics['roi']['overall']
        risk_adjusted_roi = base_roi * (1 - max_impact * 0.5)
        
        # Calculate risk-adjusted NPV
        base_npv = financial_metrics['npv']['recommended_rate']
        risk_adjusted_npv = base_npv * (1 - max_impact * 0.3)
        
        # Calculate confidence intervals
        confidence_interval_roi = [
            base_roi - max_impact,
            base_roi + max_impact
        ]
        
        confidence_interval_npv = [
            base_npv - abs(base_npv) * max_impact * 0.5,
            base_npv + abs(base_npv) * max_impact * 0.5
        ]
        
        risk_adjusted_metrics = {
            'risk_score': risk_score,
            'risk_adjusted_roi': risk_adjusted_roi,
            'risk_adjusted_npv': risk_adjusted_npv,
            'confidence_interval_roi': confidence_interval_roi,
            'confidence_interval_npv': confidence_interval_npv,
            'max_sensitivity_impact': max_impact,
            'most_sensitive_variable': sensitivity_results['most_sensitive_variable']
        }
        
        print("Risk-adjusted metrics calculated")
        
        return risk_adjusted_metrics
    
    def generate_optimization_recommendations(self, costs: Dict, benefits: Dict, 
                                           financial_metrics: Dict, sensitivity_results: Dict) -> Dict:
        """Generate optimization recommendations based on analysis results"""
        print("Generating optimization recommendations...")
        
        recommendations = {
            'cost_optimization': [],
            'benefit_maximization': [],
            'risk_mitigation': []
        }
        
        # Cost optimization recommendations
        total_costs = costs['total_costs']['total']
        fixed_costs = costs['fixed_costs']['total_fixed']['total']
        variable_costs = costs['variable_costs']['total_variable']['total']
        
        if fixed_costs / total_costs > 0.6:
            recommendations['cost_optimization'].append(
                "Consider flexible lease terms to reduce fixed costs"
            )
            recommendations['cost_optimization'].append(
                "Explore shared warehouse facilities to distribute fixed costs"
            )
        
        if variable_costs / total_costs > 0.7:
            recommendations['cost_optimization'].append(
                "Implement automation to reduce labor costs"
            )
            recommendations['cost_optimization'].append(
                "Optimize energy usage to reduce utilities costs"
            )
        
        # Benefit maximization recommendations
        total_benefits = benefits['total_benefits']['total']
        direct_benefits = sum(benefits['direct_benefits'][key]['total'] 
                            for key in benefits['direct_benefits'])
        indirect_benefits = sum(benefits['indirect_benefits'][key]['total'] 
                              for key in benefits['indirect_benefits'])
        
        if direct_benefits / total_benefits < 0.8:
            recommendations['benefit_maximization'].append(
                "Focus on direct revenue generation through capacity optimization"
            )
        
        if indirect_benefits / total_benefits < 0.2:
            recommendations['benefit_maximization'].append(
                "Invest in customer satisfaction initiatives to capture indirect benefits"
            )
        
        # Risk mitigation recommendations
        risk_score = sensitivity_results.get('risk_score', 'medium')
        most_sensitive = sensitivity_results.get('most_sensitive_variable', 'revenue_growth')
        
        if risk_score == 'high':
            recommendations['risk_mitigation'].append(
                "Implement hedging strategies for key cost drivers"
            )
            recommendations['risk_mitigation'].append(
                "Diversify revenue streams to reduce dependency on single factors"
            )
        
        if most_sensitive == 'revenue_growth':
            recommendations['risk_mitigation'].append(
                "Develop conservative revenue projections and contingency plans"
            )
        elif most_sensitive == 'cost_inflation':
            recommendations['risk_mitigation'].append(
                "Lock in long-term contracts for key cost components"
            )
        elif most_sensitive == 'capacity_utilization':
            recommendations['risk_mitigation'].append(
                "Implement flexible capacity management systems"
            )
        
        # ROI optimization
        current_roi = financial_metrics['roi']['overall']
        if current_roi < 0.2:
            recommendations['cost_optimization'].append(
                "Review and optimize all cost components to improve ROI"
            )
            recommendations['benefit_maximization'].append(
                "Focus on high-impact revenue generation activities"
            )
        
        # NPV optimization
        current_npv = financial_metrics['npv']['recommended_rate']
        if current_npv < 0:
            recommendations['cost_optimization'].append(
                "Reduce initial investment or spread costs over longer period"
            )
            recommendations['benefit_maximization'].append(
                "Accelerate benefit realization timeline"
            )
        
        print(f"Generated {sum(len(recs) for recs in recommendations.values())} optimization recommendations")
        
        return recommendations
    
    def create_sensitivity_visualization(self, sensitivity_results: Dict, output_path: str):
        """Create sensitivity analysis visualization"""
        print("Creating sensitivity analysis visualization...")
        
        # Prepare data for visualization
        variables = list(sensitivity_results['key_variables'].keys())
        roi_impacts = sensitivity_results['roi_impacts']
        
        # Create tornado chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(variables))
        pessimistic_impacts = [roi_impacts[var][0] for var in variables]
        optimistic_impacts = [roi_impacts[var][2] for var in variables]
        
        # Create horizontal bars
        ax.barh(y_pos, pessimistic_impacts, color='red', alpha=0.7, label='Pessimistic')
        ax.barh(y_pos, optimistic_impacts, color='green', alpha=0.7, label='Optimistic')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([var.replace('_', ' ').title() for var in variables])
        ax.set_xlabel('ROI Impact')
        ax.set_title('Sensitivity Analysis - ROI Impact by Variable')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Sensitivity analysis visualization created")
    
    def create_monte_carlo_visualization(self, monte_carlo_results: Dict, output_path: str):
        """Create Monte Carlo simulation visualization"""
        print("Creating Monte Carlo simulation visualization...")
        
        # Generate sample data for histogram
        np.random.seed(42)
        sample_roi = np.random.normal(monte_carlo_results['mean_roi'], 
                                    monte_carlo_results['std_roi'], 1000)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROI distribution
        ax1.hist(sample_roi, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(monte_carlo_results['mean_roi'], color='red', linestyle='--', 
                   label=f"Mean: {monte_carlo_results['mean_roi']:.3f}")
        ax1.axvline(monte_carlo_results['percentile_5_roi'], color='orange', linestyle='--',
                   label=f"5th percentile: {monte_carlo_results['percentile_5_roi']:.3f}")
        ax1.axvline(monte_carlo_results['percentile_95_roi'], color='orange', linestyle='--',
                   label=f"95th percentile: {monte_carlo_results['percentile_95_roi']:.3f}")
        ax1.set_xlabel('ROI')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Monte Carlo ROI Distribution')
        ax1.legend()
        
        # Statistics summary
        stats_text = f"""
        Monte Carlo Simulation Results:
        
        Mean ROI: {monte_carlo_results['mean_roi']:.3f}
        Std ROI: {monte_carlo_results['std_roi']:.3f}
        Min ROI: {monte_carlo_results['min_roi']:.3f}
        Max ROI: {monte_carlo_results['max_roi']:.3f}
        
        P(ROI > 0): {monte_carlo_results['probability_positive_roi']:.1%}
        P(NPV > 0): {monte_carlo_results['probability_positive_npv']:.1%}
        
        VaR (95%): {monte_carlo_results['var_95_roi']:.3f}
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