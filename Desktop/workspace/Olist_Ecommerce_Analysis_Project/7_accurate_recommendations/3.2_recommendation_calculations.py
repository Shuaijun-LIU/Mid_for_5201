"""
Task 3.2: Recommendation Calculations Module
Contains calculation methods for multi-dimensional analysis and comprehensive 
scoring for precision recommendations.
Execution date: 2025-07-19
Update date: 2025-07-26
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict
from scipy import stats

class RecommendationCalculations:
    """Recommendation calculations module with multi-dimensional analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def integrate_multi_dimensional_analysis(self, recommendation_data: Dict[str, pd.DataFrame]) -> Dict:
        """Integrate multi-dimensional analysis for recommendation generation"""
        print("Integrating multi-dimensional analysis...")
        
        integrated_analysis = {}
        
        # Capacity analysis
        if 'capacity_forecasts' in recommendation_data:
            capacity_df = recommendation_data['capacity_forecasts']
            integrated_analysis['capacity_analysis'] = {
                'current_capacity': capacity_df.get('current_capacity', pd.Series([1000])).mean(),
                'projected_demand': capacity_df.get('projected_demand', pd.Series([1200])).mean(),
                'capacity_gap': capacity_df.get('projected_demand', pd.Series([1200])).mean() - capacity_df.get('current_capacity', pd.Series([1000])).mean(),
                'utilization_rate': capacity_df.get('utilization_rate', pd.Series([0.75])).mean(),
                'growth_rate': capacity_df.get('growth_rate', pd.Series([0.15])).mean()
            }
        
        # Cost analysis
        if 'cost_analysis' in recommendation_data:
            cost_df = recommendation_data['cost_analysis']
            integrated_analysis['cost_analysis'] = {
                'total_costs': cost_df.get('total_costs', pd.Series([50000])).mean(),
                'fixed_costs': cost_df.get('fixed_costs', pd.Series([30000])).mean(),
                'variable_costs': cost_df.get('variable_costs', pd.Series([20000])).mean(),
                'cost_per_unit': cost_df.get('cost_per_unit', pd.Series([25])).mean(),
                'cost_efficiency': cost_df.get('cost_efficiency', pd.Series([0.8])).mean()
            }
        
        # Seller analysis
        if 'seller_analysis' in recommendation_data:
            seller_df = recommendation_data['seller_analysis']
            integrated_analysis['seller_analysis'] = {
                'seller_performance': seller_df.get('performance_score', pd.Series([0.75])).mean(),
                'market_position': seller_df.get('market_position', pd.Series([0.6])).mean(),
                'customer_satisfaction': seller_df.get('customer_satisfaction', pd.Series([4.2])).mean(),
                'operational_efficiency': seller_df.get('operational_efficiency', pd.Series([0.8])).mean()
            }
        
        # Inventory analysis
        if 'inventory_efficiency' in recommendation_data:
            inventory_df = recommendation_data['inventory_efficiency']
            integrated_analysis['inventory_analysis'] = {
                'inventory_turnover': inventory_df.get('inventory_turnover', pd.Series([8])).mean(),
                'holding_costs': inventory_df.get('holding_costs', pd.Series([5000])).mean(),
                'stockout_rate': inventory_df.get('stockout_rate', pd.Series([0.05])).mean(),
                'inventory_accuracy': inventory_df.get('inventory_accuracy', pd.Series([0.95])).mean()
            }
        
        # Financial analysis
        if 'payments' in recommendation_data:
            payment_df = recommendation_data['payments']
            integrated_analysis['financial_analysis'] = {
                'total_revenue': payment_df.get('payment_value', pd.Series([100000])).sum(),
                'average_order_value': payment_df.get('payment_value', pd.Series([100])).mean(),
                'payment_efficiency': payment_df.get('payment_efficiency', pd.Series([0.9])).mean(),
                'cash_flow': payment_df.get('cash_flow', pd.Series([15000])).mean()
            }
        
        print("Multi-dimensional analysis integrated")
        
        return integrated_analysis
    
    def calculate_comprehensive_scores(self, integrated_analysis: Dict) -> Dict:
        """Calculate comprehensive scores for different recommendation types"""
        print("Calculating comprehensive scores...")
        
        comprehensive_scores = {}
        
        # Capacity expansion score
        if 'capacity_analysis' in integrated_analysis:
            capacity_analysis = integrated_analysis['capacity_analysis']
            capacity_gap = capacity_analysis['capacity_gap']
            utilization_rate = capacity_analysis['utilization_rate']
            growth_rate = capacity_analysis['growth_rate']
            
            capacity_expansion_score = (
                min(capacity_gap / 500, 1.0) * 0.4 +  # Capacity gap (40%)
                (1 - utilization_rate) * 0.3 +  # Low utilization penalty (30%)
                min(growth_rate, 1.0) * 0.3  # Growth potential (30%)
            )
            
            comprehensive_scores['capacity_expansion'] = {
                'score': capacity_expansion_score,
                'priority': 'high' if capacity_expansion_score > 0.7 else 'medium' if capacity_expansion_score > 0.4 else 'low',
                'factors': {
                    'capacity_gap': min(capacity_gap / 500, 1.0),
                    'utilization_rate': utilization_rate,
                    'growth_rate': min(growth_rate, 1.0)
                }
            }
        
        # Cost optimization score
        if 'cost_analysis' in integrated_analysis:
            cost_analysis = integrated_analysis['cost_analysis']
            cost_per_unit = cost_analysis['cost_per_unit']
            cost_efficiency = cost_analysis['cost_efficiency']
            fixed_cost_ratio = cost_analysis['fixed_costs'] / cost_analysis['total_costs']
            
            cost_optimization_score = (
                (1 - cost_efficiency) * 0.4 +  # Cost efficiency gap (40%)
                min(fixed_cost_ratio, 1.0) * 0.3 +  # High fixed costs (30%)
                (1 - min(cost_per_unit / 30, 1.0)) * 0.3  # High unit costs (30%)
            )
            
            comprehensive_scores['cost_optimization'] = {
                'score': cost_optimization_score,
                'priority': 'high' if cost_optimization_score > 0.7 else 'medium' if cost_optimization_score > 0.4 else 'low',
                'factors': {
                    'cost_efficiency': cost_efficiency,
                    'fixed_cost_ratio': fixed_cost_ratio,
                    'cost_per_unit': cost_per_unit
                }
            }
        
        # Efficiency improvement score
        if 'seller_analysis' in integrated_analysis and 'inventory_analysis' in integrated_analysis:
            seller_analysis = integrated_analysis['seller_analysis']
            inventory_analysis = integrated_analysis['inventory_analysis']
            
            operational_efficiency = seller_analysis['operational_efficiency']
            inventory_turnover = inventory_analysis['inventory_turnover']
            inventory_accuracy = inventory_analysis['inventory_accuracy']
            
            efficiency_improvement_score = (
                (1 - operational_efficiency) * 0.4 +  # Operational efficiency gap (40%)
                max(0, (12 - inventory_turnover) / 12) * 0.3 +  # Low inventory turnover (30%)
                (1 - inventory_accuracy) * 0.3  # Low inventory accuracy (30%)
            )
            
            comprehensive_scores['efficiency_improvement'] = {
                'score': efficiency_improvement_score,
                'priority': 'high' if efficiency_improvement_score > 0.7 else 'medium' if efficiency_improvement_score > 0.4 else 'low',
                'factors': {
                    'operational_efficiency': operational_efficiency,
                    'inventory_turnover': inventory_turnover,
                    'inventory_accuracy': inventory_accuracy
                }
            }
        
        # Technology upgrade score
        if 'seller_analysis' in integrated_analysis:
            seller_analysis = integrated_analysis['seller_analysis']
            seller_performance = seller_analysis['seller_performance']
            customer_satisfaction = seller_analysis['customer_satisfaction']
            
            technology_upgrade_score = (
                (1 - seller_performance) * 0.5 +  # Performance gap (50%)
                max(0, (5 - customer_satisfaction) / 5) * 0.5  # Customer satisfaction gap (50%)
            )
            
            comprehensive_scores['technology_upgrade'] = {
                'score': technology_upgrade_score,
                'priority': 'high' if technology_upgrade_score > 0.7 else 'medium' if technology_upgrade_score > 0.4 else 'low',
                'factors': {
                    'seller_performance': seller_performance,
                    'customer_satisfaction': customer_satisfaction
                }
            }
        
        # Process optimization score
        if 'inventory_analysis' in integrated_analysis and 'financial_analysis' in integrated_analysis:
            inventory_analysis = integrated_analysis['inventory_analysis']
            financial_analysis = integrated_analysis['financial_analysis']
            
            stockout_rate = inventory_analysis['stockout_rate']
            holding_costs = inventory_analysis['holding_costs']
            payment_efficiency = financial_analysis['payment_efficiency']
            
            process_optimization_score = (
                stockout_rate * 0.4 +  # High stockout rate (40%)
                min(holding_costs / 10000, 1.0) * 0.3 +  # High holding costs (30%)
                (1 - payment_efficiency) * 0.3  # Low payment efficiency (30%)
            )
            
            comprehensive_scores['process_optimization'] = {
                'score': process_optimization_score,
                'priority': 'high' if process_optimization_score > 0.7 else 'medium' if process_optimization_score > 0.4 else 'low',
                'factors': {
                    'stockout_rate': stockout_rate,
                    'holding_costs': holding_costs,
                    'payment_efficiency': payment_efficiency
                }
            }
        
        print("Comprehensive scores calculated")
        
        return comprehensive_scores
    
    def calculate_roi_estimates(self, integrated_analysis: Dict, recommendation_type: str) -> Dict:
        """Calculate ROI estimates for different recommendation types"""
        print(f"Calculating ROI estimates for {recommendation_type}...")
        
        roi_estimates = {}
        
        if recommendation_type == 'capacity_expansion':
            if 'capacity_analysis' in integrated_analysis:
                capacity_gap = integrated_analysis['capacity_analysis']['capacity_gap']
                growth_rate = integrated_analysis['capacity_analysis']['growth_rate']
                
                # ROI calculation for capacity expansion
                investment_cost = capacity_gap * 100  # $100 per unit capacity
                annual_revenue_increase = capacity_gap * 150 * growth_rate  # $150 per unit revenue
                annual_cost_increase = capacity_gap * 50  # $50 per unit cost
                annual_profit_increase = annual_revenue_increase - annual_cost_increase
                
                roi = annual_profit_increase / investment_cost
                
                roi_estimates = {
                    'roi': roi,
                    'investment_cost': investment_cost,
                    'annual_revenue_increase': annual_revenue_increase,
                    'annual_cost_increase': annual_cost_increase,
                    'annual_profit_increase': annual_profit_increase,
                    'payback_period': investment_cost / annual_profit_increase if annual_profit_increase > 0 else float('inf')
                }
        
        elif recommendation_type == 'cost_optimization':
            if 'cost_analysis' in integrated_analysis:
                total_costs = integrated_analysis['cost_analysis']['total_costs']
                cost_efficiency = integrated_analysis['cost_analysis']['cost_efficiency']
                
                # ROI calculation for cost optimization
                optimization_investment = total_costs * 0.1  # 10% of current costs
                cost_savings = total_costs * (1 - cost_efficiency) * 0.3  # 30% of inefficiency
                
                roi = cost_savings / optimization_investment
                
                roi_estimates = {
                    'roi': roi,
                    'investment_cost': optimization_investment,
                    'annual_cost_savings': cost_savings,
                    'payback_period': optimization_investment / cost_savings if cost_savings > 0 else float('inf')
                }
        
        elif recommendation_type == 'efficiency_improvement':
            if 'seller_analysis' in integrated_analysis and 'inventory_analysis' in integrated_analysis:
                operational_efficiency = integrated_analysis['seller_analysis']['operational_efficiency']
                inventory_turnover = integrated_analysis['inventory_analysis']['inventory_turnover']
                
                # ROI calculation for efficiency improvement
                improvement_investment = 25000  # Fixed investment for efficiency tools
                efficiency_gains = (1 - operational_efficiency) * 50000  # Revenue from efficiency
                inventory_savings = max(0, (12 - inventory_turnover)) * 2000  # Savings from better turnover
                
                total_benefits = efficiency_gains + inventory_savings
                roi = total_benefits / improvement_investment
                
                roi_estimates = {
                    'roi': roi,
                    'investment_cost': improvement_investment,
                    'efficiency_gains': efficiency_gains,
                    'inventory_savings': inventory_savings,
                    'total_benefits': total_benefits,
                    'payback_period': improvement_investment / total_benefits if total_benefits > 0 else float('inf')
                }
        
        elif recommendation_type == 'technology_upgrade':
            if 'seller_analysis' in integrated_analysis:
                seller_performance = integrated_analysis['seller_analysis']['seller_performance']
                customer_satisfaction = integrated_analysis['seller_analysis']['customer_satisfaction']
                
                # ROI calculation for technology upgrade
                tech_investment = 50000  # Technology upgrade investment
                performance_improvement = (1 - seller_performance) * 40000  # Revenue from better performance
                satisfaction_improvement = max(0, (5 - customer_satisfaction)) * 10000  # Revenue from better satisfaction
                
                total_benefits = performance_improvement + satisfaction_improvement
                roi = total_benefits / tech_investment
                
                roi_estimates = {
                    'roi': roi,
                    'investment_cost': tech_investment,
                    'performance_improvement': performance_improvement,
                    'satisfaction_improvement': satisfaction_improvement,
                    'total_benefits': total_benefits,
                    'payback_period': tech_investment / total_benefits if total_benefits > 0 else float('inf')
                }
        
        elif recommendation_type == 'process_optimization':
            if 'inventory_analysis' in integrated_analysis and 'financial_analysis' in integrated_analysis:
                stockout_rate = integrated_analysis['inventory_analysis']['stockout_rate']
                holding_costs = integrated_analysis['inventory_analysis']['holding_costs']
                payment_efficiency = integrated_analysis['financial_analysis']['payment_efficiency']
                
                # ROI calculation for process optimization
                process_investment = 30000  # process optimization investment
                stockout_reduction = stockout_rate * 30000  # revenue from reduced stockout
                holding_cost_reduction = holding_costs * 0.2  # 20% reduction in holding costs
                payment_improvement = (1 - payment_efficiency) * 15000  # revenue from better payment
                
                total_benefits = stockout_reduction + holding_cost_reduction + payment_improvement
                roi = total_benefits / process_investment
                
                roi_estimates = {
                    'roi': roi,
                    'investment_cost': process_investment,
                    'stockout_reduction': stockout_reduction,
                    'holding_cost_reduction': holding_cost_reduction,
                    'payment_improvement': payment_improvement,
                    'total_benefits': total_benefits,
                    'payback_period': process_investment / total_benefits if total_benefits > 0 else float('inf')
                }
        
        print(f"ROI estimates calculated for {recommendation_type}")
        
        return roi_estimates 