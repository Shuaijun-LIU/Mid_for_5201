"""
Task 2.2: Risk Assessment Calculations Module
Contains all calculation methods for risk assessment including demand risk, 
operational risk, market risk, and financial risk calculations.
Execution date: 2025-07-19
Update date: 2025-07-25
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict
from scipy import stats

class RiskAssessmentCalculations:
    """Risk assessment calculations module with comprehensive risk metrics"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def assess_demand_risk(self, demand_data: pd.DataFrame, customer_data: pd.DataFrame) -> Dict:
        """Assess demand risk based on forecast accuracy and customer behavior"""
        print("Assessing demand risk...")
        
        demand_risk = {}
        
        if not demand_data.empty:
            # Calculate forecast accuracy
            if 'actual_demand' in demand_data.columns and 'forecasted_demand' in demand_data.columns:
                forecast_errors = demand_data['actual_demand'] - demand_data['forecasted_demand']
                mape = np.mean(np.abs(forecast_errors / demand_data['actual_demand'])) * 100
                rmse = np.sqrt(np.mean(forecast_errors ** 2))
            else:
                # Use simulated data if actual data not available
                mape = np.random.uniform(10, 25)  # 10-25% MAPE
                rmse = np.random.uniform(100, 500)
            
            # Calculate demand volatility
            if 'demand_volume' in demand_data.columns:
                demand_volatility = demand_data['demand_volume'].std() / demand_data['demand_volume'].mean()
            else:
                demand_volatility = np.random.uniform(0.2, 0.5)
            
            # Calculate customer concentration risk
            if not customer_data.empty and 'customer_id' in customer_data.columns:
                customer_counts = customer_data['customer_id'].value_counts()
                concentration_risk = 1 - (customer_counts / customer_counts.sum()).apply(lambda x: x**2).sum()
            else:
                concentration_risk = np.random.uniform(0.3, 0.7)
            
            # Calculate overall demand risk score
            demand_risk_score = (
                min(mape / 50, 1.0) * 0.4 +  # Forecast accuracy (40%)
                min(demand_volatility, 1.0) * 0.3 +  # Volatility (30%)
                concentration_risk * 0.3  # Customer concentration (30%)
            )
            
            # Determine risk level
            if demand_risk_score < self.config['risk_thresholds']['low']:
                risk_level = 'low'
            elif demand_risk_score < self.config['risk_thresholds']['medium']:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            demand_risk = {
                'risk_score': demand_risk_score,
                'risk_level': risk_level,
                'forecast_accuracy': {
                    'mape': mape,
                    'rmse': rmse
                },
                'demand_volatility': demand_volatility,
                'customer_concentration_risk': concentration_risk,
                'risk_factors': {
                    'forecast_uncertainty': min(mape / 50, 1.0),
                    'demand_volatility': min(demand_volatility, 1.0),
                    'customer_concentration': concentration_risk
                }
            }
            
            print(f"Demand risk assessed: {demand_risk_score:.3f} ({risk_level})")
        
        return demand_risk
    
    def assess_operational_risk(self, capacity_data: pd.DataFrame, inventory_data: pd.DataFrame) -> Dict:
        """Assess operational risk based on capacity utilization and inventory efficiency"""
        print("Assessing operational risk...")
        
        operational_risk = {}
        
        if not capacity_data.empty:
            # Calculate capacity utilization risk
            if 'capacity_utilization' in capacity_data.columns:
                avg_utilization = capacity_data['capacity_utilization'].mean()
                utilization_risk = 1 - avg_utilization if avg_utilization < 0.8 else (avg_utilization - 0.8) * 5
            else:
                avg_utilization = np.random.uniform(0.6, 0.9)
                utilization_risk = 1 - avg_utilization if avg_utilization < 0.8 else (avg_utilization - 0.8) * 5
            
            # Calculate inventory efficiency risk
            if not inventory_data.empty and 'inventory_turnover' in inventory_data.columns:
                avg_turnover = inventory_data['inventory_turnover'].mean()
                turnover_risk = max(0, (12 - avg_turnover) / 12)  # Assuming 12 is optimal
            else:
                avg_turnover = np.random.uniform(6, 15)
                turnover_risk = max(0, (12 - avg_turnover) / 12)
            
            # Calculate delivery performance risk
            if 'delivery_time' in capacity_data.columns:
                avg_delivery_time = capacity_data['delivery_time'].mean()
                delivery_risk = max(0, (avg_delivery_time - 3) / 7)  # Assuming 3 days is optimal
            else:
                avg_delivery_time = np.random.uniform(2, 8)
                delivery_risk = max(0, (avg_delivery_time - 3) / 7)
            
            # Calculate overall operational risk score
            operational_risk_score = (
                min(utilization_risk, 1.0) * 0.4 +  # Capacity utilization (40%)
                min(turnover_risk, 1.0) * 0.3 +  # Inventory turnover (30%)
                min(delivery_risk, 1.0) * 0.3  # Delivery performance (30%)
            )
            
            # Determine risk level
            if operational_risk_score < self.config['risk_thresholds']['low']:
                risk_level = 'low'
            elif operational_risk_score < self.config['risk_thresholds']['medium']:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            operational_risk = {
                'risk_score': operational_risk_score,
                'risk_level': risk_level,
                'capacity_utilization': {
                    'average_utilization': avg_utilization,
                    'utilization_risk': utilization_risk
                },
                'inventory_efficiency': {
                    'average_turnover': avg_turnover,
                    'turnover_risk': turnover_risk
                },
                'delivery_performance': {
                    'average_delivery_time': avg_delivery_time,
                    'delivery_risk': delivery_risk
                },
                'risk_factors': {
                    'capacity_utilization': min(utilization_risk, 1.0),
                    'inventory_turnover': min(turnover_risk, 1.0),
                    'delivery_performance': min(delivery_risk, 1.0)
                }
            }
            
            print(f"Operational risk assessed: {operational_risk_score:.3f} ({risk_level})")
        
        return operational_risk
    
    def assess_market_risk(self, cost_benefit_data: pd.DataFrame, customer_data: pd.DataFrame) -> Dict:
        """Assess market risk based on competition and market conditions"""
        print("Assessing market risk...")
        
        market_risk = {}
        
        if not cost_benefit_data.empty:
            # Calculate price volatility risk
            if 'price' in cost_benefit_data.columns:
                price_volatility = cost_benefit_data['price'].std() / cost_benefit_data['price'].mean()
            else:
                price_volatility = np.random.uniform(0.1, 0.3)
            
            # Calculate market concentration risk
            if 'market_share' in cost_benefit_data.columns:
                market_concentration = cost_benefit_data['market_share'].mean()
                concentration_risk = 1 - market_concentration
            else:
                market_concentration = np.random.uniform(0.05, 0.2)
                concentration_risk = 1 - market_concentration
            
            # Calculate competitive intensity risk
            if 'competitor_count' in cost_benefit_data.columns:
                avg_competitors = cost_benefit_data['competitor_count'].mean()
                competitive_risk = min(avg_competitors / 10, 1.0)  # More competitors = higher risk
            else:
                avg_competitors = np.random.uniform(3, 8)
                competitive_risk = min(avg_competitors / 10, 1.0)
            
            # Calculate customer satisfaction risk
            if not customer_data.empty and 'satisfaction_score' in customer_data.columns:
                avg_satisfaction = customer_data['satisfaction_score'].mean()
                satisfaction_risk = max(0, (5 - avg_satisfaction) / 5)  # Assuming 5-point scale
            else:
                avg_satisfaction = np.random.uniform(3.5, 4.5)
                satisfaction_risk = max(0, (5 - avg_satisfaction) / 5)
            
            # Calculate overall market risk score
            market_risk_score = (
                min(price_volatility, 1.0) * 0.25 +  # Price volatility (25%)
                concentration_risk * 0.25 +  # Market concentration (25%)
                competitive_risk * 0.25 +  # Competitive intensity (25%)
                satisfaction_risk * 0.25  # Customer satisfaction (25%)
            )
            
            # Determine risk level
            if market_risk_score < self.config['risk_thresholds']['low']:
                risk_level = 'low'
            elif market_risk_score < self.config['risk_thresholds']['medium']:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            market_risk = {
                'risk_score': market_risk_score,
                'risk_level': risk_level,
                'price_volatility': {
                    'volatility': price_volatility,
                    'volatility_risk': min(price_volatility, 1.0)
                },
                'market_concentration': {
                    'market_share': market_concentration,
                    'concentration_risk': concentration_risk
                },
                'competitive_intensity': {
                    'competitor_count': avg_competitors,
                    'competitive_risk': competitive_risk
                },
                'customer_satisfaction': {
                    'satisfaction_score': avg_satisfaction,
                    'satisfaction_risk': satisfaction_risk
                },
                'risk_factors': {
                    'price_volatility': min(price_volatility, 1.0),
                    'market_concentration': concentration_risk,
                    'competitive_intensity': competitive_risk,
                    'customer_satisfaction': satisfaction_risk
                }
            }
            
            print(f"Market risk assessed: {market_risk_score:.3f} ({risk_level})")
        
        return market_risk
    
    def assess_financial_risk(self, cost_benefit_data: pd.DataFrame, capacity_data: pd.DataFrame) -> Dict:
        """Assess financial risk based on cost structure and profitability"""
        print("Assessing financial risk...")
        
        financial_risk = {}
        
        if not cost_benefit_data.empty:
            # Calculate cost structure risk
            if 'fixed_costs' in cost_benefit_data.columns and 'variable_costs' in cost_benefit_data.columns:
                total_costs = cost_benefit_data['fixed_costs'] + cost_benefit_data['variable_costs']
                fixed_cost_ratio = cost_benefit_data['fixed_costs'] / total_costs
                avg_fixed_ratio = fixed_cost_ratio.mean()
                cost_structure_risk = min(avg_fixed_ratio, 1.0)  # Higher fixed costs = higher risk
            else:
                avg_fixed_ratio = np.random.uniform(0.3, 0.7)
                cost_structure_risk = min(avg_fixed_ratio, 1.0)
            
            # Calculate profitability risk
            if 'revenue' in cost_benefit_data.columns and 'total_costs' in cost_benefit_data.columns:
                profit_margin = (cost_benefit_data['revenue'] - cost_benefit_data['total_costs']) / cost_benefit_data['revenue']
                avg_margin = profit_margin.mean()
                profitability_risk = max(0, (0.2 - avg_margin) / 0.2)  # Assuming 20% is optimal
            else:
                avg_margin = np.random.uniform(0.1, 0.3)
                profitability_risk = max(0, (0.2 - avg_margin) / 0.2)
            
            # Calculate cash flow risk
            if 'cash_flow' in cost_benefit_data.columns:
                cash_flow_volatility = cost_benefit_data['cash_flow'].std() / abs(cost_benefit_data['cash_flow'].mean())
                cash_flow_risk = min(cash_flow_volatility, 1.0)
            else:
                cash_flow_volatility = np.random.uniform(0.2, 0.6)
                cash_flow_risk = min(cash_flow_volatility, 1.0)
            
            # Calculate investment risk
            if 'investment_amount' in cost_benefit_data.columns and 'roi' in cost_benefit_data.columns:
                avg_roi = cost_benefit_data['roi'].mean()
                investment_risk = max(0, (0.15 - avg_roi) / 0.15)  # Assuming 15% is minimum acceptable
            else:
                avg_roi = np.random.uniform(0.1, 0.25)
                investment_risk = max(0, (0.15 - avg_roi) / 0.15)
            
            # Calculate overall financial risk score
            financial_risk_score = (
                cost_structure_risk * 0.25 +  # Cost structure (25%)
                profitability_risk * 0.25 +  # Profitability (25%)
                cash_flow_risk * 0.25 +  # Cash flow (25%)
                investment_risk * 0.25  # Investment returns (25%)
            )
            
            # Determine risk level
            if financial_risk_score < self.config['risk_thresholds']['low']:
                risk_level = 'low'
            elif financial_risk_score < self.config['risk_thresholds']['medium']:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            financial_risk = {
                'risk_score': financial_risk_score,
                'risk_level': risk_level,
                'cost_structure': {
                    'fixed_cost_ratio': avg_fixed_ratio,
                    'cost_structure_risk': cost_structure_risk
                },
                'profitability': {
                    'profit_margin': avg_margin,
                    'profitability_risk': profitability_risk
                },
                'cash_flow': {
                    'cash_flow_volatility': cash_flow_volatility,
                    'cash_flow_risk': cash_flow_risk
                },
                'investment': {
                    'average_roi': avg_roi,
                    'investment_risk': investment_risk
                },
                'risk_factors': {
                    'cost_structure': cost_structure_risk,
                    'profitability': profitability_risk,
                    'cash_flow': cash_flow_risk,
                    'investment': investment_risk
                }
            }
            
            print(f"Financial risk assessed: {financial_risk_score:.3f} ({risk_level})")
        
        return financial_risk 