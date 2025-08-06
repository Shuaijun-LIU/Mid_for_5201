"""
Task 3: Precision Recommendation Generator
Based on Week6-Task 3 capacity prediction results, Week4 cost data, and Week5 seller analysis,
generate precise warehouse recommendations including implementation plans, expected benefits,
monitoring metrics and feasibility assessment to provide actionable warehouse strategy decisions.
Execution date: 2025-07-19
Update date: 2025-07-26
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

class PrecisionRecommendationGenerator:
    """Precision recommendation generator with comprehensive analysis and planning"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        
        # Create directories
        for dir_path in [output_dir, self.viz_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Recommendation parameters
        self.config = {
            'analysis_period': 3,  # years
            'confidence_level': 0.95,
            'recommendation_types': [
                'capacity_expansion',
                'cost_optimization',
                'efficiency_improvement',
                'technology_upgrade',
                'process_optimization'
            ],
            'priority_weights': {
                'roi': 0.4,
                'feasibility': 0.3,
                'risk_level': 0.2,
                'implementation_time': 0.1
            }
        }
        
        # Initialize calculation and analysis modules
        # Dynamic import for modules with numeric prefixes
        spec_calc = importlib.util.spec_from_file_location("recommendation_calculations", "3.2_recommendation_calculations.py")
        module_calc = importlib.util.module_from_spec(spec_calc)
        spec_calc.loader.exec_module(module_calc)
        
        spec_analysis = importlib.util.spec_from_file_location("recommendation_analysis", "3.3_recommendation_analysis.py")
        module_analysis = importlib.util.module_from_spec(spec_analysis)
        spec_analysis.loader.exec_module(module_analysis)
        
        self.calculations = module_calc.RecommendationCalculations(self.config)
        self.analysis = module_analysis.RecommendationAnalysis(self.config)
        
        # Data storage
        self.data = {}
        self.recommendations = {}
        self.implementation_plans = {}
        self.monitoring_metrics = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Task 1-5 outputs"""
        print("Loading data from Task 1-5 outputs...")
        
        data = {}
        
        # Load Task 1-5 outputs (Week6)
        task_outputs = [
            'week6_time_series_features_matrix.csv',  
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
    
    def prepare_recommendation_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data for recommendation generation"""
        print("Preparing recommendation data...")
        
        recommendation_data = {}
        
        # Process capacity forecasts
        if 'capacity_forecasts' in data:
            capacity_df = data['capacity_forecasts'].copy()
            recommendation_data['capacity_forecasts'] = capacity_df
            print(f"Processed capacity forecasts: {capacity_df.shape}")
        
        # Process cost analysis
        if 'capacity_cost_analysis' in data:
            cost_df = data['capacity_cost_analysis'].copy()
            recommendation_data['cost_analysis'] = cost_df
            print(f"Processed cost analysis: {cost_df.shape}")
        
        # Process seller analysis
        if 'seller_summary_analysis' in data:
            seller_df = data['seller_summary_analysis'].copy()
            recommendation_data['seller_analysis'] = seller_df
            print(f"Processed seller analysis: {seller_df.shape}")
        
        # Process inventory efficiency
        if 'inventory_efficiency_metrics' in data:
            inventory_df = data['inventory_efficiency_metrics'].copy()
            recommendation_data['inventory_efficiency'] = inventory_df
            print(f"Processed inventory efficiency: {inventory_df.shape}")
        
        # Process payment data
        if 'olist_order_payments_dataset' in data:
            payment_df = data['olist_order_payments_dataset'].copy()
            recommendation_data['payments'] = payment_df
            print(f"Processed payments: {payment_df.shape}")
        
        return recommendation_data
    
    def generate_precision_recommendations(self, recommendation_data: Dict[str, pd.DataFrame]) -> Dict:
        """Generate precision recommendations with comprehensive analysis"""
        print("Generating precision recommendations...")
        
        all_recommendations = {}
        
        if 'capacity_forecasts' in recommendation_data:
            capacity_df = recommendation_data['capacity_forecasts']
            
            # Group by seller for analysis
            for seller_id, group in capacity_df.groupby('seller_id'):
                print(f"Generating recommendations for seller {seller_id}")
                
                # Integrate multi-dimensional analysis using calculation module
                integrated_analysis = self.calculations.integrate_multi_dimensional_analysis(recommendation_data)
                
                # Calculate comprehensive scores using calculation module
                comprehensive_scores = self.calculations.calculate_comprehensive_scores(integrated_analysis)
                
                # Generate recommendations for each type
                seller_recommendations = {}
                
                for rec_type in self.config['recommendation_types']:
                    # generate implementation plan using analysis module
                    implementation_plan = self.analysis.generate_implementation_plans(seller_id, rec_type)
                    
                    # predict expected benefits using analysis module
                    expected_benefits = self.analysis.predict_expected_benefits(seller_id, rec_type)
                    
                    # design monitoring metrics using analysis module
                    monitoring_metrics = self.analysis.design_monitoring_metrics(seller_id, rec_type)
                    
                    # assess feasibility using analysis module
                    feasibility_assessment = self.analysis.assess_feasibility(seller_id, rec_type, implementation_plan)
                    
                    # calculate priority score
                    priority_score = (
                        expected_benefits['roi'] * self.config['priority_weights']['roi'] +
                        feasibility_assessment['feasibility_score'] * self.config['priority_weights']['feasibility'] +
                        (1 - feasibility_assessment['risk_level']) * self.config['priority_weights']['risk_level'] +
                        (1 - implementation_plan['implementation_time'] / 12) * self.config['priority_weights']['implementation_time']
                    )
                    
                    # create recommendation structure
                    recommendation = {
                        'seller_id': seller_id,
                        'recommendation_id': f'rec_{seller_id}_{rec_type}',
                        'recommendation_type': rec_type,
                        'priority_score': priority_score,
                        'expected_roi': expected_benefits['roi'],
                        'feasibility_score': feasibility_assessment['feasibility_score'],
                        'risk_level': feasibility_assessment['risk_level'],
                        'implementation_time': implementation_plan['implementation_time'],
                        'implementation_plan': implementation_plan,
                        'expected_benefits': expected_benefits,
                        'monitoring_metrics': monitoring_metrics,
                        'feasibility_assessment': feasibility_assessment,
                        'generation_date': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    seller_recommendations[rec_type] = recommendation
                
                all_recommendations[seller_id] = seller_recommendations
        
        return all_recommendations
    
    def create_visualizations(self, recommendations: Dict):
        """Create recommendation visualizations"""
        print("Creating recommendation visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Precision Recommendation Results', fontsize=16, fontweight='bold')
        
        # Extract data for visualization
        sellers = list(recommendations.keys())
        
        # 1. Recommendation types distribution
        rec_types = []
        for seller in sellers:
            for rec_type in recommendations[seller].keys():
                rec_types.append(rec_type)
        
        type_counts = pd.Series(rec_types).value_counts()
        axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Recommendation Types Distribution')
        
        # 2. Priority scores by seller
        priority_scores = []
        seller_labels = []
        for seller in sellers:
            for rec_type, rec in recommendations[seller].items():
                priority_scores.append(rec['priority_score'])
                seller_labels.append(f"{seller}_{rec_type}")
        
        axes[0, 1].bar(range(len(priority_scores)), priority_scores, alpha=0.8, color='green')
        axes[0, 1].set_title('Priority Scores by Recommendation')
        axes[0, 1].set_ylabel('Priority Score')
        axes[0, 1].set_xticks(range(len(seller_labels)))
        axes[0, 1].set_xticklabels(seller_labels, rotation=45, ha='right')
        
        # 3. Expected ROI comparison
        roi_values = []
        for seller in sellers:
            for rec_type, rec in recommendations[seller].items():
                roi_values.append(rec['expected_roi'])
        
        axes[0, 2].bar(range(len(roi_values)), roi_values, alpha=0.8, color='blue')
        axes[0, 2].set_title('Expected ROI by Recommendation')
        axes[0, 2].set_ylabel('Expected ROI')
        axes[0, 2].set_xticks(range(len(seller_labels)))
        axes[0, 2].set_xticklabels(seller_labels, rotation=45, ha='right')
        
        # 4. Feasibility scores
        feasibility_scores = []
        for seller in sellers:
            for rec_type, rec in recommendations[seller].items():
                feasibility_scores.append(rec['feasibility_score'])
        
        axes[1, 0].bar(range(len(feasibility_scores)), feasibility_scores, alpha=0.8, color='orange')
        axes[1, 0].set_title('Feasibility Scores by Recommendation')
        axes[1, 0].set_ylabel('Feasibility Score')
        axes[1, 0].set_xticks(range(len(seller_labels)))
        axes[1, 0].set_xticklabels(seller_labels, rotation=45, ha='right')
        
        # 5. Risk level distribution
        risk_levels = []
        for seller in sellers:
            for rec_type, rec in recommendations[seller].items():
                risk_levels.append(rec['risk_level'])
        
        risk_counts = pd.Series(risk_levels).value_counts()
        axes[1, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Risk Level Distribution')
        
        # 6. Implementation time comparison
        impl_times = []
        for seller in sellers:
            for rec_type, rec in recommendations[seller].items():
                impl_times.append(rec['implementation_time'])
        
        axes[1, 2].bar(range(len(impl_times)), impl_times, alpha=0.8, color='purple')
        axes[1, 2].set_title('Implementation Time by Recommendation')
        axes[1, 2].set_ylabel('Implementation Time (months)')
        axes[1, 2].set_xticks(range(len(seller_labels)))
        axes[1, 2].set_xticklabels(seller_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'precision_recommendations.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Precision recommendation visualizations created successfully")
    
    def save_outputs(self, recommendations: Dict):
        """Save precision recommendation outputs"""
        print("Saving precision recommendation outputs...")
        
        # Save comprehensive recommendation results
        recommendation_results = []
        for seller_id, seller_recs in recommendations.items():
            for rec_type, rec in seller_recs.items():
                result = {
                    'seller_id': seller_id,
                    'recommendation_id': rec['recommendation_id'],
                    'recommendation_type': rec['recommendation_type'],
                    'priority_score': rec['priority_score'],
                    'expected_roi': rec['expected_roi'],
                    'feasibility_score': rec['feasibility_score'],
                    'risk_level': rec['risk_level'],
                    'implementation_time': rec['implementation_time'],
                    'generation_date': rec['generation_date']
                }
                recommendation_results.append(result)
        
        recommendation_df = pd.DataFrame(recommendation_results)
        recommendation_df.to_csv(os.path.join(self.output_dir, 'precision_recommendations.csv'), index=False)
        print(f"Saved precision recommendations: {recommendation_df.shape}")
        
        # save implementation plans
        implementation_results = []
        for seller_id, seller_recs in recommendations.items():
            for rec_type, rec in seller_recs.items():
                plan = rec['implementation_plan']
                result = {
                    'seller_id': seller_id,
                    'recommendation_id': rec['recommendation_id'],
                    'implementation_time': plan['implementation_time'],
                    'resource_requirements': plan['resource_requirements'],
                    'key_milestones': len(plan['key_milestones']),
                    'success_criteria': len(plan['success_criteria'])
                }
                implementation_results.append(result)
        
        implementation_df = pd.DataFrame(implementation_results)
        implementation_df.to_csv(os.path.join(self.output_dir, 'implementation_plans.csv'), index=False)
        print(f"Saved implementation plans: {implementation_df.shape}")
        
        # monitoring metrics
        monitoring_results = []
        for seller_id, seller_recs in recommendations.items():
            for rec_type, rec in seller_recs.items():
                metrics = rec['monitoring_metrics']
                result = {
                    'seller_id': seller_id,
                    'recommendation_id': rec['recommendation_id'],
                    'kpi_count': len(metrics['kpis']),
                    'alert_thresholds': len(metrics['alert_thresholds']),
                    'reporting_frequency': metrics['reporting_frequency']
                }
                monitoring_results.append(result)
        
        monitoring_df = pd.DataFrame(monitoring_results)
        monitoring_df.to_csv(os.path.join(self.output_dir, 'monitoring_dashboards.csv'), index=False)
        print(f"Saved monitoring dashboards: {monitoring_df.shape}")
        
        # Save feasibility reports
        feasibility_results = []
        for seller_id, seller_recs in recommendations.items():
            for rec_type, rec in seller_recs.items():
                feasibility = rec['feasibility_assessment']
                result = {
                    'seller_id': seller_id,
                    'recommendation_id': rec['recommendation_id'],
                    'feasibility_score': feasibility['feasibility_score'],
                    'risk_level': feasibility['risk_level'],
                    'resource_availability': feasibility['resource_availability'],
                    'technical_feasibility': feasibility['technical_feasibility'],
                    'organizational_readiness': feasibility['organizational_readiness']
                }
                feasibility_results.append(result)
        
        feasibility_df = pd.DataFrame(feasibility_results)
        feasibility_df.to_csv(os.path.join(self.output_dir, 'feasibility_reports.csv'), index=False)
        print(f"Saved feasibility reports: {feasibility_df.shape}")
        
        print("All precision recommendation outputs saved successfully")
    
    def print_analysis_summary(self, recommendations: Dict):
        """Print precision recommendation summary"""
        print("\n" + "="*80)
        print("PRECISION RECOMMENDATION GENERATOR SUMMARY")
        print("="*80)
        
        total_recommendations = len(recommendations)
        
        print(f"\nRECOMMENDATION OVERVIEW:")
        print("-" * 50)
        print(f"Total recommendations generated: {total_recommendations}")
        
        if recommendations:
            # Calculate average metrics
            all_priority_scores = []
            all_roi_values = []
            all_feasibility_scores = []
            
            for seller_id, seller_recs in recommendations.items():
                for rec_type, rec in seller_recs.items():
                    all_priority_scores.append(rec['priority_score'])
                    all_roi_values.append(rec['expected_roi'])
                    all_feasibility_scores.append(rec['feasibility_score'])
            
            avg_priority_score = np.mean(all_priority_scores)
            avg_roi = np.mean(all_roi_values)
            avg_feasibility = np.mean(all_feasibility_scores)
            
            print(f"Average priority score: {avg_priority_score:.3f}")
            print(f"Average expected ROI: {avg_roi:.3f}")
            print(f"Average feasibility score: {avg_feasibility:.3f}")
            
            # Recommendation type distribution
            rec_types = []
            for seller_id, seller_recs in recommendations.items():
                for rec_type in seller_recs.keys():
                    rec_types.append(rec_type)
            
            rec_type_counts = pd.Series(rec_types).value_counts()
            
            print(f"\nRECOMMENDATION TYPES:")
            print("-" * 50)
            for rec_type, count in rec_type_counts.items():
                print(f"  {rec_type}: {count} recommendations")
            
            # Risk level distribution
            risk_levels = []
            for seller_id, seller_recs in recommendations.items():
                for rec_type, rec in seller_recs.items():
                    risk_levels.append(rec['risk_level'])
            
            risk_level_counts = pd.Series(risk_levels).value_counts()
            
            print(f"\nRISK LEVEL DISTRIBUTION:")
            print("-" * 50)
            for risk_level, count in risk_level_counts.items():
                # Convert risk_level to string if it's a float
                if isinstance(risk_level, float):
                    if risk_level < 0.3:
                        risk_level_str = "low"
                    elif risk_level < 0.6:
                        risk_level_str = "medium"
                    else:
                        risk_level_str = "high"
                else:
                    risk_level_str = str(risk_level)
                print(f"  {risk_level_str.capitalize()} risk: {count} recommendations")
            
            # Priority score distribution
            high_priority = sum(1 for score in all_priority_scores if score > 0.8)
            medium_priority = sum(1 for score in all_priority_scores if 0.6 <= score <= 0.8)
            low_priority = sum(1 for score in all_priority_scores if score < 0.6)
            
            print(f"\nPRIORITY DISTRIBUTION:")
            print("-" * 50)
            print(f"  High priority (>0.8): {high_priority} recommendations")
            print(f"  Medium priority (0.6-0.8): {medium_priority} recommendations")
            print(f"  Low priority (<0.6): {low_priority} recommendations")
        
        print(f"\nOutput files saved in: {self.output_dir}")
        print(f"Visualizations saved in: {self.viz_dir}")
        
        print("\n" + "="*80)

def main():
    """Main execution function"""
    print("Starting Precision Recommendation Generator")
    
    # Initialize generator
    generator = PrecisionRecommendationGenerator()
    
    try:
        data = generator.load_data()
        
        if not data:
            print("No data loaded. Exiting.")
            return
        
        # Prepare recommendation data
        recommendation_data = generator.prepare_recommendation_data(data)
        
        if not recommendation_data:
            print("No recommendation data prepared. Exiting.")
            return
        
        # Generate precision recommendations
        recommendations = generator.generate_precision_recommendations(recommendation_data)
        
        if not recommendations:
            print("No recommendations generated. Exiting.")
            return
        
        # visualizations
        generator.create_visualizations(recommendations)
        
        # Save outputs
        generator.save_outputs(recommendations)
        
        # Print summary
        generator.print_analysis_summary(recommendations)
        
        print("Precision Recommendation Generator completed successfully")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 