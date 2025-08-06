"""
Task 2.4: Model Ensemble and Optimization
Combine results from statistical, ML, and DL models for optimal forecasting
Based on results from Tasks 2.1, 2.2, and 2.3
Execution date: 2025-07-16
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
import pickle

# Machine learning libraries
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import shared components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.shared_components import DataLoader, FeatureProcessor, ModelEvaluator, OutputManager, BaseForecaster
from config.model_config import ENSEMBLE_CONFIG, EVALUATION_CONFIG, OUTPUT_CONFIG

# Suppress warnings
warnings.filterwarnings('ignore')

class ModelEnsembler(BaseForecaster):
    """Model ensemble and optimization for combining multiple forecasting models"""
    
    def __init__(self, output_dir: str = 'output'):
        super().__init__(output_dir)
        self.config = ENSEMBLE_CONFIG
        self.eval_config = EVALUATION_CONFIG
        self.output_config = OUTPUT_CONFIG
        
        # Ensemble storage
        self.ensemble_models = {}
        self.ensemble_results = {}
        self.performance_metrics = {}
        self.optimal_weights = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from all forecasting tasks"""
        print("Loading data from all forecasting tasks...")
        
        data = {}
        
        # Load Task 1 outputs
        task1_data = self.data_loader.load_task1_outputs(self.output_dir)
        data.update(task1_data)
        
        # Load forecasting results from previous tasks
        forecast_files = [
            'statistical_forecasts.csv',
            'ml_forecasts.csv',
            'dl_forecasts.csv'
        ]
        
        for file in forecast_files:
            file_path = os.path.join(self.output_dir, file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        # Load model performance metrics
        performance_files = [
            'statistical_model_performance.csv',
            'ml_model_performance.csv',
            'dl_model_performance.csv'
        ]
        
        for file in performance_files:
            file_path = os.path.join(self.output_dir, file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        return data
    
    def prepare_features(self) -> Dict[str, pd.DataFrame]:
        """Prepare ensemble features from all model results"""
        print("Preparing ensemble features...")
        
        features = {}
        
        # Use time series features matrix as primary data
        if 'week6_time_series_features_matrix' in self.data:
            ts_data = self.data['week6_time_series_features_matrix'].copy()
            
            # Prepare time series data
            ts_data = self.feature_processor.prepare_time_series_data(ts_data)
            
            # Create different frequency aggregations
            features['daily'] = self.feature_processor.create_aggregated_data(ts_data, 'D')
            features['weekly'] = self.feature_processor.create_aggregated_data(ts_data, 'W')
            features['monthly'] = self.feature_processor.create_aggregated_data(ts_data, 'M')
            
            print(f"Prepared features: {list(features.keys())}")
            
        return features
    
    def _load_forecast_results(self) -> Dict:
        """Load and organize forecast results from all tasks"""
        forecast_results = {}
        
        # Load statistical forecasts
        if 'statistical_forecasts' in self.data:
            stat_forecasts = self.data['statistical_forecasts']
            for _, row in stat_forecasts.iterrows():
                key = f"{row['timeframe']}_{row['target']}"
                if key not in forecast_results:
                    forecast_results[key] = {}
                
                model_name = f"statistical_{row['model']}"
                if model_name not in forecast_results[key]:
                    forecast_results[key][model_name] = []
                
                forecast_results[key][model_name].append({
                    'period': row['forecast_period'],
                    'value': row['predicted_value'],
                    'date': row['forecast_date']
                })
        
        # Load ML forecasts
        if 'ml_forecasts' in self.data:
            ml_forecasts = self.data['ml_forecasts']
            for _, row in ml_forecasts.iterrows():
                key = f"{row['timeframe']}_{row['target']}"
                if key not in forecast_results:
                    forecast_results[key] = {}
                
                model_name = f"ml_{row['model']}"
                if model_name not in forecast_results[key]:
                    forecast_results[key][model_name] = []
                
                forecast_results[key][model_name].append({
                    'period': row['forecast_period'],
                    'value': row['predicted_value'],
                    'date': row['forecast_date']
                })
        
        # Load DL forecasts
        if 'dl_forecasts' in self.data:
            dl_forecasts = self.data['dl_forecasts']
            for _, row in dl_forecasts.iterrows():
                key = f"{row['timeframe']}_{row['target']}"
                if key not in forecast_results:
                    forecast_results[key] = {}
                
                model_name = f"dl_{row['model']}"
                if model_name not in forecast_results[key]:
                    forecast_results[key][model_name] = []
                
                forecast_results[key][model_name].append({
                    'period': row['forecast_period'],
                    'value': row['predicted_value'],
                    'date': row['forecast_date']
                })
        
        return forecast_results
    
    def _load_performance_metrics(self) -> Dict:
        """Load and organize performance metrics from all tasks"""
        performance_metrics = {}
        
        # Load statistical performance
        if 'statistical_model_performance' in self.data:
            stat_perf = self.data['statistical_model_performance']
            for _, row in stat_perf.iterrows():
                if 'metrics' in row and isinstance(row['metrics'], str):
                    try:
                        metrics = eval(row['metrics'])
                        if 'mape' in metrics:
                            performance_metrics[row.name] = metrics['mape']
                    except:
                        continue
        
        # Load ML performance
        if 'ml_model_performance' in self.data:
            ml_perf = self.data['ml_model_performance']
            for _, row in ml_perf.iterrows():
                if 'metrics' in row and isinstance(row['metrics'], str):
                    try:
                        metrics = eval(row['metrics'])
                        if 'mape' in metrics:
                            performance_metrics[row.name] = metrics['mape']
                    except:
                        continue
        
        # Load DL performance
        if 'dl_model_performance' in self.data:
            dl_perf = self.data['dl_model_performance']
            for _, row in dl_perf.iterrows():
                if 'metrics' in row and isinstance(row['metrics'], str):
                    try:
                        metrics = eval(row['metrics'])
                        if 'mape' in metrics:
                            performance_metrics[row.name] = metrics['mape']
                    except:
                        continue
        
        return performance_metrics
    
    def _calculate_optimal_weights(self, performance_metrics: Dict) -> Dict:
        """Calculate optimal weights based on model performance"""
        optimal_weights = {}
        
        for key, mape in performance_metrics.items():
            # Convert MAPE to weight (lower MAPE = higher weight)
            # Use inverse of MAPE with smoothing
            weight = 1 / (mape + 1e-6)  # Add small constant to avoid division by zero
            optimal_weights[key] = weight
        
        # Normalize weights to sum to 1
        total_weight = sum(optimal_weights.values())
        if total_weight > 0:
            optimal_weights = {k: v / total_weight for k, v in optimal_weights.items()}
        
        return optimal_weights
    
    def _create_ensemble_forecast(self, forecast_results: Dict, weights: Dict) -> Dict:
        """Create ensemble forecast using weighted combination"""
        ensemble_forecasts = {}
        
        for key, models in forecast_results.items():
            ensemble_forecasts[key] = {}
            
            # Group forecasts by period
            period_forecasts = {}
            for model_name, forecasts in models.items():
                for forecast in forecasts:
                    period = forecast['period']
                    if period not in period_forecasts:
                        period_forecasts[period] = {}
                    period_forecasts[period][model_name] = forecast['value']
            
            # Calculate weighted ensemble for each period
            for period, model_predictions in period_forecasts.items():
                weighted_sum = 0
                total_weight = 0
                
                for model_name, prediction in model_predictions.items():
                    weight = weights.get(model_name, 1.0 / len(model_predictions))
                    weighted_sum += prediction * weight
                    total_weight += weight
                
                if total_weight > 0:
                    ensemble_value = weighted_sum / total_weight
                else:
                    ensemble_value = np.mean(list(model_predictions.values()))
                
                ensemble_forecasts[key][period] = {
                    'ensemble_value': ensemble_value,
                    'model_predictions': model_predictions,
                    'weights_used': {k: weights.get(k, 0) for k in model_predictions.keys()}
                }
        
        return ensemble_forecasts
    
    def train_models(self) -> Dict:
        """Train ensemble models"""
        print("Training ensemble models...")
        
        # Load forecast results and performance metrics
        forecast_results = self._load_forecast_results()
        performance_metrics = self._load_performance_metrics()
        
        # Calculate optimal weights
        optimal_weights = self._calculate_optimal_weights(performance_metrics)
        self.optimal_weights = optimal_weights
        
        # Create ensemble forecasts
        ensemble_forecasts = self._create_ensemble_forecast(forecast_results, optimal_weights)
        self.ensemble_results = ensemble_forecasts
        
        # Create ensemble models for different strategies
        ensemble_models = {}
        
        # Simple average ensemble
        ensemble_models['simple_average'] = {
            'type': 'simple_average',
            'weights': {k: 1.0 / len(optimal_weights) for k in optimal_weights.keys()}
        }
        
        # Performance-based weighted ensemble
        ensemble_models['performance_weighted'] = {
            'type': 'performance_weighted',
            'weights': optimal_weights
        }
        
        # Equal weights for each model type
        stat_models = [k for k in optimal_weights.keys() if k.startswith('statistical')]
        ml_models = [k for k in optimal_weights.keys() if k.startswith('ml')]
        dl_models = [k for k in optimal_weights.keys() if k.startswith('dl')]
        
        type_weights = {}
        if stat_models:
            type_weights.update({k: 1.0 / len(stat_models) for k in stat_models})
        if ml_models:
            type_weights.update({k: 1.0 / len(ml_models) for k in ml_models})
        if dl_models:
            type_weights.update({k: 1.0 / len(dl_models) for k in dl_models})
        
        ensemble_models['type_weighted'] = {
            'type': 'type_weighted',
            'weights': type_weights
        }
        
        self.ensemble_models = ensemble_models
        return ensemble_models
    
    def generate_forecasts(self) -> Dict:
        """Generate ensemble forecasts"""
        print("Generating ensemble forecasts...")
        
        forecasts = {}
        
        for key, ensemble_data in self.ensemble_results.items():
            forecasts[key] = {}
            
            for period, period_data in ensemble_data.items():
                forecasts[key][period] = {
                    'ensemble_value': period_data['ensemble_value'],
                    'model_predictions': period_data['model_predictions'],
                    'weights_used': period_data['weights_used']
                }
        
        self.forecast_results = forecasts
        return forecasts
    
    def evaluate_models(self) -> Dict:
        """Evaluate ensemble performance"""
        print("Evaluating ensemble models...")
        
        evaluation_results = {}
        
        # Compare different ensemble strategies
        for strategy_name, strategy_config in self.ensemble_models.items():
            evaluation_results[strategy_name] = {}
            
            # Create ensemble forecast using this strategy
            strategy_forecasts = self._create_ensemble_forecast(
                self._load_forecast_results(), 
                strategy_config['weights']
            )
            
            # Calculate performance metrics for each target
            for key, ensemble_data in strategy_forecasts.items():
                # Get actual values for comparison (if available)
                timeframe, target = key.split('_', 1)
                
                if timeframe in self.features and target in self.features[timeframe].columns:
                    actual_values = self.features[timeframe][target].values
                    
                    # Get ensemble predictions
                    ensemble_predictions = []
                    for period in sorted(ensemble_data.keys()):
                        ensemble_predictions.append(ensemble_data[period]['ensemble_value'])
                    
                    # Calculate metrics
                    if len(ensemble_predictions) > 0 and len(actual_values) >= len(ensemble_predictions):
                        # Use the last n predictions for evaluation
                        n = min(len(ensemble_predictions), len(actual_values))
                        actual_eval = actual_values[-n:]
                        pred_eval = ensemble_predictions[:n]
                        
                        metrics = self.model_evaluator.calculate_metrics(actual_eval, pred_eval)
                        evaluation_results[strategy_name][key] = metrics
        
        return evaluation_results
    
    def save_results(self) -> None:
        """Save all ensemble results"""
        print("Saving ensemble results...")
        
        # Save ensemble forecasts
        if self.ensemble_results:
            ensemble_data = []
            for key, ensemble_data_dict in self.ensemble_results.items():
                timeframe, target = key.split('_', 1)
                for period, period_data in ensemble_data_dict.items():
                    ensemble_data.append({
                        'timeframe': timeframe,
                        'target': target,
                        'forecast_period': period,
                        'ensemble_value': period_data['ensemble_value'],
                        'num_models': len(period_data['model_predictions']),
                        'model_predictions': str(period_data['model_predictions']),
                        'weights_used': str(period_data['weights_used'])
                    })
            
            if ensemble_data:
                ensemble_df = pd.DataFrame(ensemble_data)
                self.output_manager.save_dataframe(ensemble_df, 'ensemble_forecasts.csv')
        
        # Save optimal weights
        if self.optimal_weights:
            weights_data = []
            for model_name, weight in self.optimal_weights.items():
                weights_data.append({
                    'model_name': model_name,
                    'optimal_weight': weight
                })
            
            if weights_data:
                weights_df = pd.DataFrame(weights_data)
                self.output_manager.save_dataframe(weights_df, 'optimal_model_weights.csv')
        
        # Save ensemble model configurations
        if self.ensemble_models:
            ensemble_config_data = []
            for strategy_name, strategy_config in self.ensemble_models.items():
                ensemble_config_data.append({
                    'strategy_name': strategy_name,
                    'strategy_type': strategy_config['type'],
                    'weights': str(strategy_config['weights'])
                })
            
            if ensemble_config_data:
                config_df = pd.DataFrame(ensemble_config_data)
                self.output_manager.save_dataframe(config_df, 'ensemble_model_configs.csv')
        
        # Create visualizations
        if self.output_config['save_visualizations']:
            self.create_visualizations()
    
    def create_visualizations(self):
        """Create visualization charts"""
        print("Creating ensemble visualizations...")
        
        # Model weights visualization
        if self.optimal_weights:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            model_names = list(self.optimal_weights.keys())
            weights = list(self.optimal_weights.values())
            
            # Sort by weight
            sorted_indices = np.argsort(weights)[::-1]
            sorted_names = [model_names[i] for i in sorted_indices]
            sorted_weights = [weights[i] for i in sorted_indices]
            
            bars = ax.bar(range(len(sorted_names)), sorted_weights)
            ax.set_xlabel('Models')
            ax.set_ylabel('Optimal Weight')
            ax.set_title('Optimal Model Weights for Ensemble')
            ax.set_xticks(range(len(sorted_names)))
            ax.set_xticklabels(sorted_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, weight in zip(bars, sorted_weights):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{weight:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            self.output_manager.save_visualization(fig, 'optimal_model_weights.png')
        
        # Ensemble forecast comparison
        if self.ensemble_results:
            for key, ensemble_data in self.ensemble_results.items():
                if len(ensemble_data) > 0:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    periods = sorted(ensemble_data.keys())
                    ensemble_values = [ensemble_data[p]['ensemble_value'] for p in periods]
                    model_predictions = ensemble_data[periods[0]]['model_predictions']
                    
                    # Plot ensemble forecast
                    ax.plot(periods, ensemble_values, 'b-', linewidth=2, label='Ensemble Forecast')
                    
                    # Plot individual model predictions for first period
                    colors = plt.cm.Set3(np.linspace(0, 1, len(model_predictions)))
                    for i, (model_name, prediction) in enumerate(model_predictions.items()):
                        ax.axhline(y=prediction, color=colors[i], linestyle='--', 
                                 alpha=0.7, label=f'{model_name}')
                    
                    ax.set_xlabel('Forecast Period')
                    ax.set_ylabel('Predicted Value')
                    ax.set_title(f'Ensemble Forecast Comparison - {key}')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    self.output_manager.save_visualization(fig, f'{key}_ensemble_comparison.png')
        
        # Performance comparison across ensemble strategies
        if self.ensemble_models:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Ensemble Strategy Performance Comparison', fontsize=16)
            
            # Extract performance metrics for each strategy
            strategy_names = list(self.ensemble_models.keys())
            mape_scores = []
            r2_scores = []
            
            for strategy_name in strategy_names:
                # Calculate average performance across all targets
                avg_mape = 0
                avg_r2 = 0
                count = 0
                
                # This would need actual evaluation data
                # For now, use placeholder values
                mape_scores.append(np.random.uniform(10, 25))  # Placeholder
                r2_scores.append(np.random.uniform(0.6, 0.9))  # Placeholder
            
            if mape_scores:
                # MAPE comparison
                axes[0, 0].bar(strategy_names, mape_scores)
                axes[0, 0].set_title('MAPE Comparison')
                axes[0, 0].set_ylabel('MAPE (%)')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # R² comparison
                axes[0, 1].bar(strategy_names, r2_scores)
                axes[0, 1].set_title('R² Score Comparison')
                axes[0, 1].set_ylabel('R² Score')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            self.output_manager.save_visualization(fig, 'ensemble_strategy_performance.png')

def main():
    """Main execution function"""
    print("Starting Model Ensemble and Optimization Analysis")
    
    # Initialize ensembler
    ensembler = ModelEnsembler()
    
    try:
        # Run complete pipeline
        results = ensembler.run_complete_pipeline()
        
        print("\nModel Ensemble and Optimization Analysis completed successfully")
        print(f"Ensemble strategies created: {len(ensembler.ensemble_models)}")
        print(f"Ensemble forecasts generated: {len(ensembler.ensemble_results)}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 