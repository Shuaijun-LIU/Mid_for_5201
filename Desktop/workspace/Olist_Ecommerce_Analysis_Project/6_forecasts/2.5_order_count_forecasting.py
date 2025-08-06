"""
Task 2.5: Order Count Forecasting
Predict order_count in addition to total_sales for capacity planning
Based on results from Tasks 2.1, 2.2, 2.3, and 2.4
Execution date: 2025-07-27
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

class OrderCountForecaster(BaseForecaster):
    """Order count forecasting for capacity planning"""
    
    def __init__(self, output_dir: str = 'output'):
        super().__init__(output_dir)
        self.config = ENSEMBLE_CONFIG
        self.eval_config = EVALUATION_CONFIG
        self.output_config = OUTPUT_CONFIG
        
        # Forecast storage
        self.order_count_forecasts = {}
        self.performance_metrics = {}
        
    def load_features(self) -> Dict[str, pd.DataFrame]:
        """Load time series features for order count forecasting"""
        print("Loading features for order count forecasting...")
        
        features = {}
        
        # Load time series features matrix
        features_file = os.path.join(self.output_dir, 'week6_time_series_features_matrix.csv')
        if os.path.exists(features_file):
            features_df = pd.read_csv(features_file)
            print(f"Loaded features: {features_df.shape}")
            
            # Convert date column
            if 'date' in features_df.columns:
                features_df['date'] = pd.to_datetime(features_df['date'])
                features_df = features_df.sort_values('date')
            
            # Use all data (no timeframe grouping needed)
            features['daily'] = features_df.copy()
            print(f"  daily: {features_df.shape}")
        
        return features
    
    def create_order_count_forecasts(self) -> Dict:
        """Create order count forecasts using ensemble approach"""
        print("Creating order count forecasts...")
        
        forecasts = {}
        
        # Load features
        features = self.load_features()
        
        for timeframe, timeframe_data in features.items():
            print(f"Processing {timeframe} forecasts...")
            
            if 'order_count' not in timeframe_data.columns:
                print(f"Warning: order_count not found in {timeframe} data")
                continue
            
            # Prepare data for forecasting
            X, y = self._prepare_forecast_data(timeframe_data, 'order_count')
            
            if X is None or y is None:
                print(f"Warning: Insufficient data for {timeframe} order_count forecasting")
                continue
            
            # Create ensemble forecast
            ensemble_forecast = self._create_ensemble_forecast(X, y, timeframe)
            
            if ensemble_forecast:
                forecasts[f"{timeframe}_order_count"] = ensemble_forecast
                print(f"Created {timeframe} order_count forecast with {len(ensemble_forecast)} periods")
        
        self.order_count_forecasts = forecasts
        return forecasts
    
    def _prepare_forecast_data(self, data: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for forecasting"""
        # Select feature columns (exclude target and metadata columns)
        exclude_cols = ['date', 'timeframe', 'seller_id', 'product_category', 'state', 
                       'order_count', 'total_sales', 'avg_price', 'avg_freight']
        
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        if len(feature_cols) < 5:
            print(f"Warning: Insufficient features for {target} forecasting")
            return None, None
        
        # Prepare X and y (handle non-numeric data)
        X_data = data[feature_cols].copy()
        
        # Convert non-numeric columns to numeric or drop them
        for col in X_data.columns:
            if X_data[col].dtype == 'object':
                try:
                    X_data[col] = pd.to_numeric(X_data[col], errors='coerce')
                except:
                    X_data = X_data.drop(columns=[col])
        
        # Fill NaN values
        X_data = X_data.fillna(0)
        X = X_data.values
        y = data[target].fillna(0).values
        
        return X, y
    
    def _create_ensemble_forecast(self, X: np.ndarray, y: np.ndarray, timeframe: str) -> Dict:
        """Create ensemble forecast for order count"""
        if len(X) < 30:  # Need at least 30 data points
            print(f"Warning: Insufficient data points for {timeframe} forecasting")
            return {}
        
        # Split data for training
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create ensemble model
        ensemble_models = {
            'linear': LinearRegression(),
            'voting': VotingRegressor([
                ('lr', LinearRegression()),
                ('lr2', LinearRegression())
            ])
        }
        
        # Train models and make predictions
        predictions = {}
        for model_name, model in ensemble_models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                predictions[model_name] = pred
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        # Create ensemble forecast
        if predictions:
            # Simple average ensemble
            ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
            
            # Create forecast periods
            forecast_periods = {}
            for i, pred_value in enumerate(ensemble_pred):
                period = i + 1
                forecast_periods[period] = {
                    'ensemble_value': max(0, pred_value),  # Ensure non-negative
                    'model_predictions': {name: max(0, pred[i]) for name, pred in predictions.items()},
                    'weights_used': {name: 1.0/len(predictions) for name in predictions.keys()}
                }
            
            return forecast_periods
        
        return {}
    
    def save_results(self) -> None:
        """Save order count forecast results"""
        print("Saving order count forecast results...")
        
        if self.order_count_forecasts:
            # Save ensemble forecasts
            ensemble_data = []
            for key, ensemble_data_dict in self.order_count_forecasts.items():
                timeframe, target = key.split('_', 1)
                for period, period_data in ensemble_data_dict.items():
                    ensemble_data.append({
                        'timeframe': timeframe,
                        'target': target,
                        'forecast_period': period,
                        'ensemble_value': period_data['ensemble_value'],
                        'num_models': len(period_data['model_predictions']),
                        'model_predictions': json.dumps(period_data['model_predictions']),
                        'weights_used': json.dumps(period_data['weights_used'])
                    })
            
            ensemble_df = pd.DataFrame(ensemble_data)
            ensemble_df.to_csv(os.path.join(self.output_dir, 'order_count_forecasts.csv'), index=False)
            print(f"Saved order count forecasts: {ensemble_df.shape}")
            
            # Save performance metrics
            if self.performance_metrics:
                metrics_df = pd.DataFrame(self.performance_metrics).T
                metrics_df.to_csv(os.path.join(self.output_dir, 'order_count_performance.csv'), index=True)
                print(f"Saved order count performance metrics: {metrics_df.shape}")
        
        print("Order count forecasting results saved successfully")
    
    def print_summary(self) -> None:
        """Print order count forecasting summary"""
        print("\n" + "="*60)
        print("ORDER COUNT FORECASTING SUMMARY")
        print("="*60)
        
        if self.order_count_forecasts:
            total_forecasts = sum(len(forecast) for forecast in self.order_count_forecasts.values())
            print(f"Total forecast periods: {total_forecasts}")
            
            for key, forecast in self.order_count_forecasts.items():
                timeframe, target = key.split('_', 1)
                print(f"\n{timeframe.upper()} {target.upper()} Forecasts:")
                print(f"  Periods: {len(forecast)}")
                
                if forecast:
                    values = [period_data['ensemble_value'] for period_data in forecast.values()]
                    print(f"  Average: {np.mean(values):.2f}")
                    print(f"  Min: {np.min(values):.2f}")
                    print(f"  Max: {np.max(values):.2f}")
        else:
            print("No order count forecasts generated")
        
        print("\n" + "="*60)

def main():
    """Main execution function"""
    print("Starting Order Count Forecasting")
    
    # Initialize forecaster
    forecaster = OrderCountForecaster()
    
    try:
        # Create order count forecasts
        forecasts = forecaster.create_order_count_forecasts()
        
        if forecasts:
            # Save results
            forecaster.save_results()
            
            # Print summary
            forecaster.print_summary()
            
            print("Order count forecasting completed successfully")
        else:
            print("No order count forecasts created")
        
    except Exception as e:
        print(f"Error in order count forecasting: {e}")
        raise

if __name__ == "__main__":
    main() 