"""
Task 2.1: Statistical Forecasting Models
ARIMA, Prophet, and VAR models for time series forecasting
Based on Task 1's time series feature engineering results
Execution date: 2025-07-14
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

# Time series and statistical libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.vector_ar.var_model import VAR
from prophet import Prophet

# Import shared components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.shared_components import DataLoader, FeatureProcessor, ModelEvaluator, OutputManager, BaseForecaster
from config.model_config import STATISTICAL_MODELS, EVALUATION_CONFIG, OUTPUT_CONFIG

# Suppress warnings
warnings.filterwarnings('ignore')

class StatisticalForecaster(BaseForecaster):
    """Statistical forecasting with ARIMA, Prophet, and VAR models"""
    
    def __init__(self, output_dir: str = 'output'):
        super().__init__(output_dir)
        self.config = STATISTICAL_MODELS
        self.eval_config = EVALUATION_CONFIG
        self.output_config = OUTPUT_CONFIG
        
        # Model storage
        self.trained_models = {}
        self.forecast_results = {}
        self.performance_metrics = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Task 1 and Week5 outputs"""
        print("Loading data for statistical forecasting...")
        
        data = {}
        
        # Load Task 1 outputs
        task1_data = self.data_loader.load_task1_outputs(self.output_dir)
        data.update(task1_data)
        
        # Load Week5 outputs
        week5_data = self.data_loader.load_week5_outputs()
        data.update(week5_data)
        
        # Load Week2 outputs
        week2_data = self.data_loader.load_week2_outputs()
        data.update(week2_data)
        
        return data
    
    def prepare_features(self) -> Dict[str, pd.DataFrame]:
        """Prepare features for statistical models"""
        print("Preparing features for statistical forecasting...")
        
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
    
    def train_arima_model(self, data: pd.Series, target: str = 'total_orders') -> Tuple[ARIMA, Dict]:
        """Train ARIMA model with automatic parameter selection"""
        print(f"Training ARIMA model for {target}...")
        
        # Check stationarity
        adf_result = adfuller(data.dropna())
        is_stationary = adf_result[1] < 0.05
        
        # Determine differencing order
        d = 0
        if not is_stationary:
            d = 1
            data_diff = data.diff().dropna()
            adf_result_diff = adfuller(data_diff)
            if adf_result_diff[1] >= 0.05:
                d = 2
        
        # Grid search for optimal parameters
        best_aic = float('inf')
        best_params = (1, d, 1)
        best_model = None
        
        p_range = range(0, self.config['arima']['p_max'] + 1)
        q_range = range(0, self.config['arima']['q_max'] + 1)
        
        for p in p_range:
            for q in q_range:
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        best_model = fitted_model
                        
                except:
                    continue
        
        print(f"Best ARIMA parameters: {best_params}, AIC: {best_aic:.2f}")
        
        # Calculate performance metrics (align predictions and actuals)
        predictions = best_model.predict(start=0, end=len(data)-1)
        metrics = self.model_evaluator.calculate_metrics(data.values, predictions.values)
        
        return best_model, {
            'params': best_params,
            'aic': best_aic,
            'is_stationary': is_stationary,
            'metrics': metrics
        }
    
    def train_prophet_model(self, data: pd.DataFrame, target: str = 'total_orders') -> Tuple[Prophet, Dict]:
        """Train Prophet model with holiday effects"""
        print(f"Training Prophet model for {target}...")
        
        # Prepare data for Prophet
        prophet_data = data[['date', target]].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Create Prophet model
        model = Prophet(
            changepoint_prior_scale=self.config['prophet']['changepoint_prior_scale'],
            seasonality_prior_scale=self.config['prophet']['seasonality_prior_scale'],
            holidays_prior_scale=self.config['prophet']['holidays_prior_scale'],
            seasonality_mode=self.config['prophet']['seasonality_mode'],
            changepoint_range=self.config['prophet']['changepoint_range']
        )
        
        # Add seasonality
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='weekly', period=7, fourier_order=3)
        
        # Add Brazilian holidays if available
        if 'orders_with_weekend_holiday' in self.data:
            holiday_data = self.data['orders_with_weekend_holiday']
            if 'holiday_name' in holiday_data.columns:
                holidays = holiday_data[['date', 'holiday_name']].copy()
                holidays.columns = ['ds', 'holiday']
                model.add_country_holidays(country_name='BR')
        
        # Fit model
        model.fit(prophet_data)
        
        # Make predictions for evaluation
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        
        # Calculate performance metrics
        actual = prophet_data['y'].values
        predicted = forecast['yhat'].values[:len(actual)]
        metrics = self.model_evaluator.calculate_metrics(actual, predicted)
        
        return model, {
            'metrics': metrics,
            'forecast_components': forecast
        }
    
    def train_var_model(self, data: pd.DataFrame, targets: List[str]) -> Tuple[VAR, Dict]:
        """Train VAR model for multivariate time series"""
        print(f"Training VAR model for {targets}...")
        
        # Prepare data for VAR
        var_data = data[targets].copy()
        var_data = var_data.dropna()
        
        # Check stationarity for each variable
        stationary_data = var_data.copy()
        d_values = {}
        
        for col in targets:
            adf_result = adfuller(var_data[col])
            if adf_result[1] >= 0.05:
                stationary_data[col] = var_data[col].diff().dropna()
                d_values[col] = 1
            else:
                d_values[col] = 0
        
        # Fit VAR model
        model = VAR(stationary_data)
        
        # Select optimal lag order
        maxlags = self.config['var']['maxlags']
        lag_order = model.select_order(maxlags=maxlags)
        optimal_lags = lag_order.aic
        
        # Fit model with optimal lags
        fitted_model = model.fit(optimal_lags)
        
        # Make predictions
        lag_order = fitted_model.k_ar
        predictions = fitted_model.forecast(stationary_data.values[-lag_order:], steps=len(stationary_data))
        
        # Calculate performance metrics for each target
        metrics = {}
        for i, target in enumerate(targets):
            actual = stationary_data[target].values
            predicted = predictions[:, i]
            metrics[target] = self.model_evaluator.calculate_metrics(actual, predicted)
        
        return fitted_model, {
            'optimal_lags': optimal_lags,
            'd_values': d_values,
            'metrics': metrics
        }
    
    def train_models(self) -> Dict:
        """Train all statistical models"""
        print("Training statistical models...")
        
        models = {}
        
        for timeframe, data in self.features.items():
            print(f"\nTraining models for {timeframe} data...")
            models[timeframe] = {}
            
            # Get target columns
            target_columns = ['total_orders', 'total_sales', 'avg_order_value']
            available_targets = [col for col in target_columns if col in data.columns]
            
            for target in available_targets:
                print(f"Training models for target: {target}")
                models[timeframe][target] = {}
                
                # Train ARIMA
                try:
                    arima_model, arima_info = self.train_arima_model(data[target], target)
                    models[timeframe][target]['arima'] = arima_model
                    self.performance_metrics[f"{timeframe}_{target}_arima"] = arima_info
                except Exception as e:
                    print(f"ARIMA training failed for {target}: {e}")
                
                # Train Prophet
                try:
                    prophet_model, prophet_info = self.train_prophet_model(data, target)
                    models[timeframe][target]['prophet'] = prophet_model
                    self.performance_metrics[f"{timeframe}_{target}_prophet"] = prophet_info
                except Exception as e:
                    print(f"Prophet training failed for {target}: {e}")
            
            # Train VAR for multivariate analysis
            if len(available_targets) > 1:
                try:
                    var_model, var_info = self.train_var_model(data, available_targets)
                    models[timeframe]['var'] = var_model
                    self.performance_metrics[f"{timeframe}_var"] = var_info
                except Exception as e:
                    print(f"VAR training failed: {e}")
        
        self.trained_models = models
        return models
    
    def generate_forecasts(self) -> Dict:
        """Generate forecasts using trained models"""
        print("Generating statistical forecasts...")
        
        forecasts = {}
        
        for timeframe, timeframe_models in self.trained_models.items():
            forecasts[timeframe] = {}
            
            for target, target_models in timeframe_models.items():
                if target == 'var':
                    continue
                    
                forecasts[timeframe][target] = {}
                
                # Generate ARIMA forecasts
                if 'arima' in target_models:
                    try:
                        arima_forecast = target_models['arima'].forecast(steps=self.eval_config['forecast_horizon'])
                        forecasts[timeframe][target]['arima'] = {
                            'predictions': arima_forecast.values,
                            'dates': pd.date_range(
                                start=self.features[timeframe]['date'].iloc[-1] + timedelta(days=1),
                                periods=self.eval_config['forecast_horizon'],
                                freq='D'
                            )
                        }
                    except Exception as e:
                        print(f"ARIMA forecast failed for {target}: {e}")
                
                # Generate Prophet forecasts
                if 'prophet' in target_models:
                    try:
                        future = target_models['prophet'].make_future_dataframe(
                            periods=self.eval_config['forecast_horizon']
                        )
                        prophet_forecast = target_models['prophet'].predict(future)
                        
                        # Get forecast values
                        forecast_values = prophet_forecast['yhat'].tail(self.eval_config['forecast_horizon']).values
                        forecast_dates = prophet_forecast['ds'].tail(self.eval_config['forecast_horizon']).values
                        
                        forecasts[timeframe][target]['prophet'] = {
                            'predictions': forecast_values,
                            'dates': forecast_dates,
                            'lower_bound': prophet_forecast['yhat_lower'].tail(self.eval_config['forecast_horizon']).values,
                            'upper_bound': prophet_forecast['yhat_upper'].tail(self.eval_config['forecast_horizon']).values
                        }
                    except Exception as e:
                        print(f"Prophet forecast failed for {target}: {e}")
            
            # Generate VAR forecasts
            if 'var' in timeframe_models:
                try:
                    var_model = timeframe_models['var']
                    lag_order = var_model.k_ar
                    var_forecast = var_model.forecast(
                        self.features[timeframe].values[-lag_order:],
                        steps=self.eval_config['forecast_horizon']
                    )
                    
                    forecasts[timeframe]['var'] = {
                        'predictions': var_forecast,
                        'dates': pd.date_range(
                            start=self.features[timeframe]['date'].iloc[-1] + timedelta(days=1),
                            periods=self.eval_config['forecast_horizon'],
                            freq='D'
                        )
                    }
                except Exception as e:
                    print(f"VAR forecast failed: {e}")
        
        self.forecast_results = forecasts
        return forecasts
    
    def evaluate_models(self) -> Dict:
        """Evaluate model performance"""
        print("Evaluating statistical models...")
        
        evaluation_results = {}
        
        for timeframe, timeframe_models in self.trained_models.items():
            evaluation_results[timeframe] = {}
            
            for target, target_models in timeframe_models.items():
                if target == 'var':
                    continue
                    
                evaluation_results[timeframe][target] = {}
                
                # Get actual values for evaluation
                actual_data = self.features[timeframe][target].values
                
                for model_name, model in target_models.items():
                    if model_name == 'var':
                        continue
                        
                    # Get predictions for evaluation period
                    if model_name == 'arima':
                        predictions = model.predict(start=0, end=len(actual_data)-1)
                        pred_values = predictions.values
                    elif model_name == 'prophet':
                        future = model.make_future_dataframe(periods=0)
                        forecast = model.predict(future)
                        pred_values = forecast['yhat'].values[:len(actual_data)]
                    
                    # Align lengths just in case
                    min_len = min(len(actual_data), len(pred_values))
                    metrics = self.model_evaluator.calculate_metrics(actual_data[:min_len], pred_values[:min_len])
                    evaluation_results[timeframe][target][model_name] = metrics
        
        return evaluation_results
    
    def save_results(self) -> None:
        """Save all results"""
        print("Saving statistical forecasting results...")
        
        # Save performance metrics
        if self.performance_metrics:
            metrics_df = pd.DataFrame.from_dict(self.performance_metrics, orient='index')
            self.output_manager.save_dataframe(metrics_df, 'statistical_model_performance.csv')
        
        # Save forecasts
        if self.forecast_results:
            forecast_data = []
            for timeframe, timeframe_forecasts in self.forecast_results.items():
                for target, target_forecasts in timeframe_forecasts.items():
                    for model_name, forecast_info in target_forecasts.items():
                        if 'predictions' in forecast_info:
                            for i, (pred, date) in enumerate(zip(forecast_info['predictions'], forecast_info['dates'])):
                                forecast_data.append({
                                    'timeframe': timeframe,
                                    'target': target,
                                    'model': model_name,
                                    'forecast_period': i + 1,
                                    'predicted_value': pred,
                                    'forecast_date': date
                                })
            
            if forecast_data:
                forecast_df = pd.DataFrame(forecast_data)
                self.output_manager.save_dataframe(forecast_df, 'statistical_forecasts.csv')
        
        # Save trained models
        if self.output_config['save_models']:
            for timeframe, timeframe_models in self.trained_models.items():
                for target, target_models in timeframe_models.items():
                    for model_name, model in target_models.items():
                        model_filename = f"{timeframe}_{target}_{model_name}"
                        self.output_manager.save_model(model, model_filename, model_name)
        
        # Create visualizations
        if self.output_config['save_visualizations']:
            self.create_visualizations()
    
    def create_visualizations(self):
        """Create visualization charts"""
        print("Creating statistical forecasting visualizations...")
        
        # Model performance comparison
        if self.performance_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Statistical Model Performance Comparison', fontsize=16)
            
            # Extract metrics for plotting
            model_names = []
            mape_scores = []
            r2_scores = []
            
            for key, metrics in self.performance_metrics.items():
                if 'metrics' in metrics and 'mape' in metrics['metrics']:
                    model_names.append(key)
                    mape_scores.append(metrics['metrics']['mape'])
                    r2_scores.append(metrics['metrics']['r2_score'])
            
            if model_names:
                # MAPE comparison
                axes[0, 0].bar(model_names, mape_scores)
                axes[0, 0].set_title('MAPE Comparison')
                axes[0, 0].set_ylabel('MAPE (%)')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # R² comparison
                axes[0, 1].bar(model_names, r2_scores)
                axes[0, 1].set_title('R² Score Comparison')
                axes[0, 1].set_ylabel('R² Score')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # RMSE comparison
                rmse_scores = [self.performance_metrics[name]['metrics']['rmse'] 
                              for name in model_names if 'metrics' in self.performance_metrics[name]]
                axes[1, 0].bar(model_names[:len(rmse_scores)], rmse_scores)
                axes[1, 0].set_title('RMSE Comparison')
                axes[1, 0].set_ylabel('RMSE')
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # MAE comparison
                mae_scores = [self.performance_metrics[name]['metrics']['mae'] 
                             for name in model_names if 'metrics' in self.performance_metrics[name]]
                axes[1, 1].bar(model_names[:len(mae_scores)], mae_scores)
                axes[1, 1].set_title('MAE Comparison')
                axes[1, 1].set_ylabel('MAE')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            self.output_manager.save_visualization(fig, 'statistical_model_performance.png')
        
        # Forecast plots
        if self.forecast_results:
            for timeframe, timeframe_forecasts in self.forecast_results.items():
                for target, target_forecasts in timeframe_forecasts.items():
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical data
                    historical_data = self.features[timeframe]
                    if 'date' in historical_data.columns and target in historical_data.columns:
                        ax.plot(historical_data['date'], historical_data[target], 
                               label='Historical', linewidth=2)
                    
                    # Plot forecasts
                    for model_name, forecast_info in target_forecasts.items():
                        if 'predictions' in forecast_info and 'dates' in forecast_info:
                            ax.plot(forecast_info['dates'], forecast_info['predictions'], 
                                   label=f'{model_name} Forecast', linestyle='--')
                    
                    ax.set_title(f'{target} Forecasts - {timeframe}')
                    ax.set_xlabel('Date')
                    ax.set_ylabel(target)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    self.output_manager.save_visualization(fig, f'{timeframe}_{target}_forecasts.png')

def main():
    """Main execution function"""
    print("Starting Statistical Forecasting Analysis")
    
    # Initialize forecaster
    forecaster = StatisticalForecaster()
    
    try:
        # Run complete pipeline
        results = forecaster.run_complete_pipeline()
        
        print("\nStatistical Forecasting Analysis completed successfully")
        print(f"Models trained: {len(forecaster.trained_models)}")
        print(f"Forecasts generated: {len(forecaster.forecast_results)}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 