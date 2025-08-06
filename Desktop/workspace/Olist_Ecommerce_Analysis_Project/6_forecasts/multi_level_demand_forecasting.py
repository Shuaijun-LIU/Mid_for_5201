"""
Task 2: Multi-level Demand Forecasting Models
Based on Task 1's time series feature engineering results, build multi-level demand forecasting models
including ARIMA, Prophet, LSTM and other models, with model ensemble and performance evaluation
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

# Time series and ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Suppress warnings
warnings.filterwarnings('ignore')

class MultiLevelDemandForecaster:
    """Multi-level demand forecasting with ARIMA, Prophet, and LSTM models"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.models_dir = os.path.join(output_dir, 'models')
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        
        # Create directories
        for dir_path in [output_dir, self.models_dir, self.viz_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Model configurations
        self.config = {
            'arima': {
                'p_max': 3, 'd_max': 2, 'q_max': 3,
                'seasonal_period': 12
            },
            'prophet': {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10,
                'holidays_prior_scale': 10
            },
            'lstm': {
                'units': 50,
                'dropout': 0.2,
                'epochs': 100,
                'batch_size': 32,
                'sequence_length': 30
            },
            'ensemble': {
                'initial_weights': {'arima': 0.3, 'prophet': 0.4, 'lstm': 0.3}
            }
        }
        
        # Performance tracking
        self.model_performance = {}
        self.forecast_results = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Task 1 and Week5 outputs"""
        print("Loading data from Task 1 and Week5 outputs...")
        
        data = {}
        
        # Load Task 1 outputs
        task1_files = [
            'time_series_features_matrix.csv',
            'feature_importance_ranking.csv',
            'feature_correlation_matrix.csv'
        ]
        
        for file in task1_files:
            file_path = os.path.join(self.output_dir, file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        # Load Week5 outputs
        week5_files = [
            'four_d_main_analysis.csv',
            'seller_summary_analysis.csv',
            'product_summary_analysis.csv'
        ]
        
        for file in week5_files:
            file_path = os.path.join('../week5_seller_analysis&initial_warehouse_recommendations/output', file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        # Load Week2 time series data
        week2_files = [
            'timeseries/aov_monthly.csv',
            'holiday_sensitive/holiday_sensitive_categories.csv'
        ]
        
        for file in week2_files:
            file_path = os.path.join('../week2_eda_update/output', file)
            if os.path.exists(file_path):
                data[file.replace('/', '_').replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('/', '_').replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        return data
    
    def prepare_forecasting_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data for forecasting models"""
        print("Preparing forecasting data...")
        
        forecasting_data = {}
        
        # Use time series features matrix as primary data
        if 'time_series_features_matrix' in data:
            ts_data = data['time_series_features_matrix'].copy()
            
            # Convert date columns to datetime
            date_columns = [col for col in ts_data.columns if 'date' in col.lower()]
            for col in date_columns:
                ts_data[col] = pd.to_datetime(ts_data[col])
            
            # Create different aggregation levels
            forecasting_data['daily'] = self._create_daily_data(ts_data)
            forecasting_data['weekly'] = self._create_weekly_data(ts_data)
            forecasting_data['monthly'] = self._create_monthly_data(ts_data)
            
            print(f"Created forecasting data: {list(forecasting_data.keys())}")
        
        return forecasting_data
    
    def _create_daily_data(self, ts_data: pd.DataFrame) -> pd.DataFrame:
        """Create daily aggregated data"""
        # Group by date and aggregate
        daily_data = ts_data.groupby('date').agg({
            'total_orders': 'sum',
            'total_revenue': 'sum',
            'avg_order_value': 'mean',
            'unique_customers': 'nunique',
            'unique_sellers': 'nunique',
            'unique_products': 'nunique'
        }).reset_index()
        
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data = daily_data.sort_values('date').reset_index(drop=True)
        
        return daily_data
    
    def _create_weekly_data(self, ts_data: pd.DataFrame) -> pd.DataFrame:
        """Create weekly aggregated data"""
        ts_data['date'] = pd.to_datetime(ts_data['date'])
        ts_data['week'] = ts_data['date'].dt.to_period('W')
        
        weekly_data = ts_data.groupby('week').agg({
            'total_orders': 'sum',
            'total_revenue': 'sum',
            'avg_order_value': 'mean',
            'unique_customers': 'nunique',
            'unique_sellers': 'nunique',
            'unique_products': 'nunique'
        }).reset_index()
        
        weekly_data['week_start'] = weekly_data['week'].dt.start_time
        weekly_data = weekly_data.sort_values('week_start').reset_index(drop=True)
        
        return weekly_data
    
    def _create_monthly_data(self, ts_data: pd.DataFrame) -> pd.DataFrame:
        """Create monthly aggregated data"""
        ts_data['date'] = pd.to_datetime(ts_data['date'])
        ts_data['month'] = ts_data['date'].dt.to_period('M')
        
        monthly_data = ts_data.groupby('month').agg({
            'total_orders': 'sum',
            'total_revenue': 'sum',
            'avg_order_value': 'mean',
            'unique_customers': 'nunique',
            'unique_sellers': 'nunique',
            'unique_products': 'nunique'
        }).reset_index()
        
        monthly_data['month_start'] = monthly_data['month'].dt.start_time
        monthly_data = monthly_data.sort_values('month_start').reset_index(drop=True)
        
        return monthly_data
    
    def train_arima_model(self, data: pd.Series, target: str = 'total_orders') -> Tuple[ARIMA, Dict]:
        """Train ARIMA model with automatic parameter selection"""
        print(f"Training ARIMA model for {target}...")
        
        # Check stationarity
        adf_result = adfuller(data)
        is_stationary = adf_result[1] < 0.05
        
        # Determine differencing order
        d = 0
        if not is_stationary:
            d = 1
            data_diff = data.diff().dropna()
            adf_result_diff = adfuller(data_diff)
            if adf_result_diff[1] >= 0.05:
                d = 2
        
        # Grid search for best parameters
        best_aic = float('inf')
        best_params = (1, d, 1)
        best_model = None
        
        for p in range(0, self.config['arima']['p_max'] + 1):
            for q in range(0, self.config['arima']['q_max'] + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        best_model = fitted_model
                        
                except:
                    continue
        
        print(f"Best ARIMA parameters: {best_params}, AIC: {best_aic}")
        
        model_info = {
            'params': best_params,
            'aic': best_aic,
            'is_stationary': is_stationary,
            'adf_pvalue': adf_result[1]
        }
        
        return best_model, model_info
    
    def train_prophet_model(self, data: pd.DataFrame, target: str = 'total_orders') -> Tuple[Prophet, Dict]:
        """Train Prophet model with holiday effects"""
        print(f"Training Prophet model for {target}...")
        
        # Prepare data for Prophet
        prophet_data = data[['date', target]].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Create model
        model = Prophet(
            changepoint_prior_scale=self.config['prophet']['changepoint_prior_scale'],
            seasonality_prior_scale=self.config['prophet']['seasonality_prior_scale'],
            holidays_prior_scale=self.config['prophet']['holidays_prior_scale']
        )
        
        # Add holidays
        model.add_country_holidays(country_name='BR')
        
        # Add custom seasonalities
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='weekly', period=7, fourier_order=3)
        
        # Fit model
        model.fit(prophet_data)
        
        model_info = {
            'changepoints': len(model.changepoints),
            'holidays': len(model.holidays) if hasattr(model, 'holidays') else 0
        }
        
        return model, model_info
    
    def train_lstm_model(self, data: pd.Series, target: str = 'total_orders') -> Tuple[Sequential, Dict]:
        """Train LSTM model for time series forecasting"""
        print(f"Training LSTM model for {target}...")
        
        # Prepare data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Create sequences
        X, y = self._create_sequences(scaled_data, self.config['lstm']['sequence_length'])
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build model
        model = Sequential([
            LSTM(self.config['lstm']['units'], return_sequences=True, 
                 input_shape=(X.shape[1], X.shape[2])),
            Dropout(self.config['lstm']['dropout']),
            LSTM(self.config['lstm']['units'] // 2, return_sequences=False),
            Dropout(self.config['lstm']['dropout']),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=self.config['lstm']['epochs'],
            batch_size=self.config['lstm']['batch_size'],
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        model_info = {
            'scaler': scaler,
            'sequence_length': self.config['lstm']['sequence_length'],
            'training_history': history.history,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
        
        return model, model_info
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def evaluate_model(self, model, data: pd.Series, model_type: str, target: str) -> Dict:
        """Evaluate model performance"""
        print(f"Evaluating {model_type} model for {target}...")
        
        # Split data for evaluation
        split_idx = int(len(data) * 0.7)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        if model_type == 'arima':
            # ARIMA evaluation
            predictions = model.forecast(steps=len(test_data))
            actual = test_data.values
            
        elif model_type == 'prophet':
            # Prophet evaluation
            future = pd.DataFrame({'ds': test_data.index})
            forecast = model.predict(future)
            predictions = forecast['yhat'].values
            actual = test_data.values
            
        elif model_type == 'lstm':
            # LSTM evaluation
            scaler = model[1]['scaler']  # model is tuple (model, info)
            lstm_model = model[0]
            
            scaled_data = scaler.transform(data.values.reshape(-1, 1))
            X_test, y_test = self._create_sequences(scaled_data, self.config['lstm']['sequence_length'])
            
            predictions = lstm_model.predict(X_test)
            predictions = scaler.inverse_transform(predictions).flatten()
            actual = scaler.inverse_transform(y_test).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        r2 = r2_score(actual, predictions)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2,
            'predictions': predictions,
            'actual': actual
        }
        
        return metrics
    
    def create_ensemble_forecast(self, models: Dict, data: pd.Series, target: str) -> Dict:
        """Create ensemble forecast using multiple models"""
        print(f"Creating ensemble forecast for {target}...")
        
        # Get individual predictions
        predictions = {}
        
        for model_type, model in models.items():
            if model_type == 'arima':
                pred = model[0].forecast(steps=30)  # 30 days forecast
                predictions[model_type] = pred.values
                
            elif model_type == 'prophet':
                future = pd.DataFrame({'ds': pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=30)})
                forecast = model[0].predict(future)
                predictions[model_type] = forecast['yhat'].values
                
            elif model_type == 'lstm':
                scaler = model[1]['scaler']
                lstm_model = model[0]
                
                # Prepare last sequence
                last_sequence = data.tail(self.config['lstm']['sequence_length']).values
                last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
                
                # Generate predictions
                lstm_preds = []
                current_sequence = last_sequence_scaled[-self.config['lstm']['sequence_length']:]
                
                for _ in range(30):
                    X_pred = current_sequence.reshape(1, self.config['lstm']['sequence_length'], 1)
                    pred = lstm_model.predict(X_pred, verbose=0)
                    lstm_preds.append(pred[0, 0])
                    current_sequence = np.vstack([current_sequence[1:], pred])
                
                predictions[model_type] = scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1)).flatten()
        
        # Calculate ensemble prediction
        weights = self.config['ensemble']['initial_weights']
        ensemble_pred = np.zeros(30)
        
        for model_type, pred in predictions.items():
            if model_type in weights:
                ensemble_pred += weights[model_type] * pred
        
        # Calculate confidence intervals
        all_preds = np.array(list(predictions.values()))
        std_pred = np.std(all_preds, axis=0)
        
        confidence_intervals = {
            'lower_95': ensemble_pred - 1.96 * std_pred,
            'upper_95': ensemble_pred + 1.96 * std_pred,
            'lower_99': ensemble_pred - 2.58 * std_pred,
            'upper_99': ensemble_pred + 2.58 * std_pred
        }
        
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'weights': weights
        }
    
    def generate_forecasts(self, forecasting_data: Dict[str, pd.DataFrame]) -> Dict:
        """Generate forecasts for all levels and timeframes"""
        print("Generating multi-level forecasts...")
        
        all_forecasts = {}
        
        for timeframe, data in forecasting_data.items():
            print(f"Processing {timeframe} data...")
            
            timeframe_forecasts = {}
            
            for target in ['total_orders', 'total_revenue', 'avg_order_value']:
                if target in data.columns:
                    print(f"Forecasting {target} for {timeframe}...")
                    
                    # Prepare time series
                    ts_data = data.set_index('date' if 'date' in data.columns else 'week_start' if 'week_start' in data.columns else 'month_start')[target]
                    
                    # Train models
                    models = {}
                    
                    try:
                        arima_model, arima_info = self.train_arima_model(ts_data, target)
                        models['arima'] = (arima_model, arima_info)
                        print(f"ARIMA model trained successfully for {target}")
                    except Exception as e:
                        print(f"ARIMA training failed for {target}: {e}")
                    
                    try:
                        prophet_model, prophet_info = self.train_prophet_model(data, target)
                        models['prophet'] = (prophet_model, prophet_info)
                        print(f"Prophet model trained successfully for {target}")
                    except Exception as e:
                        print(f"Prophet training failed for {target}: {e}")
                    
                    try:
                        lstm_model, lstm_info = self.train_lstm_model(ts_data, target)
                        models['lstm'] = (lstm_model, lstm_info)
                        print(f"LSTM model trained successfully for {target}")
                    except Exception as e:
                        print(f"LSTM training failed for {target}: {e}")
                    
                    # Evaluate models
                    model_performance = {}
                    for model_type, model in models.items():
                        try:
                            performance = self.evaluate_model(model, ts_data, model_type, target)
                            model_performance[model_type] = performance
                        except Exception as e:
                            print(f"Model evaluation failed for {model_type}: {e}")
                    
                    # Create ensemble forecast
                    if len(models) > 1:
                        try:
                            ensemble_forecast = self.create_ensemble_forecast(models, ts_data, target)
                            timeframe_forecasts[target] = {
                                'models': models,
                                'performance': model_performance,
                                'ensemble_forecast': ensemble_forecast
                            }
                        except Exception as e:
                            print(f"Ensemble forecast failed for {target}: {e}")
                    else:
                        timeframe_forecasts[target] = {
                            'models': models,
                            'performance': model_performance
                        }
            
            all_forecasts[timeframe] = timeframe_forecasts
        
        return all_forecasts
    
    def create_visualizations(self, forecasting_data: Dict, all_forecasts: Dict):
        """Create visualization charts"""
        print("Creating forecast visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Time series forecasts
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Multi-level Demand Forecasts', fontsize=16, fontweight='bold')
        
        for i, (timeframe, forecasts) in enumerate(all_forecasts.items()):
            if 'total_orders' in forecasts:
                data = forecasting_data[timeframe]
                ts_data = data.set_index('date' if 'date' in data.columns else 'week_start' if 'week_start' in data.columns else 'month_start')['total_orders']
                
                axes[i].plot(ts_data.index, ts_data.values, label='Historical', linewidth=2)
                
                if 'ensemble_forecast' in forecasts['total_orders']:
                    ensemble = forecasts['total_orders']['ensemble_forecast']
                    future_dates = pd.date_range(ts_data.index[-1] + pd.Timedelta(days=1), periods=30)
                    
                    axes[i].plot(future_dates, ensemble['ensemble_prediction'], 
                               label='Ensemble Forecast', linewidth=2, color='red')
                    axes[i].fill_between(future_dates, 
                                       ensemble['confidence_intervals']['lower_95'],
                                       ensemble['confidence_intervals']['upper_95'],
                                       alpha=0.3, color='red', label='95% CI')
                
                axes[i].set_title(f'{timeframe.capitalize()} Total Orders Forecast')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Total Orders')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'time_series_forecasts.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model performance comparison
        performance_data = []
        for timeframe, forecasts in all_forecasts.items():
            for target, forecast_info in forecasts.items():
                if 'performance' in forecast_info:
                    for model_type, perf in forecast_info['performance'].items():
                        performance_data.append({
                            'timeframe': timeframe,
                            'target': target,
                            'model': model_type,
                            'mape': perf['mape'],
                            'rmse': perf['rmse'],
                            'r2_score': perf['r2_score']
                        })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
            
            metrics = ['mape', 'rmse', 'r2_score']
            titles = ['MAPE (%)', 'RMSE', 'R² Score']
            
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                pivot_data = perf_df.pivot_table(values=metric, index='model', columns='timeframe', aggfunc='mean')
                pivot_data.plot(kind='bar', ax=axes[i], width=0.8)
                axes[i].set_title(title)
                axes[i].set_xlabel('Model')
                axes[i].set_ylabel(title)
                axes[i].legend(title='Timeframe')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Residual analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Residual Analysis', fontsize=16, fontweight='bold')
        
        # Get residuals from best performing model
        best_model_residuals = None
        best_mape = float('inf')
        
        for timeframe, forecasts in all_forecasts.items():
            for target, forecast_info in forecasts.items():
                if 'performance' in forecast_info:
                    for model_type, perf in forecast_info['performance'].items():
                        if perf['mape'] < best_mape:
                            best_mape = perf['mape']
                            best_model_residuals = perf['actual'] - perf['predictions']
        
        if best_model_residuals is not None:
            # Residual plot
            axes[0, 0].scatter(range(len(best_model_residuals)), best_model_residuals, alpha=0.6)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_title('Residual Plot')
            axes[0, 0].set_xlabel('Observation')
            axes[0, 0].set_ylabel('Residual')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Residual histogram
            axes[0, 1].hist(best_model_residuals, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Residual Distribution')
            axes[0, 1].set_xlabel('Residual')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(best_model_residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Residual vs Fitted
            axes[1, 1].scatter(best_model_residuals + (perf['actual'] - best_model_residuals), best_model_residuals, alpha=0.6)
            axes[1, 1].axhline(y=0, color='red', linestyle='--')
            axes[1, 1].set_title('Residual vs Fitted')
            axes[1, 1].set_xlabel('Fitted Values')
            axes[1, 1].set_ylabel('Residual')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'residual_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations created successfully")
    
    def save_outputs(self, all_forecasts: Dict, forecasting_data: Dict):
        """Save forecast results and performance metrics"""
        print("Saving forecast outputs...")
        
        # 1. Save demand forecasts
        forecast_results = []
        
        for timeframe, forecasts in all_forecasts.items():
            for target, forecast_info in forecasts.items():
                if 'ensemble_forecast' in forecast_info:
                    ensemble = forecast_info['ensemble_forecast']
                    
                    for i in range(30):  # 30 days forecast
                        forecast_results.append({
                            'timeframe': timeframe,
                            'target': target,
                            'forecast_period': i + 1,
                            'predicted_value': ensemble['ensemble_prediction'][i],
                            'lower_95': ensemble['confidence_intervals']['lower_95'][i],
                            'upper_95': ensemble['confidence_intervals']['upper_95'][i],
                            'lower_99': ensemble['confidence_intervals']['lower_99'][i],
                            'upper_99': ensemble['confidence_intervals']['upper_99'][i],
                            'confidence_level': 0.95
                        })
        
        if forecast_results:
            forecast_df = pd.DataFrame(forecast_results)
            forecast_df.to_csv(os.path.join(self.output_dir, 'demand_forecasts.csv'), index=False)
            print(f"Saved demand forecasts: {forecast_df.shape}")
        
        # 2. Save model performance metrics
        performance_results = []
        
        for timeframe, forecasts in all_forecasts.items():
            for target, forecast_info in forecasts.items():
                if 'performance' in forecast_info:
                    for model_type, perf in forecast_info['performance'].items():
                        performance_results.append({
                            'timeframe': timeframe,
                            'target': target,
                            'model': model_type,
                            'mape': perf['mape'],
                            'rmse': perf['rmse'],
                            'mae': perf['mae'],
                            'r2_score': perf['r2_score']
                        })
        
        if performance_results:
            perf_df = pd.DataFrame(performance_results)
            perf_df.to_csv(os.path.join(self.output_dir, 'model_performance_metrics.csv'), index=False)
            print(f"Saved performance metrics: {perf_df.shape}")
        
        # 3. Save forecast confidence intervals
        confidence_results = []
        
        for timeframe, forecasts in all_forecasts.items():
            for target, forecast_info in forecasts.items():
                if 'ensemble_forecast' in forecast_info:
                    ensemble = forecast_info['ensemble_forecast']
                    
                    for i in range(30):
                        confidence_results.append({
                            'timeframe': timeframe,
                            'target': target,
                            'forecast_period': i + 1,
                            'predicted': ensemble['ensemble_prediction'][i],
                            'lower_95': ensemble['confidence_intervals']['lower_95'][i],
                            'upper_95': ensemble['confidence_intervals']['upper_95'][i],
                            'lower_99': ensemble['confidence_intervals']['lower_99'][i],
                            'upper_99': ensemble['confidence_intervals']['upper_99'][i],
                            'std_dev': np.std([ensemble['ensemble_prediction'][i]])
                        })
        
        if confidence_results:
            conf_df = pd.DataFrame(confidence_results)
            conf_df.to_csv(os.path.join(self.output_dir, 'forecast_confidence_intervals.csv'), index=False)
            print(f"Saved confidence intervals: {conf_df.shape}")
        
        # 4. Save model comparison report
        comparison_results = []
        
        for timeframe, forecasts in all_forecasts.items():
            for target, forecast_info in forecasts.items():
                if 'performance' in forecast_info:
                    best_model = min(forecast_info['performance'].items(), 
                                   key=lambda x: x[1]['mape'])[0]
                    
                    comparison_results.append({
                        'timeframe': timeframe,
                        'target': target,
                        'best_model': best_model,
                        'best_mape': forecast_info['performance'][best_model]['mape'],
                        'best_rmse': forecast_info['performance'][best_model]['rmse'],
                        'best_r2': forecast_info['performance'][best_model]['r2_score'],
                        'models_tested': len(forecast_info['performance'])
                    })
        
        if comparison_results:
            comp_df = pd.DataFrame(comparison_results)
            comp_df.to_csv(os.path.join(self.output_dir, 'model_comparison_report.csv'), index=False)
            print(f"Saved model comparison: {comp_df.shape}")
        
        # 5. Save models
        for timeframe, forecasts in all_forecasts.items():
            for target, forecast_info in forecasts.items():
                if 'models' in forecast_info:
                    for model_type, model in forecast_info['models'].items():
                        model_path = os.path.join(self.models_dir, f'{timeframe}_{target}_{model_type}_model.pkl')
                        try:
                            with open(model_path, 'wb') as f:
                                pickle.dump(model, f)
                        except Exception as e:
                            print(f"Could not save {model_type} model: {e}")
        
        print("All outputs saved successfully")
    
    def print_analysis_summary(self, all_forecasts: Dict):
        """Print analysis summary"""
        print("\n" + "="*80)
        print("MULTI-LEVEL DEMAND FORECASTING ANALYSIS SUMMARY")
        print("="*80)
        
        total_models_trained = 0
        total_forecasts_generated = 0
        best_performances = []
        
        for timeframe, forecasts in all_forecasts.items():
            print(f"\n📊 {timeframe.upper()} FORECASTS:")
            print("-" * 50)
            
            for target, forecast_info in forecasts.items():
                print(f"\n🎯 Target: {target}")
                
                if 'performance' in forecast_info:
                    models_perf = forecast_info['performance']
                    print(f"   Models trained: {len(models_perf)}")
                    total_models_trained += len(models_perf)
                    
                    # Find best model
                    if models_perf:
                        best_model = min(models_perf.items(), key=lambda x: x[1]['mape'])
                        best_performances.append({
                            'timeframe': timeframe,
                            'target': target,
                            'model': best_model[0],
                            'mape': best_model[1]['mape'],
                            'r2': best_model[1]['r2_score']
                        })
                        
                        print(f"   Best model: {best_model[0]}")
                        print(f"   Best MAPE: {best_model[1]['mape']:.2f}%")
                        print(f"   Best R²: {best_model[1]['r2_score']:.3f}")
                
                if 'ensemble_forecast' in forecast_info:
                    total_forecasts_generated += 1
                    print(f"   ✅ Ensemble forecast generated")
        
        print(f"\n📈 OVERALL SUMMARY:")
        print("-" * 50)
        print(f"Total models trained: {total_models_trained}")
        print(f"Total forecasts generated: {total_forecasts_generated}")
        
        if best_performances:
            overall_best = min(best_performances, key=lambda x: x['mape'])
            print(f"Best overall performance: {overall_best['model']} for {overall_best['target']} ({overall_best['timeframe']})")
            print(f"Best MAPE: {overall_best['mape']:.2f}%")
            print(f"Best R²: {overall_best['r2']:.3f}")
        
        print(f"\n📁 Output files saved in: {self.output_dir}")
        print(f"📊 Visualizations saved in: {self.viz_dir}")
        print(f"🤖 Models saved in: {self.models_dir}")
        
        print("\n" + "="*80)

def main():
    """Main execution function"""
    print("Starting Multi-level Demand Forecasting Analysis")
    
    # Initialize forecaster
    forecaster = MultiLevelDemandForecaster()
    
    try:
        data = forecaster.load_data()
        
        if not data:
            print("No data loaded. Exiting.")
            return
        
        # Prepare forecasting data
        forecasting_data = forecaster.prepare_forecasting_data(data)
        
        if not forecasting_data:
            print("No forecasting data prepared. Exiting.")
            return
        
        # Generate forecasts
        all_forecasts = forecaster.generate_forecasts(forecasting_data)
        
        if not all_forecasts:
            print("No forecasts generated. Exiting.")
            return
        
        # Create visualizations
        forecaster.create_visualizations(forecasting_data, all_forecasts)

        forecaster.save_outputs(all_forecasts, forecasting_data)

        forecaster.print_analysis_summary(all_forecasts)
        
        print("Multi-level Demand Forecasting Analysis completed successfully")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 