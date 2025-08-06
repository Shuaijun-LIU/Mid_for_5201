"""
Task 2.2: Machine Learning Forecasting Models
XGBoost, LightGBM, Random Forest, and Elastic Net for time series forecasting
Based on Task 1's time series feature engineering results
Execution date: 2025-07-15
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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb

# Import shared components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.shared_components import DataLoader, FeatureProcessor, ModelEvaluator, OutputManager, BaseForecaster
from config.model_config import ML_MODELS, EVALUATION_CONFIG, OUTPUT_CONFIG, FEATURE_CONFIG

# Suppress warnings
warnings.filterwarnings('ignore')

class MLForecaster(BaseForecaster):
    """Machine learning forecasting with XGBoost, LightGBM, Random Forest, and Elastic Net"""
    
    def __init__(self, output_dir: str = 'output'):
        super().__init__(output_dir)
        self.config = ML_MODELS
        self.eval_config = EVALUATION_CONFIG
        self.output_config = OUTPUT_CONFIG
        self.feature_config = FEATURE_CONFIG
        
        # Model storage
        self.trained_models = {}
        self.forecast_results = {}
        self.performance_metrics = {}
        self.feature_importance = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Task 1 and Week5 outputs"""
        print("Loading data for machine learning forecasting...")
        
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
        """Prepare features for machine learning models"""
        print("Preparing features for machine learning forecasting...")
        
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
            
            # Create ML-specific features
            for timeframe, data in features.items():
                features[timeframe] = self._create_ml_features(data)
            
            print(f"Prepared features: {list(features.keys())}")
            
        return features
    
    def _create_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create machine learning specific features"""
        ml_data = data.copy()
        
        # Convert date to features
        if 'date' in ml_data.columns:
            ml_data['date'] = pd.to_datetime(ml_data['date'])
            ml_data['year'] = ml_data['date'].dt.year
            ml_data['month'] = ml_data['date'].dt.month
            ml_data['day'] = ml_data['date'].dt.day
            ml_data['day_of_week'] = ml_data['date'].dt.dayofweek
            ml_data['quarter'] = ml_data['date'].dt.quarter
            
            # Cyclical encoding
            ml_data['month_sin'] = np.sin(2 * np.pi * ml_data['month'] / 12)
            ml_data['month_cos'] = np.cos(2 * np.pi * ml_data['month'] / 12)
            ml_data['day_of_week_sin'] = np.sin(2 * np.pi * ml_data['day_of_week'] / 7)
            ml_data['day_of_week_cos'] = np.cos(2 * np.pi * ml_data['day_of_week'] / 7)
        
        # Create lag features for target variables
        target_columns = ['total_orders', 'total_sales', 'avg_order_value']
        for target in target_columns:
            if target in ml_data.columns:
                for lag in self.feature_config['lag_periods']:
                    ml_data[f'{target}_lag_{lag}'] = ml_data[target].shift(lag)
                
                # Moving averages
                for window in self.feature_config['ma_windows']:
                    ml_data[f'{target}_ma_{window}'] = ml_data[target].rolling(window=window).mean()
                    ml_data[f'{target}_std_{window}'] = ml_data[target].rolling(window=window).std()
        
        # Create interaction features
        if 'total_orders' in ml_data.columns and 'total_sales' in ml_data.columns:
            ml_data['sales_per_order'] = ml_data['total_sales'] / ml_data['total_orders']
        
        if 'avg_price' in ml_data.columns and 'avg_freight' in ml_data.columns:
            ml_data['price_to_freight_ratio'] = ml_data['avg_price'] / ml_data['avg_freight']
        
        # Handle missing values
        ml_data = self.feature_processor.handle_missing_values(ml_data, method='forward')
        
        return ml_data
    
    def _prepare_ml_data(self, data: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for machine learning models"""
        # Select features
        id_cols = ['seller_id', 'product_category', 'state']  
        feature_columns = [col for col in data.columns 
                          if col not in ['date', target] + id_cols and not col.startswith('Unnamed')]
        # 只保留数值型特征
        feature_columns = [col for col in feature_columns if pd.api.types.is_numeric_dtype(data[col])]
        
        # Remove columns with too many missing values
        missing_ratio = data[feature_columns].isnull().sum() / len(data)
        feature_columns = [col for col in feature_columns if missing_ratio[col] < self.feature_config['missing_threshold']]
        
        # Prepare X and y
        X = data[feature_columns].fillna(0)  # Simple imputation for ML
        y = data[target].fillna(method='ffill')
        
        # Remove rows with missing target
        valid_indices = ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X.values, y.values, feature_columns
    
    def train_xgboost_model(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[xgb.XGBRegressor, Dict]:
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        model = xgb.XGBRegressor(**self.config['xgboost'])
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        
        # Train final model
        model.fit(X, y)
        
        # Feature importance
        importance = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        
        # Predictions for evaluation
        predictions = model.predict(X)
        metrics = self.model_evaluator.calculate_metrics(y, predictions)
        
        return model, {
            'cv_scores': -cv_scores,  # Convert back to positive
            'feature_importance': feature_importance,
            'metrics': metrics
        }
    
    def train_lightgbm_model(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[lgb.LGBMRegressor, Dict]:
        """Train LightGBM model"""
        print("Training LightGBM model...")
        
        model = lgb.LGBMRegressor(**self.config['lightgbm'])
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        
        # Train final model
        model.fit(X, y)
        
        # Feature importance
        importance = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        
        # Predictions for evaluation
        predictions = model.predict(X)
        metrics = self.model_evaluator.calculate_metrics(y, predictions)
        
        return model, {
            'cv_scores': -cv_scores,  # Convert back to positive
            'feature_importance': feature_importance,
            'metrics': metrics
        }
    
    def train_random_forest_model(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[RandomForestRegressor, Dict]:
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        model = RandomForestRegressor(**self.config['random_forest'])
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        
        # Train final model
        model.fit(X, y)
        
        # Feature importance
        importance = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        
        # Predictions for evaluation
        predictions = model.predict(X)
        metrics = self.model_evaluator.calculate_metrics(y, predictions)
        
        return model, {
            'cv_scores': -cv_scores,  # Convert back to positive
            'feature_importance': feature_importance,
            'metrics': metrics
        }
    
    def train_elastic_net_model(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[ElasticNet, Dict]:
        """Train Elastic Net model"""
        print("Training Elastic Net model...")
        
        # Scale features for linear models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = ElasticNet(**self.config['elastic_net'])
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_absolute_error')
        
        # Train final model
        model.fit(X_scaled, y)
        
        # Feature importance (coefficients)
        importance = np.abs(model.coef_)
        feature_importance = dict(zip(feature_names, importance))
        
        # Predictions for evaluation
        predictions = model.predict(X_scaled)
        metrics = self.model_evaluator.calculate_metrics(y, predictions)
        
        return model, {
            'cv_scores': -cv_scores,  # Convert back to positive
            'feature_importance': feature_importance,
            'metrics': metrics,
            'scaler': scaler
        }
    
    def train_models(self) -> Dict:
        """Train all machine learning models"""
        print("Training machine learning models...")
        
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
                
                # Prepare data
                X, y, feature_names = self._prepare_ml_data(data, target)
                
                if len(X) < 10:  # Need sufficient data
                    print(f"Insufficient data for {target}: {len(X)} samples")
                    continue
                
                # Train XGBoost
                try:
                    xgb_model, xgb_info = self.train_xgboost_model(X, y, feature_names)
                    models[timeframe][target]['xgboost'] = xgb_model
                    self.performance_metrics[f"{timeframe}_{target}_xgboost"] = {**xgb_info, 'feature_names': feature_names}
                    self.feature_importance[f"{timeframe}_{target}_xgboost"] = xgb_info['feature_importance']
                except Exception as e:
                    print(f"XGBoost training failed for {target}: {e}")
                
                # Train LightGBM
                try:
                    lgb_model, lgb_info = self.train_lightgbm_model(X, y, feature_names)
                    models[timeframe][target]['lightgbm'] = lgb_model
                    self.performance_metrics[f"{timeframe}_{target}_lightgbm"] = {**lgb_info, 'feature_names': feature_names}
                    self.feature_importance[f"{timeframe}_{target}_lightgbm"] = lgb_info['feature_importance']
                except Exception as e:
                    print(f"LightGBM training failed for {target}: {e}")
                
                # Train Random Forest
                try:
                    rf_model, rf_info = self.train_random_forest_model(X, y, feature_names)
                    models[timeframe][target]['random_forest'] = rf_model
                    self.performance_metrics[f"{timeframe}_{target}_random_forest"] = {**rf_info, 'feature_names': feature_names}
                    self.feature_importance[f"{timeframe}_{target}_random_forest"] = rf_info['feature_importance']
                except Exception as e:
                    print(f"Random Forest training failed for {target}: {e}")
                
                # Train Elastic Net
                try:
                    en_model, en_info = self.train_elastic_net_model(X, y, feature_names)
                    models[timeframe][target]['elastic_net'] = en_model
                    self.performance_metrics[f"{timeframe}_{target}_elastic_net"] = {**en_info, 'feature_names': feature_names}
                    self.feature_importance[f"{timeframe}_{target}_elastic_net"] = en_info['feature_importance']
                except Exception as e:
                    print(f"Elastic Net training failed for {target}: {e}")
        
        self.trained_models = models
        return models
    
    def generate_forecasts(self) -> Dict:
        """Generate forecasts using trained models"""
        print("Generating machine learning forecasts...")
        
        forecasts = {}
        
        for timeframe, timeframe_models in self.trained_models.items():
            forecasts[timeframe] = {}
            
            for target, target_models in timeframe_models.items():
                forecasts[timeframe][target] = {}
                
                # Get the latest data for forecasting
                latest_data = self.features[timeframe].tail(1)
                
                for model_name, model in target_models.items():
                    try:
                        feature_names = self.performance_metrics[f"{timeframe}_{target}_{model_name}"]['feature_names']
                        feature_names = list(feature_names)
                        X_forecast = latest_data[feature_names].fillna(0).values
                        
                        if model_name == 'elastic_net':
                            scaler = self.performance_metrics[f"{timeframe}_{target}_{model_name}"]['scaler']
                            X_forecast = scaler.transform(X_forecast)
                        
                        # Generate forecast
                        prediction = model.predict(X_forecast)[0]
                        
                        # Create forecast dates
                        forecast_dates = pd.date_range(
                            start=latest_data['date'].iloc[-1] + timedelta(days=1),
                            periods=self.eval_config['forecast_horizon'],
                            freq='D'
                        )
                        
                        # Extend forecast 
                        predictions = np.full(self.eval_config['forecast_horizon'], prediction)
                        
                        forecasts[timeframe][target][model_name] = {
                            'predictions': predictions,
                            'dates': forecast_dates
                        }
                        
                    except Exception as e:
                        print(f"Forecast generation failed for {model_name} - {target}: {e}")
        
        self.forecast_results = forecasts
        return forecasts
    
    def evaluate_models(self) -> Dict:
        """Evaluate model performance"""
        print("Evaluating machine learning models...")
        
        evaluation_results = {}
        
        for timeframe, timeframe_models in self.trained_models.items():
            evaluation_results[timeframe] = {}
            
            for target, target_models in timeframe_models.items():
                evaluation_results[timeframe][target] = {}
                
                # Get actual values for evaluation
                actual_data = self.features[timeframe][target].values
                
                for model_name, model in target_models.items():
                    # Get predictions for evaluation period
                    X_eval, y_eval, _ = self._prepare_ml_data(self.features[timeframe], target)
                    
                    if model_name == 'elastic_net':
                        # Scale features for Elastic Net
                        scaler = self.performance_metrics[f"{timeframe}_{target}_{model_name}"]['scaler']
                        X_eval = scaler.transform(X_eval)
                    
                    predictions = model.predict(X_eval)
                    
                    # Calculate metrics
                    metrics = self.model_evaluator.calculate_metrics(y_eval, predictions)
                    evaluation_results[timeframe][target][model_name] = metrics
        
        return evaluation_results
    
    def save_results(self) -> None:
        """Save all results"""
        print("Saving machine learning forecasting results...")
        
        # Save performance metrics
        if self.performance_metrics:
            metrics_df = pd.DataFrame.from_dict(self.performance_metrics, orient='index')
            self.output_manager.save_dataframe(metrics_df, 'ml_model_performance.csv')
        
        # Save feature importance
        if self.feature_importance:
            importance_data = []
            for key, importance_dict in self.feature_importance.items():
                for feature, importance in importance_dict.items():
                    importance_data.append({
                        'model_key': key,
                        'feature': feature,
                        'importance': importance
                    })
            
            if importance_data:
                importance_df = pd.DataFrame(importance_data)
                self.output_manager.save_dataframe(importance_df, 'ml_feature_importance.csv')
        
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
                self.output_manager.save_dataframe(forecast_df, 'ml_forecasts.csv')
        
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
        print("Creating machine learning forecasting visualizations...")
        
        # Model performance comparison
        if self.performance_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Machine Learning Model Performance Comparison', fontsize=16)
            
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
            self.output_manager.save_visualization(fig, 'ml_model_performance.png')
        
        # Feature importance plots
        if self.feature_importance:
            for key, importance_dict in self.feature_importance.items():
                if importance_dict:
                    # Get top features
                    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                    features, importances = zip(*sorted_features)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(range(len(features)), importances)
                    ax.set_yticks(range(len(features)))
                    ax.set_yticklabels(features)
                    ax.set_xlabel('Feature Importance')
                    ax.set_title(f'Top 10 Feature Importance - {key}')
                    
                    plt.tight_layout()
                    self.output_manager.save_visualization(fig, f'{key}_feature_importance.png')

def main():
    """Main execution function"""
    print("Starting Machine Learning Forecasting Analysis")
    
    # Initialize forecaster
    forecaster = MLForecaster()
    
    try:
        # Run complete pipeline
        results = forecaster.run_complete_pipeline()
        
        print("\nMachine Learning Forecasting Analysis completed successfully")
        print(f"Models trained: {len(forecaster.trained_models)}")
        print(f"Forecasts generated: {len(forecaster.forecast_results)}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 