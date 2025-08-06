"""
Shared Components for Week6 Forecasting Tasks
Common functionality extracted from original scripts
Execution date: 2025-07-13
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Unified data loading interface for all forecasting tasks"""
    
    def __init__(self, base_dir: str = '..'):
        self.base_dir = base_dir
        self.data = {}
        
    def load_task1_outputs(self, output_dir: str = 'output') -> Dict[str, pd.DataFrame]:
        """Load Task 1 time series feature engineering outputs"""
        print("Loading Task 1 outputs...")
        
        task1_files = [
            'week6_time_series_features_matrix.csv',
            'week6_feature_importance_ranking.csv',
            'week6_feature_correlation_matrix.csv',
            'week6_feature_statistics_summary.csv'
        ]
        
        data = {}
        for file in task1_files:
            file_path = os.path.join(output_dir, file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        return data
    
    def load_week5_outputs(self) -> Dict[str, pd.DataFrame]:
        """Load Week5 four-dimensional analysis outputs"""
        print("Loading Week5 outputs...")
        
        week5_files = [
            'four_d_main_analysis.csv',
            'seller_summary_analysis.csv',
            'product_summary_analysis.csv',
            'geo_summary_analysis.csv',
            'time_summary_analysis.csv'
        ]
        
        data = {}
        for file in week5_files:
            file_path = os.path.join(self.base_dir, 'week5_seller_analysis_and_four_d_analysis/output', file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        return data
    
    def load_week4_outputs(self) -> Dict[str, pd.DataFrame]:
        """Load Week4 inventory efficiency outputs"""
        print("Loading Week4 outputs...")
        
        week4_files = [
            'inventory_efficiency_metrics.csv',
            'inventory_policy_matrix.csv',
            'product_warehouse_summary.csv',
            'warehouse_simulation_summary.csv'
        ]
        
        data = {}
        for file in week4_files:
            file_path = os.path.join(self.base_dir, 'week4_product_warehouse_analysis/output', file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        return data
    
    def load_week3_outputs(self) -> Dict[str, pd.DataFrame]:
        """Load Week3 customer behavior outputs"""
        print("Loading Week3 outputs...")
        
        week3_files = [
            'customer_lifecycle.csv',
            'customer_logistics_features.csv',
            'final_customer_segments.csv'
        ]
        
        data = {}
        for file in week3_files:
            file_path = os.path.join(self.base_dir, 'week3_customer_behavior/output', file)
            if os.path.exists(file_path):
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[file.replace('.csv', '')].shape}")
            else:
                print(f"File not found: {file_path}")
        
        return data
    
    def load_week2_outputs(self) -> Dict[str, pd.DataFrame]:
        """Load Week2 time series and holiday data"""
        print("Loading Week2 outputs...")
        
        week2_files = [
            'weekend_holiday_analysis/orders_with_weekend_holiday.csv',
            'timeseries/aov_monthly.csv',
            'holiday_sensitive/holiday_sensitive_categories.csv'
        ]
        
        data = {}
        for file in week2_files:
            file_path = os.path.join(self.base_dir, 'week2_eda_update/output', file)
            if os.path.exists(file_path):
                key = file.replace('/', '_').replace('.csv', '')
                data[key] = pd.read_csv(file_path)
                print(f"Loaded {file}: {data[key].shape}")
            else:
                print(f"File not found: {file_path}")
        
        return data

class FeatureProcessor:
    """Common feature processing methods"""
    
    @staticmethod
    def prepare_time_series_data(ts_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series data for forecasting"""
        data = ts_data.copy()
        
        # Convert date columns to datetime
        date_columns = [col for col in data.columns if 'date' in col.lower()]
        for col in date_columns:
            data[col] = pd.to_datetime(data[col])
        
        # Sort by date
        if 'date' in data.columns:
            data = data.sort_values('date')
        
        return data
    
    @staticmethod
    def create_aggregated_data(ts_data: pd.DataFrame, freq: str = 'M') -> pd.DataFrame:
        """Create aggregated time series data"""
        if 'date' not in ts_data.columns:
            raise ValueError("Date column not found in data")
        
        # Set date as index
        data = ts_data.set_index('date')
        
        # Aggregate by frequency
        if freq == 'D':
            aggregated = data.resample('D').sum()
        elif freq == 'W':
            aggregated = data.resample('W').sum()
        elif freq == 'M':
            aggregated = data.resample('M').sum()
        else:
            raise ValueError(f"Unsupported frequency: {freq}")
        
        # Reset index
        aggregated = aggregated.reset_index()
        
        return aggregated
    
    @staticmethod
    def handle_missing_values(data: pd.DataFrame, method: str = 'median') -> pd.DataFrame:
        """Handle missing values in time series data"""
        data_clean = data.copy()
        
        # Select numeric columns
        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
        
        if method == 'median':
            data_clean[numeric_cols] = data_clean[numeric_cols].fillna(data_clean[numeric_cols].median())
        elif method == 'mean':
            data_clean[numeric_cols] = data_clean[numeric_cols].fillna(data_clean[numeric_cols].mean())
        elif method == 'forward':
            data_clean[numeric_cols] = data_clean[numeric_cols].fillna(method='ffill')
        elif method == 'backward':
            data_clean[numeric_cols] = data_clean[numeric_cols].fillna(method='bfill')
        
        return data_clean
    
    @staticmethod
    def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """Detect outliers in time series data"""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > threshold
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        return outliers

class ModelEvaluator:
    """Standard model evaluation methods"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate standard evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Calculate R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2
        }
    
    @staticmethod
    def calculate_confidence_intervals(predictions: np.ndarray, confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }

class OutputManager:
    """Output file management"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        self.models_dir = os.path.join(output_dir, 'models')
        
        # Create directories
        for dir_path in [output_dir, self.viz_dir, self.models_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, index: bool = False) -> str:
        """Save DataFrame to CSV file"""
        file_path = os.path.join(self.output_dir, filename)
        df.to_csv(file_path, index=index)
        print(f"Saved {filename}: {df.shape}")
        return file_path
    
    def save_model(self, model, model_name: str, model_type: str) -> str:
        """Save trained model using pickle"""
        file_path = os.path.join(self.models_dir, f'{model_name}_{model_type}_model.pkl')
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved model: {file_path}")
            return file_path
        except Exception as e:
            print(f"Could not save {model_type} model: {e}")
            return None
    
    def save_visualization(self, fig, filename: str, dpi: int = 300) -> str:
        """Save matplotlib figure"""
        file_path = os.path.join(self.viz_dir, filename)
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved visualization: {file_path}")
        return file_path
    
    def load_model(self, model_name: str, model_type: str):
        """Load trained model from pickle file"""
        file_path = os.path.join(self.models_dir, f'{model_name}_{model_type}_model.pkl')
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded model: {file_path}")
            return model
        except Exception as e:
            print(f"Could not load {model_type} model: {e}")
            return None

class BaseForecaster:
    """Base class for all forecasting models"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.data_loader = DataLoader()
        self.feature_processor = FeatureProcessor()
        self.model_evaluator = ModelEvaluator()
        self.output_manager = OutputManager(output_dir)
        
        # Performance tracking
        self.model_performance = {}
        self.forecast_results = {}
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data - to be implemented by subclasses"""
        raise NotImplementedError
    
    def prepare_features(self) -> Dict[str, pd.DataFrame]:
        """Prepare features - to be implemented by subclasses"""
        raise NotImplementedError
    
    def train_models(self) -> Dict:
        """Train models - to be implemented by subclasses"""
        raise NotImplementedError
    
    def generate_forecasts(self) -> Dict:
        """Generate forecasts - to be implemented by subclasses"""
        raise NotImplementedError
    
    def evaluate_models(self) -> Dict:
        """Evaluate models - to be implemented by subclasses"""
        raise NotImplementedError
    
    def save_results(self) -> None:
        """Save results - to be implemented by subclasses"""
        raise NotImplementedError
    
    def run_complete_pipeline(self) -> Dict:
        print("=" * 60)
        print(f"RUNNING {self.__class__.__name__.upper()}")
        print("=" * 60)
        
        try:
            data = self.load_data()
            self.data = data  # Ensure self.data is available to all methods
            features = self.prepare_features()
            self.features = features  # Ensure self.features is available to all methods
            models = self.train_models()
            forecasts = self.generate_forecasts()
            evaluation = self.evaluate_models()
            self.save_results()
            
            print("\n" + "=" * 60)
            print(f"{self.__class__.__name__.upper()} COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            return {
                'data': data,
                'features': features,
                'models': models,
                'forecasts': forecasts,
                'evaluation': evaluation
            }
            
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            raise 