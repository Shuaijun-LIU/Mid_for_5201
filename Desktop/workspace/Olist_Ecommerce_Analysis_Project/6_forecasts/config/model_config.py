"""
Model Configuration for Week6 Forecasting Tasks
Configuration settings for all forecasting models
Execution date: 2025-07-13
Update date: 2025-07-18
"""

# Statistical Models Configuration
STATISTICAL_MODELS = {
    'arima': {
        'p_max': 3,
        'd_max': 2,
        'q_max': 3,
        'seasonal_period': 12,
        'seasonal_p_max': 1,
        'seasonal_d_max': 1,
        'seasonal_q_max': 1
    },
    'prophet': {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10,
        'holidays_prior_scale': 10,
        'seasonality_mode': 'additive',
        'changepoint_range': 0.8
    },
    'var': {
        'maxlags': 12,
        'trend': 'c',
        'seasonal': True,
        'seasonal_periods': 12
    }
}

# Machine Learning Models Configuration
ML_MODELS = {
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    },
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'elastic_net': {
        'alpha': 0.1,
        'l1_ratio': 0.5,
        'max_iter': 1000,
        'random_state': 42
    }
}

# Deep Learning Models Configuration
DL_MODELS = {
    'lstm': {
        'sequence_length': 10,
        'units': 16,
        'dropout': 0.05,
        'epochs': 3,
        'batch_size': 256,
        'validation_split': 0.1,
        'early_stopping_patience': 1,
    },
    'gru': {
        'sequence_length': 5,
        'units': 8,
        'dropout': 0.01,
        'epochs': 3,
        'batch_size': 4,
        'validation_split': 0.1,
        'early_stopping_patience': 1,
    },
    'transformer': {
        'sequence_length': 10,
        'd_model': 16,
        'n_heads': 2,
        'n_layers': 1,
        'dropout': 0.05,
        'epochs': 3,
        'batch_size': 256,
        'validation_split': 0.1,
        'early_stopping_patience': 1,
    }
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'initial_weights': {
        'arima': 0.25,
        'prophet': 0.25,
        'var': 0.25,
        'xgboost': 0.25,
        'lightgbm': 0.25,
        'random_forest': 0.25,
        'elastic_net': 0.25,
        'lstm': 0.25,
        'gru': 0.25,
        'transformer': 0.25
    },
    'weight_update_method': 'performance_based',  # 'performance_based', 'equal', 'static'
    'confidence_level': 0.95,
    'min_models_for_ensemble': 3
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'lag_periods': [1, 2, 3, 7, 14, 30],
    'ma_windows': [3, 7, 14, 30, 60, 90],
    'holiday_windows': [1, 3, 7, 14],
    'seasonal_period': 12,
    'correlation_threshold': 0.1,
    'missing_threshold': 0.05,
    'outlier_threshold': 1.5
}

# Capacity Prediction Configuration
CAPACITY_CONFIG = {
    'safety_margin': 0.15,
    'peak_multiplier': 1.3,
    'target_utilization': 0.85,
    'emergency_capacity_ratio': 0.20,
    'warehouse_rent_per_sqm': 15.5,
    'labor_cost_per_hour': 25.0,
    'equipment_maintenance_ratio': 0.05,
    'seasonal_adjustment_factors': {
        'Q1': 0.9,  # Low season
        'Q2': 1.0,  # Normal season
        'Q3': 1.1,  # High season
        'Q4': 1.2   # Peak season
    }
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'test_size': 0.2,
    'cv_folds': 5,
    'scoring_metrics': ['mae', 'mse', 'rmse', 'mape', 'r2_score'],
    'confidence_levels': [0.90, 0.95, 0.99],
    'forecast_horizon': 30  # days
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_models': True,
    'save_visualizations': True,
    'save_predictions': True,
    'save_metrics': True,
    'output_format': 'csv',
    'visualization_dpi': 300,
    'compression': None
}

# Data Processing Configuration
DATA_CONFIG = {
    'date_column': 'date',
    'target_columns': ['total_orders', 'total_sales', 'avg_order_value'],
    'categorical_columns': ['seller_id', 'product_category', 'state'],
    'numeric_columns': ['order_count', 'total_sales', 'avg_price', 'avg_freight'],
    'time_frequencies': ['D', 'W', 'M'],  # Daily, Weekly, Monthly
    'missing_value_method': 'median',
    'outlier_detection_method': 'iqr'
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'max_memory_usage': '8GB',
    'parallel_processing': True,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': True,
    'log_level': 'INFO'
} 