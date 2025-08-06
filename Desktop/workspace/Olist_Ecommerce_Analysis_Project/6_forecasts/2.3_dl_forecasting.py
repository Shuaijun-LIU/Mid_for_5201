"""
Task 2.3: Deep Learning Forecasting Models
LSTM, GRU, and Transformer models for time series forecasting
Based on Task 1's time series feature engineering results
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

# Deep learning libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Import shared components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.shared_components import DataLoader, FeatureProcessor, ModelEvaluator, OutputManager, BaseForecaster
from config.model_config import DL_MODELS, EVALUATION_CONFIG, OUTPUT_CONFIG

# Suppress warnings
warnings.filterwarnings('ignore')

class DLForecaster(BaseForecaster):
    """Deep learning forecasting with LSTM, GRU, and Transformer models"""
    def __init__(self, output_dir: str = 'output'):
        super().__init__(output_dir)
        self.config = DL_MODELS
        self.eval_config = EVALUATION_CONFIG
        self.output_config = OUTPUT_CONFIG
        
        # Model storage
        self.trained_models = {}
        self.forecast_results = {}
        self.performance_metrics = {}
        self.scalers = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from Task 1 and Week5 outputs"""
        print("Loading data for deep learning forecasting...")
        
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
        """Prepare features for deep learning models"""
        print("Preparing features for deep learning forecasting...")
        
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
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for deep learning models"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def _prepare_dl_data(self, data: pd.DataFrame, target: str, sequence_length: int) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Prepare data for deep learning models"""
        # Select target column
        target_data = data[target].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(target_data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data, sequence_length)
        
        return X, y, scaler
    
    def build_lstm_model(self, sequence_length: int, units: int = 50, dropout: float = 0.2) -> Sequential:
        """Build LSTM model"""
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(sequence_length, 1), 
                 dropout=dropout, recurrent_dropout=dropout),
            LSTM(units // 2, dropout=dropout, recurrent_dropout=dropout),
            Dense(units // 4, activation='relu'),
            Dropout(dropout),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def build_gru_model(self, sequence_length: int, units: int = 50, dropout: float = 0.2) -> Sequential:
        """Build GRU model"""
        model = Sequential([
            GRU(units, return_sequences=True, input_shape=(sequence_length, 1), 
                dropout=dropout, recurrent_dropout=dropout),
            GRU(units // 2, dropout=dropout, recurrent_dropout=dropout),
            Dense(units // 4, activation='relu'),
            Dropout(dropout),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def build_transformer_model(self, sequence_length: int, d_model: int = 64, n_heads: int = 8, 
                               n_layers: int = 4, dropout: float = 0.1) -> tf.keras.Model:
        """Build Transformer model"""
        inputs = Input(shape=(sequence_length, 1))
        
        # Embedding layer
        x = Dense(d_model)(inputs)
        
        # Transformer blocks
        for _ in range(n_layers):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=n_heads, key_dim=d_model, dropout=dropout
            )(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed forward network
            ffn = Sequential([
                Dense(d_model * 4, activation='relu'),
                Dropout(dropout),
                Dense(d_model)
            ])
            ffn_output = ffn(x)
            x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        # Global average pooling and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(d_model // 2, activation='relu')(x)
        x = Dropout(dropout)(x)
        outputs = Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train_lstm_model(self, X: np.ndarray, y: np.ndarray, target: str) -> Tuple[Sequential, Dict]:
        """Train LSTM model"""
        print(f"Training LSTM model for {target}...")
        
        # Get model configuration
        config = self.config['lstm']
        sequence_length = config['sequence_length']
        units = config['units']
        dropout = config['dropout']
        epochs = config['epochs']
        batch_size = config['batch_size']
        validation_split = config['validation_split']
        early_stopping_patience = config['early_stopping_patience']
        
        # Build model
        model = self.build_lstm_model(sequence_length, units, dropout)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Predictions for evaluation
        predictions = model.predict(X)
        
        # Calculate metrics
        metrics = self.model_evaluator.calculate_metrics(y, predictions.flatten())
        
        return model, {
            'history': history.history,
            'metrics': metrics,
            'config': config
        }
    
    def train_gru_model(self, X: np.ndarray, y: np.ndarray, target: str) -> Tuple[Sequential, Dict]:
        """Train GRU model"""
        print(f"Training GRU model for {target}...")
        
        # Get model configuration
        config = self.config['gru']
        sequence_length = config['sequence_length']
        units = config['units']
        dropout = config['dropout']
        epochs = config['epochs']
        batch_size = config['batch_size']
        validation_split = config['validation_split']
        early_stopping_patience = config['early_stopping_patience']
        
        # Build model
        model = self.build_gru_model(sequence_length, units, dropout)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Predictions for evaluation
        predictions = model.predict(X)
        
        # Calculate metrics
        metrics = self.model_evaluator.calculate_metrics(y, predictions.flatten())
        
        return model, {
            'history': history.history,
            'metrics': metrics,
            'config': config
        }
    
    def train_transformer_model(self, X: np.ndarray, y: np.ndarray, target: str) -> Tuple[tf.keras.Model, Dict]:
        """Train Transformer model"""
        print(f"Training Transformer model for {target}...")
        
        # Get model configuration
        config = self.config['transformer']
        sequence_length = config['sequence_length']
        d_model = config['d_model']
        n_heads = config['n_heads']
        n_layers = config['n_layers']
        dropout = config['dropout']
        epochs = config['epochs']
        batch_size = config['batch_size']
        validation_split = config['validation_split']
        early_stopping_patience = config['early_stopping_patience']
        
        # Build model
        model = self.build_transformer_model(sequence_length, d_model, n_heads, n_layers, dropout)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Predictions for evaluation
        predictions = model.predict(X)
        
        # Calculate metrics
        metrics = self.model_evaluator.calculate_metrics(y, predictions.flatten())
        
        return model, {
            'history': history.history,
            'metrics': metrics,
            'config': config
        }
    
    def train_models(self) -> Dict:
        """Train all deep learning models"""
        print("Training deep learning models...")
        
        models = {}
        
        for timeframe, data in self.features.items():
            print(f"\nTraining models for {timeframe} data...")
            models[timeframe] = {}
            
            if len(data) > 500:
                data = data.tail(500)
            
            # Get target columns
            target_columns = ['total_orders', 'total_sales', 'avg_order_value']
            available_targets = [col for col in target_columns if col in data.columns]
            
            for target in available_targets:
                print(f"Training models for target: {target}")
                models[timeframe][target] = {}
                
                # Prepare data
                sequence_length = self.config['lstm']['sequence_length']
                X, y, scaler = self._prepare_dl_data(data, target, sequence_length)
                
                if len(X) < sequence_length + 10:  # Need sufficient data
                    print(f"Insufficient data for {target}: {len(X)} samples")
                    continue
                
                # Store scaler
                self.scalers[f"{timeframe}_{target}"] = scaler
                
                # Train LSTM
                try:
                    lstm_model, lstm_info = self.train_lstm_model(X, y, target)
                    models[timeframe][target]['lstm'] = lstm_model
                    self.performance_metrics[f"{timeframe}_{target}_lstm"] = lstm_info
                except Exception as e:
                    print(f"LSTM training failed for {target}: {e}")
                
                # Train GRU
                try:
                    gru_model, gru_info = self.train_gru_model(X, y, target)
                    models[timeframe][target]['gru'] = gru_model
                    self.performance_metrics[f"{timeframe}_{target}_gru"] = gru_info
                except Exception as e:
                    print(f"GRU training failed for {target}: {e}")
                
                # Train Transformer
                try:
                    transformer_model, transformer_info = self.train_transformer_model(X, y, target)
                    models[timeframe][target]['transformer'] = transformer_model
                    self.performance_metrics[f"{timeframe}_{target}_transformer"] = transformer_info
                except Exception as e:
                    print(f"Transformer training failed for {target}: {e}")
        
        self.trained_models = models
        return models
    
    def generate_forecasts(self) -> Dict:
        """Generate forecasts using trained models"""
        print("Generating deep learning forecasts...")
        
        forecasts = {}
        
        for timeframe, timeframe_models in self.trained_models.items():
            forecasts[timeframe] = {}
            
            for target, target_models in timeframe_models.items():
                forecasts[timeframe][target] = {}
                
                # Get scaler
                scaler = self.scalers.get(f"{timeframe}_{target}")
                if scaler is None:
                    continue
                
                # Get the latest sequence for forecasting
                sequence_length = self.config['lstm']['sequence_length']
                latest_data = self.features[timeframe][target].values[-sequence_length:].reshape(-1, 1)
                scaled_sequence = scaler.transform(latest_data)
                
                for model_name, model in target_models.items():
                    try:
                        # Prepare input sequence
                        X_forecast = scaled_sequence.reshape(1, sequence_length, 1)
                        
                        # Generate forecast
                        prediction = model.predict(X_forecast)[0][0]
                        
                        # Inverse transform
                        prediction_original = scaler.inverse_transform([[prediction]])[0][0]
                        
                        # Create forecast dates
                        forecast_dates = pd.date_range(
                            start=self.features[timeframe]['date'].iloc[-1] + timedelta(days=1),
                            periods=self.eval_config['forecast_horizon'],
                            freq='D'
                        )
                        
                        # Extend forecast (simple approach - use the same prediction)
                        predictions = np.full(self.eval_config['forecast_horizon'], prediction_original)
                        
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
        print("Evaluating deep learning models...")
        
        evaluation_results = {}
        
        for timeframe, timeframe_models in self.trained_models.items():
            evaluation_results[timeframe] = {}
            
            for target, target_models in timeframe_models.items():
                evaluation_results[timeframe][target] = {}
                
                # Get scaler
                scaler = self.scalers.get(f"{timeframe}_{target}")
                if scaler is None:
                    continue
                
                # Prepare evaluation data
                sequence_length = self.config['lstm']['sequence_length']
                X_eval, y_eval, _ = self._prepare_dl_data(self.features[timeframe], target, sequence_length)
                
                for model_name, model in target_models.items():
                    # Get predictions for evaluation period
                    predictions = model.predict(X_eval)
                    
                    # Inverse transform predictions
                    predictions_original = scaler.inverse_transform(predictions).flatten()
                    y_original = scaler.inverse_transform(y_eval.reshape(-1, 1)).flatten()
                    
                    # Calculate metrics
                    metrics = self.model_evaluator.calculate_metrics(y_original, predictions_original)
                    evaluation_results[timeframe][target][model_name] = metrics
        
        return evaluation_results
    
    def save_results(self) -> None:
        """Save all results"""
        print("Saving deep learning forecasting results...")
        
        # Save performance metrics
        if self.performance_metrics:
            metrics_df = pd.DataFrame.from_dict(self.performance_metrics, orient='index')
            self.output_manager.save_dataframe(metrics_df, 'dl_model_performance.csv')
        
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
                self.output_manager.save_dataframe(forecast_df, 'dl_forecasts.csv')
        
        # Save trained models
        if self.output_config['save_models']:
            for timeframe, timeframe_models in self.trained_models.items():
                for target, target_models in timeframe_models.items():
                    for model_name, model in target_models.items():
                        model_filename = f"{timeframe}_{target}_{model_name}"
                        self.output_manager.save_model(model, model_filename, model_name)
        
        # Save scalers
        if self.scalers:
            scaler_data = []
            for key, scaler in self.scalers.items():
                scaler_data.append({
                    'key': key,
                    'min_': scaler.min_[0],
                    'scale_': scaler.scale_[0],
                    'data_min_': scaler.data_min_[0],
                    'data_max_': scaler.data_max_[0]
                })
            
            if scaler_data:
                scaler_df = pd.DataFrame(scaler_data)
                self.output_manager.save_dataframe(scaler_df, 'dl_scalers.csv')
        
        # Create visualizations
        if self.output_config['save_visualizations']:
            self.create_visualizations()
    
    def create_visualizations(self):
        """Create visualization charts"""
        print("Creating deep learning forecasting visualizations...")
        
        # Model performance comparison
        if self.performance_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Deep Learning Model Performance Comparison', fontsize=16)
            
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
            self.output_manager.save_visualization(fig, 'dl_model_performance.png')
        
        # Training history plots
        if self.performance_metrics:
            for key, metrics in self.performance_metrics.items():
                if 'history' in metrics:
                    history = metrics['history']
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Loss plot
                    ax1.plot(history['loss'], label='Training Loss')
                    if 'val_loss' in history:
                        ax1.plot(history['val_loss'], label='Validation Loss')
                    ax1.set_title(f'Training History - {key}')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Metrics plot
                    if 'metrics' in metrics:
                        mape = metrics['metrics']['mape']
                        r2 = metrics['metrics']['r2_score']
                        ax2.bar(['MAPE', 'R²'], [mape, r2])
                        ax2.set_title(f'Model Performance - {key}')
                        ax2.set_ylabel('Score')
                        ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    self.output_manager.save_visualization(fig, f'{key}_training_history.png')

def main():
    """Main execution function"""
    print("Starting Deep Learning Forecasting Analysis")
    print("Select model mode:")
    print("1. Only GRU (fastest, lowest resource)")
    print("2. All models (GRU, LSTM, Transformer)")
    mode = input("Enter 1 or 2: ").strip()
    if mode not in ('1', '2'):
        print("Invalid input, defaulting to 1 (Only GRU)")
        mode = '1'

    # Initialize forecaster
    forecaster = DLForecaster()

    # Patch train_models to respect mode
    orig_train_lstm = forecaster.train_lstm_model
    orig_train_transformer = forecaster.train_transformer_model
    def train_lstm_disabled(*args, **kwargs):
        print("LSTM skipped (mode 1)")
        return None, {}
    def train_transformer_disabled(*args, **kwargs):
        print("Transformer skipped (mode 1)")
        return None, {}
    if mode == '1':
        forecaster.train_lstm_model = train_lstm_disabled
        forecaster.train_transformer_model = train_transformer_disabled

    try:
        # Run complete pipeline
        results = forecaster.run_complete_pipeline()
        print("\nDeep Learning Forecasting Analysis completed successfully")
        print(f"Models trained: {len(forecaster.trained_models)}")
        print(f"Forecasts generated: {len(forecaster.forecast_results)}")
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 