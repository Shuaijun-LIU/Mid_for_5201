"""
Week 6 - Task 1: Time Series Feature Engineering
Prepare advanced time series features for prediction models
Execution date: 2025-07-13
Update date: 2025-07-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

warnings.filterwarnings('ignore')

class TimeSeriesFeatureEngineer:
    def __init__(self):
        self.data = {}
        self.features = {}
        self.output_dir = 'output'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f'{self.output_dir}/visualizations', exist_ok=True)
        
        # Config
        self.config = {
            'lag_periods': [1, 2, 3, 7, 14, 30],
            'ma_windows': [3, 7, 14, 30, 60, 90],
            'holiday_windows': [1, 3, 7, 14],
            'seasonal_period': 12,  # Monthly period
            'correlation_threshold': 0.1,
            'missing_threshold': 0.05
        }
        
    def load_data(self):
        # Load Week5 four-dimensional analysis results
        print("Loading Week5 four-dimensional analysis results...")
        
        try:
            # Load Week5 pre-processed four-dimensional data
            self.data['four_d_main'] = pd.read_csv('../week5_seller_analysis_and_four_d_analysis/output/four_d_main_analysis.csv')
            self.data['seller_summary'] = pd.read_csv('../week5_seller_analysis_and_four_d_analysis/output/seller_summary_analysis.csv')
            self.data['product_summary'] = pd.read_csv('../week5_seller_analysis_and_four_d_analysis/output/product_summary_analysis.csv')
            self.data['geo_summary'] = pd.read_csv('../week5_seller_analysis_and_four_d_analysis/output/geo_summary_analysis.csv')
            self.data['time_summary'] = pd.read_csv('../week5_seller_analysis_and_four_d_analysis/output/time_summary_analysis.csv')
            
            # Load Week2 pre-processed time series data
            print("Loading Week2 pre-processed time series data...")
            self.data['weekend_holiday'] = pd.read_csv('../week2_eda_update/output/weekend_holiday_analysis/orders_with_weekend_holiday.csv')
            
            print("Data loading completed!")
            self._print_data_summary()
            
        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            raise
            
    def _print_data_summary(self):
        print("Data Summary:")
        for name, df in self.data.items():
            print(f"  {name}: {df.shape}")
            
    def prepare_time_series_data(self):
        print("Preparing time series data from Week5 results...")
        
        try:
            # Directly use Week5's four-dimensional main analysis as base
            four_d_data = self.data['four_d_main'].copy()
            
            # Convert month to datetime for time series operations
            four_d_data['month'] = pd.to_datetime(four_d_data['month'].astype(str))
            four_d_data['date'] = four_d_data['month'].dt.date
            
            # Use the data directly as our base aggregated data
            self.features['base_aggregated'] = four_d_data
            
            # Add additional features from Week5 summaries
            print("Adding additional features from Week5 summaries...")
            self._add_week5_features()
            
            # Handle missing values and outliers
            print("Handling missing values and outliers...")
            self._handle_missing_and_outliers()
            
            print(f"Time series data preparation completed: {self.features['base_aggregated'].shape}")
            
        except Exception as e:
            print(f"Time series data preparation failed: {str(e)}")
            raise
            
    def _add_week5_features(self):
        # add additional features from Week5 analysis results
        base_data = self.features['base_aggregated']
        
        # Add seller-level features
        seller_features = self.data['seller_summary'][['seller_id', 'product_diversity', 'geographic_coverage', 'temporal_coverage', 'avg_complexity', 'avg_retention']]
        base_data = base_data.merge(seller_features, on='seller_id', how='left')
        
        # Add product-level features
        product_features = self.data['product_summary'][['product_category', 'seller_diversity', 'geographic_distribution', 'temporal_coverage', 'dominant_lifecycle']]
        base_data = base_data.merge(product_features, on='product_category', how='left')
        
        # add geographic features
        geo_features = self.data['geo_summary'][['state', 'seller_concentration', 'product_mix', 'temporal_coverage']]
        base_data = base_data.merge(geo_features, on='state', how='left')
        
        # Add time features
        time_features = self.data['time_summary'][['month', 'active_sellers', 'active_categories', 'active_states']]
        time_features['month'] = pd.to_datetime(time_features['month'].astype(str))
        time_features['date'] = time_features['month'].dt.date
        base_data = base_data.merge(time_features[['date', 'active_sellers', 'active_categories', 'active_states']], on='date', how='left')
        
        self.features['base_aggregated'] = base_data
            
    def _ensure_time_series_continuity(self):
        # Skip this step since we're using monthly data from Week5
        # Week5 already provides continuous monthly data
        pass
        
    def _handle_missing_and_outliers(self):
        # Handle missing values and outliers
        # Check missing values
        missing_ratio = self.features['base_aggregated'].isnull().sum() / len(self.features['base_aggregated'])
        print(f"Missing value ratio: {missing_ratio}")
        
        # Fill missing values with appropriate methods
        numeric_cols = ['order_count', 'total_sales', 'avg_price', 'avg_freight']
        for col in numeric_cols:
            if col in self.features['base_aggregated'].columns:
                # Fill missing values with median
                self.features['base_aggregated'][col] = self.features['base_aggregated'][col].fillna(
                    self.features['base_aggregated'][col].median()
                )
            
    def create_lag_features(self):
        print("Starting to create lag features...")
        
        try:
            df = self.features['base_aggregated'].copy()
            lag_features = []
            
            # Create lag features by seller-product-geo groups
            for group_name, group in df.groupby(['seller_id', 'product_category', 'state']):
                group = group.sort_values('date')
                
                # Create lag features
                for lag in self.config['lag_periods']:
                    if len(group) > lag:
                        group[f'order_count_lag{lag}'] = group['order_count'].shift(lag)
                        group[f'total_sales_lag{lag}'] = group['total_sales'].shift(lag)
                        group[f'avg_price_lag{lag}'] = group['avg_price'].shift(lag)
                        group[f'avg_freight_lag{lag}'] = group['avg_freight'].shift(lag)
                
                lag_features.append(group)
            
            self.features['with_lags'] = pd.concat(lag_features, ignore_index=True)
            print("Lag features creation completed")
            
        except Exception as e:
            print(f"Lag features creation failed: {str(e)}")
            raise
            
    def create_moving_average_features(self):
        # create moving average features
        print("Starting to create moving average features...")
        
        try:
            df = self.features['with_lags'].copy()
            ma_features = []
            
            # create moving average features
            for group_name, group in df.groupby(['seller_id', 'product_category', 'state']):
                group = group.sort_values('date')
            
                for window in self.config['ma_windows']:
                    if len(group) >= window:
                        group[f'order_count_ma{window}'] = group['order_count'].rolling(window=window, min_periods=1).mean()
                        group[f'total_sales_ma{window}'] = group['total_sales'].rolling(window=window, min_periods=1).mean()
                        group[f'avg_price_ma{window}'] = group['avg_price'].rolling(window=window, min_periods=1).mean()
                        group[f'avg_freight_ma{window}'] = group['avg_freight'].rolling(window=window, min_periods=1).mean()
                
                # create weighted moving average
                group['order_count_wma7'] = group['order_count'].rolling(window=7, min_periods=1).apply(
                    lambda x: np.average(x, weights=np.arange(1, len(x) + 1))
                )
                group['total_sales_wma7'] = group['total_sales'].rolling(window=7, min_periods=1).apply(
                    lambda x: np.average(x, weights=np.arange(1, len(x) + 1))
                )
                
                ma_features.append(group)
            
            self.features['with_ma'] = pd.concat(ma_features, ignore_index=True)
            print("Moving average features creation completed")
            
        except Exception as e:
            print(f"Moving average features creation failed: {str(e)}")
            raise
            
    def create_seasonal_features(self):
        # Create seasonal features
        print("Starting to create seasonal features...")
        
        try:
            df = self.features['with_ma'].copy()
            seasonal_features = []
            
            # Create seasonal features by seller-product-geo groups
            for group_name, group in df.groupby(['seller_id', 'product_category', 'state']):
                group = group.sort_values('date')
                
                if len(group) >= self.config['seasonal_period'] * 2:
                    # STL decomposition
                    try:
                        # Ensure data is continuous
                        group_ts = group.set_index('date')['order_count']
                        decomposition = seasonal_decompose(
                            group_ts, 
                            period=self.config['seasonal_period'], 
                            extrapolate_trend='freq'
                        )
                        
                        group['trend_component'] = decomposition.trend
                        group['seasonal_component'] = decomposition.seasonal
                        group['residual_component'] = decomposition.resid
                        
                        # Calculate seasonal strength
                        seasonal_strength = np.abs(decomposition.seasonal).sum() / (np.abs(decomposition.seasonal).sum() + np.abs(decomposition.resid).sum())
                        group['seasonal_strength'] = seasonal_strength
                        
                    except Exception as e:
                        print(f"STL decomposition failed: {str(e)}")
                        group['trend_component'] = np.nan
                        group['seasonal_component'] = np.nan
                        group['residual_component'] = np.nan
                        group['seasonal_strength'] = np.nan
                
                # Create trend features
                group['linear_trend'] = np.arange(len(group))
                group['exponential_trend'] = np.exp(np.arange(len(group)) / len(group))
                
                seasonal_features.append(group)
            
            self.features['with_seasonal'] = pd.concat(seasonal_features, ignore_index=True)
            print("Seasonal features creation completed")
            
        except Exception as e:
            print(f"Seasonal features creation failed: {str(e)}")
            raise
            
    def create_holiday_features(self):
        print("Starting to create holiday features...")
        
        try:
            df = self.features['with_seasonal'].copy()
            
            # use Week2 pre-processed holiday data
            if 'weekend_holiday' in self.data:
                holiday_data = self.data['weekend_holiday'].copy()
                holiday_data['order_date'] = pd.to_datetime(holiday_data['order_purchase_timestamp']).dt.date
                
                # Ensure order_date is unique for mapping
                holiday_map = holiday_data.drop_duplicates('order_date').set_index('order_date')['holiday_name']
                daycat_map = holiday_data.drop_duplicates('order_date').set_index('order_date')['day_category']
                
                # Create holiday indicators using Week2 results
                df['is_holiday'] = df['date'].isin(holiday_data['order_date'])
                df['holiday_name'] = df['date'].map(holiday_map).fillna('Regular Day')
                df['day_category'] = df['date'].map(daycat_map).fillna('Weekday')
            
            # Holiday effect features 
            for window in self.config['holiday_windows']:
                df[f'holiday_effect_{window}d'] = (
                    df.groupby(['seller_id', 'product_category', 'state'])['is_holiday']
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                )
            
            # Holiday intensity by seller-product-state combination (unique to Week6)
            holiday_intensity = df.groupby(['seller_id', 'product_category', 'state'])['is_holiday'].mean()
            df['holiday_intensity'] = df.set_index(['seller_id', 'product_category', 'state'])['is_holiday'].map(holiday_intensity).values
            
            self.features['with_holiday'] = df
            print("Holiday features creation completed (using Week2 pre-processed data)")
            
        except Exception as e:
            print(f"Holiday features creation failed: {str(e)}")
            raise
            
    def create_cyclical_features(self):
        print("Starting to create cyclical features...")
        
        try:
            df = self.features['with_holiday'].copy()
            
            # Time cyclical features 
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['year'] = df['date'].dt.year
            
            # Cyclical encoding
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
            df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
            
            # Seasonal encoding 
            df['season'] = df['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
            
            # Seasonal one-hot encoding 
            season_dummies = pd.get_dummies(df['season'], prefix='season')
            df = pd.concat([df, season_dummies], axis=1)
            
            self.features['with_cyclical'] = df
            print("Cyclical features creation completed ")
            
        except Exception as e:
            print(f"Cyclical features creation failed: {str(e)}")
            raise
            
    def create_differential_features(self):
        # Create differential features
        print("Starting to create differential features...")
        
        try:
            df = self.features['with_cyclical'].copy()
            diff_features = []
            
            # Create differential features by seller-product-geo groups
            for group_name, group in df.groupby(['seller_id', 'product_category', 'state']):
                group = group.sort_values('date')
                
                # First-order differences
                group['order_count_diff1'] = group['order_count'].diff()
                group['total_sales_diff1'] = group['total_sales'].diff()
                group['avg_price_diff1'] = group['avg_price'].diff()
                group['avg_freight_diff1'] = group['avg_freight'].diff()
                
                # Second-order differences
                group['order_count_diff2'] = group['order_count_diff1'].diff()
                group['total_sales_diff2'] = group['total_sales_diff1'].diff()
                
                # 7-day differences
                group['order_count_diff7'] = group['order_count'].diff(7)
                group['total_sales_diff7'] = group['total_sales'].diff(7)
                
                # 30-day differences
                group['order_count_diff30'] = group['order_count'].diff(30)
                group['total_sales_diff30'] = group['total_sales'].diff(30)
                
                diff_features.append(group)
            
            self.features['with_diffs'] = pd.concat(diff_features, ignore_index=True)
            print("Differential features creation completed")
            
        except Exception as e:
            print(f"Differential features creation failed: {str(e)}")
            raise
            
    def create_transformation_features(self):
        # Create transformation features
        print("Starting to create transformation features...")
        
        try:
            df = self.features['with_diffs'].copy()
            
            # Logarithmic transformation (handle zero and negative values)
            numeric_cols = ['order_count', 'total_sales', 'avg_price', 'avg_freight']
            for col in numeric_cols:
                if col in df.columns:
                    # Add small constant to avoid log(0)
                    df[f'{col}_log'] = np.log1p(df[col])
                    
                    # Square root transformation
                    df[f'{col}_sqrt'] = np.sqrt(df[col])
                    
                    # Square transformation
                    df[f'{col}_square'] = df[col] ** 2
            
            # Ratio features
            df['sales_per_order'] = df['total_sales'] / (df['order_count'] + 1e-8)
            df['price_to_freight_ratio'] = df['avg_price'] / (df['avg_freight'] + 1e-8)
            
            # Standardization features
            scaler = StandardScaler()
            for col in numeric_cols:
                if col in df.columns:
                    df[f'{col}_standardized'] = scaler.fit_transform(df[[col]])
            
            # Min-Max standardization
            minmax_scaler = MinMaxScaler()
            for col in numeric_cols:
                if col in df.columns:
                    df[f'{col}_minmax'] = minmax_scaler.fit_transform(df[[col]])
            
            self.features['final'] = df
            print("Transformation features creation completed")
            
        except Exception as e:
            print(f"Transformation features creation failed: {str(e)}")
            raise
            
    def analyze_feature_importance(self):
        # Analyze feature importance
        print("Starting feature importance analysis...")
        
        try:
            df = self.features['final'].copy()
            
            # Select numeric features
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            target_vars = ['order_count', 'total_sales', 'avg_price', 'avg_freight']
            
            # Calculate correlation with target variables
            importance_data = []
            for target in target_vars:
                if target in numeric_features:
                    for feature in numeric_features:
                        if feature != target and not feature.startswith(target):
                            correlation = df[feature].corr(df[target])
                            if not pd.isna(correlation):
                                importance_data.append({
                                    'target_variable': target,
                                    'feature': feature,
                                    'correlation': abs(correlation),
                                    'correlation_raw': correlation
                                })
            
            self.features['importance'] = pd.DataFrame(importance_data)
            
            # Sort by correlation
            self.features['importance'] = self.features['importance'].sort_values(
                'correlation', ascending=False
            )
            
            print("Feature importance analysis completed")
            
        except Exception as e:
            print(f"Feature importance analysis failed: {str(e)}")
            raise
            
    def analyze_feature_correlation(self):
        # Analyze feature correlation
        print("Starting feature correlation analysis...")
        
        try:
            df = self.features['final'].copy()
            
            # Select numeric features
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Calculate correlation matrix
            correlation_matrix = df[numeric_features].corr()
            
            # Identify high correlation feature pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:  # High correlation threshold
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            self.features['correlation_matrix'] = correlation_matrix
            self.features['high_correlation_pairs'] = pd.DataFrame(high_corr_pairs)
            
            print("Feature correlation analysis completed")
            
        except Exception as e:
            print(f"Feature correlation analysis failed: {str(e)}")
            raise
            
    def create_visualizations(self):
        # Create visualization charts
        print("Starting to create visualization charts...")
        
        try:
            df = self.features['final'].copy()
            
            # Set chart style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Week6 Time Series Feature Engineering Analysis', fontsize=16, fontweight='bold')
            
            # 1. Lag feature effects 
            if 'order_count_lag7' in df.columns:
                lag_corr = df['order_count'].corr(df['order_count_lag7'])
                axes[0,0].scatter(df['order_count_lag7'], df['order_count'], alpha=0.6)
                axes[0,0].set_title(f'Lag 7 Correlation: {lag_corr:.3f}')
                axes[0,0].set_xlabel('Order Count (Lag 7)')
                axes[0,0].set_ylabel('Order Count')
            
            # 2. Moving average features 
            if 'order_count_ma7' in df.columns:
                sample_ma = df.groupby('date')[['order_count', 'order_count_ma7']].mean().reset_index()
                sample_ma['date'] = pd.to_datetime(sample_ma['date'])
                
                axes[0,1].plot(sample_ma['date'], sample_ma['order_count'], label='Original', alpha=0.7)
                axes[0,1].plot(sample_ma['date'], sample_ma['order_count_ma7'], label='7-day MA', linewidth=2)
                axes[0,1].set_title('Moving Average Comparison')
                axes[0,1].set_xlabel('Date')
                axes[0,1].set_ylabel('Order Count')
                axes[0,1].legend()
                axes[0,1].tick_params(axis='x', rotation=45)
            
            # 3. Seasonal decomposition  
            if 'seasonal_component' in df.columns:
                sample_seasonal = df.groupby('date')['seasonal_component'].mean().reset_index()
                sample_seasonal['date'] = pd.to_datetime(sample_seasonal['date'])
                
                axes[0,2].plot(sample_seasonal['date'], sample_seasonal['seasonal_component'])
                axes[0,2].set_title('Seasonal Component')
                axes[0,2].set_xlabel('Date')
                axes[0,2].set_ylabel('Seasonal Component')
                axes[0,2].tick_params(axis='x', rotation=45)
            
            # 4. Holiday intensity by seller-product-state  
            if 'holiday_intensity' in df.columns:
                holiday_intensity_dist = df['holiday_intensity'].hist(bins=20, ax=axes[1,0])
                axes[1,0].set_title('Holiday Intensity Distribution')
                axes[1,0].set_xlabel('Holiday Intensity')
                axes[1,0].set_ylabel('Frequency')
            
            # 5. Cyclical features  
            if 'month_sin' in df.columns:
                axes[1,1].scatter(df['month_sin'], df['order_count'], alpha=0.6)
                axes[1,1].set_title('Monthly Cyclical Effect')
                axes[1,1].set_xlabel('Month (Sine)')
                axes[1,1].set_ylabel('Order Count')
            
            # 6. Feature importance  
            if 'importance' in self.features:
                top_features = self.features['importance'].head(10)
                axes[1,2].barh(range(len(top_features)), top_features['correlation'])
                axes[1,2].set_yticks(range(len(top_features)))
                axes[1,2].set_yticklabels(top_features['feature'], fontsize=8)
                axes[1,2].set_title('Top 10 Feature Importance')
                axes[1,2].set_xlabel('Absolute Correlation')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/visualizations/week6_time_series_features.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create correlation heatmap  
            if 'correlation_matrix' in self.features:
                plt.figure(figsize=(12, 10))
                sns.heatmap(
                    self.features['correlation_matrix'].iloc[:20, :20], 
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    fmt='.2f'
                )
                plt.title('Week6 Feature Correlation Heatmap (Top 20 Features)')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/visualizations/week6_feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            print("visualization charts creation completed")
            
        except Exception as e:
            print(f"Visualization creation failed: {str(e)}")
            raise
            
    def save_outputs(self):
        print("Starting to save Week6 output results...")
        
        try:
            # Save feature matrix
            self.features['final'].to_csv(f'{self.output_dir}/week6_time_series_features_matrix.csv', index=False)
            print(f"Week6 feature matrix saved: {self.output_dir}/week6_time_series_features_matrix.csv")
            
            # Save feature importance
            if 'importance' in self.features:
                self.features['importance'].to_csv(f'{self.output_dir}/week6_feature_importance_ranking.csv', index=False)
                print(f"Week6 feature importance saved: {self.output_dir}/week6_feature_importance_ranking.csv")
            
            # Save correlation matrix
            if 'correlation_matrix' in self.features:
                self.features['correlation_matrix'].to_csv(f'{self.output_dir}/week6_feature_correlation_matrix.csv')
                print(f"Week6 correlation matrix saved: {self.output_dir}/week6_feature_correlation_matrix.csv")
            
            # Save high correlation feature pairs
            if 'high_correlation_pairs' in self.features:
                self.features['high_correlation_pairs'].to_csv(f'{self.output_dir}/week6_high_correlation_pairs.csv', index=False)
                print(f"Week6 high correlation pairs saved: {self.output_dir}/week6_high_correlation_pairs.csv")
            
            # Create feature statistics summary
            self._create_feature_statistics_summary()
            
            print("All Week6 output results saved")
            
        except Exception as e:
            print(f"Output saving failed: {str(e)}")
            raise
            
    def _create_feature_statistics_summary(self):
        try:
            df = self.features['final'].copy()
            
            # Select numeric features
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # statistics summary
            stats_summary = df[numeric_features].describe()
            
            # missing
            missing_stats = df[numeric_features].isnull().sum()
            stats_summary.loc['missing_count'] = missing_stats
            stats_summary.loc['missing_ratio'] = missing_stats / len(df)

            stats_summary.to_csv(f'{self.output_dir}/week6_feature_statistics_summary.csv')
            print(f"Week6 feature statistics summary saved: {self.output_dir}/week6_feature_statistics_summary.csv")
            
        except Exception as e:
            print(f"Feature statistics summary creation failed: {str(e)}")
            
    def run_complete_feature_engineering(self):
        print("=" * 60)
        print("TIME SERIES FEATURE ENGINEERING")
        print("=" * 60)
        
        try:
            self.load_data()
            self.prepare_time_series_data()
            self.create_lag_features()
            self.create_moving_average_features()
            self.create_seasonal_features()
            self.create_holiday_features()
            self.create_cyclical_features()
            self.create_differential_features()
            self.create_transformation_features()
            self.analyze_feature_importance()
            self.analyze_feature_correlation()
            self.create_visualizations()
            self.save_outputs()
            
            print("\n" + "=" * 60)
            print("TIME SERIES FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            return self.features['final']
            
        except Exception as e:
            print(f"Feature engineering pipeline failed: {str(e)}")
            raise

def main():
    # Main function
    try:
        # Create feature engineer
        feature_engineer = TimeSeriesFeatureEngineer()
        
        # Run complete pipeline
        final_features = feature_engineer.run_complete_feature_engineering()
        
        print("\nWeek6 feature engineering completed!")
        print(f"Final feature matrix shape: {final_features.shape}")
        print(f"Number of features: {len(final_features.columns)}")
        print(f"Number of samples: {len(final_features)}")
        
        print("\nWeek6 output files:")
        print("  - week6_time_series_features_matrix.csv")
        print("  - week6_feature_importance_ranking.csv")
        print("  - week6_feature_correlation_matrix.csv")
        print("  - week6_feature_statistics_summary.csv")
        print("  - visualizations/week6_*.png")
        
    except Exception as e:
        print(f"Program execution failed: {str(e)}")

if __name__ == "__main__":
    main() 