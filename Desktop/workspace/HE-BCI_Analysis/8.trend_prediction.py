import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path
import os
from matplotlib import cm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import xgboost as xgb
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import pmdarima as pm

warnings.filterwarnings("ignore")


def save_plot(fig, filename):
    save_dir = Path('visualizations/trend_prediction')
    save_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_dir / filename, bbox_inches='tight', dpi=300)
    plt.close()


def load_data():
    print("Loading data files...")
    
    ne_universities = [
        'University of Durham',
        'Newcastle University',
        'University of Northumbria at Newcastle',
        'Teesside University',
        'The University of Sunderland'
    ]

    data = {}
    
    # Load research income data
    data['research'] = pd.read_csv('Data/table-1.csv', skiprows=11)
    
    # Load business services data
    data['business'] = pd.read_csv('Data/table-2a.csv', skiprows=11)
    
    # Load CPD data
    data['cpd'] = pd.read_csv('Data/table-2b.csv', skiprows=11)
    
    # Load regeneration data
    data['regeneration'] = pd.read_csv('Data/table-3.csv', skiprows=11)
    
    # Load IP data
    data['ip_disclosures'] = pd.read_csv('Data/table-4a.csv', skiprows=11)
    data['ip_licenses'] = pd.read_csv('Data/table-4b.csv', skiprows=11)
    data['ip_income'] = pd.read_csv('Data/table-4c.csv', skiprows=11)
    data['ip_income_total'] = pd.read_csv('Data/table-4d.csv', skiprows=11)
    data['spinouts'] = pd.read_csv('Data/table-4e.csv', skiprows=11)
    
    # Load public engagement data
    data['public_engagement'] = pd.read_csv('Data/table-5.csv', skiprows=11)
    
    # Clean column names for all datasets
    for key in data:
        data[key].columns = data[key].columns.str.strip()
    
    print("Data files loaded successfully!")
    
    return data, ne_universities


def prepare_time_series(data, metric_name, value_column='Value'):
    """Prepare time series data for prediction"""
    # Get all unique academic years and sort them
    years = sorted(data['Academic Year'].unique())
    year_to_num = {year: i for i, year in enumerate(years)}

    # Convert academic years to numeric values
    X = np.array([year_to_num[year] for year in data['Academic Year']]).reshape(-1, 1)
    y = data[value_column].values

    # Remove NaN values
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    return X, y, years


def extract_clustering_features(data, ne_universities):
    """Extract features for clustering analysis"""
    print("\n=== Clustering Feature Extraction ===")
    
    features_data = []
    
    for univ in ne_universities:
        univ_features = {'University': univ}
        
        # Calculate features for each metric
        metrics = {
            'research': 'Value',
            'business': data['business'].columns[-1],
            'cpd': 'Value',
            'regeneration': 'Value',
            'ip_income': 'Value'
        }
        
        for metric_name, value_col in metrics.items():
            metric_data = data[metric_name]
            univ_data = metric_data[metric_data['HE Provider'] == univ]
            
            if len(univ_data) > 0:
                # Aggregate by year for CPD data
                if metric_name == 'cpd':
                    univ_data = univ_data.groupby('Academic Year')[value_col].sum().reset_index()
                
                univ_data = univ_data.sort_values('Academic Year')
                values = univ_data[value_col].values
                
                if len(values) > 1:
                    # Basic statistics
                    univ_features[f'{metric_name}_total'] = np.sum(values)
                    univ_features[f'{metric_name}_mean'] = np.mean(values)
                    univ_features[f'{metric_name}_std'] = np.std(values)
                    
                    # Growth rate (handle division by zero)
                    growth_rates = []
                    for i in range(1, len(values)):
                        if values[i-1] != 0:
                            growth_rate = (values[i] - values[i-1]) / abs(values[i-1]) * 100
                            # Handle infinite values
                            if not np.isinf(growth_rate) and not np.isnan(growth_rate):
                                growth_rates.append(growth_rate)
                    
                    if growth_rates:
                        univ_features[f'{metric_name}_growth_mean'] = np.mean(growth_rates)
                        univ_features[f'{metric_name}_growth_std'] = np.std(growth_rates)
                    else:
                        univ_features[f'{metric_name}_growth_mean'] = 0.0
                        univ_features[f'{metric_name}_growth_std'] = 0.0
                    
                    # Volatility (CV) - handle division by zero
                    if np.mean(values) != 0:
                        cv = np.std(values) / abs(np.mean(values))
                        if not np.isinf(cv) and not np.isnan(cv):
                            univ_features[f'{metric_name}_cv'] = cv
                        else:
                            univ_features[f'{metric_name}_cv'] = 0.0
                    else:
                        univ_features[f'{metric_name}_cv'] = 0.0
                else:
                    # Default values for insufficient data
                    for feature in [f'{metric_name}_total', f'{metric_name}_mean', f'{metric_name}_std',
                                  f'{metric_name}_growth_mean', f'{metric_name}_growth_std', f'{metric_name}_cv']:
                        univ_features[feature] = 0.0
            else:
                # Default values for no data
                for feature in [f'{metric_name}_total', f'{metric_name}_mean', f'{metric_name}_std',
                              f'{metric_name}_growth_mean', f'{metric_name}_growth_std', f'{metric_name}_cv']:
                    univ_features[feature] = 0.0
        
        features_data.append(univ_features)
    
    return pd.DataFrame(features_data)


def perform_clustering_analysis(features_df):
    """Perform K-Means and Hierarchical clustering"""
    print("\n=== Clustering Analysis ===")
    
    # Prepare features for clustering (exclude University name)
    feature_cols = [col for col in features_df.columns if col != 'University']
    X = features_df[feature_cols].values
    
    # Handle NaN values by replacing with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    n_samples = X.shape[0]
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means Clustering
    print("Performing K-Means clustering...")
    
    # Find optimal k using elbow method
    inertias = []
    silhouette_scores = []
    k_range = range(2, max(2, n_samples))  # 2 to n_samples-1
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        # Only compute silhouette_score if k < n_samples
        if k < n_samples:
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        else:
            silhouette_scores.append(np.nan)
    
    # Choose optimal k (simplified: use k=3 if possible, else 2)
    optimal_k = 3 if n_samples > 3 else 2
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Hierarchical Clustering
    print("Performing Hierarchical clustering...")
    linkage_matrix = linkage(X_scaled, method='ward')
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    
    # Add cluster labels to features dataframe
    features_df['KMeans_Cluster'] = kmeans_labels
    features_df['Hierarchical_Cluster'] = hierarchical_labels
    
    return features_df, kmeans, hierarchical, linkage_matrix, X_scaled


def create_clustering_visualizations(features_df, kmeans, hierarchical, linkage_matrix, X_scaled):
    """Create clustering visualization plots"""
    print("Creating clustering visualizations...")
    
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # K-Means visualization
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', s=100, alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    
    # Hierarchical clustering visualization
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical.labels_, cmap='plasma', s=100, alpha=0.7)
    plt.title('Hierarchical Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Dendrogram
    plt.subplot(1, 3, 3)
    dendrogram(linkage_matrix, labels=features_df['University'].values, leaf_rotation=45)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Universities')
    plt.ylabel('Distance')
    
    plt.tight_layout()
    save_plot(plt.gcf(), 'clustering_analysis.png')
    print("Clustering analysis plot saved as 'clustering_analysis.png'")
    
    return features_df


def fit_arima_model(time_series, university_name):
    """Fit ARIMA model with automatic parameter selection"""
    try:
        # Remove NaN values
        clean_series = time_series.dropna()
        
        if len(clean_series) < 3:
            return None, None
        
        # Check stationarity
        adf_result = adfuller(clean_series)
        is_stationary = adf_result[1] < 0.05
        
        # Auto ARIMA for parameter selection
        try:
            auto_arima = pm.auto_arima(clean_series, 
                                     seasonal=False, 
                                     suppress_warnings=True,
                                     error_action='ignore',
                                     max_p=3, max_q=3, max_d=2)
            order = auto_arima.order
        except:
            # Fallback to simple ARIMA(1,1,1)
            order = (1, 1, 1)
        
        # Fit ARIMA model
        model = ARIMA(clean_series, order=order)
        fitted_model = model.fit()
        
        return fitted_model, order
        
    except Exception as e:
        print(f"ARIMA fitting failed for {university_name}: {str(e)}")
        return None, None


def predict_arima(model, steps=3):
    """Make predictions using fitted ARIMA model"""
    try:
        if model is None:
            return None
        
        forecast = model.forecast(steps=steps)
        return forecast.values
        
    except Exception as e:
        print(f"ARIMA prediction failed: {str(e)}")
        return None


def prepare_xgboost_features(time_series_data, years):
    """Prepare features for XGBoost model"""
    features = []
    targets = []
    
    for i in range(3, len(time_series_data)):
        # Time features
        year_num = i
        features.append([
            year_num,  # Current year
            time_series_data[i-1],  # Lag 1
            time_series_data[i-2],  # Lag 2
            time_series_data[i-3],  # Lag 3
            np.mean(time_series_data[i-3:i]),  # Rolling mean
            np.std(time_series_data[i-3:i]),   # Rolling std
            np.max(time_series_data[i-3:i]),   # Rolling max
            np.min(time_series_data[i-3:i])    # Rolling min
        ])
        targets.append(time_series_data[i])
    
    return np.array(features), np.array(targets)


def fit_xgboost_model(time_series_data, university_name):
    """Fit XGBoost model for time series prediction"""
    try:
        # Remove NaN values
        clean_series = time_series_data.dropna()
        
        if len(clean_series) < 6:
            return None, None
        
        # Prepare features
        X, y = prepare_xgboost_features(clean_series.values, range(len(clean_series)))
        
        if len(X) < 2:
            return None, None
        
        # For small datasets, do not use early_stopping_rounds
        if len(X) < 20:
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            model.fit(X, y)
        else:
            # Use early stopping if enough data
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                early_stopping_rounds=10
            )
            # Use last 10% as validation set
            split = int(len(X) * 0.9)
            model.fit(X[:split], y[:split], eval_set=[(X[split:], y[split:])], verbose=False)
        return model, clean_series.values
        
    except Exception as e:
        print(f"XGBoost fitting failed for {university_name}: {str(e)}")
        return None, None


def predict_xgboost(model, last_values, steps=3):
    """Make predictions using fitted XGBoost model"""
    try:
        if model is None or last_values is None:
            return None
        
        predictions = []
        current_values = last_values[-3:].copy()  # Last 3 values
        
        for _ in range(steps):
            # Prepare features for next prediction
            features = [
                len(last_values) + len(predictions),  # Current year
                current_values[-1],  # Lag 1
                current_values[-2],  # Lag 2
                current_values[-3],  # Lag 3
                np.mean(current_values),  # Rolling mean
                np.std(current_values),   # Rolling std
                np.max(current_values),   # Rolling max
                np.min(current_values)    # Rolling min
            ]
            
            # Make prediction
            pred = model.predict([features])[0]
            predictions.append(pred)
            
            # Update current values for next iteration
            current_values = np.append(current_values[1:], pred)
        
        return np.array(predictions)
        
    except Exception as e:
        print(f"XGBoost prediction failed: {str(e)}")
        return None


def predict_trends(data, ne_universities):
    """Predict trends for key metrics using multiple methods: Linear Regression, Exponential Smoothing, Random Forest, ARIMA, XGBoost"""
    print("\n=== Trend Prediction Analysis ===\n")

    # Define metrics to analyze
    metrics = {
        'Research Income': ('research', 'Value'),
        'Business Income': ('business', data['business'].columns[-1]),
        'CPD Income': ('cpd', 'Value'),
        'Regeneration Income': ('regeneration', 'Value'),
        'IP Disclosures': ('ip_disclosures', 'Value'),
        'IP Licenses': ('ip_licenses', 'Value'),
        'IP Income': ('ip_income', 'Value'),
        'Total IP Income': ('ip_income_total', 'Value'),
        'Spin-out Companies': ('spinouts', 'Value'),
        'Public Engagement': ('public_engagement', 'Value')
    }

    methods = [
        ('Linear Regression', 'linear_regression'),
        ('Exponential Smoothing', 'exponential_smoothing'),
        ('Random Forest', 'random_forest'),
        ('ARIMA', 'arima'),
        ('XGBoost', 'xgboost')
    ]

    color_map = cm.get_cmap('tab10', 6)  # 6 color

    all_methods_summary = {}  # Collect percent change for all methods
    for method_name, method_key in methods:
        print(f"\n--- {method_name} ---")
        results_summary = {}  # Collect all results for this method
        all_methods_summary[method_name] = {}  # Init for this method
        plt.figure(figsize=(25, 12))
        for i, (metric_name, (dataset, value_col)) in enumerate(metrics.items(), 1):
            print(f"\n{metric_name} ({method_name}):")
            results_summary[metric_name] = {}
            all_methods_summary[method_name][metric_name] = {}  # Init for this metric
            plt.subplot(2, 5, i)
            try:
                df = data[dataset]
                if 'HE Provider' in df.columns and 'Academic Year' in df.columns and value_col in df.columns:
                    df_agg = df.groupby(['HE Provider', 'Academic Year'])[value_col].sum().reset_index()
                else:
                    df_agg = df.copy()
                national_avg = df_agg.groupby('Academic Year')[value_col].mean()
                X_nat, y_nat, years = prepare_time_series(
                    pd.DataFrame({'Academic Year': national_avg.index, 'Value': national_avg.values}),
                    'National Average'
                )
                color_list = [color_map(j) for j in range(6)]
                if len(X_nat) > 0:
                    future_years = np.array(range(len(years), len(years) + 3)).reshape(-1, 1)
                    if method_key == 'linear_regression':
                        model_nat = LinearRegression()
                        model_nat.fit(X_nat, y_nat)
                        future_pred_nat = model_nat.predict(future_years)
                    elif method_key == 'exponential_smoothing':
                        try:
                            model_nat = ExponentialSmoothing(y_nat, trend='add', seasonal=None,
                                                             initialization_method="estimated")
                            fit_nat = model_nat.fit()
                            future_pred_nat = fit_nat.forecast(3)
                        except Exception:
                            future_pred_nat = [np.nan] * 3
                    elif method_key == 'random_forest':
                        model_nat = RandomForestRegressor(n_estimators=100, random_state=0)
                        model_nat.fit(X_nat, y_nat)
                        future_pred_nat = model_nat.predict(future_years)
                    elif method_key == 'arima':
                        # ARIMA for national average
                        national_series = pd.Series(y_nat, index=range(len(y_nat)))
                        arima_model_nat, order_nat = fit_arima_model(national_series, 'National Average')
                        if arima_model_nat is not None:
                            future_pred_nat = predict_arima(arima_model_nat, 3)
                        else:
                            future_pred_nat = [np.nan] * 3
                    elif method_key == 'xgboost':
                        # XGBoost for national average
                        national_series = pd.Series(y_nat, index=range(len(y_nat)))
                        xgb_model_nat, last_values_nat = fit_xgboost_model(national_series, 'National Average')
                        if xgb_model_nat is not None:
                            future_pred_nat = predict_xgboost(xgb_model_nat, last_values_nat, 3)
                        else:
                            future_pred_nat = [np.nan] * 3
                    
                    plt.plot(years, y_nat, linestyle='-', marker='D', color=color_list[0], label='National Average',
                             alpha=0.9)
                    plt.plot(
                        [years[-1]] + [f'20{int(years[-1][-2:]) + i + 1}/{int(years[-1][-2:]) + i + 2}' for i in
                                       range(3)],
                        [y_nat[-1]] + list(future_pred_nat),
                        linestyle='--', marker=None, color=color_list[0], alpha=0.9
                    )
                    print("National Average:")
                    results_summary[metric_name]['National Average'] = []
                    for yv, yval in zip(years, y_nat):
                        print(f"  {yv}: {yval:.2f}")
                        results_summary[metric_name]['National Average'].append((yv, yval, False))
                    for j, pred in enumerate(future_pred_nat):
                        future_label = f"20{int(years[-1][-2:]) + j + 1}/{int(years[-1][-2:]) + j + 2}"
                        print(f"  {future_label} (pred): {pred:.2f}")
                        results_summary[metric_name]['National Average'].append((future_label, pred, True))
                for idx, university in enumerate(ne_universities):
                    univ_data = df_agg[df_agg['HE Provider'] == university]
                    if len(univ_data) == 0:
                        continue
                    X, y, _ = prepare_time_series(univ_data, university, value_col)
                    if len(X) > 0:
                        future_years = np.array(range(len(years), len(years) + 3)).reshape(-1, 1)
                        if method_key == 'linear_regression':
                            model = LinearRegression()
                            model.fit(X, y)
                            future_pred = model.predict(future_years)
                        elif method_key == 'exponential_smoothing':
                            try:
                                model = ExponentialSmoothing(y, trend='add', seasonal=None,
                                                             initialization_method="estimated")
                                fit = model.fit()
                                future_pred = fit.forecast(3)
                            except Exception:
                                future_pred = [np.nan] * 3
                        elif method_key == 'random_forest':
                            model = RandomForestRegressor(n_estimators=100, random_state=0)
                            model.fit(X, y)
                            future_pred = model.predict(future_years)
                        elif method_key == 'arima':
                            # ARIMA for individual university
                            univ_series = pd.Series(y, index=range(len(y)))
                            arima_model, order = fit_arima_model(univ_series, university)
                            if arima_model is not None:
                                future_pred = predict_arima(arima_model, 3)
                            else:
                                future_pred = [np.nan] * 3
                        elif method_key == 'xgboost':
                            # XGBoost for individual university
                            univ_series = pd.Series(y, index=range(len(y)))
                            xgb_model, last_values = fit_xgboost_model(univ_series, university)
                            if xgb_model is not None:
                                future_pred = predict_xgboost(xgb_model, last_values, 3)
                            else:
                                future_pred = [np.nan] * 3
                        
                        plt.plot(univ_data['Academic Year'], y, marker='o', linestyle='-', color=color_list[idx + 1],
                                 label=university)
                        plt.plot(
                            [univ_data['Academic Year'].iloc[-1]] + [
                                f'20{int(years[-1][-2:]) + i + 1}/{int(years[-1][-2:]) + i + 2}' for i in range(3)],
                            [y[-1]] + list(future_pred),
                            linestyle='--', marker=None, color=color_list[idx + 1], alpha=0.9
                        )
                        print(university + ":")
                        results_summary[metric_name][university] = []
                        for yv, yval in zip(univ_data['Academic Year'], y):
                            print(f"  {yv}: {yval:.2f}")
                            results_summary[metric_name][university].append((yv, yval, False))
                        for j, pred in enumerate(future_pred):
                            future_label = f"20{int(years[-1][-2:]) + j + 1}/{int(years[-1][-2:]) + j + 2}"
                            print(f"  {future_label} (pred): {pred:.2f}")
                            results_summary[metric_name][university].append((future_label, pred, True))
                plt.title(f'{metric_name} Trend Prediction ({method_name})')
                plt.xlabel('Academic Year')
                plt.ylabel('Value')
                plt.grid(True)
                plt.xticks(rotation=45)
            except Exception as e:
                print(f"Error processing {metric_name} with {method_name}: {str(e)}")
                plt.title(f'{metric_name} ({method_name}, Error)')
                plt.text(0.5, 0.5, f'Error: {str(e)}',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=plt.gca().transAxes)
            for metric, schools in results_summary.items():
                for school, values in schools.items():
                    actual_vals = [v for v in values if not v[2]]
                    pred_vals = [v for v in values if v[2]]
                    if actual_vals and pred_vals:
                        last_actual = actual_vals[-1][1]
                        last_pred = pred_vals[-1][1]
                        if last_actual is not None and last_actual != 0 and not pd.isna(last_actual):
                            pct_change = (last_pred - last_actual) / abs(last_actual) * 100
                            sign = '+' if pct_change >= 0 else ''
                            all_methods_summary[method_name][metric][school] = f"{sign}{pct_change:.2f}%"
                        else:
                            all_methods_summary[method_name][metric][school] = "n/a"
                    else:
                        all_methods_summary[method_name][metric][school] = "n/a"
        lines = []
        labels = []
        for ax in plt.gcf().axes:
            for line, label in zip(ax.get_lines(), ax.get_legend_handles_labels()[1]):
                if label not in labels:
                    lines.append(line)
                    labels.append(label)
        if lines:
            plt.figlegend(lines, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_plot(plt.gcf(), f'trend_predictions_{method_key}.png')
        print(f"Trend predictions plot saved as 'trend_predictions_{method_key}.png'")
    print("\n=== All Methods Summary ===")
    md_lines = ["# Trend Prediction Summary\n"]
    for method_name in all_methods_summary:
        print(f"\n[{method_name}]")
        md_lines.append(f"## {method_name}\n")
        for metric in all_methods_summary[method_name]:
            print(f"{metric}:")
            md_lines.append(f"### {metric}\n")
            for school in all_methods_summary[method_name][metric]:
                line = f"  {school}: {all_methods_summary[method_name][metric][school]}"
                print(line)
                md_lines.append(f"- {school}: {all_methods_summary[method_name][metric][school]}")
            md_lines.append("")
    with open("8.trend_prediction.md", "w") as f:
        f.write("\n".join(md_lines))
    print("\nSummary written to 8.trend_prediction.md")


def main():
    print("Starting Trend Prediction Analysis...")

    data, ne_universities = load_data()
    
    # Perform clustering analysis first
    print("\n" + "="*50)
    print("STEP 1: CLUSTERING ANALYSIS")
    print("="*50)
    
    # Extract features for clustering
    features_df = extract_clustering_features(data, ne_universities)
    
    # Perform clustering analysis
    features_df, kmeans, hierarchical, linkage_matrix, X_scaled = perform_clustering_analysis(features_df)
    
    # Create clustering visualizations
    features_df = create_clustering_visualizations(features_df, kmeans, hierarchical, linkage_matrix, X_scaled)
    
    # Print clustering results
    print("\n=== Clustering Results ===")
    print("\nK-Means Clustering Results:")
    for cluster_id in sorted(features_df['KMeans_Cluster'].unique()):
        cluster_universities = features_df[features_df['KMeans_Cluster'] == cluster_id]['University'].tolist()
        print(f"Cluster {cluster_id}: {', '.join(cluster_universities)}")
    
    print("\nHierarchical Clustering Results:")
    for cluster_id in sorted(features_df['Hierarchical_Cluster'].unique()):
        cluster_universities = features_df[features_df['Hierarchical_Cluster'] == cluster_id]['University'].tolist()
        print(f"Cluster {cluster_id}: {', '.join(cluster_universities)}")
    
    # Perform trend prediction analysis
    print("\n" + "="*50)
    print("STEP 2: TREND PREDICTION ANALYSIS")
    print("="*50)
    
    predict_trends(data, ne_universities)

    print("\nTrend Prediction Analysis completed!")


if __name__ == "__main__":
    main()
