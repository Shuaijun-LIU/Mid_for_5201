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

warnings.filterwarnings("ignore")


def save_plot(fig, filename):
    save_dir = Path('visualizations/trend_prediction')
    save_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_dir / filename, bbox_inches='tight', dpi=300)
    plt.close()


def load_data():
    ne_universities = [
        'University of Durham',
        'Newcastle University',
        'University of Northumbria at Newcastle',
        'Teesside University',
        'The University of Sunderland'
    ]

    data = {}

    # Income data
    data['research'] = pd.read_csv('table-1.csv', skiprows=11)
    data['business'] = pd.read_csv('table-2a.csv', skiprows=11)
    data['cpd'] = pd.read_csv('table-2b.csv', skiprows=11)
    data['regeneration'] = pd.read_csv('table-3.csv', skiprows=11)

    # IP data
    data['ip_disclosures'] = pd.read_csv('table-4a.csv', skiprows=11)
    data['ip_licenses'] = pd.read_csv('table-4b.csv', skiprows=11)
    data['ip_income'] = pd.read_csv('table-4c.csv', skiprows=11)
    data['ip_income_total'] = pd.read_csv('table-4d.csv', skiprows=11)
    data['spinouts'] = pd.read_csv('table-4e.csv', skiprows=11)

    # Public engagement data
    data['public_engagement'] = pd.read_csv('table-5.csv', skiprows=11)

    # Clean column names for all datasets
    for key in data:
        data[key].columns = data[key].columns.str.strip()

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


def predict_trends(data, ne_universities):
    """Predict trends for key metrics using three methods: Linear Regression, Exponential Smoothing, Random Forest"""
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
        ('Random Forest', 'random_forest')
    ]

    color_map = cm.get_cmap('tab10', 6)  # 6 color

    all_methods_summary = {}  # Collect percent change for all methods
    for method_name, method_key in methods:
        print(f"\n--- {method_name} ---")
        results_summary = {}  # Collect all results for this method
        all_methods_summary[method_name] = {}  # Init for this method
        plt.figure(figsize=(20, 25))
        for i, (metric_name, (dataset, value_col)) in enumerate(metrics.items(), 1):
            print(f"\n{metric_name} ({method_name}):")
            results_summary[metric_name] = {}
            all_methods_summary[method_name][metric_name] = {}  # Init for this metric
            plt.subplot(4, 3, i)
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
        print(f"Trend predictions plot saved as '1.trend_predictions_{method_key}.png'")
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
    with open("6.trend_prediction.md", "w") as f:
        f.write("\n".join(md_lines))
    print("\nSummary written to trend_prediction_summary.md")

#################################
def main():
    """Main function to run the trend prediction analysis"""
    print("Starting Trend Prediction Analysis...")

    data, ne_universities = load_data()
    predict_trends(data, ne_universities)

    print("\nTrend Prediction Analysis completed!")


if __name__ == "__main__":
    main()
