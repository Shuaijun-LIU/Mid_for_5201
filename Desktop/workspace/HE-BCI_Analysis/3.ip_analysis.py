import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def save_plot(fig, filename):
    save_dir = Path('visualizations/ip_analysis')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(save_dir / filename)
    plt.close(fig)

def analyze_ip():
    """Main function to analyze all intellectual property related data"""
    data_dir = Path('Data')
    tables = {
        'table4a': pd.read_csv(data_dir / 'table-4a.csv', skiprows=11),
        'table4b': pd.read_csv(data_dir / 'table-4b.csv', skiprows=11),
        'table4c': pd.read_csv(data_dir / 'table-4c.csv', skiprows=11),
        'table4d': pd.read_csv(data_dir / 'table-4d.csv', skiprows=11),
        'table4e': pd.read_csv(data_dir / 'table-4e.csv', skiprows=11)
    }
    
    # Define universities for comparison
    north_east_universities = [
        'University of Durham',
        'Newcastle University',
        'University of Northumbria at Newcastle',
        'The University of Sunderland',
        'Teesside University'
    ]
    
    # 1. IP Disclosures and Patents Analysis
    print("\n=== IP Disclosures and Patents Analysis ===")
    
    # 1.1 Disclosure Type Analysis (Durham only)
    durham_ip = tables['table4a'][tables['table4a']['HE Provider'] == 'University of Durham']
    disclosure_type = durham_ip.groupby('Type of disclosure or patent')['Value'].sum()
    print("\nDisclosure Type Distribution (Durham):")
    print(disclosure_type)
    
    # National statistics for IP disclosures
    national_ip = tables['table4a'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_ip.index.get_loc('University of Durham') + 1
    total_universities = len(national_ip)
    national_avg = national_ip.mean()
    print(f"\nNational Statistics for IP Disclosures:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: {national_avg:,.0f}")
    print(f"Durham's value: {national_ip['University of Durham']:,.0f}")
    print(f"Difference from national average: {national_ip['University of Durham'] - national_avg:,.0f}")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(disclosure_type)))
    ax = plt.bar(disclosure_type.index, disclosure_type.values, color=colors)
    plt.title('IP Disclosures by Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(disclosure_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '1.1_ip_disclosure_type.png')
    
    # 1.2 Time Trend Analysis (Durham only)
    yearly_disclosures = durham_ip.groupby(['Academic Year', 'Type of disclosure or patent'])['Value'].sum().reset_index()
    print("\nYearly Disclosure Trend (Durham):")
    print(yearly_disclosures)
    
    fig = plt.figure(figsize=(12, 6))
    for disc_type in yearly_disclosures['Type of disclosure or patent'].unique():
        data = yearly_disclosures[yearly_disclosures['Type of disclosure or patent'] == disc_type]
        plt.plot(data['Academic Year'], data['Value'], label=disc_type, marker='o')
    plt.title('IP Disclosures Trend (Durham)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(fig, '1.2_ip_disclosure_trend.png')
    
    # 1.3 North East Universities Comparison
    ne_ip = tables['table4a'][tables['table4a']['HE Provider'].isin(north_east_universities)]
    ne_total = ne_ip.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal IP Disclosures by University (North East):")
    print(ne_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_total)))
    ax = plt.bar(ne_total.index, ne_total.values, color=colors)
    plt.title('Total IP Disclosures - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '1.3_ip_ne_comparison.png')
    
    ###############################

    # 2. License Analysis
    print("\n=== License Analysis ===")
    
    # 2.1 License Type Analysis (Durham only)
    durham_license = tables['table4b'][tables['table4b']['HE Provider'] == 'University of Durham']
    license_type = durham_license.groupby('Type of licence granted')['Value'].sum()
    print("\nLicense Type Distribution (Durham):")
    print(license_type)
    
    # National statistics for licenses
    national_license = tables['table4b'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_license.index.get_loc('University of Durham') + 1
    total_universities = len(national_license)
    national_avg = national_license.mean()
    print(f"\nNational Statistics for Licenses:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: {national_avg:,.0f}")
    print(f"Durham's value: {national_license['University of Durham']:,.0f}")
    print(f"Difference from national average: {national_license['University of Durham'] - national_avg:,.0f}")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(license_type)))
    ax = plt.bar(license_type.index, license_type.values, color=colors)
    plt.title('Licenses by Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(license_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '2.1_license_type.png')
    
    # 2.2 Organization Type Analysis (Durham only)
    org_type = durham_license.groupby('Type of organisation')['Value'].sum()
    print("\nOrganization Type Distribution (Durham):")
    print(org_type)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(org_type)))
    ax = plt.bar(org_type.index, org_type.values, color=colors)
    plt.title('Licenses by Organisation Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(org_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '2.2_license_org_type.png')
    
    # 2.3 North East Universities Comparison
    ne_license = tables['table4b'][tables['table4b']['HE Provider'].isin(north_east_universities)]
    ne_license_total = ne_license.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal Licenses by University (North East):")
    print(ne_license_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_license_total)))
    ax = plt.bar(ne_license_total.index, ne_license_total.values, color=colors)
    plt.title('Total Licenses - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_license_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '2.3_license_ne_comparison.png')
    
    #################################
    # 3. IP Income Analysis
    print("\n=== IP Income Analysis ===")
    
    # 3.1 Income Source Analysis (Durham only)
    durham_income = pd.concat([
        tables['table4c'][tables['table4c']['HE Provider'] == 'University of Durham'],
        tables['table4d'][tables['table4d']['HE Provider'] == 'University of Durham']
    ])
    income_source = durham_income.groupby('Income source')['Value'].sum()
    print("\nIncome Source Distribution (Durham):")
    print(income_source)
    
    # National statistics for IP income
    national_income = pd.concat([
        tables['table4c'],
        tables['table4d']
    ]).groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_income.index.get_loc('University of Durham') + 1
    total_universities = len(national_income)
    national_avg = national_income.mean()
    print(f"\nNational Statistics for IP Income:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: £{national_avg:,.0f}")
    print(f"Durham's value: £{national_income['University of Durham']:,.0f}")
    print(f"Difference from national average: £{national_income['University of Durham'] - national_avg:,.0f}")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(income_source)))
    ax = plt.bar(income_source.index, income_source.values, color=colors)
    plt.title('IP Income by Source (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(income_source.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '3.1_ip_income_source.png')
    
    # 3.2 Organization Type Analysis (Durham only)
    org_type = durham_income.groupby('Type of organisation')['Value'].sum()
    print("\nOrganization Type Distribution (Durham):")
    print(org_type)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(org_type)))
    ax = plt.bar(org_type.index, org_type.values, color=colors)
    plt.title('IP Income by Organisation Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(org_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '3.2_ip_income_org_type.png')
    
    # 3.3 North East Universities Comparison
    ne_income = pd.concat([
        tables['table4c'][tables['table4c']['HE Provider'].isin(north_east_universities)],
        tables['table4d'][tables['table4d']['HE Provider'].isin(north_east_universities)]
    ])
    ne_income_total = ne_income.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal IP Income by University (North East):")
    print(ne_income_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_income_total)))
    ax = plt.bar(ne_income_total.index, ne_income_total.values, color=colors)
    plt.title('Total IP Income - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_income_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '3.3_ip_income_ne_comparison.png')
    
    ###############################
    # 4. spin-off Company Analysis
    print("\n=== Spin-off Company Analysis ===")
    
    # 4.1 Metric Type Analysis (Durham)
    durham_spin = tables['table4e'][tables['table4e']['HE Provider'] == 'University of Durham']
    metric_type = durham_spin.groupby('Metric')['Value'].sum()
    print("\nMetric Type Distribution (Durham):")
    print(metric_type)
    
    # National statistics for spin-off companies
    national_spin = tables['table4e'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_spin.index.get_loc('University of Durham') + 1
    total_universities = len(national_spin)
    national_avg = national_spin.mean()
    print(f"\nNational Statistics for Spin-off Companies:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: {national_avg:,.0f}")
    print(f"Durham's value: {national_spin['University of Durham']:,.0f}")
    print(f"Difference from national average: {national_spin['University of Durham'] - national_avg:,.0f}")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(metric_type)))
    ax = plt.bar(metric_type.index, metric_type.values, color=colors)
    plt.title('Spin-off Companies by Metric Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(metric_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '4.1_spin_metric_type.png')
    
    # 4.2 Category Analysis (Durham only)
    category = durham_spin.groupby('Category Marker')['Value'].sum()
    print("\nCategory Distribution (Durham):")
    print(category)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(category)))
    ax = plt.bar(category.index, category.values, color=colors)
    plt.title('Spin-off Companies by Category (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(category.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '4.2_spin_category.png')
    
    # 4.3 North East Universities Comparison
    ne_spin = tables['table4e'][tables['table4e']['HE Provider'].isin(north_east_universities)]
    ne_spin_total = ne_spin.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal Spin-off Companies by University (North East):")
    print(ne_spin_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_spin_total)))
    ax = plt.bar(ne_spin_total.index, ne_spin_total.values, color=colors)
    plt.title('Total Spin-off Companies - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_spin_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '4.3_spin_ne_comparison.png')

if __name__ == "__main__":
    analyze_ip() 