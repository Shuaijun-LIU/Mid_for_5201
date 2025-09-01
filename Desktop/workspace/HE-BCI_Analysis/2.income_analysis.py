import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def save_plot(fig, filename):
    save_dir = Path('visualizations/income_analysis')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(save_dir / filename)
    plt.close(fig)

def analyze_income():
    """Main function to analyze all income-related data"""
    data_dir = Path('Data')
    tables = {
        'table1': pd.read_csv(data_dir / 'table-1.csv', skiprows=11),
        'table2a': pd.read_csv(data_dir / 'table-2a.csv', skiprows=11),
        'table2b': pd.read_csv(data_dir / 'table-2b.csv', skiprows=11),
        'table3': pd.read_csv(data_dir / 'table-3.csv', skiprows=11)
    }
    
    # Define universities for comparison
    north_east_universities = [
        'University of Durham',
        'Newcastle University',
        'University of Northumbria at Newcastle',
        'The University of Sunderland',
        'Teesside University'
    ]
    
    # 1. Collaborative Research Income Analysis
    print("\n=== Collaborative Research Income Analysis ===")
    
    # 1.1 Funding Source Analysis (Durham only)
    durham_data = tables['table1'][tables['table1']['HE Provider'] == 'University of Durham']
    funding_source = durham_data.groupby('Source of public funding')['Value'].sum()
    print("\nFunding Source Distribution (Durham):")
    print(funding_source)
    
    # National statistics for research income
    national_research = tables['table1'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_research.index.get_loc('University of Durham') + 1
    total_universities = len(national_research)
    national_avg = national_research.mean()
    print(f"\nNational Statistics for Research Income:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: £{national_avg:,.0f}")
    print(f"Durham's value: £{national_research['University of Durham']:,.0f}")
    print(f"Difference from national average: £{national_research['University of Durham'] - national_avg:,.0f}")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(funding_source)))
    ax = plt.bar(funding_source.index, funding_source.values, color=colors)
    plt.title('Collaborative Research Income by Funding Source (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(funding_source.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '1.1_research_funding_source.png')
    
    # 1.2 Income Type Analysis (Durham only)
    income_type = durham_data.groupby('Type of income')['Value'].sum()
    print("\nIncome Type Distribution (Durham):")
    print(income_type)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(income_type)))
    ax = plt.bar(income_type.index, income_type.values, color=colors)
    plt.title('Collaborative Research Income by Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(income_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '1.2_research_income_type.png')
    
    # 1.3 Time Trend Analysis (Durham only)
    yearly_income = durham_data.groupby(['Academic Year', 'Source of public funding'])['Value'].sum().reset_index()
    print("\nYearly Income Trend (Durham):")
    print(yearly_income)
    
    fig = plt.figure(figsize=(12, 6))
    for source in yearly_income['Source of public funding'].unique():
        data = yearly_income[yearly_income['Source of public funding'] == source]
        plt.plot(data['Academic Year'], data['Value'], label=source, marker='o')
    plt.title('Collaborative Research Income Trend (Durham)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(fig, '1.3_research_time_trend.png')
    
    # 1.4 North East Universities Comparison
    print("\n=== North East Universities Comparison ===")
    
    # Compare total research income
    ne_research = tables['table1'][tables['table1']['HE Provider'].isin(north_east_universities)]
    ne_total = ne_research.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal Research Income by University (North East):")
    print(ne_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_total)))
    ax = plt.bar(ne_total.index, ne_total.values, color=colors)
    plt.title('Total Research Income - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '1.4_research_ne_comparison.png')
    
    # Compare with national average
    national_avg = tables['table1'].groupby('HE Provider')['Value'].sum().mean()
    print("\nNational Average Research Income:", national_avg)
    print("Durham's Research Income:", ne_total['University of Durham'])
    print("Difference from National Average:", ne_total['University of Durham'] - national_avg)
    
    # 2. Business Services Analysis
    print("\n=== Business Services Analysis ===")
    
    # 2.1 Service Type Analysis (Durham only)
    durham_services = tables['table2a'][tables['table2a']['HE Provider'] == 'University of Durham']
    service_type = durham_services.groupby('Type of service')['Number/Value'].sum()
    print("\nService Type Distribution (Durham):")
    print(service_type)
    
    # National statistics for business services
    national_services = tables['table2a'].groupby('HE Provider')['Number/Value'].sum().sort_values(ascending=False)
    durham_rank = national_services.index.get_loc('University of Durham') + 1
    total_universities = len(national_services)
    national_avg = national_services.mean()
    print(f"\nNational Statistics for Business Services:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: £{national_avg:,.0f}")
    print(f"Durham's value: £{national_services['University of Durham']:,.0f}")
    print(f"Difference from national average: £{national_services['University of Durham'] - national_avg:,.0f}")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(service_type)))
    ax = plt.bar(service_type.index, service_type.values, color=colors)
    plt.title('Business Services by Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(service_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '2.1_services_type.png')
    
    # 2.2 Organization Type Analysis (Durham only)
    org_type = durham_services.groupby('Type of organisation')['Number/Value'].sum()
    print("\nOrganization Type Distribution (Durham):")
    print(org_type)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(org_type)))
    ax = plt.bar(org_type.index, org_type.values, color=colors)
    plt.title('Business Services by Organisation Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(org_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '2.2_services_org_type.png')
    
    # 2.3 Metric Type Analysis (Durham only)
    metric_type = durham_services.groupby('Number/Value Marker')['Number/Value'].sum()
    print("\nMetric Type Distribution (Durham):")
    print(metric_type)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(metric_type)))
    ax = plt.bar(metric_type.index, metric_type.values, color=colors)
    plt.title('Business Services by Metric Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(metric_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '2.3_services_metric_type.png')
    
    # 2.4 North East Universities Business Services Comparison
    ne_services = tables['table2a'][tables['table2a']['HE Provider'].isin(north_east_universities)]
    ne_services_total = ne_services.groupby('HE Provider')['Number/Value'].sum().sort_values(ascending=False)
    print("\nTotal Business Services by University (North East):")
    print(ne_services_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_services_total)))
    ax = plt.bar(ne_services_total.index, ne_services_total.values, color=colors)
    plt.title('Total Business Services - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_services_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '2.4_services_ne_comparison.png')
    
    # 3. CPD and Continuing Education Analysis
    print("\n=== CPD and Continuing Education Analysis ===")
    
    # 3.1 Category Analysis (Durham only)
    durham_cpd = tables['table2b'][tables['table2b']['HE Provider'] == 'University of Durham']
    category = durham_cpd.groupby('Category Marker')['Value'].sum()
    print("\nCategory Distribution (Durham):")
    print(category)
    
    # National statistics for CPD
    national_cpd = tables['table2b'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_cpd.index.get_loc('University of Durham') + 1
    total_universities = len(national_cpd)
    national_avg = national_cpd.mean()
    print(f"\nNational Statistics for CPD and Continuing Education:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: £{national_avg:,.0f}")
    print(f"Durham's value: £{national_cpd['University of Durham']:,.0f}")
    print(f"Difference from national average: £{national_cpd['University of Durham'] - national_avg:,.0f}")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(category)))
    ax = plt.bar(category.index, category.values, color=colors)
    plt.title('CPD and Continuing Education by Category (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(category.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '3.1_cpd_category.png')
    
    # 3.2 Unit Type Analysis (Durham only)
    unit_type = durham_cpd.groupby('Unit')['Value'].sum()
    print("\nUnit Type Distribution (Durham):")
    print(unit_type)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unit_type)))
    ax = plt.bar(unit_type.index, unit_type.values, color=colors)
    plt.title('CPD and Continuing Education by Unit Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(unit_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '3.2_cpd_unit_type.png')
    
    # 3.3 North East Universities CPD Comparison
    ne_cpd = tables['table2b'][tables['table2b']['HE Provider'].isin(north_east_universities)]
    ne_cpd_total = ne_cpd.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal CPD Income by University (North East):")
    print(ne_cpd_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_cpd_total)))
    ax = plt.bar(ne_cpd_total.index, ne_cpd_total.values, color=colors)
    plt.title('Total CPD Income - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_cpd_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '3.3_cpd_ne_comparison.png')
    
    # 4. Regeneration and Development Analysis
    print("\n=== Regeneration and Development Analysis ===")
    
    # 4.1 Programme Analysis (Durham only)
    durham_regeneration = tables['table3'][tables['table3']['HE Provider'] == 'University of Durham']
    programme = durham_regeneration.groupby('Programme')['Value'].sum()
    print("\nProgramme Distribution (Durham):")
    print(programme)
    
    # National statistics for regeneration and development
    national_regeneration = tables['table3'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_regeneration.index.get_loc('University of Durham') + 1
    total_universities = len(national_regeneration)
    national_avg = national_regeneration.mean()
    print(f"\nNational Statistics for Regeneration and Development:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: £{national_avg:,.0f}")
    print(f"Durham's value: £{national_regeneration['University of Durham']:,.0f}")
    print(f"Difference from national average: £{national_regeneration['University of Durham'] - national_avg:,.0f}")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(programme)))
    ax = plt.bar(programme.index, programme.values, color=colors)
    plt.title('Regeneration and Development by Programme (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(programme.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '4.1_regeneration_programme.png')
    
    # 4.2 Time Trend Analysis (Durham only)
    yearly_regeneration = durham_regeneration.groupby(['Academic Year', 'Programme'])['Value'].sum().reset_index()
    print("\nYearly Regeneration Trend (Durham):")
    print(yearly_regeneration)
    
    fig = plt.figure(figsize=(12, 6))
    for prog in yearly_regeneration['Programme'].unique():
        data = yearly_regeneration[yearly_regeneration['Programme'] == prog]
        plt.plot(data['Academic Year'], data['Value'], label=prog, marker='o')
    plt.title('Regeneration and Development Trend (Durham)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(fig, '4.2_regeneration_time_trend.png')
    
    # 4.3 North East Universities Regeneration Comparison
    ne_regeneration = tables['table3'][tables['table3']['HE Provider'].isin(north_east_universities)]
    ne_regeneration_total = ne_regeneration.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal Regeneration Income by University (North East):")
    print(ne_regeneration_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_regeneration_total)))
    ax = plt.bar(ne_regeneration_total.index, ne_regeneration_total.values, color=colors)
    plt.title('Total Regeneration Income - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_regeneration_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '4.3_regeneration_ne_comparison.png')
    
    # 5. Overall Performance Summary
    print("\n=== Overall Performance Summary ===")
    
    # Calculate total income for each university
    total_income = {}
    for uni in north_east_universities:
        research = tables['table1'][tables['table1']['HE Provider'] == uni]['Value'].sum()
        services = tables['table2a'][tables['table2a']['HE Provider'] == uni]['Number/Value'].sum()
        cpd = tables['table2b'][tables['table2b']['HE Provider'] == uni]['Value'].sum()
        regeneration = tables['table3'][tables['table3']['HE Provider'] == uni]['Value'].sum()
        total_income[uni] = research + services + cpd + regeneration
    
    total_income = pd.Series(total_income).sort_values(ascending=False)
    print("\nTotal Income by University (All Sources):")
    print(total_income)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(total_income)))
    ax = plt.bar(total_income.index, total_income.values, color=colors)
    plt.title('Total Income - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(total_income.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, '5.1_overall_performance.png')

if __name__ == "__main__":
    analyze_income() 