import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def save_plot(fig, filename):
    save_dir = Path('visualizations/resilience_risk')
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
    data['research'] = pd.read_csv('Data/table-1.csv', skiprows=11)
    data['business'] = pd.read_csv('Data/table-2a.csv', skiprows=11)
    data['cpd'] = pd.read_csv('Data/table-2b.csv', skiprows=11)
    data['regeneration'] = pd.read_csv('Data/table-3.csv', skiprows=11)
    data['ip_income'] = pd.read_csv('Data/table-4c.csv', skiprows=11)

    # Clean column names for all datasets
    for key in data:
        data[key].columns = data[key].columns.str.strip()

    return data, ne_universities


def calculate_income_volatility(data, ne_universities):
    """Calculate income volatility metrics for all universities"""
    print("\n=== Income Stability and Risk Analysis ===\n")

    # Prepare volatility metrics
    volatility_data = {}

    # 1. Research Income Volatility
    print("1. Research Income Volatility Analysis")
    research_by_univ = data['research'].groupby(['HE Provider', 'Academic Year'])['Value'].sum().reset_index()
    
    # Calculate volatility (coefficient of variation) for each university
    research_volatility = research_by_univ.groupby('HE Provider')['Value'].agg(['mean', 'std']).reset_index()
    research_volatility['cv'] = research_volatility['std'] / research_volatility['mean']
    research_volatility = research_volatility.sort_values('cv', ascending=True)  # Lower CV = more stable
    
    volatility_data['Research'] = research_volatility
    print(f"Research volatility calculated for {len(research_volatility)} universities")

    # 2. Business Services Volatility
    print("2. Business Services Volatility Analysis")
    business_by_univ = data['business'].groupby(['HE Provider', 'Academic Year'])[data['business'].columns[-1]].sum().reset_index()
    
    business_volatility = business_by_univ.groupby('HE Provider')[data['business'].columns[-1]].agg(['mean', 'std']).reset_index()
    business_volatility['cv'] = business_volatility['std'] / business_volatility['mean']
    business_volatility = business_volatility.sort_values('cv', ascending=True)
    
    volatility_data['Business'] = business_volatility
    print(f"Business volatility calculated for {len(business_volatility)} universities")

    # 3. CPD Income Volatility
    print("3. CPD Income Volatility Analysis")
    cpd_by_univ = data['cpd'].groupby(['HE Provider', 'Academic Year'])['Value'].sum().reset_index()
    
    cpd_volatility = cpd_by_univ.groupby('HE Provider')['Value'].agg(['mean', 'std']).reset_index()
    cpd_volatility['cv'] = cpd_volatility['std'] / cpd_volatility['mean']
    cpd_volatility = cpd_volatility.sort_values('cv', ascending=True)
    
    volatility_data['CPD'] = cpd_volatility
    print(f"CPD volatility calculated for {len(cpd_volatility)} universities")

    # 4. IP Income Volatility
    print("4. IP Income Volatility Analysis")
    ip_by_univ = data['ip_income'].groupby(['HE Provider', 'Academic Year'])['Value'].sum().reset_index()
    
    ip_volatility = ip_by_univ.groupby('HE Provider')['Value'].agg(['mean', 'std']).reset_index()
    ip_volatility['cv'] = ip_volatility['std'] / ip_volatility['mean']
    ip_volatility = ip_volatility.sort_values('cv', ascending=True)
    
    volatility_data['IP'] = ip_volatility
    print(f"IP volatility calculated for {len(ip_volatility)} universities")

    return volatility_data


def analyze_income_concentration(data, ne_universities):
    """Analyze income concentration and diversification"""
    print("\n=== Income Concentration Analysis ===\n")

    concentration_data = {}

    # Calculate income concentration for each university
    for univ in ne_universities:
        print(f"\nIncome Concentration Analysis for {univ}:")
        
        # Get all income sources for this university
        univ_income = {}
        
        # Research income by source
        research_data = data['research'][data['research']['HE Provider'] == univ]
        if len(research_data) > 0:
            research_by_source = research_data.groupby('Source of public funding')['Value'].sum()
            univ_income['Research'] = research_by_source
        
        # Business income by type
        business_data = data['business'][data['business']['HE Provider'] == univ]
        if len(business_data) > 0:
            business_by_type = business_data.groupby('Type of service')[data['business'].columns[-1]].sum()
            univ_income['Business'] = business_by_type
        
        # CPD income
        cpd_data = data['cpd'][data['cpd']['HE Provider'] == univ]
        if len(cpd_data) > 0:
            cpd_total = cpd_data['Value'].sum()
            univ_income['CPD'] = pd.Series([cpd_total], index=['CPD Total'])
        
        # IP income
        ip_data = data['ip_income'][data['ip_income']['HE Provider'] == univ]
        if len(ip_data) > 0:
            ip_total = ip_data['Value'].sum()
            univ_income['IP'] = pd.Series([ip_total], index=['IP Total'])
        
        # Regeneration income
        regen_data = data['regeneration'][data['regeneration']['HE Provider'] == univ]
        if len(regen_data) > 0:
            regen_total = regen_data['Value'].sum()
            univ_income['Regeneration'] = pd.Series([regen_total], index=['Regeneration Total'])
        
        concentration_data[univ] = univ_income
        
        # Calculate concentration metrics
        total_income = sum([income.sum() for income in univ_income.values()])
        if total_income > 0:
            # Calculate HHI for income concentration
            shares = []
            for category, income in univ_income.items():
                for source, value in income.items():
                    if value > 0:
                        shares.append((value / total_income) ** 2)
            
            hhi = sum(shares)
            max_source_share = max([income.max() / total_income for income in univ_income.values() if income.sum() > 0])
            
            print(f"  Total Income: £{total_income:,.0f}")
            print(f"  HHI Index: {hhi:.3f}")
            print(f"  Max Source Share: {max_source_share:.1%}")
            print(f"  Number of Income Sources: {len([s for s in shares if s > 0])}")

    return concentration_data


def analyze_ne_volatility(volatility_data, ne_universities):
    """Analyze volatility within North East universities"""
    print("\n=== North East Universities Volatility Analysis ===\n")

    # Calculate volatility rankings for NE universities
    ne_volatility_summary = {}
    
    for metric_name, data in volatility_data.items():
        print(f"\n{metric_name} Volatility - North East Universities:")
        ne_data = data[data['HE Provider'].isin(ne_universities)].copy()
        
        # Sort by CV (lower is more stable)
        ne_data = ne_data.sort_values('cv')
        
        print("Volatility Ranking (Lower CV = More Stable):")
        for _, row in ne_data.iterrows():
            print(f"  {row['HE Provider']}: CV = {row['cv']:.3f}")
        
        ne_volatility_summary[metric_name] = ne_data

    # Create visualization for NE volatility comparison
    plt.figure(figsize=(20, 6))
    
    for i, (metric_name, ne_data) in enumerate(ne_volatility_summary.items(), 1):
        plt.subplot(1, 4, i)
        
        # Create bar chart (lower CV = better)
        universities = list(ne_data['HE Provider'])
        cvs = list(ne_data['cv'])
        colors = plt.cm.rainbow(np.linspace(0, 1, len(universities)))
        
        bars = plt.bar(universities, cvs, color=colors)
        plt.title(f'{metric_name} Volatility - North East Universities')
        plt.xlabel('University')
        plt.ylabel('Coefficient of Variation (Lower = More Stable)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, cv in zip(bars, cvs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{cv:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    save_plot(plt.gcf(), '1.ne_volatility_comparison.png')
    print("\nNorth East volatility comparison plot saved as '1.ne_volatility_comparison.png'")

    return ne_volatility_summary


def analyze_national_volatility_ranking(volatility_data, ne_universities):
    """Analyze national volatility rankings"""
    print("\n=== National Volatility Ranking Analysis ===\n")

    national_rankings = {}
    
    for metric_name, data in volatility_data.items():
        print(f"\n{metric_name} Volatility - National Rankings:")
        
        # Find NE universities' rankings
        ne_rankings = {}
        for univ in ne_universities:
            if univ in data['HE Provider'].values:
                univ_data = data[data['HE Provider'] == univ].iloc[0]
                rank = data.index.get_loc(data[data['HE Provider'] == univ].index[0]) + 1
                cv = univ_data['cv']
                ne_rankings[univ] = {'rank': rank, 'cv': cv}
                print(f"  {univ}: Rank {rank}/{len(data)}, CV = {cv:.3f}")
        
        national_rankings[metric_name] = ne_rankings

    # Create visualization for national rankings
    plt.figure(figsize=(20, 6))
    
    for i, (metric_name, rankings) in enumerate(national_rankings.items(), 1):
        plt.subplot(1, 4, i)
        
        universities = list(rankings.keys())
        ranks = [rankings[univ]['rank'] for univ in universities]
        cvs = [rankings[univ]['cv'] for univ in universities]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(universities)))
        
        # Create scatter plot (rank vs CV)
        plt.scatter(ranks, cvs, c=colors, s=100, alpha=0.7)
        
        # Add university labels
        for j, univ in enumerate(universities):
            plt.annotate(univ.split()[-1], (ranks[j], cvs[j]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title(f'{metric_name} Volatility - National Rankings')
        plt.xlabel('National Rank (Lower = More Stable)')
        plt.ylabel('Coefficient of Variation')
        plt.grid(True, alpha=0.3)
        
        # Invert x-axis so better rank (lower number) is on the left
        plt.gca().invert_xaxis()

    plt.tight_layout()
    save_plot(plt.gcf(), '2.national_volatility_rankings.png')
    print("\nNational volatility rankings plot saved as '2.national_volatility_rankings.png'")

    return national_rankings


def analyze_volatility_trends(data, ne_universities):
    """Analyze volatility trends over time using tables instead of charts"""
    print("\n=== Volatility Trends Analysis ===\n")

    # Create summary tables for each metric
    trend_summary = {}
    
    for metric_name, table_name in [('Research', 'research'), ('Business', 'business'), 
                                   ('CPD', 'cpd'), ('IP', 'ip_income')]:
        print(f"\n{metric_name} Income Trends - North East Universities:")
        print("=" * 80)
        
        metric_data = data[table_name]
        if table_name == 'business':
            value_col = metric_data.columns[-1]
        else:
            value_col = 'Value'
        
        # Create summary table for this metric
        summary_data = []
        
        for univ in ne_universities:
            # For CPD data, we need to aggregate by year like in calculate_income_volatility
            if metric_name == 'CPD':
                univ_data = metric_data[metric_data['HE Provider'] == univ]
                if len(univ_data) > 0:
                    # Group by year and sum values
                    univ_data = univ_data.groupby('Academic Year')['Value'].sum().reset_index()
                    univ_data = univ_data.sort_values('Academic Year')
            else:
                univ_data = metric_data[metric_data['HE Provider'] == univ]
                if len(univ_data) > 0:
                    # Sort by year
                    univ_data = univ_data.sort_values('Academic Year')
            
            if len(univ_data) > 0:
                # Debug: Check Durham CPD data specifically
                if univ == 'University of Durham' and metric_name == 'CPD':
                    print(f"\nDEBUG - Durham CPD data:")
                    print(f"Total rows: {len(univ_data)}")
                    print(f"Value column: {value_col}")
                    print(f"Years: {univ_data['Academic Year'].tolist()}")
                    print(f"Values: {univ_data[value_col].tolist()}")
                
                # Calculate basic statistics
                total_income = univ_data[value_col].sum()
                avg_income = univ_data[value_col].mean()
                min_income = univ_data[value_col].min()
                max_income = univ_data[value_col].max()
                
                # Calculate year-over-year changes
                univ_data['pct_change'] = univ_data[value_col].pct_change() * 100
                valid_changes = univ_data['pct_change'].dropna()
                
                # Debug: Check Durham CPD changes specifically
                if univ == 'University of Durham' and metric_name == 'CPD':
                    print(f"Valid changes count: {len(valid_changes)}")
                    print(f"Valid changes: {valid_changes.tolist()}")
                
                if len(valid_changes) > 0:
                    # Handle infinite values
                    valid_changes_clean = valid_changes.replace([np.inf, -np.inf], np.nan)
                    
                    # Check if we have any non-NaN values after cleaning
                    non_nan_changes = valid_changes_clean.dropna()
                    
                    # Debug: Check Durham CPD cleaned changes specifically
                    if univ == 'University of Durham' and metric_name == 'CPD':
                        print(f"Non-NaN changes count: {len(non_nan_changes)}")
                        print(f"Non-NaN changes: {non_nan_changes.tolist()}")
                    
                    if len(non_nan_changes) > 0:
                        avg_change = non_nan_changes.mean()
                        max_increase = non_nan_changes.max()
                        max_decrease = non_nan_changes.min()
                        volatility = non_nan_changes.std()
                    else:
                        # All values were infinite, set to 0 or a default value
                        avg_change = 0.0
                        max_increase = 0.0
                        max_decrease = 0.0
                        volatility = 0.0
                else:
                    avg_change = max_increase = max_decrease = volatility = np.nan
                
                # Calculate trend if enough data points
                if len(valid_changes) > 1:
                    years = range(len(valid_changes))
                    changes = valid_changes.values
                    trend = np.polyfit(years, changes, 1)[0]
                    trend_summary[f"{metric_name}_{univ}"] = trend
                else:
                    trend = np.nan
                
                # Debug: Check Durham CPD trend specifically
                if univ == 'University of Durham' and metric_name == 'CPD':
                    print(f"Trend calculation: len(valid_changes) = {len(valid_changes)}, trend = {trend}")
                
                summary_data.append({
                    'University': univ,
                    'Total Income (£)': f"{total_income:,.0f}",
                    'Avg Income (£)': f"{avg_income:,.0f}",
                    'Min Income (£)': f"{min_income:,.0f}",
                    'Max Income (£)': f"{max_income:,.0f}",
                    'Avg Change (%)': f"{avg_change:.1f}" if not pd.isna(avg_change) else "N/A",
                    'Max Increase (%)': f"{max_increase:.1f}" if not pd.isna(max_increase) else "N/A",
                    'Max Decrease (%)': f"{max_decrease:.1f}" if not pd.isna(max_decrease) else "N/A",
                    'Volatility (%)': f"{volatility:.1f}" if not pd.isna(volatility) else "N/A",
                    'Trend (%/year)': f"{trend:.1f}" if not pd.isna(trend) else "N/A"
                })
        
        # Print table
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            print(df_summary.to_string(index=False))
        
        print("\n" + "=" * 80)

    return trend_summary


def main():
    """Main function to run the resilience and risk analysis"""
    print("Starting Resilience and Risk Analysis...")
    
    # Load data
    data, ne_universities = load_data()
    
    # Calculate volatility metrics
    volatility_data = calculate_income_volatility(data, ne_universities)
    
    # Analyze income concentration
    concentration_data = analyze_income_concentration(data, ne_universities)
    
    # Analyze NE volatility
    ne_volatility_summary = analyze_ne_volatility(volatility_data, ne_universities)
    
    # Analyze national rankings
    national_rankings = analyze_national_volatility_ranking(volatility_data, ne_universities)
    
    # Analyze volatility trends
    trend_summary = analyze_volatility_trends(data, ne_universities)
    
    # Generate markdown report
    md_content = "# Resilience and Risk Analysis\n\n"
    
    # NE Volatility Summary
    md_content += "## North East Universities Volatility Summary\n\n"
    for metric_name, ne_data in ne_volatility_summary.items():
        md_content += f"### {metric_name} Volatility\n\n"
        md_content += "| University | Coefficient of Variation | Mean Income | Std Dev |\n"
        md_content += "|------------|---------------------------|-------------|---------|\n"
        for _, row in ne_data.iterrows():
            md_content += f"| {row['HE Provider']} | {row['cv']:.3f} | £{row['mean']:,.0f} | £{row['std']:,.0f} |\n"
        md_content += "\n"
    
    # National Rankings
    md_content += "## National Volatility Rankings\n\n"
    for metric_name, rankings in national_rankings.items():
        md_content += f"### {metric_name} Volatility Rankings\n\n"
        md_content += "| University | National Rank | Coefficient of Variation |\n"
        md_content += "|------------|---------------|---------------------------|\n"
        for univ, info in rankings.items():
            md_content += f"| {univ} | {info['rank']} | {info['cv']:.3f} |\n"
        md_content += "\n"
    
    # Add trend analysis tables to markdown
    md_content += "## Income Trends Analysis\n\n"
    for metric_name, table_name in [('Research', 'research'), ('Business', 'business'), 
                                   ('CPD', 'cpd'), ('IP', 'ip_income')]:
        md_content += f"### {metric_name} Income Trends - North East Universities\n\n"
        
        metric_data = data[table_name]
        if table_name == 'business':
            value_col = metric_data.columns[-1]
        else:
            value_col = 'Value'
        
        # Create summary table for this metric
        summary_data = []
        
        for univ in ne_universities:
            # For CPD data, we need to aggregate by year like in calculate_income_volatility
            if metric_name == 'CPD':
                univ_data = metric_data[metric_data['HE Provider'] == univ]
                if len(univ_data) > 0:
                    # Group by year and sum values
                    univ_data = univ_data.groupby('Academic Year')['Value'].sum().reset_index()
                    univ_data = univ_data.sort_values('Academic Year')
            else:
                univ_data = metric_data[metric_data['HE Provider'] == univ]
                if len(univ_data) > 0:
                    # Sort by year
                    univ_data = univ_data.sort_values('Academic Year')
            
            if len(univ_data) > 0:
                # Calculate basic statistics
                total_income = univ_data[value_col].sum()
                avg_income = univ_data[value_col].mean()
                min_income = univ_data[value_col].min()
                max_income = univ_data[value_col].max()
                
                # Calculate year-over-year changes
                univ_data['pct_change'] = univ_data[value_col].pct_change() * 100
                valid_changes = univ_data['pct_change'].dropna()
                
                if len(valid_changes) > 0:
                    # Handle infinite values
                    valid_changes_clean = valid_changes.replace([np.inf, -np.inf], np.nan)
                    
                    # Check if we have any non-NaN values after cleaning
                    non_nan_changes = valid_changes_clean.dropna()
                    
                    if len(non_nan_changes) > 0:
                        avg_change = non_nan_changes.mean()
                        max_increase = non_nan_changes.max()
                        max_decrease = non_nan_changes.min()
                        volatility = non_nan_changes.std()
                    else:
                        # All values were infinite, set to 0 or a default value
                        avg_change = 0.0
                        max_increase = 0.0
                        max_decrease = 0.0
                        volatility = 0.0
                else:
                    avg_change = max_increase = max_decrease = volatility = np.nan
                
                # Calculate trend if enough data points
                if len(valid_changes) > 1:
                    years = range(len(valid_changes))
                    changes = valid_changes.values
                    trend = np.polyfit(years, changes, 1)[0]
                else:
                    trend = np.nan
                
                summary_data.append({
                    'University': univ,
                    'Total Income (£)': f"{total_income:,.0f}",
                    'Avg Income (£)': f"{avg_income:,.0f}",
                    'Min Income (£)': f"{min_income:,.0f}",
                    'Max Income (£)': f"{max_income:,.0f}",
                    'Avg Change (%)': f"{avg_change:.1f}" if not pd.isna(avg_change) else "N/A",
                    'Max Increase (%)': f"{max_increase:.1f}" if not pd.isna(max_increase) else "N/A",
                    'Max Decrease (%)': f"{max_decrease:.1f}" if not pd.isna(max_decrease) else "N/A",
                    'Volatility (%)': f"{volatility:.1f}" if not pd.isna(volatility) else "N/A",
                    'Trend (%/year)': f"{trend:.1f}" if not pd.isna(trend) else "N/A"
                })
        
        # Add table to markdown
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            md_content += df_summary.to_markdown(index=False) + "\n\n"
    
    # Save markdown report
    with open('8.resilience_risk_analysis.md', 'w') as f:
        f.write(md_content)
    
    print("\nResilience and Risk Analysis completed!")
    print("Results have been saved to '8.resilience_risk_analysis.md'")


if __name__ == "__main__":
    main() 