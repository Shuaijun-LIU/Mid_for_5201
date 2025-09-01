import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def save_plot(fig, filename):
    save_dir = Path('visualizations/performance_efficiency')
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
    data['public_engagement'] = pd.read_csv('Data/table-5.csv', skiprows=11)

    # Clean column names for all datasets
    for key in data:
        data[key].columns = data[key].columns.str.strip()

    return data, ne_universities


def calculate_efficiency_metrics(data, ne_universities):
    """Calculate efficiency metrics for all universities"""
    print("\n=== Performance Efficiency Analysis ===\n")

    # Prepare efficiency metrics
    efficiency_data = {}

    # 1. Research Income Efficiency (per unit of staff time)
    print("1. Research Income Efficiency Analysis")
    research_by_univ = data['research'].groupby(['HE Provider', 'Academic Year'])['Value'].sum().reset_index()
    
    # Get staff time from public engagement data
    staff_time = data['public_engagement'][data['public_engagement']['Metric'] == 'Academic staff time (days)']
    staff_time_by_univ = staff_time.groupby(['HE Provider', 'Academic Year'])['Value'].sum().reset_index()
    
    # Merge research income with staff time
    research_efficiency = research_by_univ.merge(staff_time_by_univ, 
                                                on=['HE Provider', 'Academic Year'], 
                                                how='left', 
                                                suffixes=('_income', '_staff_time'))
    
    # Calculate efficiency (income per staff day)
    research_efficiency['efficiency'] = research_efficiency['Value_income'] / research_efficiency['Value_staff_time']
    research_efficiency = research_efficiency.replace([np.inf, -np.inf], np.nan)
    
    efficiency_data['Research'] = research_efficiency
    print(f"Research efficiency calculated for {len(research_efficiency)} observations")

    # 2. Business Services Efficiency
    print("2. Business Services Efficiency Analysis")
    business_by_univ = data['business'].groupby(['HE Provider', 'Academic Year'])[data['business'].columns[-1]].sum().reset_index()
    business_efficiency = business_by_univ.merge(staff_time_by_univ, 
                                               on=['HE Provider', 'Academic Year'], 
                                               how='left')
    business_efficiency['efficiency'] = business_efficiency[data['business'].columns[-1]] / business_efficiency['Value']
    business_efficiency = business_efficiency.replace([np.inf, -np.inf], np.nan)
    
    efficiency_data['Business'] = business_efficiency
    print(f"Business efficiency calculated for {len(business_efficiency)} observations")

    # 3. IP Income Efficiency
    print("3. IP Income Efficiency Analysis")
    ip_by_univ = data['ip_income'].groupby(['HE Provider', 'Academic Year'])['Value'].sum().reset_index()
    ip_efficiency = ip_by_univ.merge(staff_time_by_univ, 
                                    on=['HE Provider', 'Academic Year'], 
                                    how='left')
    ip_efficiency['efficiency'] = ip_efficiency['Value_x'] / ip_efficiency['Value_y']
    ip_efficiency = ip_efficiency.replace([np.inf, -np.inf], np.nan)
    
    efficiency_data['IP'] = ip_efficiency
    print(f"IP efficiency calculated for {len(ip_efficiency)} observations")

    return efficiency_data


def analyze_ne_efficiency(efficiency_data, ne_universities):
    """Analyze efficiency within North East universities"""
    print("\n=== North East Universities Efficiency Analysis ===\n")

    # Calculate average efficiency for each university across all metrics
    ne_efficiency_summary = {}
    
    for metric_name, data in efficiency_data.items():
        print(f"\n{metric_name} Efficiency - North East Universities:")
        ne_data = data[data['HE Provider'].isin(ne_universities)]
        
        # Calculate average efficiency by university
        avg_efficiency = ne_data.groupby('HE Provider')['efficiency'].mean().sort_values(ascending=False)
        
        print("Average Efficiency (Income per Staff Day):")
        for univ, eff in avg_efficiency.items():
            print(f"  {univ}: £{eff:.2f}")
        
        ne_efficiency_summary[metric_name] = avg_efficiency

    # Create visualization for NE efficiency comparison
    plt.figure(figsize=(20, 6))
    
    for i, (metric_name, avg_efficiency) in enumerate(ne_efficiency_summary.items(), 1):
        plt.subplot(1, 3, i)
        
        # Create bar chart
        universities = list(avg_efficiency.index)
        efficiencies = list(avg_efficiency.values)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(universities)))
        
        bars = plt.bar(universities, efficiencies, color=colors)
        plt.title(f'{metric_name} Efficiency - North East Universities')
        plt.xlabel('University')
        plt.ylabel('Efficiency (£ per staff day)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, eff in zip(bars, efficiencies):
            if not np.isnan(eff):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'£{eff:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    save_plot(plt.gcf(), '1.ne_efficiency_comparison.png')
    print("\nNorth East efficiency comparison plot saved as '1.ne_efficiency_comparison.png'")

    return ne_efficiency_summary


def analyze_national_efficiency_ranking(efficiency_data, ne_universities):
    """Analyze national efficiency rankings"""
    print("\n=== National Efficiency Ranking Analysis ===\n")

    national_rankings = {}
    
    for metric_name, data in efficiency_data.items():
        print(f"\n{metric_name} Efficiency - National Rankings:")
        
        # Calculate average efficiency for all universities
        avg_efficiency = data.groupby('HE Provider')['efficiency'].mean().sort_values(ascending=False)
        
        # Find NE universities' rankings
        ne_rankings = {}
        for univ in ne_universities:
            if univ in avg_efficiency.index:
                rank = avg_efficiency.index.get_loc(univ) + 1
                efficiency = avg_efficiency[univ]
                ne_rankings[univ] = {'rank': rank, 'efficiency': efficiency}
                print(f"  {univ}: Rank {rank}/{len(avg_efficiency)}, Efficiency: £{efficiency:.2f}")
        
        national_rankings[metric_name] = ne_rankings

    # Create visualization for national rankings
    plt.figure(figsize=(20, 6))
    
    for i, (metric_name, rankings) in enumerate(national_rankings.items(), 1):
        plt.subplot(1, 3, i)
        
        universities = list(rankings.keys())
        ranks = [rankings[univ]['rank'] for univ in universities]
        efficiencies = [rankings[univ]['efficiency'] for univ in universities]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(universities)))
        
        # Create scatter plot (rank vs efficiency)
        plt.scatter(ranks, efficiencies, c=colors, s=100, alpha=0.7)
        
        # Add university labels
        for j, univ in enumerate(universities):
            plt.annotate(univ.split()[-1], (ranks[j], efficiencies[j]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title(f'{metric_name} Efficiency - National Rankings')
        plt.xlabel('National Rank')
        plt.ylabel('Efficiency (£ per staff day)')
        plt.grid(True, alpha=0.3)
        
        # Invert x-axis so better rank (lower number) is on the left
        plt.gca().invert_xaxis()

    plt.tight_layout()
    save_plot(plt.gcf(), '2.national_efficiency_rankings.png')
    print("\nNational efficiency rankings plot saved as '2.national_efficiency_rankings.png'")

    return national_rankings


def analyze_efficiency_trends(efficiency_data, ne_universities):
    """Analyze efficiency trends over time"""
    print("\n=== Efficiency Trends Analysis ===\n")

    # Create trend analysis for each metric
    plt.figure(figsize=(20, 6))
    
    lines = []
    labels = []
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_universities)))
    
    for i, (metric_name, data) in enumerate(efficiency_data.items(), 1):
        plt.subplot(1, 3, i)
        
        # Filter for NE universities
        ne_data = data[data['HE Provider'].isin(ne_universities)]
        
        # Plot trends for each university
        for j, univ in enumerate(ne_universities):
            univ_data = ne_data[ne_data['HE Provider'] == univ]
            if len(univ_data) > 0:
                line, = plt.plot(univ_data['Academic Year'], univ_data['efficiency'], 
                        marker='o', color=colors[j], linewidth=2, markersize=6)
                if i == 1:  # Only collect lines and labels from first subplot
                    lines.append(line)
                    labels.append(univ)
        
        plt.title(f'{metric_name} Efficiency Trends - North East Universities')
        plt.xlabel('Academic Year')
        plt.ylabel('Efficiency (£ per staff day)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # Add single legend for all subplots
    plt.figlegend(lines, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    save_plot(plt.gcf(), '3.efficiency_trends.png')
    print("\nEfficiency trends plot saved as '3.efficiency_trends.png'")

    # Calculate trend statistics
    trend_summary = {}
    for metric_name, data in efficiency_data.items():
        print(f"\n{metric_name} Efficiency Trends:")
        ne_data = data[data['HE Provider'].isin(ne_universities)]
        
        for univ in ne_universities:
            univ_data = ne_data[ne_data['HE Provider'] == univ]
            if len(univ_data) > 1:
                # Calculate trend (simple linear trend)
                years = range(len(univ_data))
                efficiencies = univ_data['efficiency'].values
                if not np.all(np.isnan(efficiencies)):
                    trend = np.polyfit(years, efficiencies, 1)[0]
                    trend_summary[f"{metric_name}_{univ}"] = trend
                    print(f"  {univ}: Trend = {trend:.2f} £/year")

    return trend_summary


def main():
    """Main function to run the performance efficiency analysis"""
    print("Starting Performance Efficiency Analysis...")
    
    # Load data
    data, ne_universities = load_data()
    
    # Calculate efficiency metrics
    efficiency_data = calculate_efficiency_metrics(data, ne_universities)
    
    # Analyze NE efficiency
    ne_efficiency_summary = analyze_ne_efficiency(efficiency_data, ne_universities)
    
    # Analyze national rankings
    national_rankings = analyze_national_efficiency_ranking(efficiency_data, ne_universities)
    
    # Analyze efficiency trends
    trend_summary = analyze_efficiency_trends(efficiency_data, ne_universities)
    
    # Generate markdown report
    md_content = "# Performance Efficiency Analysis\n\n"
    
    # NE Efficiency Summary
    md_content += "## North East Universities Efficiency Summary\n\n"
    for metric_name, avg_efficiency in ne_efficiency_summary.items():
        md_content += f"### {metric_name} Efficiency\n\n"
        md_content += "| University | Average Efficiency (£/staff day) |\n"
        md_content += "|------------|----------------------------------|\n"
        for univ, eff in avg_efficiency.items():
            md_content += f"| {univ} | £{eff:.2f} |\n"
        md_content += "\n"
    
    # National Rankings
    md_content += "## National Efficiency Rankings\n\n"
    for metric_name, rankings in national_rankings.items():
        md_content += f"### {metric_name} Efficiency Rankings\n\n"
        md_content += "| University | National Rank | Efficiency (£/staff day) |\n"
        md_content += "|------------|---------------|---------------------------|\n"
        for univ, info in rankings.items():
            md_content += f"| {univ} | {info['rank']} | £{info['efficiency']:.2f} |\n"
        md_content += "\n"
    
    # Save markdown report
    with open('7.performance_efficiency_analysis.md', 'w') as f:
        f.write(md_content)
    
    print("\nPerformance Efficiency Analysis completed!")
    print("Results have been saved to '7.performance_efficiency_analysis.md'")


if __name__ == "__main__":
    main() 