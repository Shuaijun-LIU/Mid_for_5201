import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def save_plot(fig, filename):
    save_dir = Path('visualizations/regional_analysis')
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

    # Load all datasets
    data = {}

    base_dir = Path(__file__).resolve().parent  

    # Income data
    data['research'] = pd.read_csv(base_dir / 'table-1.csv', skiprows=11)
    data['business'] = pd.read_csv(base_dir / 'table-2a.csv', skiprows=11)
    data['cpd'] = pd.read_csv(base_dir / 'table-2b.csv', skiprows=11)
    data['regeneration'] = pd.read_csv(base_dir / 'table-3.csv', skiprows=11)

    # IP data
    data['ip_disclosures'] = pd.read_csv(base_dir / 'table-4a.csv', skiprows=11)
    data['ip_licenses'] = pd.read_csv(base_dir / 'table-4b.csv', skiprows=11)
    data['ip_income'] = pd.read_csv(base_dir / 'table-4c.csv', skiprows=11)
    data['ip_income_total'] = pd.read_csv(base_dir / 'table-4d.csv', skiprows=11)
    data['spinouts'] = pd.read_csv(base_dir / 'table-4e.csv', skiprows=11)

    # Public engagement data
    data['public_engagement'] = pd.read_csv(base_dir / 'table-5.csv', skiprows=11)

    # Clean column names for all datasets
    for key in data:
        data[key].columns = data[key].columns.str.strip()

    # Filter for North East universities
    for key in data:
        data[key] = data[key][data[key]['HE Provider'].isin(ne_universities)]

    return data


def calculate_correlations(data):
    """Calculate correlations between different metrics"""
    print("\n=== Correlation Analysis ===\n")

    # Prepare metrics for correlation analysis
    metrics = {}

    # Research income
    metrics['Research Income'] = data['research'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()

    # Business services income
    # For table-2a, the value column is the last column (unnamed)
    metrics['Business Income'] = data['business'].groupby(['HE Provider', 'Academic Year'])[
        data['business'].columns[-1]].sum()

    # CPD income
    metrics['CPD Income'] = data['cpd'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()

    # Regeneration income
    metrics['Regeneration Income'] = data['regeneration'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()

    # IP metrics
    metrics['IP Disclosures'] = data['ip_disclosures'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()
    metrics['IP Licenses'] = data['ip_licenses'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()
    metrics['IP Income'] = data['ip_income'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()

    # Spin-out employment
    metrics['Spin-out Employment'] = data['spinouts'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()

    # Public engagement
    metrics['Public Engagement'] = data['public_engagement'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()

    # Create correlation matrix
    corr_matrix = pd.DataFrame(metrics).corr()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')

    # Add correlation values as text
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                     ha='center', va='center',
                     color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')

    # Set labels
    plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_matrix)), corr_matrix.index)
    plt.title('Correlation Matrix of Key Metrics')
    plt.tight_layout()
    save_plot(plt.gcf(), '1.correlation_matrix.png')

    print("Correlation matrix saved as '1.correlation_matrix.png'")
    print("\nTop 5 strongest correlations:")
    corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
    corr_pairs = corr_pairs[corr_pairs != 1.0]  # Remove self-correlations
    print(corr_pairs.head())


def analyze_rankings(data):
    """Analyze ranking changes over time and return detailed ranking tables"""
    metrics = {}
    metrics['Research Income'] = data['research'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()
    metrics['Business Income'] = data['business'].groupby(['HE Provider', 'Academic Year'])[
        data['business'].columns[-1]].sum()
    metrics['IP Disclosures'] = data['ip_disclosures'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()
    metrics['IP Income'] = data['ip_income'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()
    metrics['Public Engagement'] = data['public_engagement'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()
    rankings = {}
    for metric_name, metric_data in metrics.items():
        # Calculate rank for each year
        ranking = metric_data.groupby('Academic Year', group_keys=False).rank(ascending=False, method='min')
        ranking_df = ranking.to_frame(name='Rank').reset_index()
        ranking_pivot = ranking_df.pivot(index='Academic Year', columns='HE Provider', values='Rank')
        rankings[metric_name] = ranking_pivot
    # Plot ranking changes
    plt.figure(figsize=(20, 10))
    lines = []
    labels = []
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data['research']['HE Provider'].unique())))

    for i, (metric_name, ranking_data) in enumerate(rankings.items(), 1):
        plt.subplot(2, 3, i)
        for j, university in enumerate(data['research']['HE Provider'].unique()):
            university_ranks = ranking_data[university]
            line, = plt.plot(ranking_data.index, university_ranks.values,
                             marker='o', label=university, color=colors[j])
            if i == 1:
                lines.append(line)
                labels.append(university)
        plt.title(f'{metric_name} Rankings')
        plt.xlabel('Academic Year')
        plt.ylabel('Rank')
        plt.grid(True)

    plt.figlegend(lines, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_plot(plt.gcf(), '2.ranking_changes.png')
    print("Ranking changes plot saved as '2.ranking_changes.png'")
    return rankings


def analyze_specialization(data):
    """Analyze specialization patterns and return detailed HHI tables"""
    specialization = {}
    research_by_source = data['research'].pivot_table(
        index=['HE Provider', 'Academic Year'],
        columns='Type of income',
        values='Value',
        aggfunc='sum'
    ).fillna(0)
    business_by_type = data['business'].pivot_table(
        index=['HE Provider', 'Academic Year'],
        columns='Type of service',
        values=data['business'].columns[-1],
        aggfunc='sum'
    ).fillna(0)
    ip_by_type = data['ip_disclosures'].pivot_table(
        index=['HE Provider', 'Academic Year'],
        columns='Type of disclosure or patent',
        values='Value',
        aggfunc='sum'
    ).fillna(0)
    engagement_by_type = data['public_engagement'].pivot_table(
        index=['HE Provider', 'Academic Year'],
        columns='Type of event',
        values='Value',
        aggfunc='sum'
    ).fillna(0)

    def calculate_hhi(df):
        shares = df.div(df.sum(axis=1), axis=0)
        return (shares ** 2).sum(axis=1)

    specialization['Research'] = calculate_hhi(research_by_source).unstack('HE Provider')
    specialization['Business'] = calculate_hhi(business_by_type).unstack('HE Provider')
    specialization['IP'] = calculate_hhi(ip_by_type).unstack('HE Provider')
    specialization['Engagement'] = calculate_hhi(engagement_by_type).unstack('HE Provider')
    # Plot specialization indices
    plt.figure(figsize=(20, 10))
    lines = []
    labels = []
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data['research']['HE Provider'].unique())))

    for i, (area, indices) in enumerate(specialization.items(), 1):
        plt.subplot(2, 2, i)
        for j, university in enumerate(data['research']['HE Provider'].unique()):
            university_indices = indices[university]
            line, = plt.plot(indices.index, university_indices.values,
                             marker='o', label=university, color=colors[j])
            if i == 1:
                lines.append(line)
                labels.append(university)
        plt.title(f'{area} Specialization Index')
        plt.xlabel('Academic Year')
        plt.ylabel('HHI')
        plt.grid(True)

    plt.figlegend(lines, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_plot(plt.gcf(), '3.specialization_indices.png')
    print("Specialization indices plot saved as '3.specialization_indices.png'")
    return specialization


def main():
    """Main function to run the regional analysis and output markdown summary"""
    print("Starting Regional Analysis...")
    data = load_data()
    calculate_correlations(data)
    rankings = analyze_rankings(data)
    specialization = analyze_specialization(data)

    # markdown
    md_content = "# Regional Analysis\n\n"

    # Ranking Analysis
    md_content += "## Ranking Analysis\n\n"
    for metric, df in rankings.items():
        md_content += f"### {metric}\n\n"
        md_content += df.to_string() + "\n\n"

    # Specialization Analysis
    md_content += "## Specialization Analysis\n\n"
    for area, df in specialization.items():
        md_content += f"### {area}\n\n"
        md_content += df.to_string() + "\n\n"

    with open('5.extral_regional_analysis.md', 'w') as f:
        f.write(md_content)

    ###############################
    print("\n# Regional Analysis\n")
    # Ranking Analysis
    print("## Ranking Analysis\n")
    for metric, df in rankings.items():
        print(f"### {metric}\n")
        print(df)
        print("")
    # Specialization Analysis
    print("## Specialization Analysis\n")
    for area, df in specialization.items():
        print(f"### {area}\n")
        print(df)
        print("")
    print("\nRegional Analysis completed!")
    print("Results have been saved to '5.extral_regional_analysis.md'")


if __name__ == "__main__":
    main()

