import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def save_plot(fig, filename):
    save_dir = Path('visualizations/public_engagement')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(save_dir / filename, bbox_inches='tight', dpi=300)
    plt.close()

def analyze_public_engagement():
    """Analyze public engagement data from Table 5"""
    df = pd.read_csv('Data/table-5.csv', skiprows=11)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Filter for Durham University
    durham_data = df[df['HE Provider'] == 'University of Durham']
    
    print("\n=== Public Engagement Analysis ===\n")
    
    # 1. Analysis by Nature of Event
    print("Nature of Event Distribution (Durham):")
    nature_dist = durham_data.groupby('Nature of Event')['Value'].sum()
    print(nature_dist)
    
    # Calculate national statistics
    total_universities = df['HE Provider'].nunique()
    durham_total = nature_dist.sum()
    national_avg = df.groupby('HE Provider')['Value'].sum().mean()
    durham_rank = (df.groupby('HE Provider')['Value'].sum() > durham_total).sum() + 1
    
    print("\nNational Statistics for Public Engagement:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: {national_avg:,.0f}")
    print(f"Durham's value: {durham_total:,.0f}")
    print(f"Difference from national average: {durham_total - national_avg:,.0f}")
    
    # Plot nature of event distribution
    plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(nature_dist)))
    ax = plt.bar(nature_dist.index, nature_dist.values, color=colors)
    plt.title('Public Engagement by Nature of Event (Durham)')
    plt.xlabel('Nature of Event')
    plt.ylabel('Total Value')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(nature_dist.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_plot(plt.gcf(), '1.1_nature_of_event.png')
    
    # 2. Analysis by Type of Event
    print("\nType of Event Distribution (Durham):")
    type_dist = durham_data.groupby('Type of event')['Value'].sum()
    print(type_dist)
    
    # Plot type of event distribution
    plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(type_dist)))
    ax = plt.bar(type_dist.index, type_dist.values, color=colors)
    plt.title('Public Engagement by Type of Event (Durham)')
    plt.xlabel('Type of Event')
    plt.ylabel('Total Value')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(type_dist.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_plot(plt.gcf(), '1.2_type_of_event.png')
    
    # 3. Attendees Analysis
    print("\nAttendees Analysis (Durham):")
    attendees_data = durham_data[durham_data['Metric'] == 'Attendees']
    attendees_by_type = attendees_data.groupby('Type of event')['Value'].sum()
    print(attendees_by_type)
    
    # Plot attendees distribution
    plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(attendees_by_type)))
    ax = plt.bar(attendees_by_type.index, attendees_by_type.values, color=colors)
    plt.title('Public Engagement Attendees by Event Type (Durham)')
    plt.xlabel('Type of Event')
    plt.ylabel('Number of Attendees')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(attendees_by_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_plot(plt.gcf(), '1.3_attendees_by_type.png')
    
    # 4. Academic Staff Time Analysis
    print("\nAcademic Staff Time Analysis (Durham):")
    staff_time_data = durham_data[durham_data['Metric'] == 'Academic staff time (days)']
    staff_time_by_type = staff_time_data.groupby('Type of event')['Value'].sum()
    print(staff_time_by_type)
    
    # Plot academic staff time distribution
    plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(staff_time_by_type)))
    ax = plt.bar(staff_time_by_type.index, staff_time_by_type.values, color=colors)
    plt.title('Academic Staff Time by Event Type (Durham)')
    plt.xlabel('Type of Event')
    plt.ylabel('Staff Time (Days)')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(staff_time_by_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_plot(plt.gcf(), '1.4_staff_time_by_type.png')
    
    # 5. North East Universities Comparison
    print("\nNorth East Universities Comparison:")
    ne_universities = ['University of Durham', 'Newcastle University', 
                      'University of Northumbria at Newcastle', 'Teesside University',
                      'The University of Sunderland']
    ne_data = df[df['HE Provider'].isin(ne_universities)]
    ne_totals = ne_data.groupby('HE Provider')['Value'].sum()
    print(ne_totals)
    
    # Plot North East universities comparison
    plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_totals)))
    ax = plt.bar(ne_totals.index, ne_totals.values, color=colors)
    plt.title('Public Engagement Comparison - North East Universities')
    plt.xlabel('University')
    plt.ylabel('Total Value')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(ne_totals.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_plot(plt.gcf(), '1.5_ne_comparison.png')

if __name__ == "__main__":
    analyze_public_engagement() 