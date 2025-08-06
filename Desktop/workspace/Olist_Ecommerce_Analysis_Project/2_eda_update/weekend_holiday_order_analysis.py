'''
Execution date: 2025-06-10
update date: 2025-06-17
Weekend and Holiday Order Analysis for Olist E-commerce
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import os
import matplotlib.dates as mdates

# pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# style
plt.style.use('default')
sns.set_theme()

# create output directory if it doesn't exist
os.makedirs('output/weekend_holiday_analysis', exist_ok=True)

# Brazilian holidays data
BRAZIL_HOLIDAYS = {
    # 2016 holidays
    '2016-01-01': 'New Year\'s Day',
    '2016-03-25': 'Good Friday',
    '2016-03-27': 'Easter Sunday',
    '2016-04-21': 'Tiradentes Day',
    '2016-05-01': 'Labor Day',
    '2016-05-08': 'Mother\'s Day',
    '2016-06-12': 'Valentine\'s Day',
    '2016-08-14': 'Father\'s Day',
    '2016-09-07': 'Brazil Independence Day',
    '2016-10-12': 'Nossa Senhora Aparecida',
    '2016-11-02': 'All Souls\' Day',
    '2016-11-15': 'Republic Proclamation Day',
    '2016-11-25': 'Black Friday',
    '2016-11-28': 'Cyber Monday',
    '2016-12-12': 'Green Monday',
    '2016-12-25': 'Christmas Day',
    
    # 2017 holidays
    '2017-01-01': 'New Year\'s Day',
    '2017-04-14': 'Good Friday',
    '2017-04-16': 'Easter Sunday',
    '2017-04-21': 'Tiradentes Day',
    '2017-05-01': 'Labor Day',
    '2017-05-14': 'Mother\'s Day',
    '2017-06-12': 'Valentine\'s Day',
    '2017-08-13': 'Father\'s Day',
    '2017-09-07': 'Brazil Independence Day',
    '2017-10-12': 'Nossa Senhora Aparecida',
    '2017-11-02': 'All Souls\' Day',
    '2017-11-15': 'Republic Proclamation Day',
    '2017-11-24': 'Black Friday',
    '2017-11-27': 'Cyber Monday',
    '2017-12-11': 'Green Monday',
    '2017-12-25': 'Christmas Day',
    
    # 2018 holidays
    '2018-01-01': 'New Year\'s Day',
    '2018-03-30': 'Good Friday',
    '2018-04-01': 'Easter Sunday',
    '2018-04-21': 'Tiradentes Day',
    '2018-05-01': 'Labor Day',
    '2018-05-13': 'Mother\'s Day',
    '2018-06-12': 'Valentine\'s Day',
    '2018-08-12': 'Father\'s Day',
    '2018-09-07': 'Brazil Independence Day',
    '2018-10-12': 'Nossa Senhora Aparecida',
    '2018-11-02': 'All Souls\' Day',
    '2018-11-15': 'Republic Proclamation Day',
    '2018-11-23': 'Black Friday',
    '2018-11-26': 'Cyber Monday',
    '2018-12-10': 'Green Monday',
    '2018-12-25': 'Christmas Day'
}

def flag_holiday(date):
    date_str = date.strftime('%Y-%m-%d')
    if date_str in BRAZIL_HOLIDAYS:
        return True, BRAZIL_HOLIDAYS[date_str]
    return False, None

def flag_weekend(date):
    """Flag if a date is a weekend and return weekend type"""
    weekday = date.weekday()
    if weekday == 5:  # Saturday
        return True, "Saturday"
    elif weekday == 6:  # Sunday
        return True, "Sunday"
    return False, None

def get_weekday_name(date):
    """Get the specific day of the week name"""
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return weekday_names[date.weekday()]

def get_day_category(is_holiday, weekday_name):
    """Determine the day category based on holiday and weekday"""
    if is_holiday and weekday_name in ['Saturday', 'Sunday']:
        return "Holiday + Weekend"
    elif is_holiday:
        return "Holiday"
    elif weekday_name in ['Saturday', 'Sunday']:
        return "Weekend"
    else:
        return "Weekday"

def annotate_orders():
    """
    Part 1: Annotate orders with holiday and weekday information
    """
    orders_df = pd.read_csv('../data/processed_missing/olist_orders_dataset.csv')
    
    # convert date columns
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    
    # extract order date
    orders_df['order_date'] = orders_df['order_purchase_timestamp'].dt.date
    
    # annotate holidays
    holiday_info = orders_df['order_date'].apply(flag_holiday)
    orders_df['is_holiday'] = holiday_info.apply(lambda x: x[0])
    orders_df['holiday_name'] = holiday_info.apply(lambda x: x[1])
    
    # annotate weekday name
    orders_df['weekday_name'] = orders_df['order_purchase_timestamp'].apply(get_weekday_name)
    
    # create day category
    orders_df['day_category'] = orders_df.apply(
        lambda row: get_day_category(row['is_holiday'], row['weekday_name']), axis=1
    )
    
    # save annotated orders
    orders_df.to_csv('output/weekend_holiday_analysis/orders_with_weekend_holiday.csv', index=False)
    
    # print sample rows
    print("\nSample of Annotated Orders:")
    print(orders_df[['order_id', 'order_date', 'is_holiday', 'holiday_name', 'weekday_name', 'day_category']].head(10))
    
    return orders_df

def analyze_weekend_metrics():
    """
    Part 2: Analyze weekend vs weekday metrics
    """
    orders_df = pd.read_csv('output/weekend_holiday_analysis/orders_with_weekend_holiday.csv')
    order_items_df = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    
    # convert date columns
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    
    # handle invalid dates
    orders_df['order_delivered_customer_date'] = pd.to_datetime(
        orders_df['order_delivered_customer_date'], 
        errors='coerce'
    )
    orders_df['order_estimated_delivery_date'] = pd.to_datetime(
        orders_df['order_estimated_delivery_date'], 
        errors='coerce'
    )
    
    # merge with order items
    merged_df = orders_df.merge(order_items_df, on='order_id', how='left')
    
    # calculate delivery delay only for rows with valid dates
    merged_df['delivery_delay'] = (
        merged_df['order_delivered_customer_date'] - 
        merged_df['order_estimated_delivery_date']
    ).dt.days
    
    # create is_weekend flag for analysis
    merged_df['is_weekend'] = merged_df['weekday_name'].isin(['Saturday', 'Sunday'])
    
    # group by weekend status
    weekend_stats = merged_df.groupby('is_weekend').agg({
        'order_id': 'count',
        'price': 'sum',
        'delivery_delay': 'mean'
    }).reset_index()
    
    # calculate AOV
    weekend_stats['aov'] = weekend_stats['price'] / weekend_stats['order_id']
    
    # rename columns
    weekend_stats.columns = ['is_weekend', 'order_count', 'total_revenue', 'avg_delivery_delay', 'aov']
    
    # print summary statistics
    print("\nWeekend vs Weekday Statistics:")
    print(weekend_stats)
    
    # Calculate percentages
    total_orders = weekend_stats['order_count'].sum()
    total_revenue = weekend_stats['total_revenue'].sum()
    
    print("\nWeekend vs Weekday Percentage Analysis:")
    print(f"Total Orders: {total_orders:,}")
    print(f"Total Revenue: {total_revenue:,.2f}")
    print("\nWeekend vs Weekday Percentages:")
    for _, row in weekend_stats.iterrows():
        day_type = "Weekend" if row['is_weekend'] else "Weekday"
        order_pct = (row['order_count'] / total_orders) * 100
        revenue_pct = (row['total_revenue'] / total_revenue) * 100
        print(f"\n{day_type} Days:")
        print(f"Orders: {row['order_count']:,} ({order_pct:.1f}%)")
        print(f"Revenue: {row['total_revenue']:,.2f} ({revenue_pct:.1f}%)")
        print(f"AOV: {row['aov']:.2f}")
        print(f"Avg Delivery Delay: {row['avg_delivery_delay']:.2f} days")
    
    # perform t-tests
    order_level = merged_df.groupby('order_id').agg({
        'price': 'sum',
        'is_weekend': 'first'
    }).reset_index()
    
    weekend_aov = order_level[order_level['is_weekend']]['price']
    weekday_aov = order_level[~order_level['is_weekend']]['price']
    aov_ttest = stats.ttest_ind(weekend_aov, weekday_aov)
    
    # Only include rows with valid delivery delay for the delay t-test
    valid_delays = merged_df.dropna(subset=['delivery_delay'])
    weekend_delay = valid_delays[valid_delays['is_weekend']]['delivery_delay']
    weekday_delay = valid_delays[~valid_delays['is_weekend']]['delivery_delay']
    delay_ttest = stats.ttest_ind(weekend_delay, weekday_delay)
    
    print("\nStatistical Tests (Weekend vs Weekday):")
    print(f"AOV t-test: t={aov_ttest.statistic:.2f}, p={aov_ttest.pvalue:.4f}")
    print(f"Delivery Delay t-test: t={delay_ttest.statistic:.2f}, p={delay_ttest.pvalue:.4f}")
    
    # Weekday analysis
    print("\nWeekday Analysis:")
    weekday_types = merged_df.groupby('weekday_name').agg({
        'order_id': 'count',
        'price': 'sum'
    }).sort_values('price', ascending=False)
    
    total_orders_all = weekday_types['order_id'].sum()
    total_revenue_all = weekday_types['price'].sum()
    
    weekday_types['order_pct'] = (weekday_types['order_id'] / total_orders_all * 100).round(1)
    weekday_types['revenue_pct'] = (weekday_types['price'] / total_revenue_all * 100).round(1)
    weekday_types['avg_order_value'] = (weekday_types['price'] / weekday_types['order_id']).round(2)
    
    for weekday, row in weekday_types.iterrows():
        print(f"\n{weekday}:")
        print(f"Orders: {row['order_id']:,} ({row['order_pct']}% of total orders)")
        print(f"Revenue: {row['price']:,.2f} ({row['revenue_pct']}% of total revenue)")
        print(f"Average Order Value: {row['avg_order_value']:.2f}")
    
    # create weekend visualizations
    # 1. Weekend vs Weekday AOV comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=weekend_stats, x='is_weekend', y='aov', hue='is_weekend', palette='Set1', legend=False)
    plt.title('Average Order Value: Weekend vs Weekday')
    plt.xlabel('Is Weekend')
    plt.ylabel('Average Order Value')
    plt.xticks([0, 1], ['Weekday', 'Weekend'])
    plt.tight_layout()
    plt.savefig('output/weekend_holiday_analysis/weekend_aov_comparison.png')
    plt.close()
    
    # 2. Weekend delivery delay comparison
    plt.figure(figsize=(10, 6))
    sns.histplot(data=valid_delays, x='delivery_delay', hue='is_weekend', 
                multiple='dodge', bins=30, palette='Set1')
    plt.title('Delivery Delay Distribution: Weekend vs Weekday')
    plt.xlabel('Delivery Delay (days)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('output/weekend_holiday_analysis/weekend_delivery_delay_comparison.png')
    plt.close()
    
    # 3. Weekly sales pattern
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    merged_df['weekday'] = pd.Categorical(merged_df['weekday_name'], categories=weekday_order, ordered=True)
    
    weekly_sales = merged_df.groupby('weekday', observed=True)['price'].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=weekly_sales, x='weekday', y='price', hue='weekday', palette='Set1', legend=False)
    plt.title('Sales by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/weekend_holiday_analysis/weekly_sales_pattern.png')
    plt.close()

def analyze_holiday_metrics():
    """
    Part 3: Analyze holiday vs regular day metrics (original function)
    """
    orders_df = pd.read_csv('output/weekend_holiday_analysis/orders_with_weekend_holiday.csv')
    order_items_df = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    
    # convert date columns
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    
    # invalid dates to NaT (Not a Time)
    orders_df['order_delivered_customer_date'] = pd.to_datetime(
        orders_df['order_delivered_customer_date'], 
        errors='coerce'
    )
    orders_df['order_estimated_delivery_date'] = pd.to_datetime(
        orders_df['order_estimated_delivery_date'], 
        errors='coerce'
    )
    
    # merge with order items
    merged_df = orders_df.merge(order_items_df, on='order_id', how='left')
    
    # calculate delivery delay only for rows with valid dates
    merged_df['delivery_delay'] = (
        merged_df['order_delivered_customer_date'] - 
        merged_df['order_estimated_delivery_date']
    ).dt.days
    
    # group by holiday status
    holiday_stats = merged_df.groupby('is_holiday').agg({
        'order_id': 'count',
        'price': 'sum',
        'delivery_delay': 'mean'
    }).reset_index()
    
    # calculate AOV
    holiday_stats['aov'] = holiday_stats['price'] / holiday_stats['order_id']
    
    # rename columns
    holiday_stats.columns = ['is_holiday', 'order_count', 'total_revenue', 'avg_delivery_delay', 'aov']
    
    # print summary statistics
    print("\nHoliday vs Regular Day Statistics:")
    print(holiday_stats)
    
    # Calculate percentages
    total_orders = holiday_stats['order_count'].sum()
    total_revenue = holiday_stats['total_revenue'].sum()
    
    print("\nPercentage Analysis:")
    print(f"Total Orders: {total_orders:,}")
    print(f"Total Revenue: {total_revenue:,.2f}")
    print("\nHoliday vs Regular Day Percentages:")
    for _, row in holiday_stats.iterrows():
        day_type = "Holiday" if row['is_holiday'] else "Regular"
        order_pct = (row['order_count'] / total_orders) * 100
        revenue_pct = (row['total_revenue'] / total_revenue) * 100
        print(f"\n{day_type} Days:")
        print(f"Orders: {row['order_count']:,} ({order_pct:.1f}%)")
        print(f"Revenue: {row['total_revenue']:,.2f} ({revenue_pct:.1f}%)")
        print(f"AOV: {row['aov']:.2f}")
        print(f"Avg Delivery Delay: {row['avg_delivery_delay']:.2f} days")
    
    # perform t-tests
    # For AOV, we'll use the order-level data
    order_level = merged_df.groupby('order_id').agg({
        'price': 'sum',
        'is_holiday': 'first'
    }).reset_index()
    
    holiday_aov = order_level[order_level['is_holiday']]['price']
    regular_aov = order_level[~order_level['is_holiday']]['price']
    aov_ttest = stats.ttest_ind(holiday_aov, regular_aov)
    
    # Only include rows with valid delivery delay for the delay t-test
    valid_delays = merged_df.dropna(subset=['delivery_delay'])
    holiday_delay = valid_delays[valid_delays['is_holiday']]['delivery_delay']
    regular_delay = valid_delays[~valid_delays['is_holiday']]['delivery_delay']
    delay_ttest = stats.ttest_ind(holiday_delay, regular_delay)
    
    print("\nStatistical Tests:")
    print(f"AOV t-test: t={aov_ttest.statistic:.2f}, p={aov_ttest.pvalue:.4f}")
    print(f"Delivery Delay t-test: t={delay_ttest.statistic:.2f}, p={delay_ttest.pvalue:.4f}")
    
    # Additional statistics
    print("\nHoliday Orders by Type:")
    holiday_types = merged_df[merged_df['is_holiday']].groupby('holiday_name').agg({
        'order_id': 'count',
        'price': 'sum'
    }).sort_values('price', ascending=False)
    
    # Calculate percentages for holiday types
    total_holiday_orders = holiday_types['order_id'].sum()
    total_holiday_revenue = holiday_types['price'].sum()
    
    holiday_types['order_pct'] = (holiday_types['order_id'] / total_holiday_orders * 100).round(1)
    holiday_types['revenue_pct'] = (holiday_types['price'] / total_holiday_revenue * 100).round(1)
    holiday_types['avg_order_value'] = (holiday_types['price'] / holiday_types['order_id']).round(2)
    
    print("\nDetailed Holiday Analysis:")
    for holiday, row in holiday_types.iterrows():
        print(f"\n{holiday}:")
        print(f"Orders: {row['order_id']:,} ({row['order_pct']}% of holiday orders)")
        print(f"Revenue: {row['price']:,.2f} ({row['revenue_pct']}% of holiday revenue)")
        print(f"Average Order Value: {row['avg_order_value']:.2f}")
    
    # create visualizations
    # 1. AOV comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=holiday_stats, x='is_holiday', y='aov', hue='is_holiday', palette='Set2', legend=False)
    plt.title('Average Order Value: Holiday vs Regular Days')
    plt.xlabel('Is Holiday')
    plt.ylabel('Average Order Value')
    plt.xticks([0, 1], ['Regular Days', 'Holidays'])
    plt.tight_layout()
    plt.savefig('output/weekend_holiday_analysis/aov_comparison.png')
    plt.close()
    
    # 2. Delivery delay comparison (only for valid delays)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=valid_delays, x='delivery_delay', hue='is_holiday', 
                multiple='dodge', bins=30, palette='Set2')
    plt.title('Delivery Delay Distribution: Holiday vs Regular Days')
    plt.xlabel('Delivery Delay (days)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('output/weekend_holiday_analysis/delivery_delay_comparison.png')
    plt.close()
    
    # 3. Holiday sales trend
    daily_sales = merged_df.groupby(['order_date', 'is_holiday'])['price'].sum().reset_index()
    daily_sales['order_date'] = pd.to_datetime(daily_sales['order_date'])
    
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=daily_sales, x='order_date', y='price', hue='is_holiday', palette='Set2')
    plt.title('Daily Sales Trend: Holiday vs Regular Days')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    
    # set x-axis format
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  
    plt.xticks(rotation=45, ha='right')
    
    # add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('output/weekend_holiday_analysis/holiday_sales_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Top holidays by sales
    plt.figure(figsize=(12, 6))
    top_holidays = holiday_types.head(10).reset_index()
    sns.barplot(data=top_holidays, x='holiday_name', y='price', hue='holiday_name', palette='Set2', legend=False)
    plt.title('Top 10 Holidays by Sales')
    plt.xlabel('Holiday')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/weekend_holiday_analysis/top_holidays_sales.png')
    plt.close()

def analyze_day_category_metrics():
    """
    Part 4: Analyze all day categories (Holiday, Weekend, Weekday, Holiday+Weekend)
    """
    orders_df = pd.read_csv('output/weekend_holiday_analysis/orders_with_weekend_holiday.csv')
    order_items_df = pd.read_csv('../data/processed_missing/olist_order_items_dataset.csv')
    
    # convert date columns
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    
    # handle invalid dates
    orders_df['order_delivered_customer_date'] = pd.to_datetime(
        orders_df['order_delivered_customer_date'], 
        errors='coerce'
    )
    orders_df['order_estimated_delivery_date'] = pd.to_datetime(
        orders_df['order_estimated_delivery_date'], 
        errors='coerce'
    )
    
    # merge with order items
    merged_df = orders_df.merge(order_items_df, on='order_id', how='left')
    
    # calculate delivery delay only for rows with valid dates
    merged_df['delivery_delay'] = (
        merged_df['order_delivered_customer_date'] - 
        merged_df['order_estimated_delivery_date']
    ).dt.days
    
    # group by day category
    category_stats = merged_df.groupby('day_category').agg({
        'order_id': 'count',
        'price': 'sum',
        'delivery_delay': 'mean'
    }).reset_index()
    
    # calculate AOV
    category_stats['aov'] = category_stats['price'] / category_stats['order_id']
    
    # rename columns
    category_stats.columns = ['day_category', 'order_count', 'total_revenue', 'avg_delivery_delay', 'aov']
    
    # print summary statistics
    print("\nDay Category Statistics:")
    print(category_stats)
    
    # Calculate percentages
    total_orders = category_stats['order_count'].sum()
    total_revenue = category_stats['total_revenue'].sum()
    
    print("\nDay Category Percentage Analysis:")
    print(f"Total Orders: {total_orders:,}")
    print(f"Total Revenue: {total_revenue:,.2f}")
    print("\nDay Category Percentages:")
    for _, row in category_stats.iterrows():
        order_pct = (row['order_count'] / total_orders) * 100
        revenue_pct = (row['total_revenue'] / total_revenue) * 100
        print(f"\n{row['day_category']}:")
        print(f"Orders: {row['order_count']:,} ({order_pct:.1f}%)")
        print(f"Revenue: {row['total_revenue']:,.2f} ({revenue_pct:.1f}%)")
        print(f"AOV: {row['aov']:.2f}")
        print(f"Avg Delivery Delay: {row['avg_delivery_delay']:.2f} days")
    
    # create day category visualizations
    # 1. Day category AOV comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=category_stats, x='day_category', y='aov', hue='day_category', palette='Set3', legend=False)
    plt.title('Average Order Value by Day Category')
    plt.xlabel('Day Category')
    plt.ylabel('Average Order Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/weekend_holiday_analysis/day_category_aov_comparison.png')
    plt.close()
    
    # 2. Day category order count comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=category_stats, x='day_category', y='order_count', hue='day_category', palette='Set3', legend=False)
    plt.title('Order Count by Day Category')
    plt.xlabel('Day Category')
    plt.ylabel('Order Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/weekend_holiday_analysis/day_category_order_count.png')
    plt.close()

def main():
    print("Part 1: Annotating orders with holiday and weekend information...")
    orders_df = annotate_orders()
    
    print("\nPart 2: Analyzing weekend vs weekday metrics...")
    analyze_weekend_metrics()
    
    print("\nPart 3: Analyzing holiday vs regular day metrics...")
    analyze_holiday_metrics()
    
    print("\nPart 4: Analyzing all day categories...")
    analyze_day_category_metrics()

if __name__ == "__main__":
    main() 