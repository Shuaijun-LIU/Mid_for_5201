# Week 2 - Step 4: Weekend and Holiday Order Analysis

## Dataset Overview

This step assesses how weekends and public holidays influence sales, revenue, delivery delays, and order value, using the following datasets:

- `olist_orders_dataset.csv`: Contains timestamped purchase and delivery information per order.
- `olist_order_items_dataset.csv`: Contains order-level item pricing and quantities.
- Brazilian holidays data: Includes traditional holidays and e-commerce events from 2016-2018.

Orders are enriched with holiday flags, weekday information, and then merged with item-level data for comprehensive temporal analysis.

## Data Annotation

Each order is flagged with:
- `is_holiday`: Boolean indicating if the purchase occurred on a public holiday.
- `holiday_name`: Name of the corresponding holiday if applicable.
- `weekday_name`: Specific day of the week (Monday through Sunday).
- `day_category`: Combined classification (Weekday, Weekend, Holiday, Holiday + Weekend).

Annotated dataset `orders_with_weekend_holiday.csv` was saved for further analysis.

## Weekend vs Weekday Analysis

### Sales & Delivery Metrics Comparison

| Metric                | Weekday Days       | Weekend Days       |
|----------------------|-------------------|--------------------|
| Order Count          | 87,695 (77.3%)    | 25,730 (22.7%)     |
| Total Revenue (BRL)  | 10,498,346.45     | 3,093,297.25       |
| Average Order Value  | 119.71            | 120.22             |
| Avg Delivery Delay   | -11.87 days       | -12.59 days        |

### Statistical Tests (Weekend vs Weekday)
- **AOV t-test**: *t = -1.06*, *p = 0.2910* → No significant difference in average order value.
- **Delivery Delay t-test**: *t = -9.94*, *p < 0.0001* → Statistically significant longer delivery delays on weekends.

### Day-by-Day Analysis

| Day of Week | Orders | Revenue (BRL) | AOV (BRL) | % of Total Orders |
|-------------|--------|---------------|-----------|-------------------|
| Monday      | 18,521 | 2,230,812.51  | 120.45    | 16.3%             |
| Tuesday     | 18,369 | 2,172,647.82  | 118.28    | 16.2%             |
| Wednesday   | 17,727 | 2,113,843.59  | 119.24    | 15.6%             |
| Thursday    | 16,919 | 2,018,615.78  | 119.31    | 14.9%             |
| Friday      | 16,159 | 1,962,426.75  | 121.44    | 14.2%             |
| Sunday      | 13,488 | 1,589,278.89  | 117.83    | 11.9%             |
| Saturday    | 12,242 | 1,504,018.36  | 122.86    | 10.8%             |

**Key Findings:**
- Monday has the highest order volume (16.3% of total orders)
- Saturday has the lowest order volume (10.8%) but highest AOV (122.86)
- Friday and Saturday show the highest AOV values
- Sunday has the lowest AOV among all days

## Holiday vs Regular Day Analysis

### Sales & Delivery Metrics Comparison

| Metric                | Regular Days      | Holidays           |
|----------------------|-------------------|--------------------|
| Order Count          | 108,121 (95.3%)   | 5,304 (4.7%)       |
| Total Revenue (BRL)  | 12,975,282.34     | 616,361.36         |
| Average Order Value  | 120.01            | 116.21             |
| Avg Delivery Delay   | -12.11 days       | -10.43 days        |

### Statistical Tests (Holiday vs Regular Day)
- **AOV t-test**: *t = -1.06*, *p = 0.2877* → No significant difference in average order value.
- **Delivery Delay t-test**: *t = 11.58*, *p < 0.0001* → Statistically significant shorter delivery delays on holidays.

## Comprehensive Day Category Analysis

| Day Category        | Orders | Revenue (BRL) | AOV (BRL) | Avg Delivery Delay |
|---------------------|--------|---------------|-----------|-------------------|
| Weekday             | 83,560 (73.7%) | 10,026,195.67 | 119.99    | -11.96 days       |
| Weekend             | 24,561 (21.7%) | 2,949,086.67  | 120.07    | -12.62 days       |
| Holiday             | 4,135 (3.6%)   | 472,150.78    | 114.18    | -9.97 days        |
| Holiday + Weekend   | 1,169 (1.0%)   | 144,210.58     | **123.36** | -12.07 days       |

**Key Insight:** Holiday + Weekend combination shows the highest AOV (123.36), suggesting optimal timing for premium promotions.

## Holiday-Level Breakdown

Top-performing holidays by total sales:

| Holiday           | Orders | Revenue (BRL) | AOV (BRL) | % of Holiday Orders |
|------------------|--------|---------------|-----------|---------------------|
| Black Friday      | 1,376  | 152,653.74    | 110.94    | 25.9%              |
| Valentine's Day   | 438    | 55,172.11     | 125.96    | 8.3%               |
| Cyber Monday      | 477    | 47,923.66     | 100.47    | 9.0%               |
| Mother's Day      | 376    | 46,533.22     | 123.76    | 7.1%               |
| Labor Day         | 410    | 45,744.07     | 111.57    | 7.7%               |
| Father's Day      | 340    | 43,858.40     | 129.00    | 6.4%               |
| Easter Sunday     | 284    | 34,852.62     | 122.72    | 5.4%               |
| Green Monday      | 296    | 33,990.39     | 114.83    | 5.6%               |
| Good Friday       | 246    | 30,256.80     | 123.00    | 4.6%               |
| Tiradentes Day    | 246    | 27,424.05     | 111.48    | 4.6%               |
| Republic Proclamation Day | 217 | 23,711.29 | 109.27 | 4.1%               |
| All Souls' Day    | 148    | 21,263.73     | 143.67    | 2.8%               |
| Nossa Senhora Aparecida | 161 | 18,754.23 | 116.49 | 3.0%               |
| Brazil Independence Day | 106 | 16,893.50 | 159.37 | 2.0%               |
| Christmas Day     | 101    | 10,381.34     | 102.79    | 1.9%               |
| New Year's Day    | 82     | 6,948.21      | 84.73     | 1.5%               |

**Notable Findings:**
- Black Friday accounts for 24.8% of total holiday revenue
- Brazil Independence Day shows the highest AOV (159.37) among all holidays
- E-commerce events (Black Friday, Cyber Monday, Green Monday) collectively account for 40.5% of holiday orders
- All Souls' Day shows high AOV (143.67) despite low order volume

## Visual Outputs (Saved)

| File Name                           | Description                                     |
|------------------------------------|-------------------------------------------------|
| `weekend_aov_comparison.png`       | Bar plot comparing AOV between weekend/weekday  |
| `weekend_delivery_delay_comparison.png` | Histogram of delivery delays by weekend status |
| `weekly_sales_pattern.png`         | Sales by day of week (Monday-Sunday)           |
| `aov_comparison.png`               | Bar plot comparing AOV between holiday/regular |
| `delivery_delay_comparison.png`    | Histogram of delivery delays by holiday status |
| `holiday_sales_trend.png`          | Daily sales trend with holiday overlay         |
| `top_holidays_sales.png`           | Top holidays ranked by revenue                 |
| `day_category_aov_comparison.png`  | AOV comparison across all day categories       |
| `day_category_order_count.png`     | Order count comparison across day categories   |

## Key Observations

### Weekend Patterns
1. **Monday Effect**: Highest order volume (16.3%) suggests weekend shopping momentum carries over
2. **Weekend AOV**: Saturday (122.86) > Sunday (117.83), indicating different weekend shopping behaviors
3. **Delivery Performance**: Weekends show longer delivery delays (-12.59 days vs -11.87 days)

### Holiday Patterns
1. **Holiday Volume**: 4.7% of total orders occur on holidays, showing significant impact
2. **Delivery Efficiency**: Holidays show better delivery performance (-10.43 days vs -12.11 days)
3. **AOV Stability**: No significant AOV difference between holidays and regular days

### Combined Insights
1. **Optimal Timing**: Holiday + Weekend combination yields highest AOV (123.36)
2. **Day Selection**: Friday and Saturday are optimal for high-value promotions
3. **Logistics Planning**: Weekend orders require special attention for delivery optimization

## Insights & Implications

### Marketing Strategy
- **Monday Promotions**: Leverage high order volume with targeted campaigns
- **Weekend Premium**: Focus high-value products on Friday-Saturday
- **Holiday + Weekend**: Optimal timing for premium promotions and exclusive offers
- **E-commerce Events**: Black Friday, Cyber Monday, and Green Monday drive significant volume

### Operations Planning
- **Weekend Logistics**: Prepare for longer delivery times and increased customer expectations
- **Holiday Efficiency**: Maintain high delivery performance during holidays
- **Inventory Management**: Stock up for Monday demand and holiday events
- **Customer Service**: Increase weekend support to match order volume patterns

### Revenue Optimization
- **AOV Strategy**: No need for AOV-based pricing during holidays, focus on volume
- **Day-Specific Offers**: Tailor promotions to day-specific AOV patterns
- **Premium Timing**: Use Holiday + Weekend combination for luxury/premium products
- **Seasonal Planning**: Integrate holiday and weekend insights for comprehensive demand forecasting

This comprehensive analysis provides a foundation for data-driven temporal marketing and operations strategies, enabling Olist sellers to optimize their business performance across different time periods.