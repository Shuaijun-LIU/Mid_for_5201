# Customer Geolocation Mapping 

## Overview
- **Execution Date**: 2025-06-20
- **Update Date**: 2025-06-23
- **Main Purpose**: Analyze geographic distribution of RFM customer segments across Brazilian states and cities
- **Input Data**: rfm_segmented_customers.csv, olist_customers_dataset.csv
- **Output Files**: customer_cluster_geolocation.csv, geographic visualizations

## Analysis Process

### 1. Data Integration
I merged RFM-segmented customer data with geographic information from the customers dataset. The integration process combined customer behavioral segments with their geographic locations, creating a comprehensive view of customer distribution across Brazil.

### 2. Geographic Coverage Analysis
The analysis covered:
- **27 Brazilian states** with customer presence
- **4,085 unique cities** across the country
- **96,683 customers** with complete geographic and behavioral data
- **93,358 unique customers** after deduplication

### 3. Cluster Distribution Analysis
I analyzed how each RFM segment is distributed across states and cities, creating both absolute counts and proportional distributions to understand regional customer behavior patterns.

### 4. High-Value Customer Geographic Analysis
I focused specifically on the "Recent Customers" segment (high-value customers) to identify geographic hotspots and opportunities for targeted marketing and service optimization.

## Key Results

### Geographic Distribution Overview
- **Total Coverage**: 27 states, 4,085 cities
- **Data Quality**: 96,683 customers with complete geographic data
- **Geographic Diversity**: Wide spread across Brazil with concentration in major urban areas

### State-Level Cluster Distribution
The analysis revealed distinct geographic patterns for each customer segment:

1. **Recent Customers** (High-Value Segment)
   - **São Paulo**: 4,976 customers (highest concentration)
   - **Rio de Janeiro**: 2,499 customers
   - **Belo Horizonte**: 981 customers
   - **Proportional dominance**: 38-49% in most states

2. **About to Sleep** (Medium-Risk Segment)
   - **São Paulo**: 5,326 customers (highest absolute count)
   - **Rio de Janeiro**: 1,783 customers
   - **Belo Horizonte**: 855 customers
   - **Proportional range**: 12-30% across states

3. **Lost** (Churned Segment)
   - **São Paulo**: 3,761 customers
   - **Rio de Janeiro**: 1,873 customers
   - **Belo Horizonte**: 688 customers
   - **Proportional range**: 18-32% across states

4. **At Risk** (Critical Segment)
   - **São Paulo**: 1,015 customers
   - **Rio de Janeiro**: 464 customers
   - **Belo Horizonte**: 181 customers
   - **Proportional range**: 3-10% across states

### City-Level Insights
The top cities by customer segment revealed:
- **São Paulo** dominates across all segments (5,326-4,976 customers)
- **Rio de Janeiro** is consistently the second-largest market
- **Belo Horizonte** represents the third-largest concentration
- Clear urban concentration pattern with major cities leading

## Business Insights

### Insight 1: Geographic Concentration in Major Urban Centers
The analysis shows strong concentration in Brazil's major urban centers, with São Paulo, Rio de Janeiro, and Belo Horizonte accounting for the majority of customers across all segments. This suggests that Olist's market penetration is primarily urban-focused, with significant opportunities for expansion into smaller cities and rural areas.

### Insight 2: Regional Variation in Customer Quality
Different states show varying proportions of high-value customers. States like AP (Amapá) show 49.3% Recent Customers, while others like AM (Amazonas) show only 33.8%. This regional variation suggests that market maturity and local economic factors significantly influence customer behavior and value.

### Insight 3: At-Risk Customer Geographic Patterns
The "At Risk" segment shows relatively consistent proportions (3-10%) across states, indicating that customer churn risk is not geographically isolated but represents a systemic challenge across all markets. However, the absolute numbers vary significantly, with São Paulo having the highest concentration of at-risk customers.

### Insight 4: Market Penetration Opportunities
The analysis reveals significant geographic expansion opportunities:
- **Rural and smaller city markets** show lower customer density
- **Northern states** (AC, AP, AM) have smaller customer bases but higher proportions of high-value customers
- **Regional expansion** could target underserved areas with high-value customer potential

## Technical Details

### Data Processing Methodology
I used inner joins to ensure data quality, merging RFM segments with geographic data while maintaining referential integrity. The analysis included both absolute counts and proportional distributions to provide comprehensive geographic insights.

### Visualization Approach
I created multiple visualization types:
- **Stacked bar charts** for state-level cluster distribution
- **Proportional heatmaps** for relative segment distribution
- **Choropleth maps** for geographic density visualization
- **City-level analysis** for urban concentration patterns

### Geographic Coverage Assessment
The analysis covered 27 Brazilian states with varying levels of customer penetration, providing a comprehensive view of Olist's geographic footprint and identifying both strong markets and expansion opportunities.

## Output File Description

- `customer_cluster_geolocation.csv`: Complete dataset with RFM segments and geographic information
- `cluster_distribution_by_state.png`: State-level distribution of customer segments
- `top_states_high_value_customers.png`: Geographic concentration of high-value customers
- `brazil_map_cluster_density.png`: Choropleth map showing customer density by state
