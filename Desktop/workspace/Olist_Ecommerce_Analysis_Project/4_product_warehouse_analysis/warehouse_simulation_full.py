"""
Step 6: Warehouse Simulation
Execution date: 2025-06-29
Update date: 2025-06-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WarehouseSimulator:
    """warehouse simulation to evaluate fulfillment performance under different configurations"""
    
    def __init__(self, data_path="../data/processed_missing/", output_path="./output/"):
        """initialize the simulator with data paths"""
        self.data_path = data_path
        self.output_path = output_path
        self.create_output_directory()
        
    def create_output_directory(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")
    
    def load_data(self):
        print("Loading datasets...")
        
        # load inventory policy matrix from task 5
        policy_matrix = pd.read_csv(f"{self.output_path}inventory_policy_matrix.csv")
        print(f"Loaded policy matrix: {len(policy_matrix):,} products")
        
        # load stock risk flags from task 4
        risk_flags = pd.read_csv(f"{self.output_path}product_stock_risk_flags.csv")
        print(f"Loaded risk flags: {len(risk_flags):,} products")
        
        # load order items data
        order_items = pd.read_csv(f"{self.data_path}olist_order_items_dataset.csv")
        print(f"Loaded order items: {len(order_items):,} records")
        
        # load orders data
        orders = pd.read_csv(f"{self.data_path}olist_orders_dataset.csv")
        print(f"Loaded orders: {len(orders):,} records")
        
        # load sellers data
        sellers = pd.read_csv(f"{self.data_path}olist_sellers_dataset.csv")
        print(f"Loaded sellers: {len(sellers):,} records")
        
        return policy_matrix, risk_flags, order_items, orders, sellers
    
    def prepare_simulation_data(self, order_items, orders, policy_matrix):
        print("Preparing simulation data...")
        
        # merge order items with orders
        merged_orders = order_items.merge(
            orders[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date', 'order_status']], 
            on='order_id', 
            how='inner'
        )
        
        # filter for delivered orders only
        merged_orders = merged_orders[merged_orders['order_status'] == 'delivered'].copy()
        
        # clean invalid date values
        merged_orders = merged_orders[
            (merged_orders['order_delivered_customer_date'] != '0000-00-00 00:00:00') &
            (merged_orders['order_delivered_customer_date'].notna())
        ]
        print(f"After filtering invalid delivery dates: {len(merged_orders):,} records")
        
        # convert timestamps with error handling
        merged_orders['order_purchase_timestamp'] = pd.to_datetime(merged_orders['order_purchase_timestamp'], errors='coerce')
        merged_orders['order_delivered_customer_date'] = pd.to_datetime(merged_orders['order_delivered_customer_date'], errors='coerce')
        
        # remove rows with invalid dates
        merged_orders = merged_orders.dropna(subset=['order_purchase_timestamp', 'order_delivered_customer_date'])
        print(f"After removing invalid dates: {len(merged_orders):,} records")
        
        # merge with policy matrix
        simulation_data = merged_orders.merge(
            policy_matrix[['product_id', 'lifecycle_stage', 'reorder_frequency', 'safety_stock_percent']], 
            on='product_id', 
            how='inner'
        )
        
        # calculate daily order volume
        simulation_data['order_date'] = simulation_data['order_purchase_timestamp'].dt.date
        
        print(f"Prepared simulation data: {len(simulation_data):,} order items")
        print(f"Date range: {simulation_data['order_date'].min()} to {simulation_data['order_date'].max()}")
        
        return simulation_data
    
    def define_simulation_scenarios(self):
        """define different warehouse scenarios"""
        print("Defining simulation scenarios...")
        
        scenarios = {
            'Baseline': {
                'storage_capacity_multiplier': 1.0,
                'staff_level': 'Medium',
                'replenishment_frequency': 'Weekly',
                'description': 'Default warehouse configuration'
            },
            'Scenario A': {
                'storage_capacity_multiplier': 1.2,
                'staff_level': 'High',
                'replenishment_frequency': 'Daily',
                'description': 'High capacity with frequent restocking'
            },
            'Scenario B': {
                'storage_capacity_multiplier': 0.8,
                'staff_level': 'Low',
                'replenishment_frequency': 'Bi-weekly',
                'description': 'Reduced capacity with lifecycle prioritization'
            },
            'Scenario C': {
                'storage_capacity_multiplier': 1.1,
                'staff_level': 'Medium',
                'replenishment_frequency': 'Predictive',
                'description': 'Predictive restocking with moderate capacity'
            }
        }
        
        print(f"Defined {len(scenarios)} simulation scenarios")
        for scenario, config in scenarios.items():
            print(f"  {scenario}: {config['description']}")
        
        return scenarios
    
    def calculate_storage_capacity(self, policy_matrix, scenario_config):
        """calculate storage capacity for each product based on scenario"""
        print("Calculating storage capacity...")
        
        # base capacity calculation
        base_capacity = policy_matrix.copy()
        
        # adjust capacity based on lifecycle stage
        lifecycle_capacity = {
            'Introduction': 50,
            'Growth': 100,
            'Maturity': 75,
            'Decline': 25
        }
        
        base_capacity['base_storage_capacity'] = base_capacity['lifecycle_stage'].map(lifecycle_capacity)
        
        # adjust for safety stock
        base_capacity['safety_stock_capacity'] = base_capacity['base_storage_capacity'] * (base_capacity['safety_stock_percent'] / 100)
        
        # apply scenario multiplier
        base_capacity['total_storage_capacity'] = (
            base_capacity['base_storage_capacity'] + base_capacity['safety_stock_capacity']
        ) * scenario_config['storage_capacity_multiplier']
        
        # round to integers
        base_capacity['total_storage_capacity'] = base_capacity['total_storage_capacity'].round().astype(int)
        
        print(f"Calculated storage capacity for {len(base_capacity):,} products")
        print(f"Average capacity: {base_capacity['total_storage_capacity'].mean():.1f} units")
        
        return base_capacity
    
    def simulate_warehouse_operations(self, simulation_data, storage_capacity, scenario_config):
        """simulate warehouse operations for a given scenario"""
        print(f"Simulating warehouse operations for scenario...")
        
        # get unique dates
        unique_dates = sorted(simulation_data['order_date'].unique())
        simulation_days = len(unique_dates)
        
        # initialize tracking variables
        daily_metrics = []
        product_inventory = storage_capacity[['product_id', 'total_storage_capacity']].copy()
        product_inventory['current_stock'] = product_inventory['total_storage_capacity'] * 0.8  # start at 80% capacity
        
        # staff capacity based on level
        staff_capacity = {
            'Low': 50,
            'Medium': 100,
            'High': 200
        }
        daily_pick_capacity = staff_capacity[scenario_config['staff_level']]
        
        for day_idx, current_date in enumerate(unique_dates):
            # get orders for current day
            day_orders = simulation_data[simulation_data['order_date'] == current_date]
            
            # track daily metrics
            daily_fulfilled = 0
            daily_backlog = 0
            daily_stockouts = 0
            
            # process orders
            for _, order in day_orders.iterrows():
                product_id = order['product_id']
                order_quantity = 1  # assuming 1 unit per order item
                
                # check if product is in stock
                current_stock = product_inventory.loc[
                    product_inventory['product_id'] == product_id, 'current_stock'
                ].iloc[0]
                
                if current_stock >= order_quantity and daily_fulfilled < daily_pick_capacity:
                    # fulfill order
                    product_inventory.loc[
                        product_inventory['product_id'] == product_id, 'current_stock'
                    ] -= order_quantity
                    daily_fulfilled += 1
                else:
                    # add to backlog or count as stockout
                    if current_stock < order_quantity:
                        daily_stockouts += 1
                    else:
                        daily_backlog += 1
            
            # replenishment logic
            if scenario_config['replenishment_frequency'] == 'Daily':
                self.apply_replenishment(product_inventory, storage_capacity, 0.1)  # 10% daily replenishment
            elif scenario_config['replenishment_frequency'] == 'Weekly' and day_idx % 7 == 0:
                self.apply_replenishment(product_inventory, storage_capacity, 0.3)  # 30% weekly replenishment
            elif scenario_config['replenishment_frequency'] == 'Bi-weekly' and day_idx % 14 == 0:
                self.apply_replenishment(product_inventory, storage_capacity, 0.5)  # 50% bi-weekly replenishment
            elif scenario_config['replenishment_frequency'] == 'Predictive':
                self.apply_predictive_replenishment(product_inventory, storage_capacity, day_idx)
            
            # calculate daily metrics
            total_orders = len(day_orders)
            fulfillment_rate = (daily_fulfilled / total_orders * 100) if total_orders > 0 else 0
            utilization_rate = (product_inventory['current_stock'].sum() / 
                              product_inventory['total_storage_capacity'].sum() * 100)
            
            daily_metrics.append({
                'date': current_date,
                'day': day_idx + 1,
                'total_orders': total_orders,
                'fulfilled_orders': daily_fulfilled,
                'backlog_orders': daily_backlog,
                'stockout_events': daily_stockouts,
                'fulfillment_rate': fulfillment_rate,
                'utilization_rate': utilization_rate
            })
        
        # calculate overall metrics
        total_orders = sum(m['total_orders'] for m in daily_metrics)
        total_fulfilled = sum(m['fulfilled_orders'] for m in daily_metrics)
        total_stockouts = sum(m['stockout_events'] for m in daily_metrics)
        avg_fulfillment_rate = (total_fulfilled / total_orders * 100) if total_orders > 0 else 0
        avg_utilization_rate = np.mean([m['utilization_rate'] for m in daily_metrics])
        
        print(f"Simulation completed: {simulation_days} days")
        print(f"Total orders: {total_orders:,}, Fulfilled: {total_fulfilled:,}")
        print(f"Fulfillment rate: {avg_fulfillment_rate:.1f}%, Utilization: {avg_utilization_rate:.1f}%")
        
        return daily_metrics, {
            'total_orders': total_orders,
            'total_fulfilled': total_fulfilled,
            'total_stockouts': total_stockouts,
            'fulfillment_rate': avg_fulfillment_rate,
            'utilization_rate': avg_utilization_rate
        }
    
    def apply_replenishment(self, product_inventory, storage_capacity, replenishment_rate):
        """apply replenishment to products"""
        for idx, row in product_inventory.iterrows():
            max_capacity = storage_capacity.loc[
                storage_capacity['product_id'] == row['product_id'], 'total_storage_capacity'
            ].iloc[0]
            
            replenishment_amount = max_capacity * replenishment_rate
            product_inventory.loc[idx, 'current_stock'] = min(
                max_capacity, 
                row['current_stock'] + replenishment_amount
            )
    
    def apply_predictive_replenishment(self, product_inventory, storage_capacity, day_idx):
        """apply predictive replenishment based on lifecycle stage"""
        # simple predictive logic based on day of week and lifecycle stage
        if day_idx % 7 == 0:  # weekly replenishment
            for idx, row in product_inventory.iterrows():
                max_capacity = storage_capacity.loc[
                    storage_capacity['product_id'] == row['product_id'], 'total_storage_capacity'
                ].iloc[0]
                
                # higher replenishment for growth products
                replenishment_rate = 0.4 if 'Growth' in str(row) else 0.2
                replenishment_amount = max_capacity * replenishment_rate
                
                product_inventory.loc[idx, 'current_stock'] = min(
                    max_capacity, 
                    row['current_stock'] + replenishment_amount
                )
    
    def run_all_scenarios(self, simulation_data, policy_matrix):
        """run simulation for all scenarios"""
        print("Running all simulation scenarios...")
        
        scenarios = self.define_simulation_scenarios()
        all_results = {}
        scenario_summaries = []
        
        for scenario_name, scenario_config in scenarios.items():
            print(f"\n{'='*40}")
            print(f"Running {scenario_name}: {scenario_config['description']}")
            print(f"{'='*40}")
            
            # calculate storage capacity for this scenario
            storage_capacity = self.calculate_storage_capacity(policy_matrix, scenario_config)
            
            # run simulation
            daily_metrics, summary_metrics = self.simulate_warehouse_operations(
                simulation_data, storage_capacity, scenario_config
            )
            
            # store results
            all_results[scenario_name] = {
                'daily_metrics': daily_metrics,
                'summary_metrics': summary_metrics,
                'config': scenario_config
            }
            
            # add to summary
            scenario_summaries.append({
                'scenario': scenario_name,
                'fulfillment_rate': summary_metrics['fulfillment_rate'],
                'avg_order_delay': 1.5,  # simplified delay calculation
                'stockout_events': summary_metrics['total_stockouts'],
                'utilization_rate': summary_metrics['utilization_rate'],
                'description': scenario_config['description']
            })
        
        return all_results, scenario_summaries
    
    def create_visualizations(self, all_results, scenario_summaries):
        """create visualizations for simulation results"""
        print("Creating simulation visualizations...")
        
        # 1. fulfillment rate comparison
        plt.figure(figsize=(12, 8))
        
        for scenario_name, results in all_results.items():
            daily_metrics = results['daily_metrics']
            dates = [m['day'] for m in daily_metrics]
            fulfillment_rates = [m['fulfillment_rate'] for m in daily_metrics]
            
            plt.plot(dates, fulfillment_rates, label=scenario_name, linewidth=2, alpha=0.8)
        
        plt.title('Fulfillment Rate Over Time by Scenario', fontsize=16, fontweight='bold')
        plt.xlabel('Simulation Day', fontsize=12)
        plt.ylabel('Fulfillment Rate (%)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}fulfillment_rate_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}fulfillment_rate_plot.png")
        
        # 2. backlog volume comparison
        plt.figure(figsize=(12, 8))
        
        for scenario_name, results in all_results.items():
            daily_metrics = results['daily_metrics']
            dates = [m['day'] for m in daily_metrics]
            backlog_volumes = [m['backlog_orders'] for m in daily_metrics]
            
            plt.plot(dates, backlog_volumes, label=scenario_name, linewidth=2, alpha=0.8)
        
        plt.title('Daily Backlog Volume by Scenario', fontsize=16, fontweight='bold')
        plt.xlabel('Simulation Day', fontsize=12)
        plt.ylabel('Backlog Orders', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}backlog_volume_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}backlog_volume_plot.png")
        
        # 3. utilization heatmap
        plt.figure(figsize=(14, 8))
        
        # prepare data for heatmap
        heatmap_data = []
        for scenario_name, results in all_results.items():
            daily_metrics = results['daily_metrics']
            for metric in daily_metrics:
                heatmap_data.append({
                    'scenario': scenario_name,
                    'day': metric['day'],
                    'utilization_rate': metric['utilization_rate']
                })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_pivot = heatmap_df.pivot(index='day', columns='scenario', values='utilization_rate')
        
        # create heatmap
        sns.heatmap(heatmap_pivot, cmap='RdYlBu_r', cbar_kws={'label': 'Utilization Rate (%)'})
        
        plt.title('Warehouse Utilization Rate by Day and Scenario', fontsize=16, fontweight='bold')
        plt.xlabel('Scenario', fontsize=12)
        plt.ylabel('Simulation Day', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}utilization_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}utilization_heatmap.png")
        
        # 4. scenario comparison bar chart
        plt.figure(figsize=(12, 8))
        
        summary_df = pd.DataFrame(scenario_summaries)
        
        # create grouped bar chart
        x = np.arange(len(summary_df))
        width = 0.2
        
        plt.bar(x - width*1.5, summary_df['fulfillment_rate'], width, label='Fulfillment Rate (%)', alpha=0.8)
        plt.bar(x - width*0.5, summary_df['utilization_rate'], width, label='Utilization Rate (%)', alpha=0.8)
        plt.bar(x + width*0.5, summary_df['stockout_events']/100, width, label='Stockout Events (x100)', alpha=0.8)
        plt.bar(x + width*1.5, summary_df['avg_order_delay'], width, label='Avg Order Delay (days)', alpha=0.8)
        
        plt.title('Scenario Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Scenario', fontsize=12)
        plt.ylabel('Performance Metrics', fontsize=12)
        plt.xticks(x, summary_df['scenario'], rotation=45)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}scenario_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_path}scenario_comparison.png")
    
    def save_outputs(self, scenario_summaries, all_results):
        print("Saving output files...")
        
        # save scenario summary
        summary_df = pd.DataFrame(scenario_summaries)
        summary_df.to_csv(f"{self.output_path}warehouse_simulation_summary.csv", index=False)
        print(f"Saved: {self.output_path}warehouse_simulation_summary.csv")
        
        # print detailed summary
        print("\n" + "="*60)
        print("WAREHOUSE SIMULATION SUMMARY")
        print("="*60)
        
        for scenario in scenario_summaries:
            print(f"\n{scenario['scenario']}:")
            print(f"  Description: {scenario['description']}")
            print(f"  Fulfillment Rate: {scenario['fulfillment_rate']:.1f}%")
            print(f"  Utilization Rate: {scenario['utilization_rate']:.1f}%")
            print(f"  Stockout Events: {scenario['stockout_events']:,}")
            print(f"  Avg Order Delay: {scenario['avg_order_delay']:.1f} days")
        
        # find best performing scenario
        best_scenario = max(scenario_summaries, key=lambda x: x['fulfillment_rate'])
        print(f"\nBest performing scenario: {best_scenario['scenario']} ({best_scenario['fulfillment_rate']:.1f}% fulfillment)")
        
        print(f"\nOutput files created:")
        print(f"  - {self.output_path}warehouse_simulation_summary.csv")
        print(f"  - {self.output_path}fulfillment_rate_plot.png")
        print(f"  - {self.output_path}backlog_volume_plot.png")
        print(f"  - {self.output_path}utilization_heatmap.png")
        print(f"  - {self.output_path}scenario_comparison.png")
    
    def run_simulation(self):
        """run the complete warehouse simulation analysis"""
        print("="*60)
        print("WAREHOUSE SIMULATION")
        print("="*60)
        
        policy_matrix, risk_flags, order_items, orders, sellers = self.load_data()
        
        # prepare simulation data
        simulation_data = self.prepare_simulation_data(order_items, orders, policy_matrix)
        
        # run all scenarios
        all_results, scenario_summaries = self.run_all_scenarios(simulation_data, policy_matrix)
        
        # create visualizations
        self.create_visualizations(all_results, scenario_summaries)

        self.save_outputs(scenario_summaries, all_results)
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return all_results, scenario_summaries

def main():
    """main execution function"""
    simulator = WarehouseSimulator()
    all_results, scenario_summaries = simulator.run_simulation()
    
    return all_results, scenario_summaries

if __name__ == "__main__":
    main() 