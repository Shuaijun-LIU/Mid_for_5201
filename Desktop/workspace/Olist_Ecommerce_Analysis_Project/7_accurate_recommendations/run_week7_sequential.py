#!/usr/bin/env python3
"""
Week7 Sequential Execution Script
Execution date: 2025-07-27
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_name, description):
    """Run single script"""
    print(f"\n{'='*60}")
    print(f"Executing: {description}")
    print(f"Script file: {script_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        # Run script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"SUCCESS: {description} executed successfully")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"FAILED: {description} execution failed")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to execute {script_name}: {e}")
        return False
    
    return True

def main():
    """Main execution function"""
    print("Week7 Sequential Script Execution Started")
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define execution order
    execution_order = [
        # Task 1: Cost-benefit analysis
        ("1.2_cost_benefit_calculations.py", "Task 1.2 - Cost-benefit calculations module"),
        ("1.3_cost_benefit_analysis.py", "Task 1.3 - Cost-benefit analysis module"),
        ("1.1_cost_benefit_analysis_engine.py", "Task 1.1 - Cost-benefit analysis main program"),
        
        # Task 2: Risk assessment and mitigation
        ("2.2_risk_assessment_calculations.py", "Task 2.2 - Risk assessment calculations module"),
        ("2.3_risk_analysis_mitigation.py", "Task 2.3 - Risk analysis mitigation module"),
        ("2.1_risk_assessment_mitigation.py", "Task 2.1 - Risk assessment mitigation main program"),
        
        # Task 3: Precision recommendation generation
        ("3.2_recommendation_calculations.py", "Task 3.2 - Recommendation calculations module"),
        ("3.3_recommendation_analysis.py", "Task 3.3 - Recommendation analysis module"),
        ("3.1_precision_recommendation_generator.py", "Task 3.1 - Precision recommendation generator main program"),
        
        # Task 4: Recommendation validation and optimization
        ("4.2_validation_calculations.py", "Task 4.2 - Validation calculations module"),
        ("4.3_validation_analysis.py", "Task 4.3 - Validation analysis module"),
        ("4.1_recommendation_validation_optimization.py", "Task 4.1 - Recommendation validation optimization main program"),
    ]
    
    # Check if files exist
    print("\nChecking script files...")
    missing_files = []
    for script_name, description in execution_order:
        if not os.path.exists(script_name):
            missing_files.append(script_name)
            print(f"MISSING: {script_name}")
    
    if missing_files:
        print(f"\nERROR: Found {len(missing_files)} missing files, please check:")
        for file in missing_files:
            print(f"  - {file}")
        return
    
    print("SUCCESS: All script files checked")
    
    # Execute scripts
    success_count = 0
    total_count = len(execution_order)
    
    for script_name, description in execution_order:
        if run_script(script_name, description):
            success_count += 1
        else:
            print(f"\nWARNING: Script {script_name} execution failed, continue? (y/n)")
            response = input().lower()
            if response != 'y':
                print("User chose to stop execution")
                break
        
        # Add brief delay to avoid high system load
        time.sleep(1)
    
    # Execution result summary
    print(f"\n{'='*60}")
    print("EXECUTION RESULT SUMMARY")
    print(f"{'='*60}")
    print(f"Total scripts: {total_count}")
    print(f"Successfully executed: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Success rate: {success_count/total_count*100:.1f}%")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_count:
        print("SUCCESS: All scripts executed successfully!")
    else:
        print("WARNING: Some scripts failed, please check error messages")

if __name__ == "__main__":
    main() 