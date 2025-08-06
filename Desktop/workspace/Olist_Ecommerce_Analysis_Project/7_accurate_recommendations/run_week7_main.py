#!/usr/bin/env python3
"""
Week7 Main Scripts Execution
只运行主文件
Execution date: 2025-07-27
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_main_script(script_name, description):
    """Run main script"""
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
    print("Week7 Main Scripts Execution Started")
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define main script execution order
    main_scripts = [
        ("1.1_cost_benefit_analysis_engine.py", "Task 1 - Cost-benefit analysis engine"),
        ("2.1_risk_assessment_mitigation.py", "Task 2 - Risk assessment and mitigation model"),
        ("3.1_precision_recommendation_generator.py", "Task 3 - Precision recommendation generator"),
        ("4.1_recommendation_validation_optimization.py", "Task 4 - Recommendation validation and optimization"),
    ]
    
    # Check if files exist
    print("\nChecking main script files...")
    missing_files = []
    for script_name, description in main_scripts:
        if not os.path.exists(script_name):
            missing_files.append(script_name)
            print(f"MISSING: {script_name}")
    
    if missing_files:
        print(f"\nERROR: Found {len(missing_files)} missing files, please check:")
        for file in missing_files:
            print(f"  - {file}")
        return
    
    print("SUCCESS: All main script files checked")
    
    # Execute main scripts
    success_count = 0
    total_count = len(main_scripts)
    
    for script_name, description in main_scripts:
        if run_main_script(script_name, description):
            success_count += 1
        else:
            print(f"\nWARNING: Script {script_name} execution failed, continue? (y/n)")
            response = input().lower()
            if response != 'y':
                print("User chose to stop execution")
                break
        
        # Add brief delay
        time.sleep(2)
    
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
        print("SUCCESS: All main scripts executed successfully!")
    else:
        print("WARNING: Some scripts failed, please check error messages")

if __name__ == "__main__":
    main() 