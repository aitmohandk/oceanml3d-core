#!/usr/bin/env python3
"""
Master Demo Script: Run All DA Baseline Visualizations

This script runs all 4 demonstration scripts sequentially with
recommended parameters for fast execution.

Usage:
    python demos/run_all_demos.py [--mode MODE]
    
Modes:
    - fast (default): 2-second windows, 3 windows (~5 minutes)
    - full: 5-second windows, 10 windows (~15 minutes)
    
Author: Agent B - Visualization Specialist
Date: June 25, 2026
"""

import sys
import os
import argparse
import subprocess
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_script(script_name, args_list, description):
    """Run a demo script with given arguments"""
    print("\n" + "=" * 80)
    print(f"  {description}")
    print("=" * 80)
    print(f"Running: python demos/{script_name} {' '.join(args_list)}\n")
    
    start_time = time.time()
    
    cmd = [sys.executable, f"demos/{script_name}"] + args_list
    result = subprocess.run(cmd, capture_output=False)
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ Completed in {elapsed:.1f} seconds")
        return True
    else:
        print(f"\n✗ Failed (exit code: {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all DA baseline demonstration scripts')
    parser.add_argument('--mode', type=str, default='fast', 
                        choices=['fast', 'full'],
                        help='Execution mode: fast (2s, 3 windows) or full (5s, 10 windows)')
    args = parser.parse_args()
    
    # Set parameters based on mode
    if args.mode == 'fast':
        duration = '2.0'
        num_windows = '3'
        print("\nMode: FAST (recommended for testing)")
        print("  - Window duration: 2.0 seconds")
        print("  - Number of windows: 3")
        print("  - Expected runtime: ~5 minutes")
    else:
        duration = '5.0'
        num_windows = '10'
        print("\nMode: FULL (publication quality)")
        print("  - Window duration: 5.0 seconds")
        print("  - Number of windows: 10")
        print("  - Expected runtime: ~15 minutes")
    
    print("\nThis will generate:")
    print("  - 9 publication-quality figures (300 DPI)")
    print("  - 3 quantitative result files")
    print("\nPress Ctrl+C to cancel, or wait 3 seconds to continue...")
    
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return
    
    total_start = time.time()
    results = []
    
    # Demo 1: Lorenz-63 Dynamics
    results.append(run_script(
        'demo_lorenz63_dynamics.py',
        [],
        'Demo 1: Lorenz-63 Dynamics Visualization'
    ))
    
    # Demo 2: CS1 Baselines
    results.append(run_script(
        'demo_cs1_baselines.py',
        ['--duration', duration],
        'Demo 2: Case Study 1 - Good Reconstruction'
    ))
    
    # Demo 3: CS2 Baselines
    results.append(run_script(
        'demo_cs2_baselines.py',
        ['--duration', duration],
        'Demo 3: Case Study 2 - Poor Reconstruction'
    ))
    
    # Demo 4: Comparison
    results.append(run_script(
        'demo_comparison.py',
        ['--num-windows', num_windows, '--duration', duration],
        'Demo 4: CS1 vs CS2 Comparison'
    ))
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    print(f"Total runtime: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f} seconds)")
    print(f"\nCompleted scripts: {sum(results)}/4")
    
    if all(results):
        print("\n✓ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("\nOutputs:")
        print("  - Figures: outputs/figures/ (9 PNG files)")
        print("  - Results: outputs/results/ (3 TXT files)")
        print("\nNext steps:")
        print("  1. Review figures in outputs/figures/")
        print("  2. Check metrics in outputs/results/")
        print("  3. Read demos/README.md for detailed documentation")
    else:
        print("\n⚠ Some demos failed. Check error messages above.")
        return 1
    
    print("=" * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
