#!/usr/bin/env python3
"""
Comprehensive profiling script for string_correlators.py

This script runs various profiling tests to identify performance bottlenecks
in the string correlator calculations.

Usage: python run_profiling.py
"""

import cProfile
import pstats
import io
import time
import sys
import os
import numpy as np

# Add the current directory to the path so we can import string_correlators
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_with_profiling():
    """Run the main string_correlators script with profiling enabled"""
    print("=== Running main script with profiling ===")

    # Import after adding to path
    import string_correlators

    # Profile the main execution
    pr = cProfile.Profile()
    pr.enable()

    # Run a smaller version of the main computation for profiling
    print("Running reduced computation for profiling...")

    # Use smaller grids for profiling
    k_min_prof = 1e-4
    k_max_prof = 1
    nk_prof = 10

    ktau_min_prof = 1e-2
    ktau_max_prof = 1e2
    nktau_prof = 32

    nmodes_prof = 16

    k_grid_prof = np.logspace(np.log10(k_min_prof), np.log10(k_max_prof), nk_prof)
    ktau_grid_prof = np.logspace(np.log10(ktau_min_prof), np.log10(ktau_max_prof), nktau_prof)

    # Profile single k calculation
    k_val = k_grid_prof[0]
    tau_grid = ktau_grid_prof / k_val

    result, _ = string_correlators.calculate_eigenvectors(
        k_val, string_correlators.SPRa, tau_grid,
        string_correlators.weighting, nmodes_prof
    )

    pr.disable()

    # Save and display results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(50)

    print(s.getvalue())

    # Save to file
    with open('profiling_results.txt', 'w') as f:
        f.write("=== PROFILING RESULTS ===\n")
        f.write(s.getvalue())

    print("\nProfiling results saved to 'profiling_results.txt'")

def profile_individual_functions():
    """Profile individual functions to identify bottlenecks"""
    print("\n=== Profiling individual functions ===")

    import string_correlators as sc

    # Test parameters
    tau1, tau2, k = 1e-2, 1e-1, 0.1
    x, rho = 1.0, 0.5
    n_terms = 50

    functions_to_profile = [
        ("get_correlators", lambda: sc.get_correlators(tau1, tau2, k, sc.SPRa), 100),
        ("spher_bessel", lambda: sc.spher_bessel(1, x), 10000),
        ("I1_int", lambda: sc.I1_int(x, rho, n_terms), 1000),
        ("I2_int", lambda: sc.I2_int(x, rho), 1000),
        ("I3_int", lambda: sc.I3_int(x, rho), 1000),
        ("I4_int", lambda: sc.I4_int(x, rho, n_terms), 1000),
        ("I5_int", lambda: sc.I5_int(x, rho), 1000),
        ("I6_int", lambda: sc.I6_int(x, rho), 1000),
        ("I1_int_a", lambda: sc.I1_int_a(x, rho), 1000),
        ("I4_int_a", lambda: sc.I4_int_a(x, rho), 1000),
    ]

    for func_name, func, n_calls in functions_to_profile:
        print(f"\n--- Profiling {func_name} ({n_calls} calls) ---")

        # Time the function
        start_time = time.time()
        for i in range(n_calls):
            try:
                result = func()
            except Exception as e:
                print(f"Error in {func_name}: {e}")
                break
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / n_calls

        print(f"Total time: {total_time:.4f}s")
        print(f"Average per call: {avg_time:.6f}s")
        print(f"Calls per second: {n_calls/total_time:.0f}")

def memory_profiling():
    """Basic memory usage profiling"""
    print("\n=== Memory Usage Analysis ===")

    import string_correlators as sc
    import psutil
    import gc

    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    print(f"Initial memory usage: {initial_memory:.1f} MB")

    # Test memory usage for different grid sizes
    grid_sizes = [16, 32, 64, 128, 256]

    for nktau in grid_sizes:
        gc.collect()  # Force garbage collection

        k_val = 0.1
        ktau_grid = np.logspace(-2, 2, nktau)
        tau_grid = ktau_grid / k_val

        memory_before = process.memory_info().rss / 1024 / 1024

        try:
            result, _ = sc.calculate_eigenvectors(
                k_val, sc.SPRa, tau_grid, sc.weighting, min(32, nktau)
            )

            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before

            print(f"Grid size {nktau}: Memory used = {memory_used:.1f} MB, "
                  f"Total = {memory_after:.1f} MB")

        except Exception as e:
            print(f"Error with grid size {nktau}: {e}")

        # Clean up
        if 'result' in locals():
            del result
        gc.collect()

def scaling_analysis():
    """Analyze how performance scales with problem size"""
    print("\n=== Scaling Analysis ===")

    import string_correlators as sc

    # Test different problem sizes
    sizes = [16, 32, 64, 128]

    for nktau in sizes:
        print(f"\n--- Testing with nktau = {nktau} ---")

        k_val = 0.1
        ktau_grid = np.logspace(-2, 2, nktau)
        tau_grid = ktau_grid / k_val
        nmodes = min(16, nktau)

        start_time = time.time()

        try:
            result, _ = sc.calculate_eigenvectors(
                k_val, sc.SPRa, tau_grid, sc.weighting, nmodes
            )

            end_time = time.time()
            total_time = end_time - start_time

            print(f"Time: {total_time:.2f}s")
            print(f"Time per tau point: {total_time/nktau:.4f}s")
            print(f"Time per matrix element: {total_time/(nktau*nktau):.6f}s")

        except Exception as e:
            print(f"Error: {e}")

def create_performance_report():
    """Create a comprehensive performance report"""
    print("\n=== Creating Performance Report ===")

    report_filename = "performance_report.txt"

    with open(report_filename, 'w') as f:
        f.write("STRING CORRELATORS PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"NumPy version: {np.__version__}\n\n")

        # Redirect stdout to file for some tests
        original_stdout = sys.stdout
        sys.stdout = f

        try:
            profile_individual_functions()
            scaling_analysis()
            memory_profiling()
        finally:
            sys.stdout = original_stdout

    print(f"Performance report saved to '{report_filename}'")

if __name__ == "__main__":
    print("Starting comprehensive profiling of string_correlators.py")
    print("=" * 60)

    try:
        # Run the profiling tests
        run_with_profiling()
        profile_individual_functions()
        scaling_analysis()
        memory_profiling()
        create_performance_report()

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure string_correlators.py is in the same directory")
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Profiling complete!")
    print("Check 'profiling_results.txt' and 'performance_report.txt' for detailed results.")