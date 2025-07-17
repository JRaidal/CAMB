#!/usr/bin/env python3
"""
Profiling script for string_correlators.py
This script profiles the performance of key functions to identify bottlenecks.
"""

import cProfile
import pstats
import io
import time
import numpy as np
from string_correlators import *

def profile_get_correlators():
    """Profile the get_correlators function with typical parameters"""
    print("=== Profiling get_correlators function ===")

    # Typical parameters
    tau1 = 1e-2
    tau2 = 1e-1
    k = 0.1
    SPR_base = SPRa

    # Profile multiple calls
    pr = cProfile.Profile()
    pr.enable()

    for i in range(100):  # Run 100 times to get meaningful statistics
        result = get_correlators(tau1, tau2, k, SPR_base)

    pr.disable()

    # Print results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

def profile_single_k_calculation():
    """Profile calculation for a single k value"""
    print("\n=== Profiling single k calculation ===")

    k_val = 0.05 * hH
    tau_grid = np.logspace(np.log10(10**-2/k_val), np.log10(10**2/k_val), 64)  # Smaller grid for profiling

    pr = cProfile.Profile()
    pr.enable()

    result, _ = calculate_eigenvectors(
        k_val, SPRa, tau_grid, weighting, min(32, len(tau_grid))
    )

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

def profile_integral_functions():
    """Profile the integral calculation functions"""
    print("\n=== Profiling integral functions ===")

    # Test parameters
    x_vals = np.logspace(-2, 2, 50)
    rho_vals = np.logspace(-2, 2, 50)
    n_terms = 50

    functions_to_test = [
        ('I1_int', lambda x, rho: I1_int(x, rho, n_terms)),
        ('I2_int', I2_int),
        ('I3_int', I3_int),
        ('I4_int', lambda x, rho: I4_int(x, rho, n_terms)),
        ('I5_int', I5_int),
        ('I6_int', I6_int),
        ('I1_int_a', I1_int_a),
        ('I4_int_a', I4_int_a),
    ]

    for func_name, func in functions_to_test:
        print(f"\n--- Profiling {func_name} ---")

        pr = cProfile.Profile()
        pr.enable()

        for x in x_vals[:10]:  # Test with first 10 values
            for rho in rho_vals[:10]:
                try:
                    result = func(x, rho)
                except:
                    pass  # Skip problematic parameter combinations

        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        print(s.getvalue())

def profile_spherical_bessel():
    """Profile the spherical bessel function"""
    print("\n=== Profiling spherical bessel function ===")

    x_vals = np.logspace(-3, 3, 1000)
    n_vals = [-2, -1, 0, 1, 2, 3, 4, 5]

    pr = cProfile.Profile()
    pr.enable()

    for n in n_vals:
        for x in x_vals:
            result = spher_bessel(n, x)

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(15)
    print(s.getvalue())

def time_comparison_test():
    """Compare timing of different approaches"""
    print("\n=== Timing comparison tests ===")

    # Test parameters
    tau1, tau2, k = 1e-2, 1e-1, 0.1

    # Time get_correlators
    start_time = time.time()
    for i in range(1000):
        result = get_correlators(tau1, tau2, k, SPRa)
    end_time = time.time()
    print(f"get_correlators (1000 calls): {end_time - start_time:.4f} seconds")
    print(f"Average per call: {(end_time - start_time)/1000:.6f} seconds")

    # Time individual integral functions
    x, rho = 1.0, 0.5
    n_terms = 50

    integral_funcs = [
        ('I1_int', lambda: I1_int(x, rho, n_terms)),
        ('I2_int', lambda: I2_int(x, rho)),
        ('I3_int', lambda: I3_int(x, rho)),
        ('I4_int', lambda: I4_int(x, rho, n_terms)),
        ('I5_int', lambda: I5_int(x, rho)),
        ('I6_int', lambda: I6_int(x, rho)),
        ('I1_int_a', lambda: I1_int_a(x, rho)),
        ('I4_int_a', lambda: I4_int_a(x, rho)),
    ]

    for func_name, func in integral_funcs:
        start_time = time.time()
        for i in range(10000):
            result = func()
        end_time = time.time()
        print(f"{func_name} (10000 calls): {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    print("Starting profiling of string_correlators.py...")
    print("=" * 60)

    # Run profiling tests
    profile_get_correlators()
    profile_integral_functions()
    profile_spherical_bessel()
    profile_single_k_calculation()
    time_comparison_test()

    print("\n" + "=" * 60)
    print("Profiling complete!")