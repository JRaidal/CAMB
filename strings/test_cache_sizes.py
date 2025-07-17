#!/usr/bin/env python3
"""
Test script to analyze optimal cache size for spherical Bessel function
"""

import time
import numpy as np
from functools import lru_cache
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cache_performance():
    """Test different cache sizes for spherical Bessel function"""

    # Import the original function
    from string_correlators import spher_bessel, SPRa, weighting, calculate_eigenvectors

    # Test different cache sizes
    cache_sizes = [100, 500, 1000, 2000, 5000, 10000]

    print("Testing spherical Bessel cache performance...")
    print("=" * 60)

    # Test parameters - similar to what's used in the actual computation
    k_val = 0.1
    ktau_grid = np.logspace(-2, 2, 64)
    tau_grid = ktau_grid / k_val
    nmodes = 32

    results = {}

    for cache_size in cache_sizes:
        print(f"\nTesting cache size: {cache_size}")

        # Create a cached version with this size
        @lru_cache(maxsize=cache_size)
        def spher_bessel_cached_test(n, x_tuple):
            x = np.array(x_tuple)
            return spher_bessel(n, x)

        # Temporarily replace the cached function in the module
        import string_correlators
        original_cached = string_correlators.spher_bessel_cached
        string_correlators.spher_bessel_cached = spher_bessel_cached_test

        # Run the test
        start_time = time.time()
        try:
            result, _ = calculate_eigenvectors(
                k_val, SPRa, tau_grid, weighting, nmodes
            )
            end_time = time.time()

            # Get cache statistics
            cache_info = spher_bessel_cached_test.cache_info()
            total_time = end_time - start_time

            results[cache_size] = {
                'time': total_time,
                'hits': cache_info.hits,
                'misses': cache_info.misses,
                'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0,
                'cache_size': cache_info.currsize
            }

            print(f"  Time: {total_time:.3f}s")
            print(f"  Cache hits: {cache_info.hits}")
            print(f"  Cache misses: {cache_info.misses}")
            print(f"  Hit rate: {results[cache_size]['hit_rate']:.1%}")
            print(f"  Cache size used: {cache_info.currsize}")

        except Exception as e:
            print(f"  Error: {e}")
            results[cache_size] = None

        # Restore original function
        string_correlators.spher_bessel_cached = original_cached

    return results

def analyze_cache_usage():
    """Analyze what values are actually being cached"""

    from string_correlators import spher_bessel, SPRa, weighting, calculate_eigenvectors

    print("\n" + "=" * 60)
    print("Analyzing cache usage patterns...")

    # Track all calls to spherical Bessel
    call_log = []

    def spher_bessel_logged(n, x):
        call_log.append((n, x))
        return spher_bessel(n, x)

    # Temporarily replace the function
    import string_correlators
    original_spher_bessel = string_correlators.spher_bessel
    string_correlators.spher_bessel = spher_bessel_logged

    # Run a small test
    k_val = 0.1
    ktau_grid = np.logspace(-2, 2, 32)  # Smaller for analysis
    tau_grid = ktau_grid / k_val
    nmodes = 16

    try:
        result, _ = calculate_eigenvectors(
            k_val, SPRa, tau_grid, weighting, nmodes
        )

        # Analyze the calls
        print(f"Total spherical Bessel calls: {len(call_log)}")

        # Count unique calls
        unique_calls = set(call_log)
        print(f"Unique (n, x) combinations: {len(unique_calls)}")

        # Count by n value
        n_values = [call[0] for call in call_log]
        unique_n = set(n_values)
        print(f"Unique n values used: {sorted(unique_n)}")

        # Count frequency of each n
        from collections import Counter
        n_counter = Counter(n_values)
        print("Frequency by n value:")
        for n in sorted(n_counter.keys()):
            print(f"  n={n}: {n_counter[n]} calls")

        # Analyze x value ranges
        x_values = [call[1] for call in call_log]
        print(f"x value range: {min(x_values):.6f} to {max(x_values):.6f}")

        # Count duplicates
        call_counter = Counter(call_log)
        duplicates = {call: count for call, count in call_counter.items() if count > 1}
        print(f"Duplicate calls: {len(duplicates)}")
        if duplicates:
            print("Most frequent duplicates:")
            for call, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {call}: {count} times")

    except Exception as e:
        print(f"Error in analysis: {e}")

    # Restore original function
    string_correlators.spher_bessel = original_spher_bessel

def recommend_cache_size(results):
    """Recommend optimal cache size based on results"""

    print("\n" + "=" * 60)
    print("CACHE SIZE ANALYSIS")
    print("=" * 60)

    if not results:
        print("No results to analyze")
        return

    # Find the best performing cache size
    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        print("No valid results")
        return

    # Sort by performance
    sorted_by_time = sorted(valid_results.items(), key=lambda x: x[1]['time'])

    print("Performance ranking (fastest first):")
    for i, (cache_size, result) in enumerate(sorted_by_time):
        print(f"{i+1}. Cache size {cache_size}: {result['time']:.3f}s "
              f"(hit rate: {result['hit_rate']:.1%}, "
              f"cache used: {result['cache_size']})")

    # Find diminishing returns point
    print("\nDiminishing returns analysis:")
    baseline_time = sorted_by_time[0][1]['time']

    for cache_size, result in sorted_by_time:
        time_diff = result['time'] - baseline_time
        if time_diff < 0.001:  # Less than 1ms difference
            print(f"Cache size {cache_size}: essentially same performance as best")
        else:
            print(f"Cache size {cache_size}: {time_diff:.3f}s slower than best")

    # Recommendation
    best_cache_size = sorted_by_time[0][0]
    best_result = sorted_by_time[0][1]

    print(f"\nRECOMMENDATION:")
    print(f"Optimal cache size: {best_cache_size}")
    print(f"Hit rate: {best_result['hit_rate']:.1%}")
    print(f"Cache utilization: {best_result['cache_size']}/{best_cache_size} "
          f"({best_result['cache_size']/best_cache_size:.1%})")

    # Check if we need a larger cache
    if best_result['cache_size'] == best_cache_size:
        print("WARNING: Cache is full - consider testing larger sizes")
    elif best_result['cache_size'] < best_cache_size * 0.5:
        print("INFO: Cache is underutilized - could use smaller size")

if __name__ == "__main__":
    print("Spherical Bessel Cache Size Optimization")
    print("=" * 60)

    # Test different cache sizes
    results = test_cache_performance()

    # Analyze usage patterns
    analyze_cache_usage()

    # Make recommendation
    recommend_cache_size(results)