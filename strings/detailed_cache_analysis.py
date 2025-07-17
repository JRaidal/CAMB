#!/usr/bin/env python3
"""
Detailed cache analysis for spherical Bessel function
"""

import time
import numpy as np
from functools import lru_cache
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_cache_usage_detailed():
    """Analyze what values are actually being cached"""

    from string_correlators import spher_bessel, SPRa, weighting, calculate_eigenvectors

    print("Detailed Cache Usage Analysis")
    print("=" * 60)

    # Track all calls to spherical Bessel
    call_log = []

    def spher_bessel_logged(n, x):
        # Convert x to tuple if it's an array, otherwise keep as is
        if hasattr(x, '__iter__') and not isinstance(x, str):
            x_key = tuple(x) if hasattr(x, '__len__') else x
        else:
            x_key = x
        call_log.append((n, x_key))
        return spher_bessel(n, x)

    # Temporarily replace the function
    import string_correlators
    original_spher_bessel = string_correlators.spher_bessel
    string_correlators.spher_bessel = spher_bessel_logged

    # Run a test
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

        # Count duplicates
        call_counter = Counter(call_log)
        duplicates = {call: count for call, count in call_counter.items() if count > 1}
        print(f"Duplicate calls: {len(duplicates)}")

        if duplicates:
            print("Most frequent duplicates:")
            for call, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  n={call[0]}, x={call[1]}: {count} times")

        # Calculate theoretical cache effectiveness
        total_calls = len(call_log)
        unique_calls_count = len(unique_calls)
        potential_cache_hits = total_calls - unique_calls_count
        theoretical_hit_rate = potential_cache_hits / total_calls if total_calls > 0 else 0

        print(f"\nCache effectiveness analysis:")
        print(f"Total calls: {total_calls}")
        print(f"Unique calls: {unique_calls_count}")
        print(f"Potential cache hits: {potential_cache_hits}")
        print(f"Theoretical hit rate: {theoretical_hit_rate:.1%}")

        # Recommend cache size based on unique calls
        print(f"\nRecommended minimum cache size: {unique_calls_count}")

        # Test different cache sizes with actual data
        print(f"\nTesting cache sizes with actual call pattern:")
        test_sizes = [50, 100, 200, 500, 1000, unique_calls_count]

        for cache_size in test_sizes:
            hits = 0
            misses = 0
            cache = {}

            for call in call_log:
                if call in cache:
                    hits += 1
                else:
                    misses += 1
                    if len(cache) < cache_size:
                        cache[call] = True
                    else:
                        # LRU eviction - remove oldest (simplified)
                        # In reality, LRU is more complex, but this gives an estimate
                        pass

            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
            print(f"  Cache size {cache_size}: {hit_rate:.1%} hit rate ({hits} hits, {misses} misses)")

    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

    # Restore original function
    string_correlators.spher_bessel = original_spher_bessel

def test_optimal_cache_sizes():
    """Test a more focused range of cache sizes"""

    from string_correlators import spher_bessel, SPRa, weighting, calculate_eigenvectors

    print("\n" + "=" * 60)
    print("Testing focused cache size range")
    print("=" * 60)

    # Test smaller range around the optimal
    cache_sizes = [50, 100, 150, 200, 300, 500]

    # Test parameters
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

        # Run multiple times for better statistics
        times = []
        for _ in range(3):
            start_time = time.time()
            try:
                result, _ = calculate_eigenvectors(
                    k_val, SPRa, tau_grid, weighting, nmodes
                )
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"  Error: {e}")
                break

        if times:
            avg_time = sum(times) / len(times)
            cache_info = spher_bessel_cached_test.cache_info()

            results[cache_size] = {
                'time': avg_time,
                'hits': cache_info.hits,
                'misses': cache_info.misses,
                'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0,
                'cache_size': cache_info.currsize
            }

            print(f"  Avg time: {avg_time:.3f}s")
            print(f"  Hit rate: {results[cache_size]['hit_rate']:.1%}")
            print(f"  Cache used: {cache_info.currsize}/{cache_size}")

        # Restore original function
        string_correlators.spher_bessel_cached = original_cached

    # Find the best
    if results:
        best_cache_size = min(results.keys(), key=lambda k: results[k]['time'])
        best_result = results[best_cache_size]

        print(f"\nBest performing cache size: {best_cache_size}")
        print(f"Time: {best_result['time']:.3f}s")
        print(f"Hit rate: {best_result['hit_rate']:.1%}")
        print(f"Cache utilization: {best_result['cache_size']}/{best_cache_size}")

if __name__ == "__main__":
    analyze_cache_usage_detailed()
    test_optimal_cache_sizes()