import time
import math
import numpy as np
from numba import njit, float64, int32
import scipy.special as sp
import pandas as pd

# --------------------------------------------------------------------
# Original pure‑Python/SciPy helper
# --------------------------------------------------------------------

def spher_bessel(n, x):
    """
    Spherical bessel functions
    """
    x = np.asarray(x); result = np.zeros_like(x, dtype=float); mask_nz = x != 0; mask_z = ~mask_nz
    if n == -1:
        if np.any(mask_nz): result[mask_nz] = np.cos(x[mask_nz]) / x[mask_nz]
        if np.any(mask_z): result[mask_z] = 0.0 # Should be 1/x -> undefined at 0, or limit to 0.
    elif n == -2:
        if np.any(mask_nz): xn = x[mask_nz]; result[mask_nz] = (-np.sin(xn)/xn - np.cos(xn)) / xn
        if np.any(mask_z): result[mask_z] = -1.0/3.0
    elif n >= 0:
        if np.any(mask_nz): result[mask_nz] = sp.spherical_jn(n, x[mask_nz])
        if np.any(mask_z): result[mask_z] = 1.0 if n == 0 else 0.0
    else: result = np.zeros_like(x, dtype=float)
    return np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)

def factorial(n):
    try: return sp.gamma(n + 1.0)
    except ValueError: return np.inf

def I1_int(x, rho, n_terms):
    val = 0.0; rho_safe = max(rho, 1e-12); x2_safe = max(x**2, 1e-12)
    base = -x2_safe / (2.0 * rho_safe)
    for i in range(1, int(n_terms) + 1):
        fact_i = factorial(i);
        if fact_i == 0 or fact_i == np.inf or fact_i > 1e300: continue
        if base == 0: power_term = 0.0 if i > 0 else 1.0
        elif i * abs(np.log(abs(base) if base != 0 else 1)) < 700: power_term = base**i
        else: power_term = np.sign(base**i) * np.inf if base != 0 else 0.0
        if not np.isfinite(power_term): continue
        term_val = (1.0 / fact_i * (rho_safe / (2.0 * i - 1.0)) * power_term * spher_bessel(i - 1, rho_safe))
        if not np.isfinite(term_val): continue
        new_val = val + term_val
        if not np.isfinite(new_val): break
        val = new_val
    return np.nan_to_num(val)

def I2_int(x, rho):
    px = rho**2 + x**2;
    if px < 1e-12: return 1.0
    rpx = np.sqrt(px); srpx = np.sin(rpx); return np.divide(srpx, rpx, out=np.ones_like(srpx), where=rpx!=0)

def I3_int(x, rho):
    px = rho**2 + x**2;
    if px < 1e-12: return -1/3.0
    rpx = np.sqrt(px); srpx = np.sin(rpx); crpx = np.cos(rpx)
    term1_factor = (1.0 - 3.0 * x**2 / px); term2_factor = (1.0 - (1.0 + x**2) / px + 3.0 * x**2 / px**2)
    term1 = np.divide(crpx * term1_factor, px, out=np.zeros_like(px), where=px!=0)
    term2 = np.divide(srpx * term2_factor, rpx, out=np.zeros_like(rpx), where=rpx!=0)
    val = term1 + term2; return np.nan_to_num(val, nan=-1/3.0, posinf=0.0, neginf=0.0)

def I4_int(x, rho, n_terms):
    rho_safe = max(rho, 1e-12);
    if rho_safe == 0 : return 0.0
    x2_safe = max(x**2, 1e-12)
    val = np.cos(x) / rho_safe**2
    if not np.isfinite(val): val = 0.0
    base = -x2_safe / (2.0 * rho_safe)
    for i in range(1, int(n_terms) + 1):
        fact_i = factorial(i);
        if fact_i == 0 or fact_i == np.inf or fact_i > 1e300: continue
        if base == 0: power_term = 0.0 if i > 0 else 1.0
        elif i * abs(np.log(abs(base) if base != 0 else 1)) < 700: power_term = base**i
        else: power_term = np.sign(base**i) * np.inf if base != 0 else 0.0
        if not np.isfinite(power_term): continue
        term_val = - (1.0 / fact_i * (1.0 / (2.0 * i - 1.0)) * power_term * spher_bessel(i - 2, rho_safe))
        if not np.isfinite(term_val): continue
        new_val = val + term_val
        if not np.isfinite(new_val): break
        val = new_val
    return np.nan_to_num(val)

def I5_int(x, rho):
    rho_safe = max(rho, 1e-12); px = rho_safe**2 + x**2
    if rho_safe**2 < 1e-15 * x**2 and abs(x) > 1e-6: return np.divide(np.sin(x), 2.0*x, out=np.full_like(x, -0.5), where=x!=0)
    elif px < 1e-12: return -0.5
    rpx = np.sqrt(px); crpx = np.cos(rpx); val = (np.cos(x) - crpx) / rho_safe**2
    return np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

def I6_int(x, rho):
    px = rho**2 + x**2;
    if px < 1e-12: return 1/3.0
    rpx = np.sqrt(px); srpx = np.sin(rpx); crpx = np.cos(rpx)
    term1 = np.divide(srpx, rpx, out=np.ones_like(srpx), where=rpx!=0)
    val = np.divide(term1 - crpx, px, out=np.full_like(px, 1/3.0), where=px!=0)
    return np.nan_to_num(val, nan=1/3.0, posinf=0.0, neginf=0.0)

#Integral approximations
def I1_int_a(x, rho):
    if rho == 0: return np.pi * x / 2.0
    j0_rho = sp.jv(0, rho); return (np.pi * x / 2.0) * j0_rho

def I4_int_a(x, rho):
    if rho == 0: return np.pi*x/4.0
    rho_safe = max(rho, 1e-12); j1_rho = sp.jv(1, rho_safe)
    return (np.pi * x * j1_rho) / (2.0 * rho_safe)

# --------------------------------------------------------------------
# Numerically stable implementation of integrals with Numba
# --------------------------------------------------------------------

@njit(float64(float64, float64, int32), fastmath=True, cache=True)
def I1_int_numba(x, rho, n_terms):
    """
    Stable evaluation of the I1 integral series.

    Strategy
    --------
    * For moderate / large rho  (>= 1e‑3) we build all needed spherical‑Bessel
      values j_ℓ(ρ) using **downward recursion**, which is numerically stable.
    * For tiny rho  (< 1e‑3) the downward scheme overflows, so we switch to an
      analytic small‑z expansion where j_ℓ(ρ) ≈ ρ^ℓ / (2ℓ+1)!!  (leading term).
    """
    # Safeguards
    if rho < 1e-12:
        rho = 1e-12
    x2 = x * x
    if x2 < 1e-12:
        x2 = 1e-12
    base = -x2 / (2.0 * rho)

    # --- small‑rho branch ----------------------------------------------------
    if rho < 1e-3:
        fact_inv = 1.0
        power_term = 1.0
        val = 0.0
        # leading‑order: j_0 ≈ 1, j_ℓ (ℓ>0) ≈ ρ^ℓ / (2ℓ+1)!!
        for i in range(1, n_terms + 1):
            fact_inv /= i          # 1/i!
            power_term *= base     # base^i
            # compute leading‑order j_{i‑1}
            l = i - 1
            if l == 0:
                j = 1.0
            else:
                # double factorial (2l+1)!!
                df = 1.0
                k = 2 * l + 1
                while k > 1:
                    df *= k
                    k -= 2
                j = (rho ** l) / df
            term = fact_inv * (rho / (2.0 * i - 1.0)) * power_term * j
            val += term
            if abs(term) < 1e-15 * abs(val):
                break
        return val

    # --- regular (downward‑recursion) branch --------------------------------
    max_l = n_terms - 1
    j = np.zeros(max_l + 2, dtype=np.float64)
    j[max_l + 1] = 0.0
    j[max_l] = 1.0
    # Downward recursion: j_{ℓ-1} = ((2ℓ+1)/ρ) j_ℓ − j_{ℓ+1}
    for l in range(max_l, 0, -1):
        j[l - 1] = ((2.0 * l + 1.0) / rho) * j[l] - j[l + 1]
    # Scale so that j_0 matches the analytic value sin(ρ)/ρ
    j0_exact = math.sin(rho) / rho
    scale = j0_exact / j[0] if j[0] != 0.0 else 0.0
    for idx in range(max_l + 1):
        j[idx] *= scale

    # Accumulate the series
    fact_inv = 1.0
    power_term = 1.0
    val = 0.0
    for i in range(1, n_terms + 1):
        fact_inv /= i
        power_term *= base
        term = fact_inv * (rho / (2.0 * i - 1.0)) * power_term * j[i - 1]
        val += term
        if abs(term) < 1e-15 * abs(val):
            break
    return val

@njit(float64(float64, float64), fastmath=True, cache=True)
def I2_int_numba(x, rho):
    px = rho * rho + x * x
    if px < 1e-12:
        return 1.0
    rpx = math.sqrt(px)
    return math.sin(rpx) / rpx

@njit(float64(float64, float64), fastmath=True, cache=True)
def I3_int_numba(x, rho):
    px = rho * rho + x * x
    if px < 1e-12:                 # original small‑argument guard
        return -1.0 / 3.0          # analytic limit

    rpx   = math.sqrt(px)
    srpx  = math.sin(rpx)
    crpx  = math.cos(rpx)

    inv_px      = 1.0 / px
    x2_over_px  = x * x * inv_px
    inv_px2     = inv_px * inv_px  # (1/px)²

    term1_factor = 1.0 - 3.0 * x2_over_px
    term2_factor = 1.0 - (1.0 + x * x) * inv_px + 3.0 * x * x * inv_px2

    term1 = crpx * term1_factor * inv_px               # (cos · …) / px
    term2 = srpx * term2_factor / rpx                  # (sin · …) / √px

    val = term1 + term2
    if not math.isfinite(val):                         # keep policy of np.nan_to_num
        return -1.0 / 3.0
    return val

@njit(float64(float64, float64, int32), fastmath=True, cache=True)
def I4_int_numba(x, rho, n_terms):
    if rho < 1e-12:
        rho = 1e-12
    x2 = x * x
    if x2 < 1e-12:
        x2 = 1e-12
    base = -x2 / (2.0 * rho)
    val = math.cos(x) / (rho * rho)

    # small rho branch
    if rho < 1e-3:
        fact_inv = 1.0
        power_term = 1.0
        j_neg1 = 1.0 / rho
        for i in range(1, n_terms + 1):
            fact_inv /= i
            power_term *= base

            # compute j_{i-2} leading term
            if i == 1:
                j_val = j_neg1
            else:
                l = i - 2
                if l == 0:
                    j_val = 1.0
                else:
                    dbl_fact = 1.0
                    k = 2 * l + 1
                    while k > 1:
                        dbl_fact *= k
                        k -= 2
                    j_val = (rho ** l) / dbl_fact

            term = -fact_inv * (1.0 / (2.0 * i - 1.0)) * power_term * j_val
            if not math.isfinite(term):
                break
            val += term
            if abs(term) < 1e-15 * abs(val):
                break
        return val

    # downward recursion branch
    max_l = n_terms
    j = np.zeros(max_l + 2, dtype=np.float64)
    j[max_l + 1] = 0.0
    j[max_l] = 1.0
    for l in range(max_l, 0, -1):
        j[l - 1] = ((2.0 * l + 1.0) / rho) * j[l] - j[l + 1]
    scale = (math.sin(rho) / rho) / j[0] if j[0] != 0.0 else 0.0
    for idx in range(max_l + 1):
        j[idx] *= scale
    j_neg1 = math.cos(rho) / rho

    fact_inv = 1.0
    power_term = 1.0
    for i in range(1, n_terms + 1):
        fact_inv /= i
        power_term *= base
        j_val = j_neg1 if i == 1 else j[i - 2]
        term = -fact_inv * (1.0 / (2.0 * i - 1.0)) * power_term * j_val
        if not math.isfinite(term):
            break
        val += term
        if abs(term) < 1e-15 * abs(val):
            break
    return val

@njit(float64(float64, float64), fastmath=True, cache=True)
def I5_int_numba(x, rho):
    # guard against exact zero
    rho_safe = rho if rho > 1e-12 else 1e-12

    # px = ρ² + x²
    px = rho_safe * rho_safe + x * x

    # ---- small‑parameter branches -----------------------------------------
    if rho_safe * rho_safe < 1e-15 * x * x and abs(x) > 1e-6:
        if x != 0.0:
            return math.sin(x) / (2.0 * x)
        else:
            return -0.5

    if px < 1e-12:
        return -0.5

    # ---- main expression ---------------------------------------------------
    rpx  = math.sqrt(px)
    crpx = math.cos(rpx)

    val = (math.cos(x) - crpx) / (rho_safe * rho_safe)

    # replicate np.nan_to_num behaviour
    if not math.isfinite(val):
        return 0.0
    return val

@njit(float64(float64, float64), fastmath=True, cache=True)
def I6_int_numba(x, rho):
    px = rho * rho + x * x
    if px < 1e-12:                     # small‑argument limit
        return 1.0 / 3.0

    rpx  = math.sqrt(px)
    srpx = math.sin(rpx)
    crpx = math.cos(rpx)

    # sin(√px)/√px  guard (√px never zero here, px ≥ 1e‑12)
    term1 = srpx / rpx
    val   = (term1 - crpx) / px

    # replicate np.nan_to_num guard behaviour
    if not math.isfinite(val):
        return 1.0 / 3.0
    return val

# --------------------------------------------------------------------
# Benchmarking
# --------------------------------------------------------------------

def time_call(fn, *args, repeats=5):
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats * 1e3  # ms

def time_call_random(fn, n_terms, n_samples=100, repeats=5):
    """Time function calls with random x and rho values"""
    # Generate random test cases
    np.random.seed(42)  # For reproducible results
    x_vals = np.random.uniform(1e-3, 10.0, n_samples)
    rho_vals = np.random.uniform(1e-3, 10.0, n_samples)

    t0 = time.perf_counter()
    for _ in range(repeats):
        for x, rho in zip(x_vals, rho_vals):
            fn(x, rho, n_terms)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats * 1e3  # ms

def time_call_random_2arg(fn, n_samples=100, repeats=5):
    """Time function calls with random x and rho values (for 2-arg functions)"""
    # Generate random test cases
    np.random.seed(42)  # For reproducible results
    x_vals = np.random.uniform(1e-3, 10.0, n_samples)
    rho_vals = np.random.uniform(1e-3, 10.0, n_samples)

    t0 = time.perf_counter()
    for _ in range(repeats):
        for x, rho in zip(x_vals, rho_vals):
            fn(x, rho)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats * 1e3  # ms


if __name__ == "__main__":

    print("=== Performance Benchmarks with Random Parameters ===\n")

    # All integral benchmarks
    print("Integral Performance Comparison (100 random samples):")
    functions = [
        ("I1", I1_int, I1_int_numba, True),    # True indicates needs n_terms
        ("I2", I2_int, I2_int_numba, False),
        ("I3", I3_int, I3_int_numba, False),
        ("I4", I4_int, I4_int_numba, True),    # True indicates needs n_terms
        ("I5", I5_int, I5_int_numba, False),
        ("I6", I6_int, I6_int_numba, False),
    ]

    # Test different n_terms values for functions that need them
    n_terms_values = [100, 1_000, 10_000]

    all_rows = []
    for name, plain_fn, numba_fn, needs_n_terms in functions:
        if needs_n_terms:
            # Test different n_terms values
            for n_terms in n_terms_values:
                plain_time = time_call_random(plain_fn, n_terms=n_terms, n_samples=100, repeats=3)
                numba_time = time_call_random(numba_fn, n_terms=n_terms, n_samples=100, repeats=3)
                speedup = plain_time / numba_time if numba_time > 0 else float('inf')

                all_rows.append({
                    "function": f"{name}(n={n_terms})",
                    "plain ms": f"{plain_time:.2f}",
                    "numba ms": f"{numba_time:.2f}",
                    "speedup": f"{speedup:.1f}x"
                })
        else:
            # Functions that don't need n_terms
            plain_time = time_call_random_2arg(plain_fn, n_samples=100, repeats=3)
            numba_time = time_call_random_2arg(numba_fn, n_samples=100, repeats=3)
            speedup = plain_time / numba_time if numba_time > 0 else float('inf')

            all_rows.append({
                "function": name,
                "plain ms": f"{plain_time:.2f}",
                "numba ms": f"{numba_time:.2f}",
                "speedup": f"{speedup:.1f}x"
            })

    df = pd.DataFrame(all_rows)
    print(df)
    print()

    # Accuracy comparison with random values for all functions
    print("=== Accuracy Comparison (Random Samples) ===")

    # Generate test cases with different scales
    np.random.seed(123)
    test_cases = []

    # Small values
    test_cases.extend([(x, rho) for x, rho in zip(
        np.random.uniform(1e-4, 1e-2, 5),
        np.random.uniform(1e-4, 1e-2, 5)
    )])

    # Medium values
    test_cases.extend([(x, rho) for x, rho in zip(
        np.random.uniform(1e-2, 1.0, 5),
        np.random.uniform(1e-2, 1.0, 5)
    )])

    # Large values
    test_cases.extend([(x, rho) for x, rho in zip(
        np.random.uniform(1.0, 10.0, 5),
        np.random.uniform(1.0, 10.0, 5)
    )])

    for name, plain_fn, numba_fn, needs_n_terms in functions:
        print(f"\nTesting {name} integral accuracy across parameter space:")
        print(f"{'x':>10} {'rho':>10} {'plain':>12} {'numba':>12} {'abs_diff':>12} {'rel_diff':>12}")
        print("-" * 70)

        for x, rho in test_cases:
            if needs_n_terms:
                slow_val = plain_fn(x, rho, n_terms=100)
                fast_val = numba_fn(x, rho, n_terms=100)
            else:
                slow_val = plain_fn(x, rho)
                fast_val = numba_fn(x, rho)

            abs_diff = abs(slow_val - fast_val)
            rel_diff = abs_diff / abs(slow_val) if abs(slow_val) > 1e-15 else 0.0

            print(f"{x:10.2e} {rho:10.2e} {slow_val:12.6e} {fast_val:12.6e} {abs_diff:12.6e} {rel_diff:12.6e}")

    print("\n=== Summary ===")
    print("Performance tests completed with random parameter sampling.")
    print("This provides a more comprehensive view of performance across the parameter space.")
