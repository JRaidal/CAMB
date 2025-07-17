"""
This is a Python implementation of the I1, I2, I3, I4, I5, and I6 integrals.

The original Fortran code versions are:

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  function spher_bessel(n,x)
    implicit none
    real(DP) spher_bessel
    real(DP) x
    integer n
    real(DP), parameter :: pi = 3.1415926535897932384626433832795d0
    COMPLEX (KIND=nag_wp)           :: z,cy(1)
    REAL (KIND=nag_wp)              :: fnu
    INTEGER                         :: nz,ifail
    CHARACTER (1)                   :: scal
    if (n.eq.-1) then
        spher_bessel=cos(x)/x
       return
    end if
    fnu=0.5d0+n
    z = dcmplx(x,0.0d0)
    scal='u'
    ifail=0
    CALL s17def(fnu,z,1,scal,cy,nz,ifail)
    if (ifail.ne.0) stop
    spher_bessel=sqrt(pi/(2.0d0*x))*dreal(cy(1))
  end function spher_bessel
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  function factorial(n)
    implicit none
    real(DP) factorial,x
    integer n,ifail
    x=1.0d0+n
    factorial = s14aaf(x,ifail)
    if (ifail.ne.0) stop
  end function factorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  function I1_int(x,rho,n)
    implicit none
    real(DP) I1_int
    real(DP) x,rho,term
    integer i,n
    I1_int = 0.0d0
    do i=1,n
       term = 1.0d0/factorial(i)*(rho/(2.0d0*i-1.0d0))*(-x**2/(2.0d0*rho))**i*spher_bessel(i-1,rho)
       I1_int = I1_int + term
    end do
  end function I1_int
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 function I2_int(x,rho)
    implicit none
    real(DP) I2_int
    real(DP) x,rho,px,rpx,srpx
    px = rho**2+x**2
    rpx = sqrt(px)
    srpx = sin(rpx)
    I2_int = srpx/rpx
  end function I2_int
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 function I3_int(x,rho)
    implicit none
    real(DP) I3_int
    real(DP) x,rho,px,rpx,srpx,crpx
    px = rho**2+x**2
    rpx = sqrt(px)
    srpx = sin(rpx)
    crpx = cos(rpx)
    I3_int = crpx/px*(1.0d0-3.0d0*x**2/px)+srpx/rpx*(1.0d0-(1.0d0+x**2)/px+3.0d0*x**2/px**2)
  end function I3_int
 !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  function I4_int(x,rho,n)
    implicit none
    real(DP) I4_int
    real(DP) x,rho,term
    integer i,n
    I4_int = cos(x)/rho**2
    do i=1,n
       term = - 1.0d0/factorial(i)*(1.0d0/(2.0d0*i-1.0d0))*(-x**2/(2.0d0*rho))**i*spher_bessel(i-2,rho)
       I4_int = I4_int + term
    end do
  end function I4_int
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 function I5_int(x,rho)
    implicit none
    real(DP) I5_int
    real(DP) x,rho,px,rpx,srpx,crpx
    px = rho**2+x**2
    rpx = sqrt(px)
    crpx = cos(rpx)
    I5_int = (cos(x)-crpx)/rho**2
end function I5_int
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 function I6_int(x,rho)
    implicit none
    real(DP) I6_int
    real(DP) x,rho,px,rpx,srpx,crpx
    px = rho**2+x**2
    rpx = sqrt(px)
    srpx = sin(rpx)
    crpx = cos(rpx)
    I6_int = (srpx/rpx-crpx)/px
end function I6_int
"""

import time
import math
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.special as sp
from numba import njit, float64, int32

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

SMALL_RHO_LIMIT_NUMBA = 0.01                      # triggers analytic branch

@njit(cache=True, fastmath=True)
def _safe_downward_j(max_l: int, rho: float) -> np.ndarray:
    """
    Down‑ward recursion for spherical‑Bessel values with automatic
    rescaling so that |j| never exceeds 1e150 (prevents overflow that
    would later turn into NaNs when multiplied by zero).
    """
    j = np.empty(max_l + 2, dtype=np.float64)
    j[max_l + 1] = 0.0
    j[max_l]     = 1.0

    for l in range(max_l, 0, -1):
        j_prev = ((2.0 * l + 1.0) / rho) * j[l] - j[l + 1]

        if abs(j_prev) > 1.0e150:     # rescale entire array by 1e‑150
            j *= 1.0e-150
            j_prev = ((2.0 * l + 1.0) / rho) * j[l] - j[l + 1]

        j[l - 1] = j_prev
    return j


@njit(cache=True, fastmath=True)
def _next_power_term(pt: float, base: float, i: int) -> Tuple[float, bool]:
    """
    Safe update  pt ← pt * base .  If that multiplication would overflow
    IEEE 754 double precision (≈ 1e308) the function returns the *old*
    pt and an overflow flag so that the outer summation can terminate.
    """
    if base == 0.0:
        return 0.0, False

    LOG_LIMIT = 700.0                       # slightly below ln(1e308)
    ln_base   = math.log(abs(base))

    if i * ln_base > LOG_LIMIT:
        return pt, True                     # signal overflow to caller

    return pt * base, False


@njit(float64(float64, float64, int32), fastmath=True, cache=True)
def I1_int_numba(x, rho, n_terms):
    rho = max(rho, 1e-12)
    x2 = x * x
    base = -x2 / (2.0 * rho)

    # ---------- analytic small‑ρ branch ----------
    if rho < SMALL_RHO_LIMIT_NUMBA:
        fact_inv   = 1.0
        power_term = 1.0
        val        = 0.0
        overflow   = False

        for i in range(1, n_terms + 1):
            fact_inv /= i
            power_term, overflow = _next_power_term(power_term, base, i)
            if overflow:
                break

            # leading‑order j_{i‑1}
            l = i - 1
            if l == 0:
                j_leading = 1.0
            else:
                dbl_fact = 1.0
                k = 2 * l + 1
                while k > 1:
                    dbl_fact *= k
                    k -= 2
                j_leading = (rho ** l) / dbl_fact

            term = fact_inv * (rho / (2.0 * i - 1.0)) * power_term * j_leading
            val += term
            if abs(term) < 1e-15 * abs(val):
                break
        return val

    # ---------- downward‑recursion branch ----------
    j = _safe_downward_j(n_terms - 1, rho)
    scale = (math.sin(rho) / rho) / j[0] if j[0] != 0.0 else 0.0
    for idx in range(n_terms):
        j[idx] *= scale

    fact_inv   = 1.0
    power_term = 1.0
    val        = 0.0
    overflow   = False

    for i in range(1, n_terms + 1):
        fact_inv /= i
        power_term, overflow = _next_power_term(power_term, base, i)
        if overflow:
            break
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
    if px < 1e-12:
        return 2.0 / 3.0

    rpx   = math.sqrt(px)
    srpx  = math.sin(rpx)
    crpx  = math.cos(rpx)
    inv_px = 1.0 / px

    term1 = crpx * (1.0 - 3.0 * x * x * inv_px) * inv_px
    term2 = srpx / rpx * (1.0 - (1.0 + x * x) * inv_px + 3.0 * x * x * inv_px * inv_px)
    return term1 + term2


@njit(float64(float64, float64, int32), fastmath=True, cache=True)
def I4_int_numba(x, rho, n_terms):
    rho = max(rho, 1e-12)
    x2 = x * x
    base = -x2 / (2.0 * rho)

    val = math.cos(x) / (rho * rho)

    # -------- analytic small‑ρ branch --------
    if rho < SMALL_RHO_LIMIT_NUMBA:
        fact_inv   = 1.0
        power_term = 1.0
        overflow   = False
        j_neg1     = math.cos(rho) / rho   # leading term for j_{‑1}

        for i in range(1, n_terms + 1):
            fact_inv /= i
            power_term, overflow = _next_power_term(power_term, base, i)
            if overflow:
                break

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
            val += term
            if abs(term) < 1e-15 * abs(val):
                break
        return val

    # -------- downward‑recursion branch --------
    j = _safe_downward_j(n_terms, rho)
    scale = (math.sin(rho) / rho) / j[0] if j[0] != 0.0 else 0.0
    for idx in range(n_terms + 1):
        j[idx] *= scale
    j_neg1 = math.cos(rho) / rho

    fact_inv   = 1.0
    power_term = 1.0
    overflow   = False

    for i in range(1, n_terms + 1):
        fact_inv /= i
        power_term, overflow = _next_power_term(power_term, base, i)
        if overflow:
            break
        j_val = j_neg1 if i == 1 else j[i - 2]
        term  = -fact_inv * (1.0 / (2.0 * i - 1.0)) * power_term * j_val
        val  += term
        if abs(term) < 1e-15 * abs(val):
            break
    return val


@njit(float64(float64, float64), fastmath=True, cache=True)
def I5_int_numba(x, rho):
    px = rho * rho + x * x
    if px < 1e-12:
        return 0.5
    rpx = math.sqrt(px)
    w = (x + rpx) / 2.0
    z = (rpx - x) / 2.0
    if w == 0.0:
        sinc_w = 1.0
    else:
        sinc_w = math.sin(w) / w
    if z == 0.0:
        sinc_z = 1.0
    else:
        sinc_z = math.sin(z) / z
    return 0.5 * sinc_w * sinc_z


@njit(float64(float64, float64), fastmath=True, cache=True)
def I6_int_numba(x, rho):
    px = rho * rho + x * x
    if px < 1e-12:
        return 1.0 / 3.0
    rpx  = math.sqrt(px)
    srpx = math.sin(rpx)
    crpx = math.cos(rpx)
    return (srpx / rpx - crpx) / px

# --------------------------------------------------------------------
# Benchmarking
# --------------------------------------------------------------------

def time_call_random(fn, n_terms, n_samples=100, repeats=5):
    """Time function calls with random x and rho values"""
    # Generate random test cases with log-uniform distribution
    np.random.seed(42)  # For reproducible results
    x_vals = np.logspace(np.log10(1e-4), np.log10(1e2), n_samples)
    rho_vals = np.logspace(np.log10(1e-4), np.log10(1e2), n_samples)
    # Shuffle to avoid correlation between x and rho
    np.random.shuffle(x_vals)
    np.random.shuffle(rho_vals)

    t0 = time.perf_counter()
    for _ in range(repeats):
        for x, rho in zip(x_vals, rho_vals):
            fn(x, rho, n_terms)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats * 1e3  # ms


def time_call_random_2arg(fn, n_samples=100, repeats=5):
    """Time function calls with random x and rho values (for 2-arg functions)"""
    # Generate random test cases with log-uniform distribution
    np.random.seed(42)  # For reproducible results
    x_vals = np.logspace(np.log10(1e-4), np.log10(1e2), n_samples)
    rho_vals = np.logspace(np.log10(1e-4), np.log10(1e2), n_samples)
    # Shuffle to avoid correlation between x and rho
    np.random.shuffle(x_vals)
    np.random.shuffle(rho_vals)

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

        # Generate test cases with log-uniform distribution
    np.random.seed(123)
    n_test_cases = 100
    x_vals = np.logspace(np.log10(1e-4), np.log10(1e2), n_test_cases)
    rho_vals = np.logspace(np.log10(1e-4), np.log10(1e2), n_test_cases)
    # Shuffle to avoid correlation between x and rho
    np.random.shuffle(x_vals)
    np.random.shuffle(rho_vals)
    test_cases = list(zip(x_vals, rho_vals))

        # Accuracy comparison with summary metrics
    rel_diff_threshold = 1e-8  # Only show cases above this threshold

    for name, plain_fn, numba_fn, needs_n_terms in functions:
        print(f"\nTesting {name} integral accuracy:")

        rel_diffs = []
        abs_diffs = []
        problem_cases = []

        for x, rho in test_cases:
            if needs_n_terms:
                slow_val = plain_fn(x, rho, n_terms=1000)
                fast_val = numba_fn(x, rho, n_terms=1000)
            else:
                slow_val = plain_fn(x, rho)
                fast_val = numba_fn(x, rho)

            abs_diff = abs(slow_val - fast_val)
            rel_diff = abs_diff / abs(slow_val) if abs(slow_val) > 1e-15 else 0.0

            rel_diffs.append(rel_diff)
            abs_diffs.append(abs_diff)

            # Only store cases above threshold
            if rel_diff > rel_diff_threshold:
                problem_cases.append((x, rho, slow_val, fast_val, abs_diff, rel_diff))

        # Summary statistics
        rel_diffs = np.array(rel_diffs)
        abs_diffs = np.array(abs_diffs)

        print(f"  Summary: {len(test_cases)} test cases")
        print(f"  Max relative error: {np.max(rel_diffs):.2e}")
        print(f"  Mean relative error: {np.mean(rel_diffs):.2e}")
        print(f"  Cases above threshold ({rel_diff_threshold:.0e}): {len(problem_cases)}")

        # Show problem cases if any
        if problem_cases:
            print(f"  Problem cases (rel_diff > {rel_diff_threshold:.0e}):")
            print(f"  {'x':>10} {'rho':>10} {'plain':>12} {'numba':>12} {'abs_diff':>12} {'rel_diff':>12}")
            print("  " + "-" * 68)
            for x, rho, slow_val, fast_val, abs_diff, rel_diff in problem_cases:
                print(f"  {x:10.2e} {rho:10.2e} {slow_val:12.6e} {fast_val:12.6e} {abs_diff:12.6e} {rel_diff:12.6e}")
        else:
            print(f"  ✓ All cases within tolerance")

    print("\n=== Summary ===")
    print("Performance tests completed with random parameter sampling.")
    print("This provides a more comprehensive view of performance across the parameter space.")
