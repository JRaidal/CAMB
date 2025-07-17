import time
import math
import numpy as np
import pandas as pd
import scipy.special as sp

from numba import njit, float64, int32


def factorial(n):
    try: return sp.gamma(n + 1.0)
    except ValueError: return np.inf

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

# ---------- numba‑JIT version ----------
@njit(float64(float64, float64, int32), fastmath=True, cache=True)
def I1_int_numba(x, rho, n_terms):
    rho = rho if rho > 1e-12 else 1e-12
    x2 = x * x if x * x > 1e-12 else 1e-12
    base = -x2 / (2.0 * rho)

    # j_0 and j_1
    j_prev = math.sin(rho) / rho
    j_curr = (math.sin(rho) - rho * math.cos(rho)) / (rho * rho)

    fact_inv = 1.0
    power_term = 1.0
    val = 0.0

    for i in range(1, n_terms + 1):
        fact_inv /= i
        power_term *= base

        if i > 1:
            j_next = ((2 * (i - 1) - 1) / rho) * j_curr - j_prev
            j_prev, j_curr = j_curr, j_next

        term = fact_inv * (rho / (2.0 * i - 1.0)) * power_term * j_curr
        val += term
        if abs(term) < 1e-15 * abs(val):
            break

    return val

# --------------- benchmark ---------------
def time_call(fn, *args, repeats=5):
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats * 1e3  # ms

# compile numba first (warm‑up)
I1_int_numba(1.0, 1.0, 10)

sizes = [100, 1_000, 10_000]
rows = []
for n in sizes:
    rows.append({
        "n_terms": n,
        "plain ms": time_call(I1_int, 1.0, 1.0, n),
        "numba ms": time_call(I1_int_numba, 1.0, 1.0, n),
    })

df = pd.DataFrame(rows)
print(df)
