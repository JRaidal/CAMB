import time
import math
import numpy as np
import pandas as pd
import scipy.special as sp
import matplotlib.pyplot as plt
from numba import njit, float64, int32

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

@njit(float64(float64, float64, int32), fastmath=True, cache=True)
def I2_int_numba(x, rho, n_terms):
    px = rho * rho + x * x
    if px < 1e-12:
        return 1.0
    rpx = math.sqrt(px)
    return math.sin(rpx) / rpx

@njit(float64(float64, float64, int32), fastmath=True, cache=True)
def I3_int_numba(x, rho, n_terms):
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

@njit(float64(float64, float64, int32), fastmath=True, cache=True)
def I5_int_numba(x, rho, n_terms):
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


@njit(float64(float64, float64, int32), fastmath=True, cache=True)
def I6_int_numba(x, rho, n_terms):
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
# Original pure‑Python/SciPy helper (for reference timing)
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

def I1_int_original(x, rho, n_terms):
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

def I2_int_original(x, rho, n_terms):
    px = rho**2 + x**2;
    if px < 1e-12: return 1.0
    rpx = np.sqrt(px); srpx = np.sin(rpx); return np.divide(srpx, rpx, out=np.ones_like(srpx), where=rpx!=0)

def I3_int_original(x, rho, n_terms):
    px = rho**2 + x**2;
    if px < 1e-12: return -1/3.0
    rpx = np.sqrt(px); srpx = np.sin(rpx); crpx = np.cos(rpx)
    term1_factor = (1.0 - 3.0 * x**2 / px); term2_factor = (1.0 - (1.0 + x**2) / px + 3.0 * x**2 / px**2)
    term1 = np.divide(crpx * term1_factor, px, out=np.zeros_like(px), where=px!=0)
    term2 = np.divide(srpx * term2_factor, rpx, out=np.zeros_like(rpx), where=rpx!=0)
    val = term1 + term2; return np.nan_to_num(val, nan=-1/3.0, posinf=0.0, neginf=0.0)

def I4_int_original(x, rho, n_terms):
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

def I5_int_original(x, rho, n_terms):
    rho_safe = max(rho, 1e-12); px = rho_safe**2 + x**2
    if rho_safe**2 < 1e-15 * x**2 and abs(x) > 1e-6: return np.divide(np.sin(x), 2.0*x, out=np.full_like(x, -0.5), where=x!=0)
    elif px < 1e-12: return -0.5
    rpx = np.sqrt(px); crpx = np.cos(rpx); val = (np.cos(x) - crpx) / rho_safe**2
    return np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

def I6_int_original(x, rho, n_terms):
    px = rho**2 + x**2;
    if px < 1e-12: return 1/3.0
    rpx = np.sqrt(px); srpx = np.sin(rpx); crpx = np.cos(rpx)
    term1 = np.divide(srpx, rpx, out=np.ones_like(srpx), where=rpx!=0)
    val = np.divide(term1 - crpx, px, out=np.full_like(px, 1/3.0), where=px!=0)
    return np.nan_to_num(val, nan=1/3.0, posinf=0.0, neginf=0.0)

# --------------------------------------------------------------------
# Glue into the correlator routine
# --------------------------------------------------------------------
def correlator_factory(I1_int_impl):
    """Return a correlator function that uses the supplied I1_int implementation."""
    def _corr(tau1, tau2, k, n_terms=75, xapr=2.5):
        v1 = 0.65
        xi1 = 0.13
        v2 = 0.65
        xi2 = 0.13

        x1 = k * tau1 * xi1
        x2 = k * tau2 * xi2
        xp = 0.5 * (x1 + x2)
        xm = 0.5 * (x1 - x2)
        rho = k * abs(v1 * tau1 - v2 * tau2)
        rho_safe = rho if rho > 1e-12 else 1e-12

        if abs(x1 - x2) >= xapr:          # use approximation
            I1 = (math.pi * min(x1, x2) / 2.0) * sp.jv(0, rho_safe)
        else:                             # full series
            I1 = I1_int_impl(xm, rho_safe, n_terms) - I1_int_impl(xp, rho_safe, n_terms)
        return I1
    return _corr

corr_slow  = correlator_factory(I6_int_original)
corr_fast  = correlator_factory(I6_int_numba)

# --------------------------------------------------------------------
# Profiling sweep (64×64 τ grid, same as before)
# --------------------------------------------------------------------
tau_full = np.logspace(np.log10(1e-4), np.log10(1e3), 256)
tau      = tau_full[::4]                 # 64 points
k        = 1.0
n_terms_list = [10, 25, 50, 75, 100]
total_calls  = len(tau) ** 2             # 4096 evaluations per n_terms

records = []
# warm‑up JIT so compilation time is not counted
corr_fast(tau[0], tau[1], k, n_terms=10)

for n in n_terms_list:
    # --- slow reference ---
    t0 = time.perf_counter()
    for t1 in tau:
        for t2 in tau:
            corr_slow(t1, t2, k, n_terms=n)
    t_slow = time.perf_counter() - t0

    # --- numba‑accelerated ---
    t0 = time.perf_counter()
    for t1 in tau:
        for t2 in tau:
            corr_fast(t1, t2, k, n_terms=n)
    t_fast = time.perf_counter() - t0

    records.append(
        {
            "n_terms": n,
            "slow_s":  t_slow,
            "fast_s":  t_fast,
            "speed‑up": round(t_slow / t_fast, 2),
            "avg_ms_per_call (fast)": round(1e3 * t_fast / total_calls, 3),
        }
    )

df = pd.DataFrame(records)
print(df)

# --------------------------------------------------------------------
# Numerical accuracy comparison
# --------------------------------------------------------------------
print("\n" + "="*60)
print("NUMERICAL ACCURACY COMPARISON")
print("="*60)

# Test on a subset of tau values for detailed comparison
tau_test = np.logspace(np.log10(1e-4), np.log10(1e3), 16)  # 16 points for detailed analysis
n_terms_test = 75  # Use a representative n_terms value

max_abs_diff = 0.0
max_rel_diff = 0.0
differences = []

print(f"Testing {len(tau_test)}×{len(tau_test)} = {len(tau_test)**2} evaluations with n_terms={n_terms_test}")
print(f"{'tau1':>10} {'tau2':>10} {'slow':>12} {'fast':>12} {'abs_diff':>12} {'rel_diff':>12}")
print("-" * 70)

for i, t1 in enumerate(tau_test):
    for j, t2 in enumerate(tau_test):
        slow_val = corr_slow(t1, t2, k, n_terms=n_terms_test)
        fast_val = corr_fast(t1, t2, k, n_terms=n_terms_test)

        abs_diff = abs(slow_val - fast_val)
        rel_diff = abs_diff / abs(slow_val) if abs(slow_val) > 1e-15 else 0.0

        differences.append({
            'tau1': t1,
            'tau2': t2,
            'slow': slow_val,
            'fast': fast_val,
            'abs_diff': abs_diff,
            'rel_diff': rel_diff
        })

        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)

        # Print a few representative cases
        if (i + j) % 32 == 0:  # Print every 32nd case
            print(f"{t1:10.2e} {t2:10.2e} {slow_val:12.6e} {fast_val:12.6e} {abs_diff:12.6e} {rel_diff:12.6e}")

print("-" * 70)
print(f"Maximum absolute difference: {max_abs_diff:.6e}")
print(f"Maximum relative difference: {max_rel_diff:.6e}")

# Statistics on differences
diff_array = np.array([d['abs_diff'] for d in differences])
rel_diff_array = np.array([d['rel_diff'] for d in differences])

print(f"\nAbsolute difference statistics:")
print(f"  Mean: {np.mean(diff_array):.6e}")
print(f"  Median: {np.median(diff_array):.6e}")
print(f"  Std: {np.std(diff_array):.6e}")
print(f"  95th percentile: {np.percentile(diff_array, 95):.6e}")

print(f"\nRelative difference statistics:")
print(f"  Mean: {np.mean(rel_diff_array):.6e}")
print(f"  Median: {np.median(rel_diff_array):.6e}")
print(f"  Std: {np.std(rel_diff_array):.6e}")
print(f"  95th percentile: {np.percentile(rel_diff_array, 95):.6e}")

# Check if differences are within acceptable tolerance
tolerance_abs = 1e-12
tolerance_rel = 1e-10

n_bad_abs = np.sum(diff_array > tolerance_abs)
n_bad_rel = np.sum(rel_diff_array > tolerance_rel)

print(f"\nTolerance check:")
print(f"  Cases with |diff| > {tolerance_abs:.0e}: {n_bad_abs}/{len(differences)} ({100*n_bad_abs/len(differences):.2f}%)")
print(f"  Cases with rel_diff > {tolerance_rel:.0e}: {n_bad_rel}/{len(differences)} ({100*n_bad_rel/len(differences):.2f}%)")

if n_bad_abs == 0 and n_bad_rel == 0:
    print("✓ All differences within acceptable tolerance!")
else:
    print("⚠ Some differences exceed tolerance - check implementation")

# --------------------------------------------------------------------
# Plot performance and accuracy
# --------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Performance plot
ax1.plot(df["n_terms"], df["slow_s"],  marker="o", label="original SciPy")
ax1.plot(df["n_terms"], df["fast_s"],  marker="o", label="Numba‑stable")
ax1.set_xlabel("n_terms")
ax1.set_ylabel("Runtime (s) on 64×64 τ grid")
ax1.set_title("Performance comparison")
ax1.grid(True)
ax1.legend()

# Accuracy plot
ax2.hist(np.log10(diff_array + 1e-16), bins=30, alpha=0.7, label="Absolute differences")
ax2.set_xlabel("log₁₀(absolute difference)")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of absolute differences")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# Additional plot: relative differences vs tau values
plt.figure(figsize=(10, 6))
diff_df = pd.DataFrame(differences)
scatter = plt.scatter(diff_df['tau1'], diff_df['tau2'], c=np.log10(diff_df['rel_diff'] + 1e-16),
                     cmap='viridis', s=20, alpha=0.7)
plt.colorbar(scatter, label='log₁₀(relative difference)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('tau1')
plt.ylabel('tau2')
plt.title('Relative differences across tau parameter space')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
