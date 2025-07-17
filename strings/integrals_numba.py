import time
import math
import numpy as np
from numba import njit, float64, int32

# --------------------------------------------------------------------
# Numerically stable implementation of integrals with Numba
# --------------------------------------------------------------------

@njit(float64(float64, float64), fastmath=True, cache=True)
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
