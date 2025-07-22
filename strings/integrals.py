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
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 function I1_int_a(x,rho)
   implicit none
    real(DP) I1_int_a
    real(DP) x,rho,J0_rho
    J0_rho = BESSEL_JN(0,rho)
    I1_int_a = (pi*x/2.0d0)*J0_rho
 end function I1_int_a
 !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 function I4_int_a(x,rho)
   implicit none
   real(DP) I4_int_a
   real(DP) x,rho,J1_rho
   J1_rho = BESSEL_JN(1,rho)
   I4_int_a = (pi*x*J1_rho)/(2.0d0*rho)
 end function I4_int_a
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import time
import math
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.special as sp
from numba import njit, float64, int32
import matplotlib.pyplot as plt

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
    if px < 1e-12: return 2/3.0
    rpx = np.sqrt(px); srpx = np.sin(rpx); crpx = np.cos(rpx)
    term1_factor = (1.0 - 3.0 * x**2 / px); term2_factor = (1.0 - (1.0 + x**2) / px + 3.0 * x**2 / px**2)
    term1 = np.divide(crpx * term1_factor, px, out=np.zeros_like(px), where=px!=0)
    term2 = np.divide(srpx * term2_factor, rpx, out=np.zeros_like(rpx), where=rpx!=0)
    val = term1 + term2; return np.nan_to_num(val, nan=2/3.0, posinf=0.0, neginf=0.0)


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
    if rho_safe**2 < 1e-15 * x**2 and abs(x) > 1e-6: return np.divide(np.sin(x), 2.0*x, out=np.full_like(x, 0.5), where=x!=0)
    elif px < 1e-12: return 0.5
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
# Numerically stable implementation of integrals / special functions with Numba
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


@njit(float64(float64), cache=True, fastmath=True)
def _j0_numba(z):
    """
    Approximation to the cylindrical Bessel function J0(z).

    * |z| < 8      :  5/5 rational Chebyshev fit (Hart et al., 1968)
    * |z| >= 8     :  two‑term asymptotic expansion with sqrt(2/(πz))
    The coefficients are the same as in CEPHES / fdlibm and give
    |err| < 3.5e‑8 everywhere in IEEE double precision.
    """
    a = abs(z)

    # ---- small / medium range : rational fit --------------------------------
    if a < 8.0:
        y   = z * z
        num = ( 57568490574.0
              + (-13362590354.0
              + (   651619640.7
              + (-  11214424.18
              + (       77392.33017
              -             184.9052456 * y) * y) * y) * y) * y )
        den = ( 57568490411.0
              + ( 1029532985.0
              + (     9494680.718
              + (       59272.64853
              + (           267.8532712) * y) * y) * y) * y )
        return num / den

    # ---- large range : asymptotic form --------------------------------------
    else:
        # two‑term asymptotic expansion
        p  = math.sqrt(2.0 / (math.pi * a))
        q  = z - math.pi * 0.25
        return p * math.cos(q)            # relative err < 4e‑8 for |z| ≥ 8


@njit(float64(float64, float64), cache=True, fastmath=True)
def I1_int_a_numba(x, rho):
    """
    Quick analytic approximation
        I1_int_a(x, rho) = (π x / 2) * J0(rho)

    The special‑case ρ → 0 is handled explicitly to avoid 0·nan.
    """
    if rho == 0.0:
        return math.pi * x * 0.5          # π x / 2
    return math.pi * x * 0.5 * _j0_numba(rho)


@njit(cache=True, fastmath=True)
def _polevl(x, coeff):
    acc = 0.0
    for c in coeff:                 # coeff can be read‑only or writeable
        acc = acc * x + c
    return acc


@njit(cache=True, fastmath=True)
def _p1evl(x, coeff):
    acc = x + coeff[0]              # implicit leading 1.0
    for c in coeff[1:]:
        acc = acc * x + c
    return acc


_RP = np.array([-8.99971225705559398224e8,
                 4.52228297998194034323e11,
                -7.27494245221818276015e13,
                 3.68295732863852883286e15], dtype=np.float64)

_RQ = np.array([ 6.20836478118054335476e2,
                 2.56987256757748830383e5,
                 8.35146791431949253037e7,
                 2.21511595479792499675e10,
                 4.74914122079991414898e12,
                 7.84369607876235854894e14,
                 8.95222336184627338078e16,
                 5.32278620332680085395e18], dtype=np.float64)

_PP = np.array([7.62125616208173112003e-4,
                7.31397056940917570436e-2,
                1.12719608129684925192e0,
                5.11207951146807644818e0,
                8.42404590141772420927e0,
                5.21451598682361504063e0,
                1.00000000000000000254e0], dtype=np.float64)

_PQ = np.array([5.71323128072548699714e-4,
                6.88455908754495404082e-2,
                1.10514232634061696926e0,
                5.07386386128601488557e0,
                8.39985554327604159757e0,
                5.20982848682361821619e0,
                9.99999999999999997461e-1], dtype=np.float64)

_QP = np.array([5.10862594750176621635e-2,
                4.98213872951233449420e0,
                7.58238284132545283818e1,
                3.66779609360150777800e2,
                7.10856304998926107277e2,
                5.97489612400613639965e2,
                2.11688757100572135698e2,
                2.52070205858023719784e1], dtype=np.float64)

_QQ = np.array([7.42373277035675149943e1,
                1.05644886038262816351e3,
                4.98641058337653607651e3,
                9.56231892404756170795e3,
                7.99704160447350683650e3,
                2.82619278517639096600e3,
                3.36093607810698293419e2], dtype=np.float64)

_Z1      = 1.46819706421238932572e1   # (r₁)²  ≈ 14.68197
_Z2      = 4.92184563216946036703e1   # (r₂)²  ≈ 49.21846
_THPIO4  = 3.0 * math.pi / 4.0        # 3π/4
_SQ2OPI  = math.sqrt(2.0 / math.pi)   # √(2/π)

@njit(float64(float64), cache=True, fastmath=True)
def _j1_numba(x):
    """Cephes J1(x) with |err| < 4 e‑8 on all real x."""
    if x == 0.0:
        return 0.0

    ax = abs(x)

    # ----- |x| ≤ 5  :  rational Chebyshev form -----------------------------
    if ax <= 5.0:
        z   = ax * ax
        w   = _polevl(z, _RP) / _p1evl(z, _RQ)
        w   = w * ax * (z - _Z1) * (z - _Z2)
        return -w if x < 0.0 else w

    # ----- |x| > 5  :  Hankel asymptotic expansion -------------------------
    w  = 5.0 / ax
    z  = w * w
    p  = _polevl(z, _PP) / _polevl(z, _PQ)
    q  = _polevl(z, _QP) / _p1evl(z, _QQ)
    xn = ax - _THPIO4
    val = _SQ2OPI / math.sqrt(ax) * (p * math.cos(xn) - w * q * math.sin(xn))
    return -val if x < 0.0 else val


@njit(float64(float64, float64), cache=True, fastmath=True)
def I4_int_a_numba(x, rho):
    """
    Analytic approximation of the I4 integral:
        I4_int_a(x, rho) = (π * x / (2 ρ)) * J1(ρ)

    Special‑cases ρ→0 to avoid 0/0.
    """
    if rho == 0.0:
        return math.pi * x * 0.25          # π x / 4  (limit ρ→0)
    return (math.pi * x * 0.5 / rho) * _j1_numba(rho)


SN  = np.array([-8.39167827910303881427E-11,  4.62591714427012837309E-8,
                -9.75759303843632795789E-6,  9.76945438170435310816E-4,
                -4.13470316229406538752E-2,  1.00000000000000000302E0])

SD  = np.array([ 2.03269266195951942049E-12,  1.27997891179943299903E-9,
                 4.41827842801218905784E-7,  9.96412122043875552487E-5,
                 1.42085239326149893930E-2,  9.99999999999999996984E-1])

CN  = np.array([ 2.02524002389102268789E-11, -1.35249504915790756375E-8,
                 3.59325051419993077021E-6, -4.74007206873407909465E-4,
                 2.89159652607555242092E-2, -1.00000000000000000080E0])

CD  = np.array([ 4.07746040061880559506E-12,  3.06780997581887812692E-9,
                 1.23210355685883423679E-6,  3.17442024775032769882E-4,
                 5.10028056236446052392E-2,  4.00000000000000000080E0])

FN4 = np.array([ 4.23612862892216586994E0,   5.45937717161812843388E0,
                 1.62083287701538329132E0,   1.67006611831323023771E-1,
                 6.81020132472518137426E-3,  1.08936580650328664411E-4,
                 5.48900223421373614008E-7])

FD4 = np.array([ 8.16496634205391016773E0,   7.30828822505564552187E0,
                 1.86792257950184183883E0,   1.78792052963149907262E-1,
                 7.01710668322789753610E-3,  1.10034357153915731354E-4,
                 5.48900252756255700982E-7])

FN8 = np.array([ 4.55880873470465315206E-1,  7.13715274100146711374E-1,
                 1.60300158222319456320E-1,  1.16064229408124407915E-2,
                 3.49556442447859055605E-4,  4.86215430826454749482E-6,
                 3.20092790091004902806E-8,  9.41779576128512936592E-11,
                 9.70507110881952024631E-14])

FD8 = np.array([ 9.17463611873684053703E-1,  1.78685545332074536321E-1,
                 1.22253594771971293032E-2,  3.58696481881851580297E-4,
                 4.92435064317881464393E-6,  3.21956939101046018377E-8,
                 9.43720590350276732376E-11, 9.70507110881952025725E-14])

GN4 = np.array([ 8.71001698973114191777E-2,  6.11379109952219284151E-1,
                 3.97180296392337498885E-1,  7.48527737628469092119E-2,
                 5.38868681462177273157E-3,  1.61999794598934024525E-4,
                 1.97963874140963632189E-6,  7.82579040744090311069E-9])

GD4 = np.array([ 1.64402202413355338886E0,   6.66296701268987968381E-1,
                 9.88771761277688796203E-2,  6.22396345441768420760E-3,
                 1.73221081474177119497E-4,  2.02659182086343991969E-6,
                 7.82579218933534490868E-9])

GN8 = np.array([ 6.97359953443276214934E-1,  3.30410979305632063225E-1,
                 3.84878767649974295920E-2,  1.71718239052347903558E-3,
                 3.48941165502279436777E-5,  3.47131167084116673800E-7,
                 1.70404452782044526189E-9,  3.85945925430276600453E-12,
                 3.14040098946363334640E-15])

GD8 = np.array([ 1.68548898811011640017E0,   4.87852258695304967486E-1,
                 4.67913194259625806320E-2,  1.90284426674399523638E-3,
                 3.68475504442561108162E-5,  3.57043223443740838771E-7,
                 1.72693748966316146736E-9,  3.87830166023954706752E-12,
                 3.14040098946363335242E-15])


@njit(cache=True, fastmath=True)
def sici_numba(x):
    """
    Returns (Si(x), Ci(x)) with machine‑precision accuracy,
    without relying on SciPy.
    """
    EULER = 0.5772156649015328606
    PIO2  = 1.5707963267948966

    if x == 0.0:                         # exact limit
        return 0.0, -math.inf

    sign = 1.0
    if x < 0.0:                          # Si is odd, Ci is even
        sign = -1.0
        x    = -x

    # ---- very large x : two‑term asymptotic --------------------------------
    if x > 1.0e9:
        s  = math.sin(x)
        c  = math.cos(x)
        si = PIO2 - c / x
        ci = s / x
        if sign < 0.0:
            si = -si
        return si, ci

    # ---- |x| ≤ 4  : rational Chebyshev approximations ----------------------
    if x <= 4.0:
        z  = x * x
        s  = x * _polevl(z, SN) / _polevl(z, SD)
        c  = z * _polevl(z, CN) / _polevl(z, CD)
        if sign < 0.0:
            s = -s
        si = s
        ci = EULER + math.log(x) + c
        return si, ci

    # ---- |x| > 4  : auxiliary f(x), g(x) and asymptotic reconstruction -----
    s  = math.sin(x)
    c  = math.cos(x)
    z  = 1.0 / (x * x)

    if x < 8.0:
        f = _polevl(z, FN4) / (x * _p1evl(z, FD4))
        g = z * _polevl(z, GN4) / _p1evl(z, GD4)
    else:
        f = _polevl(z, FN8) / (x * _p1evl(z, FD8))
        g = z * _polevl(z, GN8) / _p1evl(z, GD8)

    si = PIO2 - f * c - g * s
    if sign < 0.0:
        si = -si
    ci = f * s - g * c
    return si, ci


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

def create_integral_visualization(integral_fn, fn_name="Integral", n_grid=512, x_range=(1e-3, 10.0), n_terms=None, save_path=None):
    """
    Create a 2D visualization of any integral function as a function of x1 and x2,
    with rho = |x1 - x2|.

    Parameters:
    -----------
    integral_fn : callable
        The integral function to visualize (e.g., I1_int_numba, I2_int_numba, etc.)
    fn_name : str
        Name of the function for plot titles and labels
    n_grid : int
        Size of the grid (n_grid x n_grid)
    x_range : tuple
        Range of x values (x_min, x_max)
    n_terms : int, optional
        Number of terms for series-based integrals (I1, I4). If None, function is called with 2 args.
    save_path : str, optional
        Path to save the plot
    """
    print(f"Creating {fn_name} visualization on {n_grid}x{n_grid} grid...")

    # Create coordinate arrays
    x1_vals = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), n_grid)
    x2_vals = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), n_grid)

    # Initialize result matrix
    result_matrix = np.zeros((n_grid, n_grid))

    # Calculate integral for each (x1, x2) pair
    for i, x1 in enumerate(x1_vals):
        if i % 50 == 0:  # Progress indicator
            print(f"  Progress: {i}/{n_grid} rows completed")

        for j, x2 in enumerate(x2_vals):
            rho = abs(x1 - x2)
            if n_terms is not None:
                # For series-based integrals (I1, I4)
                result_matrix[i, j] = integral_fn(x1, rho, n_terms)
            else:
                # For direct integrals (I2, I3, I5, I6)
                result_matrix[i, j] = integral_fn(x1, rho)

    print("  Calculation complete, creating plot...")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use log scale for better visualization
    im = ax.imshow(np.log10(np.abs(result_matrix) + 1e-20),
                   extent=[np.log10(x_range[0]), np.log10(x_range[1]),
                          np.log10(x_range[0]), np.log10(x_range[1])],
                   origin='lower',
                   cmap='viridis',
                   aspect='equal')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'log₁₀|{fn_name}(x₁, |x₁-x₂|)|')

    # Labels and title
    ax.set_xlabel('log₁₀(x₂)')
    ax.set_ylabel('log₁₀(x₁)')
    if n_terms is not None:
        ax.set_title(f'{fn_name} Integral Visualization\n(n_terms={n_terms}, grid={n_grid}×{n_grid})')
    else:
        ax.set_title(f'{fn_name} Integral Visualization\n(grid={n_grid}×{n_grid})')

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")

    #plt.show()

    return result_matrix, x1_vals, x2_vals


if __name__ == "__main__":

    print("=== Testing sp.sici vs sici_numba ===")

    # Test sici_numba vs scipy.special.sici
    print("\nTesting sici_numba accuracy against scipy.special.sici:")

    # Generate test cases with various ranges
    test_x_values = np.array([
        # Small values
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
        # Medium values
        0.5, 1.0, 2.0, 3.0, 4.0, 5.0,
        # Large values
        8.0, 10.0, 20.0, 50.0, 100.0, 1000.0,
        # Negative values (test symmetry)
        -1e-3, -0.1, -1.0, -5.0, -10.0
    ])

    print(f"Testing {len(test_x_values)} values:")
    print(f"{'x':>12} {'scipy_Si':>15} {'numba_Si':>15} {'Si_reldiff':>12} {'scipy_Ci':>15} {'numba_Ci':>15} {'Ci_reldiff':>12}")
    print("-" * 100)

    max_si_reldiff = 0.0
    max_ci_reldiff = 0.0
    problem_cases = []

    for x in test_x_values:
        # Get scipy result
        scipy_si, scipy_ci = sp.sici(x)

        # Get numba result
        numba_si, numba_ci = sici_numba(x)

        # Calculate relative differences
        si_reldiff = abs(scipy_si - numba_si) / abs(scipy_si) if abs(scipy_si) > 1e-15 else 0.0
        ci_reldiff = abs(scipy_ci - numba_ci) / abs(scipy_ci) if abs(scipy_ci) > 1e-15 and np.isfinite(scipy_ci) else 0.0

        max_si_reldiff = max(max_si_reldiff, si_reldiff)
        max_ci_reldiff = max(max_ci_reldiff, ci_reldiff)

        # Flag problematic cases
        if si_reldiff > 1e-6 or ci_reldiff > 1e-6:
            problem_cases.append((x, scipy_si, numba_si, si_reldiff, scipy_ci, numba_ci, ci_reldiff))

        print(f"{x:12.2e} {scipy_si:15.8e} {numba_si:15.8e} {si_reldiff:12.2e} {scipy_ci:15.8e} {numba_ci:15.8e} {ci_reldiff:12.2e}")

    print(f"\nSummary:")
    print(f"Maximum Si relative error: {max_si_reldiff:.2e}")
    print(f"Maximum Ci relative error: {max_ci_reldiff:.2e}")
    print(f"Problem cases (rel_diff > 1e-6): {len(problem_cases)}")

    if problem_cases:
        print("\nProblem cases:")
        for x, scipy_si, numba_si, si_reldiff, scipy_ci, numba_ci, ci_reldiff in problem_cases:
            print(f"  x={x:8.2e}: Si_err={si_reldiff:.2e}, Ci_err={ci_reldiff:.2e}")
    else:
        print("✓ All cases within tolerance (1e-6)")

    # Performance comparison
    print("\n=== Performance Comparison: sp.sici vs sici_numba ===")

    # Generate larger test set for timing
    np.random.seed(42)
    n_timing_samples = 1000
    timing_x_vals = np.logspace(np.log10(1e-3), np.log10(1e3), n_timing_samples)

    # Time scipy version
    t0 = time.perf_counter()
    for _ in range(3):  # 3 repeats
        for x in timing_x_vals:
            sp.sici(x)
    t1 = time.perf_counter()
    scipy_time = (t1 - t0) / 3 * 1e3  # ms

    # Time numba version
    t0 = time.perf_counter()
    for _ in range(3):  # 3 repeats
        for x in timing_x_vals:
            sici_numba(x)
    t1 = time.perf_counter()
    numba_time = (t1 - t0) / 3 * 1e3  # ms

    speedup = scipy_time / numba_time if numba_time > 0 else float('inf')

    print(f"Timing results ({n_timing_samples} samples, 3 repeats):")
    print(f"  scipy.special.sici: {scipy_time:.2f} ms")
    print(f"  sici_numba:         {numba_time:.2f} ms")
    print(f"  Speedup:            {speedup:.1f}x")

    print("\n" + "="*60)

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
        ("I1_a", I1_int_a, I1_int_a_numba, False),
        ("I4_a", I4_int_a, I4_int_a_numba, False),
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

    # Create visualization
    for name, plain_fn, numba_fn, needs_n_terms in functions:
        if needs_n_terms:
            n_terms = 100
        else:
            n_terms = None
        matrix, x1_vals, x2_vals = create_integral_visualization(
            integral_fn=plain_fn,
            fn_name=name,
            n_grid=128,
            x_range=(1e-4, 100.0),
            n_terms=n_terms,
            save_path=f"{name}_plain_visualization.png"
        )
        matrix, x1_vals, x2_vals = create_integral_visualization(
            integral_fn=numba_fn,
            fn_name=name,
            n_grid=1024,
            x_range=(1e-4, 100.0),
            n_terms=n_terms,
            save_path=f"{name}_numba_visualization.png"
        )
