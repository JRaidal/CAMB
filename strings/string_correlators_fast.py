"""
Fast generation of UETC eigenmode tables for the Unconnected Segment Model
==========================================================================
This is a fast version of the string correlator code.






Author : Juhan Raidal and Adam Moss
Date   : 2025‑07‑17
"""

import os, time, warnings, math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.linalg
from numba import njit, prange
from tqdm import tqdm                                   # Progress bar

from integrals import I1_int_numba, I2_int_numba, I3_int_numba, I4_int_numba, I5_int_numba, I6_int_numba, I1_int_a_numba, I4_int_a_numba, sici_numba

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------- #
# Cosmological parameters
# --------------------------------------------------------------------- #
Omega_Lambda = 0.685
Omega_R      = 9.2434441243835e-05
Omega_M      = 1.0 - Omega_Lambda - Omega_R
Omega_K      = 0.0
OMEGAS       = {'R': Omega_R, 'M': Omega_M, 'L': Omega_Lambda, 'K': Omega_K}

Mpc_in_m     = 3.085677577849e22
c_m_per_s    = 299792458.0
hH           = 0.673
H0           = (100 * 1000 * hH) / c_m_per_s           # Mpc⁻¹

# --------------------------------------------------------------------- #
# String parameters and run‑time options
# --------------------------------------------------------------------- #

mu = 1.0
alpha = 1.9
L = 0.95

# ––– table ranges –––
k_min, k_max, nk               = 1e-6, 10.0, 100
ktau_min, ktau_max, nktau      = 1e-4, 1e3, 256
nmodes                          = nktau                 # include all
weighting_gamma                 = 0.25                  # γ in papers

# integration knobs (same as before)
xmin, xmax, xapr, etcmin        = 0.15, 20.0, 2.5, 1e-3
min_terms, scale_terms          = 10, 15.0
MAX_N_TERMS, scaling_option     = 75, 2

# --------------------------------------------------------------------- #
# Solve VOS, tabulate ξ(τ) and v(τ)
# --------------------------------------------------------------------- #
def _solve_cosmology_and_vos(ntau=5000,
                             tau_min_s=1e-4,
                             tau_max_s=8e17,
                             cr=0.23):
    """
    Returns arrays:
        tau_grid  – logarithmic grid in conformal‑time [Mpc]
        a_arr, Hc_arr – scale factor and conformal H on same grid
        xi_arr, v_arr – VOS solution on same grid
    """

    s_per_mpc = Mpc_in_m / c_m_per_s
    tau_min   = tau_min_s / s_per_mpc
    tau_max   = tau_max_s / s_per_mpc
    tau_grid  = np.logspace(np.log10(tau_min), np.log10(tau_max), ntau)

    # --- solve a(τ) ---
    def friedmann_rhs(tau, a):
        r = OMEGAS['R'] / a**2 + OMEGAS['M'] / a + OMEGAS['K'] + OMEGAS['L'] * a**2
        return a * H0 * math.sqrt(r)

    a0 = math.sqrt(OMEGAS['R']) * H0 * tau_grid[0]
    sol = solve_ivp(friedmann_rhs,
                    (tau_grid[0], tau_grid[-1]),
                    (a0,),
                    t_eval=tau_grid,
                    atol=1e-12, rtol=1e-10, method='DOP853')
    if not sol.success:
        raise RuntimeError("Friedmann solver failed: " + sol.message)
    a_arr = sol.y[0]
    Hc_arr = H0 * np.sqrt(OMEGAS['R']/a_arr**2 + OMEGAS['M']/a_arr +
                          OMEGAS['K'] + OMEGAS['L']*a_arr**2)

    # --- VOS equations ---
    def k_tilde(v):
        v6 = v**6
        return (2.0*np.sqrt(2)/np.pi)*(1.0 - 8.0*v6)/(1.0 + 8.0*v6)

    def vos_rhs(tau, y):
        xi, v = y
        Hc     = np.interp(tau, tau_grid, Hc_arr)
        dxi    = 1/tau * (v**2 * xi * tau * Hc - xi + cr*v/2)
        dv     = (1 - v**2)*(k_tilde(v)/(xi*tau) - 2*v*Hc)
        return (dxi, dv)

    sol_vos = solve_ivp(vos_rhs,
                        (tau_grid[0], tau_grid[-1]),
                        (0.13, 0.65),
                        t_eval=tau_grid,
                        atol=1e-12, rtol=1e-10, method='LSODA')
    if not sol_vos.success:
        raise RuntimeError("VOS solver failed: " + sol_vos.message)

    xi_arr, v_arr = sol_vos.y
    return tau_grid, a_arr, Hc_arr, xi_arr, v_arr

# global pre‑tabulation (takes O(1 s))
_TAU, _A, _HC, _XI, _V = _solve_cosmology_and_vos()

# Provide fast O(1) look‑up helpers (Numba needs pure arrays)
@njit(inline='always')
def _lookup(arr, tau):
    # assumes log‑spaced _TAU; a cheap manual log‑interp
    idx = np.searchsorted(_TAU, tau) - 1
    if idx < 0:
        return arr[0]
    if idx >= _TAU.size - 1:
        return arr[-1]
    w = (tau - _TAU[idx]) / (_TAU[idx+1] - _TAU[idx])
    return (1 - w)*arr[idx] + w*arr[idx+1]

# --------------------------------------------------------------------- #
# 6. Core kernel: single‑pair correlator
# --------------------------------------------------------------------- #
@njit(fastmath=True, cache=True)
def _scaling_factor(t1, t2, xi1, xi2, L):
    if scaling_option == 1:
        return 1.0
    denom = max(xi1*t1, xi2*t2, 1e-30)
    return 1.0 / (denom**3)

@njit(fastmath=True, cache=True)
def _get_correlator_pair(tau1, tau2, k,
                         mu1, alpha1, mu2, alpha2, L):
    v1  = _lookup(_V,  tau1)
    xi1 = _lookup(_XI, tau1)
    v2  = _lookup(_V,  tau2)
    xi2 = _lookup(_XI, tau2)

    # Find x and rho as in paper
    x1=k*tau1*xi1; x2=k*tau2*xi2; xp=(x1+x2)/2.0; xm=(x1-x2)/2.0
    rho=k*abs(v1*tau1-v2*tau2); rho_safe=max(rho, 1e-12)

    # Common factors for all correlators
    norm_denom_sq = (1.0 - v1**2)*(1.0 - v2**2)
    norm_denom = np.sqrt(norm_denom_sq)
    sf = _scaling_factor(tau1, tau2, xi1, xi2, L)
    common_factor_base = sf / (k**2*norm_denom)
    common_factor = mu1*mu2*common_factor_base

    uetc_val = [0.0, 0.0, 0.0, 0.0, 0.0]

    # --- Regime 1: Small x---
    if x1 <= xmin and x2 <= xmin:
        if alpha1==0 or alpha2==0: return uetc_val
        term00=-(alpha1*alpha2*mu1*mu2*(-6.0 + rho**2)*x1*x2)/(6.0*k**2*norm_denom)
        term_s_num=(rho**2*(-10.0+(10.0-11.0*alpha2**2)*v2**2+v1**2*(10.0-11.0*alpha1**2+(-10.0+11.0*alpha1**2+11.0*(1.0-2.0*alpha1**2)*alpha2**2)*v2**2))+42.0*(2.0+(-2.0+alpha2**2)*v2**2+v1**2*(-2.0+alpha1**2+(2.0-alpha1**2+(-1.0+2.0*alpha1**2)*alpha2**2)*v2**2)))*x1*x2
        termS=(mu1*mu2*term_s_num)/(420.0*alpha1*alpha2*k**2*norm_denom) if alpha1*alpha2 != 0 else 0.0
        term_v_num=(8.0+4.0*(-2.0+alpha1**2)*v1**2+(-8.0-4.0*(-2.0+alpha1**2)*v1**2+alpha2**2*(4.0+(-4.0+7.0*alpha1**2)*v1**2))*v2**2)*x1*x2
        termV=(mu1*mu2*2.3*term_v_num)/(256.0*alpha1*alpha2*k**2*norm_denom) if alpha1*alpha2 != 0 else 0.0
        term_t_num=-(mu1*mu2*(-28.0+6.0*rho**2+28.0*v1**2-14.0*alpha1**2*v1**2-6.0*rho**2*v1**2+alpha1**2*rho**2*v1**2+28.0*v2**2-14.0*alpha2**2*v2**2-6.0*rho**2*v2**2+alpha2**2*rho**2*v2**2-28.0*v1**2*v2**2+14.0*alpha1**2*v1**2*v2**2+14.0*alpha2**2*v1**2*v2**2-28.0*alpha1**2*alpha2**2*v1**2*v2**2+6.0*rho**2*v1**2*v2**2-alpha1**2*rho**2*v1**2*v2**2-alpha2**2*rho**2*v1**2*v2**2+2.0*alpha1**2*alpha2**2*rho**2*v1**2*v2**2)*x1*x2)
        termT=term_t_num/(420.0*alpha1*alpha2*k**2*norm_denom) if alpha1*alpha2 != 0 else 0.0
        term_cross_num=-(mu1*mu2*rho**2*(alpha1**2*(1.0+(-1.0+2.0*alpha2**2)*v2**2)+alpha2**2*(1.0+(-1.0+2.0*alpha1**2)*v1**2))*x1*x2)
        term00S=term_cross_num/(60.0*alpha1*alpha2*k**2*norm_denom) if alpha1*alpha2 != 0 else 0.0
        return [term00*sf, termS*sf, termV*sf, termT*sf, term00S*sf]

    # --- Regime 2: ETC ---
    if abs(x1 - x2) <= etcmin:
        x = xp; alpha=(alpha1+alpha2)/2.0; v=(v1+v2)/2.0; mu=(mu1+mu2)/2.0
        if alpha==0: return uetc_val
        norm_denom_etc_sq = (1.0 - v**2);
        if norm_denom_etc_sq <= 1e-16 : return uetc_val
        norm_denom_etc = math.sqrt(norm_denom_etc_sq)

        mu = (mu1 + mu2) / 2.0; tau=(tau1+tau2)/2; xi=(xi1+xi2)/2
        base_etc_factor=common_factor

        six, _ = sici_numba(x)

        cosx=math.cos(x); sinx=math.sin(x)
        if abs(x) < 1e-12: sinx_over_x = 1.0
        else: sinx_over_x = sinx / x

        term00_num=2.0*alpha**2*(-1.0+cosx+x*six);
        uetc_val[0]=term00_num * base_etc_factor
        if x==0 or alpha==0: termS=0.0
        else:
            term1_s=(8*(-18+x**2)+8*(-2+alpha**2)*v**2*(-18+x**2)+v**4*(8*(-18+x**2)-8*alpha**2*(-18+x**2)+alpha**4*(-54+11*x**2)))*cosx
            term2_s_num=(-32*(1+(-2+alpha**2)*v**2+(1-alpha**2+alpha**4)*v**4)*x**3+3*(-8*(-6+x**2)-8*(-2+alpha**2)*v**2*(-6+x**2)+v**4*(-8*(-6+x**2)+8*alpha**2*(-6+x**2)+alpha**4*(18+x**2)))*sinx)
            if abs(x) < 1e-12:
                term2_s = 0.0
            else: term2_s=term2_s_num / x
            term3_s=(8+8*(-2+alpha**2)*v**2+(8-8*alpha**2+11*alpha**4)*v**4)*x**3*six
            termS_num=term1_s+term2_s+term3_s
            termS=termS_num/(16.0*alpha**2*x**2) if (x!=0 and alpha!=0) else 0.0
        uetc_val[1]=termS * base_etc_factor
        if x==0 or alpha==0: termV=0.0
        else:
             if abs(x) < 1e-12:
                 tV1_sub = 0.0
             else: tV1_sub = (x**3 + 3.0*x*cosx - 3.0*sinx) / (3.0 * x**3)
             term1_v=(2.0*(8.0+8.0*(-2.0+alpha**2)*v**2+(8.0-8.0*alpha**2+3*alpha**4)*v**4))*tV1_sub
             term2_v=alpha**4*v**4*(-2.0+cosx+sinx_over_x+x*six)
             termV_num=term1_v+term2_v; termV=termV_num/(8.0*alpha**2) if alpha!=0 else 0.0
        uetc_val[2]=termV * base_etc_factor
        if x==0 or alpha==0: termT=0.0
        else:
            term1_t=3*(8+8*(-2+alpha**2)*v**2+(8-8*alpha**2+3*alpha**4)*v**4)*(-2+x**2)*cosx
            term2_t_num=(64*(-1+v**2)*(1+(-1+alpha**2)*v**2)*x**3-3*(-8*(2+x**2)-8*(-2+alpha**2)*v**2*(2+x**2)+v**4*(-8*(2+x**2)+8*alpha**2*(2+x**2)+alpha**4*(-6+5*x**2)))*sinx)
            if abs(x) < 1e-12:
                term2_t = 0.0
            else: term2_t = term2_t_num / x
            term3_t=3*(8+8*(-2+alpha**2)*v**2+(8-8*alpha**2+3*alpha**4)*v**4)*x**3*six
            termT_num=term1_t+term2_t+term3_t
            termT=termT_num/(96.0*alpha**2*x**2) if (x!=0 and alpha!=0) else 0.0
        uetc_val[3]=termT * base_etc_factor
        if x==0:
            term00S_num=0.0
        else: term00S_num=(mu**2*(2+(-2+alpha**2)*v**2)*(-4+cosx+3*sinx_over_x+x*six))
        term00S=term00S_num/(2.*k**2*norm_denom_etc) if norm_denom_etc>1e-12 else 0.0
        uetc_val[4]=term00S
        return [val * sf for val in uetc_val]

    # --- Regime 3: General Case ---
    n_terms_raw = max(min_terms, int(scale_terms * xp))
    n_terms = min(n_terms_raw, MAX_N_TERMS)
    use_approx = abs(x1 - x2) >= xapr
    small_rho = rho < 1e-2
    if use_approx: I1=I1_int_a_numba(min(x1,x2),rho_safe); I4=I4_int_a_numba(min(x1,x2),rho_safe)
    else: I1=I1_int_numba(xm,rho_safe,n_terms)-I1_int_numba(xp,rho_safe,n_terms); I4=I4_int_numba(xm,rho_safe,n_terms)-I4_int_numba(xp,rho_safe,n_terms);
    if not use_approx and small_rho: I4 = I1 / 2.0
    I2=I2_int_numba(xm,rho_safe)-I2_int_numba(xp,rho_safe); I3=I3_int_numba(xm,rho_safe)-I3_int_numba(xp,rho_safe)
    if not use_approx and small_rho: I5=I2/2.0; I6=I3/2.0
    else: I5=I5_int_numba(xm,rho_safe)-I5_int_numba(xp,rho_safe); I6=I6_int_numba(xm,rho_safe)-I6_int_numba(xp,rho_safe)
    #integrals = [I1, I2, I3, I4, I5, I6];
    #if not all(np.isfinite(i) for i in integrals): return np.zeros(5)
    safe_a1a2rho2=max(2.*alpha1*alpha2*rho_safe**2,1e-30); safe_a1a2=max(2.*alpha1*alpha2,1e-30)

    sum00=2*alpha1*alpha2*I1; uetc_val[0]=sum00*common_factor
    c_term1=(-27*(alpha1*alpha2*v1*v2)**2+rho_safe**2*(1+(-1+2*alpha1**2)*v1**2)*(1+(-1+2*alpha2**2)*v2**2))/safe_a1a2rho2
    c_term2=(-3*(-9*(alpha1*alpha2*v1*v2)**2+rho_safe**2*(-1+v2**2+v1**2*(1+(-1+(alpha1*alpha2)**2)*v2**2))))/safe_a1a2rho2
    c_term3=(-9*(1+(-1+alpha1**2)*v1**2)*(1+(-1+alpha2**2)*v2**2))/safe_a1a2
    c_term4=(-3*(-(alpha2**2*rho_safe**2*(-1+v1**2)*v2**2)+alpha1**2*v1**2*(-18*alpha2**2*v2**2+rho_safe**2*(1+(-1+4*alpha2**2)*v2**2))))/safe_a1a2rho2
    c_term5=(3*(-(alpha2**2*rho_safe**2*(-1+v1**2)*v2**2)+alpha1**2*v1**2*(-18*alpha2**2*v2**2+rho_safe**2*(1+(-1+4*alpha2**2)*v2**2))))/safe_a1a2rho2
    c_term6=(9*(-(alpha2**2*(-1+v1**2)*v2**2)+alpha1**2*v1**2*(1+(-1+2*alpha2**2)*v2**2)))/safe_a1a2
    sumS=c_term1*I1+c_term2*I2+c_term3*I3+c_term4*I4+c_term5*I5+c_term6*I6; uetc_val[1]=sumS*common_factor
    safe_rho2_local=max(rho_safe**2,1e-30); safe_a1a2_local=max(alpha1*alpha2,1e-30)
    c_term1=(3*alpha1*alpha2*v1**2*v2**2)/safe_rho2_local; c_term2=(-3*alpha1*alpha2*v1**2*v2**2)/safe_rho2_local
    c_term3=((1+(-1+alpha1**2)*v1**2)*(1+(-1+alpha2**2)*v2**2))/safe_a1a2_local
    c_term4=(alpha1*alpha2*(-6+rho_safe**2)*v1**2*v2**2)/safe_rho2_local; c_term5=-((alpha1*alpha2*(-6+rho_safe**2)*v1**2*v2**2)/safe_rho2_local)
    c_term6=(alpha2**2*(-1+v1**2)*v2**2-alpha1**2*v1**2*(1+(-1+2*alpha2**2)*v2**2))/safe_a1a2_local
    sumV=c_term1*I1+c_term2*I2+c_term3*I3+c_term4*I4+c_term5*I5+c_term6*I6; uetc_val[2]=sumV*common_factor
    safe_4a1a2rho2=max(4.0*alpha1*alpha2*rho_safe**2,1e-30); safe_4a1a2=max(4.0*alpha1*alpha2,1e-30)
    c_term1=(-3.0*(alpha1*alpha2*v1*v2)**2+rho_safe**2*(-1.0+v1**2)*(-1.0+v2**2))/safe_4a1a2rho2
    c_term2=(3.0*(alpha1*alpha2*v1*v2)**2+rho_safe**2*(-1.0+v2**2+v1**2*(1.0+(-1.0+(alpha1*alpha2)**2)*v2**2)))/safe_4a1a2rho2
    c_term3=-((1.0+(-1.0+alpha1**2)*v1**2)*(1.0+(-1.0+alpha2**2)*v2**2))/safe_4a1a2
    c_term4=(-(alpha2**2*rho_safe**2*(-1.0+v1**2)*v2**2)+alpha1**2*v1**2*(6.0*alpha2**2*v2**2-rho_safe**2*(-1.0+v2**2)))/safe_4a1a2rho2
    c_term5=(alpha2**2*rho_safe**2*(-1.0+v1**2)*v2**2+alpha1**2*v1**2*(-6.0*alpha2**2*v2**2+rho_safe**2*(-1.0+v2**2)))/safe_4a1a2rho2
    c_term6=(-(alpha2**2*(-1.0+v1**2)*v2**2)+alpha1**2*v1**2*(1.0+(-1.0+2.0*alpha2**2)*v2**2))/safe_4a1a2
    sumT=c_term1*I1+c_term2*I2+c_term3*I3+c_term4*I4+c_term5*I5+c_term6*I6; uetc_val[3]=sumT*common_factor
    c_term1=(-(alpha2**2*(-1+v1**2))+alpha1**2*(1-v2**2+2*alpha2**2*(v1**2+v2**2)))/safe_a1a2
    c_term2=(-3*(-(alpha2**2*(-1+v1**2))+alpha1**2*(1-v2**2+alpha2**2*(v1**2+v2**2))))/safe_a1a2
    c_term3=0.0; c_term4=(-3*alpha1*alpha2*(v1**2+v2**2))/2.0; c_term5=(3*alpha1*alpha2*(v1**2+v2**2))/2.0; c_term6=0.0
    sumC=c_term1*I1+c_term2*I2+c_term3*I3+c_term4*I4+c_term5*I5+c_term6*I6; uetc_val[4]=sumC*common_factor

    return uetc_val

# --------------------------------------------------------------------- #
# Kernel that fills *all five* UETC matrices for a single k
# --------------------------------------------------------------------- #
@njit(parallel=True, cache=True, fastmath=True)
def build_uetc_mats(tau_vec, k, mu, alpha, L):
    n = tau_vec.size
    m00  = np.zeros((n, n), dtype=np.float64)
    mS   = np.zeros_like(m00)
    mV   = np.zeros_like(m00)
    mT   = np.zeros_like(m00)
    m00S = np.zeros_like(m00)
    for i in prange(n):
        t1 = tau_vec[i]
        for j in range(i, n):
            t2 = tau_vec[j]
            uetc_val = _get_correlator_pair(t1, t2, k, mu, alpha, mu, alpha, L)
            m00[i, j]  = uetc_val[0]
            mS[i, j]   = uetc_val[1]
            mV[i, j]   = uetc_val[2]
            mT[i, j]   = uetc_val[3]
            m00S[i, j] = uetc_val[4]
            if i != j:
                # symmetry
                m00[j, i]  = uetc_val[0]
                mS[j,  i]  = uetc_val[1]
                mV[j,  i]  = uetc_val[2]
                mT[j,  i]  = uetc_val[3]
                m00S[j, i] = uetc_val[4]
    return m00, mS, mV, mT, m00S

# --------------------------------------------------------------------- #
# Diagonalisation (unchanged, numpy/SciPy – fast)
# --------------------------------------------------------------------- #
def _diagonalise(mats, tau_vec, k, gamma, nmodes):
    m00, mS, mV, mT, m00S = mats
    n  = tau_vec.size
    if nmodes > n:
        nmodes = n

    tau_i, tau_j = np.meshgrid(tau_vec, tau_vec, indexing='ij')
    W = (k*k * tau_i*tau_j)**gamma * np.sqrt(tau_i*tau_j)

    # Vector / tensor separate
    Vw = W * mV
    Tw = W * mT
    evalV, evecV = scipy.linalg.eigh(Vw)
    evalT, evecT = scipy.linalg.eigh(Tw)

    # Scalar 2×2 block
    Sbig = np.zeros((2*n, 2*n))
    Sbig[:n, :n]      = W * m00
    Sbig[n:, n:]      = W * mS
    Sbig[:n, n:]      = W * m00S
    Sbig[n:, :n]      = Sbig[:n, n:].T
    evalS, evecSbig = scipy.linalg.eigh(Sbig)

    # collect
    slc  = slice(2*n-nmodes, 2*n)              # top nmodes
    svals = evalS[slc][::-1]
    svecs = evecSbig[:, slc][:, ::-1]
    out = {
        'eval_S':  svals,
        'eval_V':  evalV[-nmodes:][::-1],
        'eval_T':  evalT[-nmodes:][::-1],
        'eval_00': svals[:nmodes],             # keep for completeness
        'evec_00': svecs[:n, :].T,
        'evec_S':  svecs[n:, :].T,
        'evec_V':  evecV[:, -nmodes:][:, ::-1].T,
        'evec_T':  evecT[:, -nmodes:][:, ::-1].T,
    }
    return out

# --------------------------------------------------------------------- #
# Public driver: eigen decomposition for one k
# --------------------------------------------------------------------- #
def eig_for_single_k(k_val, tau_grid, gamma=weighting_gamma, nmodes=nmodes):
    mats = build_uetc_mats(tau_grid, k_val, mu, alpha, L)
    diag = _diagonalise(mats, tau_grid, k_val, gamma, nmodes)
    return diag

# --------------------------------------------------------------------- #
# Main table build loop – threaded, progress bar
# --------------------------------------------------------------------- #
def build_table(filename="correlator_table_fast.npz"):
    k_grid   = np.logspace(np.log10(k_min), np.log10(k_max), nk)
    ktau_grid= np.logspace(np.log10(ktau_min), np.log10(ktau_max), nktau)

    ntypes   = 4
    efuncs   = np.zeros((nk, ntypes, nmodes, nktau))
    evals_S  = np.zeros((nk, nmodes))
    evals_00 = np.zeros((nk, nmodes))
    evals_V  = np.zeros_like(evals_S)
    evals_T  = np.zeros_like(evals_S)

    for ik, k in enumerate(tqdm(k_grid, desc="k‑loop", ncols=80)):
        tau_vec = ktau_grid / k
        diag = eig_for_single_k(k, tau_vec)

        efuncs[ik, 0] = diag['evec_00']
        efuncs[ik, 1] = diag['evec_S']
        efuncs[ik, 2] = diag['evec_V']
        efuncs[ik, 3] = diag['evec_T']

        evals_S [ik] = diag['eval_S']
        evals_00[ik] = diag['eval_00']
        evals_V [ik] = diag['eval_V']
        evals_T [ik] = diag['eval_T']

    np.savez(filename,
             k_grid=k_grid,
             ktau_grid=ktau_grid,
             eigenfunctions=efuncs,
             eigenvalues_S=evals_S,
             eigenvalues_00=evals_00,
             eigenvalues_V=evals_V,
             eigenvalues_T=evals_T,
             string_mu=mu, string_alpha=alpha, string_L=L,
             nmodes=nmodes, weighting_gamma=weighting_gamma)
    print(f"Saved {filename}")

# --------------------------------------------------------------------- #
# Plotting function to visualize UETC results
# --------------------------------------------------------------------- #
def plot_uetc(filename="correlator_table_fast.npz", k_indices=None, mode_indices=None,
              plot_filename=None):
    """
    Plot UETC eigenfunctions and eigenvalues to verify they look reasonable.

    Parameters:
    -----------
    filename : str
        Path to the saved .npz file
    k_indices : list or None
        Which k values to plot (indices). If None, plots a few representative ones
    mode_indices : list or None
        Which modes to plot (indices). If None, plots first few modes
    plot_filename : str or None
        Filename for saved plot. If None, auto-generates based on data filename
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - cannot plot")
        return

    # Load the data
    try:
        data = np.load(filename)
    except FileNotFoundError:
        print(f"File {filename} not found. Run build_table() first.")
        return

    k_grid = data['k_grid']
    ktau_grid = data['ktau_grid']
    efuncs = data['eigenfunctions']  # shape: (nk, 4, nmodes, nktau)
    evals_S = data['eigenvalues_S']
    evals_00 = data['eigenvalues_00']
    evals_V = data['eigenvalues_V']
    evals_T = data['eigenvalues_T']

    nk, ntypes, nmodes_saved, nktau = efuncs.shape
    type_names = ['Scalar (00)', 'Scalar (S)', 'Vector (V)', 'Tensor (T)']

    # Default selections
    if k_indices is None:
        k_indices = [0, nk//4, nk//2, 3*nk//4, nk-1]  # spread across k range
    if mode_indices is None:
        mode_indices = [0, 1, 2, 3, 4] if nmodes_saved >= 5 else list(range(nmodes_saved))

    # Limit to available data
    k_indices = [i for i in k_indices if 0 <= i < nk]
    mode_indices = [i for i in mode_indices if 0 <= i < nmodes_saved]

    print(f"Plotting k indices: {k_indices}")
    print(f"Plotting mode indices: {mode_indices}")
    k_values = [k_grid[i] for i in k_indices]
    print(f"k values: {[f'{k:.3e}' for k in k_values]}")

    # Create figure with subplots - rearranged layout
    fig = plt.figure(figsize=(20, 16))

    # Get sample k for matrix visualization
    sample_k_idx = k_indices[len(k_indices)//2]  # middle k value
    sample_k = k_grid[sample_k_idx]
    tau_vec = ktau_grid / sample_k

    print(f"\nGenerating 2D UETC matrices for k = {sample_k:.3e} Mpc⁻¹")

    # Build the correlation matrices for this k
    mats = build_uetc_mats(tau_vec, sample_k, mu, alpha, L)
    m00, mS, mV, mT, m00S = mats

    # ROW 1: 4 UETC correlation matrices (plots 1-4)
    matrix_data = [m00, mS, mV, mT]
    matrix_names = ['C₀₀(τ₁,τ₂)', 'Cₛ(τ₁,τ₂)', 'Cᵥ(τ₁,τ₂)', 'Cₜ(τ₁,τ₂)']

    for i, (mat, name) in enumerate(zip(matrix_data, matrix_names)):
        ax = plt.subplot(3, 4, 1 + i)

        # Use symmetric colormap and handle potential zeros
        vmax = np.max(np.abs(mat))
        if vmax > 0:
            im = plt.imshow(mat, extent=[tau_vec[0], tau_vec[-1], tau_vec[0], tau_vec[-1]],
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower',
                           aspect='auto')
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            plt.imshow(np.zeros_like(mat), extent=[tau_vec[0], tau_vec[-1], tau_vec[0], tau_vec[-1]],
                      cmap='gray', origin='lower', aspect='auto')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('τ₁ [Mpc]')
        plt.ylabel('τ₂ [Mpc]')
        plt.title(f'{name}\nk = {sample_k:.2e} Mpc⁻¹')
        plt.grid(True, alpha=0.3)

    # ROW 2: Eigenfunctions for different types (plots 5-8)
    for itype in range(4):
        ax = plt.subplot(3, 4, 5 + itype)

        for ik in k_indices[:3]:  # plot first 3 k values
            tau_vec_plot = ktau_grid / k_grid[ik]

            for imode in mode_indices[:2]:  # plot first 2 modes
                efunc = efuncs[ik, itype, imode, :]
                alpha_val = 0.8 if imode == 0 else 0.5
                linestyle = '-' if imode == 0 else '--'
                plt.semilogx(tau_vec_plot, efunc, linestyle, alpha=alpha_val,
                           label=f'k={k_grid[ik]:.2e}, mode {imode}' if ik < 3 and imode < 2 else '')

        plt.xlabel('τ [Mpc]')
        plt.ylabel('Eigenfunction')
        plt.title(f'{type_names[itype]} Eigenfunctions')
        if itype == 0:  # only show legend for first plot to avoid clutter
            plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

    # ROW 3: Analysis plots (plots 9-12)

    # Plot 9: Eigenvalues vs k
    ax = plt.subplot(3, 4, 9)
    for i in mode_indices[:3]:  # plot first 3 modes
        plt.loglog(k_grid, np.abs(evals_S[:, i]), 'b-', alpha=0.7, label=f'S mode {i}' if i < 3 else '')
        plt.loglog(k_grid, np.abs(evals_V[:, i]), 'r-', alpha=0.7, label=f'V mode {i}' if i < 3 else '')
        plt.loglog(k_grid, np.abs(evals_T[:, i]), 'g-', alpha=0.7, label=f'T mode {i}' if i < 3 else '')
    plt.xlabel('k [Mpc⁻¹]')
    plt.ylabel('|Eigenvalue|')
    plt.title('Eigenvalues vs k')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 10: Eigenvalue distribution for one k
    ax = plt.subplot(3, 4, 10)
    k_idx = k_indices[len(k_indices)//2]  # middle k
    modes = np.arange(len(mode_indices))
    plt.semilogy(modes, np.abs(evals_S[k_idx, mode_indices]), 'bo-', label='Scalar')
    plt.semilogy(modes, np.abs(evals_V[k_idx, mode_indices]), 'ro-', label='Vector')
    plt.semilogy(modes, np.abs(evals_T[k_idx, mode_indices]), 'go-', label='Tensor')
    plt.xlabel('Mode index')
    plt.ylabel('|Eigenvalue|')
    plt.title(f'Eigenvalue spectrum at k={k_grid[k_idx]:.3e}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 11: Cross-correlation matrix C₀₀,ₛ
    ax = plt.subplot(3, 4, 11)
    vmax = np.max(np.abs(m00S))
    if vmax > 0:
        im = plt.imshow(m00S, extent=[tau_vec[0], tau_vec[-1], tau_vec[0], tau_vec[-1]],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower',
                       aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
    else:
        plt.imshow(np.zeros_like(m00S), extent=[tau_vec[0], tau_vec[-1], tau_vec[0], tau_vec[-1]],
                  cmap='gray', origin='lower', aspect='auto')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('τ₁ [Mpc]')
    plt.ylabel('τ₂ [Mpc]')
    plt.title(f'C₀₀,ₛ(τ₁,τ₂)\nk = {sample_k:.2e} Mpc⁻¹')
    plt.grid(True, alpha=0.3)

    # Plot 12: Matrix statistics
    ax = plt.subplot(3, 4, 12)
    matrix_stats = []
    for mat, name in zip([m00, mS, mV, mT, m00S], ['C₀₀', 'Cₛ', 'Cᵥ', 'Cₜ', 'C₀₀,ₛ']):
        diag_vals = np.diag(mat)
        off_diag = mat - np.diag(diag_vals)
        matrix_stats.append({
            'name': name,
            'max_diag': np.max(np.abs(diag_vals)),
            'max_off_diag': np.max(np.abs(off_diag)),
            'trace': np.trace(mat),
            'frobenius': np.linalg.norm(mat, 'fro')
        })

    names = [s['name'] for s in matrix_stats]
    max_diag = [s['max_diag'] for s in matrix_stats]
    max_off_diag = [s['max_off_diag'] for s in matrix_stats]

    x = np.arange(len(names))
    width = 0.35

    plt.bar(x - width/2, max_diag, width, label='Max diagonal', alpha=0.8)
    plt.bar(x + width/2, max_off_diag, width, label='Max off-diagonal', alpha=0.8)

    plt.yscale('log')
    plt.xlabel('Matrix type')
    plt.ylabel('Max |value|')
    plt.title(f'Matrix element magnitudes\nk = {sample_k:.2e} Mpc⁻¹')
    plt.xticks(x, names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if plot_filename is None:
        # Auto-generate filename based on data file
        base_name = filename.replace('.npz', '')
        plot_filename = f"{base_name}_uetc_plot.png"

    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")

    # Print some statistics
    print(f"\nUETC Statistics:")
    print(f"k range: {k_grid[0]:.3e} to {k_grid[-1]:.3e} Mpc⁻¹")
    print(f"ktau range: {ktau_grid[0]:.3e} to {ktau_grid[-1]:.3e}")
    print(f"Number of modes: {nmodes_saved}")
    print(f"String parameters: μ={data['string_mu']}, α={data['string_alpha']}, L={data['string_L']}")

    # Check for potential issues
    max_evals = [np.max(np.abs(evals_S)), np.max(np.abs(evals_V)), np.max(np.abs(evals_T))]
    min_evals = [np.min(np.abs(evals_S[evals_S != 0])),
                 np.min(np.abs(evals_V[evals_V != 0])),
                 np.min(np.abs(evals_T[evals_T != 0]))]

    print(f"\nEigenvalue ranges:")
    for i, name in enumerate(['Scalar', 'Vector', 'Tensor']):
        print(f"{name}: {min_evals[i]:.3e} to {max_evals[i]:.3e}")

    # Check for NaN or inf
    if np.any(np.isnan(efuncs)) or np.any(np.isinf(efuncs)):
        print("\nWARNING: Found NaN or inf values in eigenfunctions!")

    eval_arrays = [evals_S, evals_V, evals_T]
    if any(np.any(np.isnan(arr)) or np.any(np.isinf(arr)) for arr in eval_arrays):
        print("WARNING: Found NaN or inf values in eigenvalues!")

# --------------------------------------------------------------------- #
# Run if executed as a script
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate UETC eigenmode tables for cosmic strings')
    parser.add_argument('--plot', action='store_true',
                       help='Plot the results after generation (requires matplotlib)')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only plot existing results, don\'t generate new table')
    parser.add_argument('--filename', default='correlator_table_fast.npz',
                       help='Filename for the correlator table (default: correlator_table_fast.npz)')

    args = parser.parse_args()

    if args.plot_only:
        # Just plot existing results
        print(f"Plotting existing results from {args.filename}")
        try:
            plot_uetc(args.filename)
        except Exception as e:
            print(f"Error plotting: {e}")
    else:
        # Generate table (and optionally plot)
        t0 = time.time()
        print("Generating UETC eigenmode table...")
        build_table(args.filename)
        elapsed = time.time() - t0
        print(f"Total wall time: {elapsed:.1f} s")

        if args.plot:
            print("\nGenerating plots...")
            try:
                plot_uetc(args.filename)
            except Exception as e:
                print(f"Error plotting: {e}")
                print("You can plot later with: python string_correlators_fast.py --plot-only")