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

from integrals import I1_int_numba, I2_int_numba, I3_int_numba, I4_int_numba, I5_int_numba, I6_int_numba, I1_int_a_numba, I4_int_a_numba

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
def _get_correlator_pair(t1, t2, k,
                         mu1, alpha1, mu2, alpha2, L):
    v1  = _lookup(_V,  t1)
    xi1 = _lookup(_XI, t1)
    v2  = _lookup(_V,  t2)
    xi2 = _lookup(_XI, t2)

    # Find x and rho as in paper
    x1=k*t1*xi1; x2=k*t2*xi2; xp=(x1+x2)/2.0; xm=(x1-x2)/2.0
    rho=k*abs(v1*t1-v2*t2); rho_safe=max(rho, 1e-12)

    # Common factors for all correlators
    norm_denom_sq = (1.0 - v1**2)*(1.0 - v2**2)
    norm_denom = np.sqrt(norm_denom_sq)
    sf = _scaling_factor(t1, t2, xi1, xi2, L)
    common_factor_base = sf / (k**2*norm_denom)
    common_factor = mu*mu*common_factor_base

    uetc_val = 0.0, 0.0, 0.0, 0.0, 0.0

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
        #uetc_val=np.array([term00, termS, termV, termT, term00S])*sf
        return uetc_val

    return 0.0, 0.0, 0.0, 0.0, 0.0   # (00,S,V,T,00S)

# --------------------------------------------------------------------- #
# 7. Kernel that fills *all five* UETC matrices for a single k
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
            c00, cS, cV, cT, cXS = _get_correlator_pair(t1, t2, k, mu, alpha, mu, alpha, L)
            m00[i, j]  = c00
            mS[i, j]   = cS
            mV[i, j]   = cV
            mT[i, j]   = cT
            m00S[i, j] = cXS
            if i != j:
                # symmetry
                m00[j, i]  = c00
                mS[j,  i]  = cS
                mV[j,  i]  = cV
                mT[j,  i]  = cT
                m00S[j, i] = cXS
    return m00, mS, mV, mT, m00S

# --------------------------------------------------------------------- #
# 8. Diagonalisation (unchanged, numpy/SciPy – fast)
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
# 9. Public driver: eigen decomposition for one k
# --------------------------------------------------------------------- #
def eig_for_single_k(k_val, tau_grid, gamma=weighting_gamma, nmodes=nmodes):
    mats = build_uetc_mats(tau_grid, k_val, mu, alpha, L)
    diag = _diagonalise(mats, tau_grid, k_val, gamma, nmodes)
    return diag

# --------------------------------------------------------------------- #
# 10. Main table build loop – threaded, progress bar
# --------------------------------------------------------------------- #
def build_table():
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

    np.savez("correlator_table_fast.npz",
             k_grid=k_grid,
             ktau_grid=ktau_grid,
             eigenfunctions=efuncs,
             eigenvalues_S=evals_S,
             eigenvalues_00=evals_00,
             eigenvalues_V=evals_V,
             eigenvalues_T=evals_T,
             string_mu=mu, string_alpha=alpha, string_L=L,
             nmodes=nmodes, weighting_gamma=weighting_gamma)
    print("Saved correlator_table_fast.npz")

# --------------------------------------------------------------------- #
# 11. Run if executed as a script
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    t0 = time.time()
    build_table()
    print(f"Total wall time: {time.time()-t0:.1f} s")