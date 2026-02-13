# string_simulation.py

# Main script computing UETC correlators, diagonalising them, passing them to CAMB and calculating CMB spectra using our CAMB active sources module.

import os, sys, time, warnings, math, argparse
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import scipy.linalg
from numba import njit, prange
from tqdm import tqdm  # Already imported, now we will use it

try:
    import camb
    from camb.active_sources import ActiveSources
except ImportError:
    print("Error: The 'camb' package is required but not found.")
    print("Please install it, e.g., via 'pip install camb'")
    sys.exit(1)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    # Assuming integrals.py is in the same directory or python path
    from integrals import I1_int_numba, I2_int_numba, I3_int_numba, I4_int_numba, I5_int_numba, I6_int_numba, I1_int_a_numba, I4_int_a_numba, sici_numba
except ImportError:
    print("Error: The 'integrals.py' file with Numba functions is required but not found.")
    sys.exit(1)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ===================================================================== #
# SECTION 1 & 2: CORE COMPUTATION FUNCTIONS
# ===================================================================== #
def _solve_cosmology_and_vos(ntau, tau_min_s, tau_max_s, cr, omegas, h0):
    s_per_mpc = 3.085677577849e22 / 299792458.0
    tau_min, tau_max = tau_min_s / s_per_mpc, tau_max_s / s_per_mpc
    tau_grid = np.logspace(np.log10(tau_min), np.log10(tau_max), ntau)

    def friedmann_rhs(tau, a):
        r = omegas['R'] / a**2 + omegas['M'] / a + omegas['K'] + omegas['L'] * a**2
        return a * h0 * np.sqrt(r)

    a0 = np.sqrt(omegas['R']) * h0 * tau_grid[0]
    sol = solve_ivp(friedmann_rhs, (tau_grid[0], tau_grid[-1]), [a0], t_eval=tau_grid, atol=1e-12, rtol=1e-10, method='DOP853')
    if not sol.success: raise RuntimeError("Friedmann solver failed: " + sol.message)
    a_arr = sol.y[0]
    Hc_arr = h0 * np.sqrt(omegas['R']/a_arr**2 + omegas['M']/a_arr + omegas['K'] + omegas['L']*a_arr**2)
    k_tilde = lambda v: (2.0*np.sqrt(2)/np.pi)*(1.0 - 8.0*v**6)/(1.0 + 8.0*v**6)

    def vos_rhs(tau, y):
        xi, v = y
        Hc = np.interp(tau, tau_grid, Hc_arr)
        dxi = 1/tau * (v**2 * xi * tau * Hc - xi + cr * v / 2)
        dv = (1 - v**2) * (k_tilde(v) / (xi * tau) - 2 * v * Hc)
        return (dxi, dv)

    sol_vos = solve_ivp(vos_rhs, (tau_grid[0], tau_grid[-1]), (0.13, 0.65), t_eval=tau_grid, atol=1e-12, rtol=1e-10, method='LSODA')
    if not sol_vos.success: raise RuntimeError("VOS solver failed: " + sol_vos.message)
    return tau_grid, sol_vos.y[0], sol_vos.y[1]

@njit(inline='always')
def _lookup(arr, tau, tau_grid):
    idx = np.searchsorted(tau_grid, tau) - 1
    if idx < 0: return arr[0]
    if idx >= len(tau_grid) - 1: return arr[-1]
    w = (tau - tau_grid[idx]) / (tau_grid[idx+1] - tau_grid[idx])
    return (1 - w) * arr[idx] + w * arr[idx+1]

@njit(fastmath=True, cache=True)
def _scaling_factor(t1, t2, xi1, xi2, scaling_option):
    if scaling_option == 1: return 1.0
    denom = max(xi1 * t1, xi2 * t2, 1e-30)
    return 1.0 / (denom**3)

@njit(fastmath=True, cache=True)
def _get_correlator_pair(tau1, tau2, k, mu, alpha, L, vos_v, vos_xi, vos_tau, cfg):
    scaling_option, xmin, etcmin, xapr, min_terms, scale_terms, MAX_N_TERMS = cfg
    v1, xi1 = _lookup(vos_v, tau1, vos_tau), _lookup(vos_xi, tau1, vos_tau)
    v2, xi2 = _lookup(vos_v, tau2, vos_tau), _lookup(vos_xi, tau2, vos_tau)
    x1, x2 = k*tau1*xi1, k*tau2*xi2
    xp, xm = (x1+x2)/2.0, (x1-x2)/2.0
    rho = k*abs(v1*tau1-v2*tau2)
    rho_safe = max(rho, 1e-12)
    sf = _scaling_factor(tau1, tau2, xi1, xi2, scaling_option)
    norm_denom_sq = (1.0 - v1**2)*(1.0 - v2**2)
    norm_denom = math.sqrt(norm_denom_sq) if norm_denom_sq > 1e-16 else 0.0
    uetc_val = [0.0]*5
    if norm_denom == 0.0: return uetc_val
    common_factor = mu**2 * sf / (k**2 * norm_denom)
    if x1 <= xmin and x2 <= xmin:
        if alpha == 0: return uetc_val
        uetc_val[0]=-(alpha**2*mu**2*(-6.0+rho**2)*x1*x2)/(6.0*k**2*norm_denom)*sf
        s_num=(rho**2*(-10.0+(10.0-11.0*alpha**2)*v2**2+v1**2*(10.0-11.0*alpha**2+(-10.0+11.0*alpha**2+11.0*(1.0-2.0*alpha**2)*alpha**2)*v2**2))+42.0*(2.0+(-2.0+alpha**2)*v2**2+v1**2*(-2.0+alpha**2+(2.0-alpha**2+(-1.0+2.0*alpha**2)*alpha**2)*v2**2)))*x1*x2
        uetc_val[1]=(mu**2*s_num)/(420.0*alpha**2*k**2*norm_denom)*sf
        v_num=(8.0+4.0*(-2.0+alpha**2)*v1**2+(-8.0-4.0*(-2.0+alpha**2)*v1**2+alpha**2*(4.0+(-4.0+7.0*alpha**2)*v1**2))*v2**2)*x1*x2
        uetc_val[2]=(mu**2*2.3*v_num)/(256.0*alpha**2*k**2*norm_denom)*sf
        t_num=-(mu**2*(-28.0+6.0*rho**2+28.0*v1**2-14.0*alpha**2*v1**2-6.0*rho**2*v1**2+alpha**2*rho**2*v1**2+28.0*v2**2-14.0*alpha**2*v2**2-6.0*rho**2*v2**2+alpha**2*rho**2*v2**2-28.0*v1**2*v2**2+14.0*alpha**2*v1**2*v2**2+14.0*alpha**2*v1**2*v2**2-28.0*alpha**4*v1**2*v2**2+6.0*rho**2*v1**2*v2**2-alpha**2*rho**2*v1**2*v2**2-alpha**2*rho**2*v1**2*v2**2+2.0*alpha**4*rho**2*v1**2*v2**2)*x1*x2)
        uetc_val[3]=t_num/(420.0*alpha**2*k**2*norm_denom)*sf
        cross_num=-(mu**2*rho**2*(alpha**2*(1.0+(-1.0+2.0*alpha**2)*v2**2)+alpha**2*(1.0+(-1.0+2.0*alpha**2)*v1**2))*x1*x2)
        uetc_val[4]=cross_num/(60.0*alpha**2*k**2*norm_denom)*sf
        return uetc_val
    if abs(x1-x2)/max(x1,x2,1e-12)<=etcmin:
        x,v,mu_avg,alpha_avg=xp,(v1+v2)/2.0,mu,(alpha+alpha)/2.0
        if alpha_avg==0: return uetc_val
        norm_denom_etc_sq=(1.0-v**2)
        if norm_denom_etc_sq<=1e-16: return uetc_val
        sf_etc=_scaling_factor(tau1,tau2,xi1,xi2,scaling_option)
        six, _ = sici_numba(x); cosx, sinx = math.cos(x), math.sin(x)
        sinx_over_x = sinx/x if abs(x)>1e-12 else 1.0
        uetc_val[0]=(mu_avg**2*2.0*alpha_avg**2*(-1.0+cosx+x*six))/(k**2*(1.0-v**2))*sf_etc
        if x!=0:
            s1=(8*(-18+x**2)+8*(-2+alpha_avg**2)*v**2*(-18+x**2)+v**4*(8*(-18+x**2)-8*alpha_avg**2*(-18+x**2)+alpha_avg**4*(-54+11*x**2)))*cosx
            s2n=(-32*(1+(-2+alpha_avg**2)*v**2+(1-alpha_avg**2+alpha_avg**4)*v**4)*x**3+3*(-8*(-6+x**2)-8*(-2+alpha_avg**2)*v**2*(-6+x**2)+v**4*(-8*(-6+x**2)+8*alpha_avg**2*(-6+x**2)+alpha_avg**4*(18+x**2)))*sinx)
            s3=(8+8*(-2+alpha_avg**2)*v**2+(8-8*alpha_avg**2+11*alpha_avg**4)*v**4)*x**3*six
            uetc_val[1]=(mu_avg**2*(s1+s2n/x+s3))/(16.*alpha_avg**2*k**2*(1-v**2)*x**2)*sf_etc
            v1_sub=(x**3+3.0*x*cosx-3.0*sinx)/(3.0*x**3)
            v1_term=(2.0*(8.0+8.0*(-2.0+alpha_avg**2)*v**2+(8.0-8.0*alpha_avg**2+3*alpha_avg**4)*v**4))*v1_sub
            v2_term=alpha_avg**4*v**4*(-2.0+cosx+sinx_over_x+x*six)
            uetc_val[2]=(mu_avg**2*(v1_term+v2_term))/(8.0*alpha_avg**2*k**2*(1-v**2))*sf_etc
            t1=3*(8+8*(-2+alpha_avg**2)*v**2+(8-8*alpha_avg**2+3*alpha_avg**4)*v**4)*(-2+x**2)*cosx
            t2n=(64*(-1+v**2)*(1+(-1+alpha_avg**2)*v**2)*x**3-3*(-8*(2+x**2)-8*(-2+alpha_avg**2)*v**2*(2+x**2)+v**4*(-8*(2+x**2)+8*alpha_avg**2*(2+x**2)+alpha_avg**4*(-6+5*x**2)))*sinx)
            t3=3*(8+8*(-2+alpha_avg**2)*v**2+(8-8*alpha_avg**2+3*alpha_avg**4)*v**4)*x**3*six
            uetc_val[3]=(mu_avg**2*(t1+t2n/x+t3))/(96.*alpha_avg**2*k**2*(1-v**2)*x**2)*sf_etc
            uetc_val[4]=(mu_avg**2*(2+(-2+alpha_avg**2)*v**2)*(-4+cosx+(3*sinx)/x+x*six))/(2.*k**2*(1-v**2))*sf_etc
        return uetc_val
    n_terms=min(max(min_terms,int(scale_terms*xp)),MAX_N_TERMS)
    I1,I4=(I1_int_a_numba(min(x1,x2),rho_safe),I4_int_a_numba(min(x1,x2),rho_safe)) if abs(x1-x2)>=xapr else (I1_int_numba(xm,rho_safe,n_terms)-I1_int_numba(xp,rho_safe,n_terms),I4_int_numba(xm,rho_safe,n_terms)-I4_int_numba(xp,rho_safe,n_terms))
    I2,I3,I5,I6=I2_int_numba(xm,rho_safe)-I2_int_numba(xp,rho_safe),I3_int_numba(xm,rho_safe)-I3_int_numba(xp,rho_safe),I5_int_numba(xm,rho_safe)-I5_int_numba(xp,rho_safe),I6_int_numba(xm,rho_safe)-I6_int_numba(xp,rho_safe)

    uetc_val[0]=(2*alpha**2*I1)*common_factor
    safe_a1a2rho2,safe_a1a2=max(2.*alpha**2*rho_safe**2,1e-30),max(2.*alpha**2,1e-30)
    a1s=(-27*(alpha**2*v1*v2)**2+rho_safe**2*(1+(-1+2*alpha**2)*v1**2)*(1+(-1+2*alpha**2)*v2**2))/safe_a1a2rho2
    a2s=(-3*(-9*(alpha**2*v1*v2)**2+rho_safe**2*(-1+v2**2+v1**2*(1+(-1+(alpha**2)**2)*v2**2))))/safe_a1a2rho2
    a3s=(-9*(1+(-1+alpha**2)*v1**2)*(1+(-1+alpha**2)*v2**2))/safe_a1a2
    a4s=(-3*(-(alpha**2*rho_safe**2*(-1+v1**2)*v2**2)+alpha**2*v1**2*(-18*alpha**2*v2**2+rho_safe**2*(1+(-1+4*alpha**2)*v2**2))))/safe_a1a2rho2
    a5s=(3*(-(alpha**2*rho_safe**2*(-1+v1**2)*v2**2)+alpha**2*v1**2*(-18*alpha**2*v2**2+rho_safe**2*(1+(-1+4*alpha**2)*v2**2))))/safe_a1a2rho2
    a6s=(9*(-(alpha**2*(-1+v1**2)*v2**2)+alpha**2*v1**2*(1+(-1+2*alpha**2)*v2**2)))/safe_a1a2
    uetc_val[1]=(a1s*I1+a2s*I2+a3s*I3+a4s*I4+a5s*I5+a6s*I6)*common_factor
    safe_rho2,safe_a1a2_v=max(rho_safe**2,1e-30),max(alpha**2,1e-30)
    a1v,a2v=(3*alpha**2*v1**2*v2**2)/safe_rho2,-(3*alpha**2*v1**2*v2**2)/safe_rho2
    a3v=((1+(-1+alpha**2)*v1**2)*(1+(-1+alpha**2)*v2**2))/safe_a1a2_v
    a4v,a5v=(alpha**2*(-6+rho_safe**2)*v1**2*v2**2)/safe_rho2,-(alpha**2*(-6+rho_safe**2)*v1**2*v2**2)/safe_rho2
    a6v=(alpha**2*(-1+v1**2)*v2**2-alpha**2*v1**2*(1+(-1+2*alpha**2)*v2**2))/safe_a1a2_v
    uetc_val[2]=(a1v*I1+a2v*I2+a3v*I3+a4v*I4+a5v*I5+a6v*I6)*common_factor
    safe_4a1a2rho2,safe_4a1a2=max(4.*alpha**2*rho_safe**2,1e-30),max(4.*alpha**2,1e-30)
    a1t=(-3.*(alpha**2*v1*v2)**2+rho_safe**2*(-1.+v1**2)*(-1.+v2**2))/safe_4a1a2rho2
    a2t=(3.*(alpha**2*v1*v2)**2+rho_safe**2*(-1.+v2**2+v1**2*(1.+(-1.+(alpha**2)**2)*v2**2)))/safe_4a1a2rho2
    a3t=-((1.+(-1.+alpha**2)*v1**2)*(1.+(-1.+alpha**2)*v2**2))/safe_4a1a2
    a4t=(-(alpha**2*rho_safe**2*(-1.+v1**2)*v2**2)+alpha**2*v1**2*(6.*alpha**2*v2**2-rho_safe**2*(-1.+v2**2)))/safe_4a1a2rho2
    a5t=(alpha**2*rho_safe**2*(-1.+v1**2)*v2**2+alpha**2*v1**2*(-6.*alpha**2*v2**2+rho_safe**2*(-1.+v2**2)))/safe_4a1a2rho2
    a6t=(-(alpha**2*(-1.+v1**2)*v2**2)+alpha**2*v1**2*(1.+(-1.+2.*alpha**2)*v2**2))/safe_4a1a2
    uetc_val[3]=(a1t*I1+a2t*I2+a3t*I3+a4t*I4+a5t*I5+a6t*I6)*common_factor
    a1c=(-(alpha*(-1+v1**2))+alpha*(1-v2**2+2*alpha**2*(v1**2+v2**2)))/(2.*alpha)
    a2c=(-3*(-(alpha*(-1+v1**2))+alpha*(1-v2**2+alpha**2*(v1**2+v2**2))))/(2.*alpha)
    a4c=(-3*alpha**2*(v1**2+v2**2))/2.
    uetc_val[4]=(a1c*I1+a2c*I2+a4c*I4-a4c*I5)*common_factor
    return uetc_val

@njit(parallel=True, cache=True, fastmath=True)
def build_uetc_mats(tau_vec, k, mu, alpha, L, vos_v, vos_xi, vos_tau, cfg):
    n = tau_vec.size
    mats = (np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)))
    for i in prange(n):
        for j in range(i, n):
            uetc_val = _get_correlator_pair(tau_vec[i], tau_vec[j], k, mu, alpha, L, vos_v, vos_xi, vos_tau, cfg)
            for mat_idx in range(5):
                mats[mat_idx][i, j] = uetc_val[mat_idx]
                if i != j: mats[mat_idx][j, i] = uetc_val[mat_idx]
    return mats[0], mats[1], mats[2], mats[3], mats[4]

def _diagonalise(mats, tau_vec, k, gamma, nmodes):
    m00, mS, mV, mT, m00S = mats
    n = len(tau_vec)
    nmodes = min(nmodes, n)
    tau_i, tau_j = np.meshgrid(tau_vec, tau_vec, indexing='ij')
    W = (k**2 * tau_i * tau_j)**gamma * np.sqrt(tau_i * tau_j)
    evalV, evecV = scipy.linalg.eigh(W*mV, subset_by_index=[n-nmodes, n-1])
    evalT, evecT = scipy.linalg.eigh(W*mT, subset_by_index=[n-nmodes, n-1])
    Sbig = np.zeros((2*n, 2*n))
    Sbig[:n,:n]=W*m00; Sbig[n:,n:]=W*mS; Sbig[:n,n:]=W*m00S; Sbig[n:,:n]=Sbig[:n,n:].T
    evalS, evecSbig = scipy.linalg.eigh(Sbig, subset_by_index=[2*n-nmodes, 2*n-1])
    svals, svecs = evalS[::-1], evecSbig[:, ::-1]
    return {'eval_S': svals, 'eval_V': evalV[::-1], 'eval_T': evalT[::-1], 'eval_00': svals, 'evec_00': svecs[:n,:].T, 'evec_S': svecs[n:,:].T, 'evec_V': evecV[:,::-1].T, 'evec_T': evecT[:,::-1].T}

@njit(parallel=True, cache=True, fastmath=True)
def align_eigenvector_signs_numba(eigenfunctions, derivatives):
    nk, ntypes, nmodes, nktau = eigenfunctions.shape
    for type_idx in prange(ntypes):
        for mode_idx in range(nmodes):
            for k_idx in range(1, nk):
                v_prev, v_curr = eigenfunctions[k_idx-1,type_idx,mode_idx], eigenfunctions[k_idx,type_idx,mode_idx]
                if np.all(np.isnan(v_prev)): continue
                dot_product = 0.0
                for tau_idx in range(nktau):
                    if not (math.isnan(v_curr[tau_idx]) or math.isnan(v_prev[tau_idx])):
                        dot_product += v_curr[tau_idx] * v_prev[tau_idx]
                if dot_product < 0:
                    eigenfunctions[k_idx, type_idx, mode_idx] *= -1.0
                    derivatives[k_idx, type_idx, mode_idx]    *= -1.0
    return eigenfunctions, derivatives

# ===================================================================== #
# SECTION 3: CALCULATION AND PLOTTING
# ===================================================================== #

def plot_uetc_evecs_reconstruction(k_to_plot, tau_values, alpha, mu, gamma, nmodes_diag, vos_v, vos_xi, vos_tau, cfg, uetc_n_levels=15):
    """Ported from string_correlators.py: Sanity check visualization."""
    if plt is None: return
    print(f"--- Generating UETC, E-vec, & Reconstruction plots for k = {k_to_plot:.4e} ---")
    ntau = len(tau_values)
    log_kt_axis = np.log10(k_to_plot * tau_values)
    mu_sq = mu**2
    mats = build_uetc_mats(tau_values, k_to_plot, mu, alpha, 0.95, vos_v, vos_xi, vos_tau, cfg)
    diag = _diagonalise(mats, tau_values, k_to_plot, gamma, nmodes_diag)
    
    correlator_data_scaled = [np.full((ntau, ntau), np.nan) for _ in range(5)]
    for comp_idx, raw_matrix in enumerate(mats):
        for i in range(ntau):
            for j in range(ntau):
                plot_display_scaling = (tau_values[i] * tau_values[j])**0.5 / mu_sq if mu_sq != 0 else 0
                correlator_data_scaled[comp_idx][j, i] = raw_matrix[j, i] * plot_display_scaling

    reconstructed_correlator = [np.full((ntau, ntau), np.nan) for _ in range(5)]
    tau_i_mesh, tau_j_mesh = np.meshgrid(tau_values, tau_values, indexing='ij')
    W_unweight = np.power((k_to_plot**2 * tau_i_mesh * tau_j_mesh), gamma) * np.sqrt(tau_i_mesh * tau_j_mesh)
    
    # Reconstruct Scalar
    u00, uS, lS = diag['evec_00'], diag['evec_S'], diag['eval_S']
    reconstructed_correlator[0] = (np.einsum('p,pi,pj->ij', lS, u00, u00) / W_unweight).T * np.sqrt(tau_i_mesh * tau_j_mesh) / mu_sq
    reconstructed_correlator[1] = (np.einsum('p,pi,pj->ij', lS, uS, uS) / W_unweight).T * np.sqrt(tau_i_mesh * tau_j_mesh) / mu_sq
    reconstructed_correlator[4] = (np.einsum('p,pi,pj->ij', lS, u00, uS) / W_unweight).T * np.sqrt(tau_i_mesh * tau_j_mesh) / mu_sq
    # Vectors/Tensors
    reconstructed_correlator[2] = (np.einsum('p,pi,pj->ij', diag['eval_V'], diag['evec_V'], diag['evec_V']) / W_unweight).T * np.sqrt(tau_i_mesh * tau_j_mesh) / mu_sq
    reconstructed_correlator[3] = (np.einsum('p,pi,pj->ij', diag['eval_T'], diag['evec_T'], diag['evec_T']) / W_unweight).T * np.sqrt(tau_i_mesh * tau_j_mesh) / mu_sq

    fig, axes = plt.subplots(5, 3, figsize=(15, 20), constrained_layout=True)
    titles = ["00 Type", "S Type", "V Type", "T Type", "00S Cross"]
    evec_keys = ['evec_00', 'evec_S', 'evec_V', 'evec_T', None]
    for comp_idx in range(5):
        axes[comp_idx, 0].contourf(log_kt_axis, log_kt_axis, correlator_data_scaled[comp_idx], levels=uetc_n_levels, cmap='jet')
        axes[comp_idx, 0].set_title(f"Original {titles[comp_idx]}")
        if comp_idx < 4:
            for m in range(min(5, nmodes_diag)): axes[comp_idx, 1].plot(log_kt_axis, diag[evec_keys[comp_idx]][m, :])
            axes[comp_idx, 1].set_title(f"Top E-vecs {titles[comp_idx]}")
        axes[comp_idx, 2].contourf(log_kt_axis, log_kt_axis, reconstructed_correlator[comp_idx], levels=uetc_n_levels, cmap='jet')
        axes[comp_idx, 2].set_title(f"Reco {titles[comp_idx]}")
    plt.show()

def generate_correlators_and_eigensystem(args, verbose=True):
    if verbose: print("1. Generating UETC correlators and eigensystem...")
    k_grid = np.logspace(np.log10(args.k_min), np.log10(args.k_max), args.nk)
    ktau_grid = np.logspace(np.log10(args.ktau_min), np.log10(args.ktau_max), args.nktau)
    nmodes = args.nmodes
    H0_Mpc_inv = args.H0 / 299792.458
    omegas = {'R': args.Omega_rad, 'M': args.Omega_matter, 'L': args.Omega_lambda, 'K': 0.0}
    cfg = (2, 0.15, 1e-1, 2.5, 10, 15.0, 75)
    if verbose: print("   - Solving background cosmology and VOS model...")
    vos_tau, vos_xi, vos_v = _solve_cosmology_and_vos(5000, 1e-4, 8e17, args.cr, omegas, H0_Mpc_inv)
    if verbose: print("   - VOS solution complete.")
    
    if hasattr(args, 'plot_uetc') and args.plot_uetc:
        k_ref = 0.05 * (args.H0 / 100.0)
        plot_uetc_evecs_reconstruction(k_ref, ktau_grid/k_ref, args.alpha, 1.0, args.weighting_gamma, nmodes, vos_v, vos_xi, vos_tau, cfg)

    ntypes, efuncs, efuncs_derivs = 4, np.zeros((args.nk, 4, nmodes, args.nktau)), np.zeros((args.nk, 4, nmodes, args.nktau))
    evals = {name: np.zeros((args.nk, nmodes)) for name in ['S','00','V','T']}
    fixed_mu, fixed_L = 1.0, 0.95
    if verbose: print(f"   - Using fixed internal string parameters: mu={fixed_mu}, L={fixed_L}")

    iterator = tqdm(k_grid, desc="   - Building correlators (k-loop)", ncols=100, unit="k", disable=not verbose)
    for ik, k in enumerate(iterator):
        tau_vec = ktau_grid/k
        try:
            mats = build_uetc_mats(tau_vec, k, fixed_mu, args.alpha, fixed_L, vos_v, vos_xi, vos_tau, cfg)
            diag = _diagonalise(mats, tau_vec, k, args.weighting_gamma, nmodes)
            efuncs[ik,0],efuncs[ik,1],efuncs[ik,2],efuncs[ik,3] = diag['evec_00'],diag['evec_S'],diag['evec_V'],diag['evec_T']
            evals['S'][ik],evals['00'][ik],evals['V'][ik],evals['T'][ik] = diag['eval_S'],diag['eval_00'],diag['eval_V'],diag['eval_T']
            log_ktau_axis = np.log(ktau_grid)
            for type_idx in range(ntypes):
                for mode_idx in range(nmodes):
                    ef_1d, valid = efuncs[ik,type_idx,mode_idx,:], ~np.isnan(efuncs[ik,type_idx,mode_idx,:])
                    if np.sum(valid)<4: efuncs_derivs[ik,type_idx,mode_idx,:] = np.nan
                    else:
                        spl = CubicSpline(log_ktau_axis[valid], ef_1d[valid], extrapolate=False)
                        efuncs_derivs[ik,type_idx,mode_idx,:] = spl.derivative(1)(log_ktau_axis)
        except Exception as e:
            iterator.write(f"\nWarning: Error processing k={k:.4e}: {e}")
            efuncs[ik,...],efuncs_derivs[ik,...] = np.nan,np.nan
            for name in evals: evals[name][ik,...] = np.nan

    if verbose: print("\n   - Aligning eigenvector signs...")
    efuncs, efuncs_derivs = align_eigenvector_signs_numba(efuncs, efuncs_derivs)
    if verbose: print("   - Sign alignment complete.")

    if hasattr(args, 'save_correlators') and args.save_correlators:
        out_file = "correlator_table.npz"
        np.savez(out_file, k_grid=k_grid, ktau_grid=ktau_grid, eigenfunctions=efuncs, eigenfunctions_d_dlogkt=efuncs_derivs, 
                 eigenvalues_S=evals['S'], eigenvalues_00=evals['00'], eigenvalues_V=evals['V'], eigenvalues_T=evals['T'], 
                 string_params_mu=fixed_mu, nmodes=nmodes, weighting_gamma=args.weighting_gamma)
        if verbose: print(f"   - Correlator table saved to {out_file}")

    return {'k_grid':k_grid, 'ktau_grid':ktau_grid, 'eigenfunctions':efuncs, 'eigenfunctions_d_dlogkt':efuncs_derivs, 'eigenvalues_S':evals['S'], 'eigenvalues_00':evals['00'], 'eigenvalues_V':evals['V'], 'eigenvalues_T':evals['T'], 'string_params_mu':fixed_mu, 'nmodes':nmodes, 'weighting_gamma':args.weighting_gamma}

def setup_camb_params(args, verbose=True):
    pars = camb.CAMBparams()
    h = args.H0 / 100.0
    baryon_fraction = 0.155
    if verbose: print(f"   - Assuming fixed baryon fraction of {baryon_fraction} to derive ombh2 and omch2.")
    omega_b = args.Omega_matter * baryon_fraction
    omega_c = args.Omega_matter * (1.0 - baryon_fraction)
    ombh2, omch2 = omega_b * h**2, omega_c * h**2
    pars.set_cosmology(H0=args.H0, ombh2=ombh2, omch2=omch2, omk=0.0, tau=args.tau, mnu=0.0, num_massive_neutrinos=0)
    pars.omk = 0.0
    pars.max_l_tensor, pars.max_l = args.lmax, args.lmax
    pars.WantScalars, pars.WantVectors, pars.WantTensors = args.scalar, args.vector, args.tensor
    pars.DoLensing = False
    return pars

def setup_active_sources(correlator_data):
    active_sources = ActiveSources()
    active_sources.set_correlator_table(k_grid=correlator_data['k_grid'],tau_grid=correlator_data['ktau_grid'],eigenfunctions=correlator_data['eigenfunctions'],eigenfunctions_d_dlogkt=correlator_data['eigenfunctions_d_dlogkt'],eigenvalues_S=correlator_data['eigenvalues_S'],eigenvalues_00=correlator_data['eigenvalues_00'],eigenvalues_V=correlator_data['eigenvalues_V'],eigenvalues_T=correlator_data['eigenvalues_T'],string_params_mu=correlator_data['string_params_mu'],nmodes_param=correlator_data['nmodes'],weighting_param=correlator_data['weighting_gamma'])
    return active_sources

def calculate_string_cls(pars, args, correlator_data, verbose=True):
    if verbose: print("\n2. Calculating string C_l spectra with CAMB...")
    actual_n_modes_to_sum = min(args.nmodes, correlator_data['nmodes'])
    if verbose: print(f"   - Summing {actual_n_modes_to_sum} eigenmodes...")
    pars.ActiveSources = setup_active_sources(correlator_data)
    pars.ActiveSources.set_active_eigenmode(0)
    dummy_results = camb.get_results(pars)
    dummy_cls = dummy_results.get_cmb_power_spectra(pars, CMB_unit=args.units, raw_cl=True)['total']
    cl_strings_sum_all = np.zeros_like(dummy_cls)

    iterator = tqdm(range(1, actual_n_modes_to_sum + 1),
                    desc="     - Summing modes in CAMB",
                    ncols=100,
                    unit="mode",
                    disable=not verbose)

    for i_mode in iterator:
        pars.ActiveSources.set_active_eigenmode(i_mode)
        results_mode_i = camb.get_results(pars)
        cl_mode_i_all = results_mode_i.get_cmb_power_spectra(pars, CMB_unit=args.units, raw_cl=True)['total']
        max_len = min(len(cl_strings_sum_all), len(cl_mode_i_all))
        cl_strings_sum_all[:max_len] += cl_mode_i_all[:max_len]

    return np.arange(cl_strings_sum_all.shape[0]), cl_strings_sum_all

def plot_final_cls(ls_calc, cl_strings_all, args):
    if args.no_plot or plt is None: return
    print("\n3. Plotting final unscaled C_l results...")
    fig, axs = plt.subplots(4, 1, figsize=(8, 11), sharex=True, constrained_layout=True)
    plot_mask = (ls_calc >= 2) & (ls_calc <= args.lmax)
    ls_plot = ls_calc[plot_mask]
    l_factor = ls_plot * (ls_plot + 1) / (2 * np.pi)

    enabled_modes = [m for m,a in [('S',args.scalar),('V',args.vector),('T',args.tensor)] if a]
    mode_str = '+'.join(enabled_modes) if enabled_modes else "None"
    pol_map = {'TT': 0, 'EE': 1, 'BB': 2, 'TE': 3}
    for i, pol_name in enumerate(['TT', 'EE', 'TE', 'BB']):
        ax = axs[i]
        cl_plot = cl_strings_all[plot_mask, pol_map[pol_name]] * (args.gmu)**2
        plot_data = l_factor * cl_plot
        ax.plot(ls_plot, plot_data, label=f'Strings ({mode_str})')
        ax.set_ylabel(rf'$\ell(\ell+1)C_\ell^{{{pol_name}}}/2\pi$')
        ax.set_xscale('log')
        if pol_name in ['TT', 'TE']: ax.set_yscale('linear')
        else: ax.set_yscale('log')
        ax.grid(True, which="both", ls=":", alpha=0.6)
        ax.legend(loc='upper right')
    axs[-1].set_xlabel(r'Multipole moment, $\ell$')
    axs[0].set_xlim([2, args.lmax])
    fig.suptitle(f'CMB Anisotropies from Cosmic Strings (Gμ = {args.gmu:.2e})', fontsize=16)
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"--> Final C_l plot saved to {args.output}")
    else:
        plt.show()

# ===================================================================== #
# SECTION 4: CALLABLE WRAPPER AND MAIN DRIVER
# ===================================================================== #

def run_string_simulation(args, verbose=True):
    """
    Callable function that runs the entire simulation for a given set of args.
    Progress bars are now inside the functions below and controlled by 'verbose'.
    """
    correlator_data = generate_correlators_and_eigensystem(args, verbose=verbose)

    pars_for_strings = setup_camb_params(args, verbose=False)

    ls_calc, cl_strings_all = calculate_string_cls(pars_for_strings, args, correlator_data, verbose=verbose)
    return ls_calc, cl_strings_all

def main():
    parser = argparse.ArgumentParser(description='Generate unscaled C_l spectra from UETC string sources for ML training.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g_calc = parser.add_argument_group('Calculation Type')
    g_calc.add_argument('--scalar', '-s', action='store_true', help='Enable scalar perturbations')
    g_calc.add_argument('--vector', '-v', action='store_true', help='Enable vector perturbations')
    g_calc.add_argument('--tensor', '-t', action='store_true', help='Enable tensor perturbations')
    g_phys = parser.add_argument_group('Physical Parameters')
    g_phys.add_argument('--alpha', type=float, default=1.0, help='String model parameter alpha')
    g_phys.add_argument('--cr', type=float, default=0.5, help='VOS loop chopping efficiency c_r')
    g_cosmo = parser.add_argument_group('Cosmological Parameters')
    # Use standard Planck 2018 values by default
    g_cosmo.add_argument('--H0', type=float, default=67.5, help='Hubble constant [km/s/Mpc]')
    g_cosmo.add_argument('--Omega_matter', type=float, default=0.315, help='Total matter density Omega_m')
    g_cosmo.add_argument('--Omega_lambda', type=float, default=0.685, help='Dark energy density Omega_Lambda')
    g_cosmo.add_argument('--Omega_rad', type=float, default=9.24e-5, help='Total radiation density Omega_r')
    g_cosmo.add_argument('--tau', type=float, default=0.06, help='Reionization optical depth')
    g_num = parser.add_argument_group('Numerical and Grid Parameters')
    g_num.add_argument('--nmodes', '-n', type=int, default=32, help='Number of UETC eigenmodes')
    g_num.add_argument('--lmax', '-l', type=int, default=4000, help='Max multipole')
    g_num.add_argument('--nk', type=int, default=100, help='Number of k bins')
    g_num.add_argument('--nktau', type=int, default=128, help='Number of k*tau bins')
    g_num.add_argument('--k-min', type=float, default=1e-6, help='Min k [Mpc^-1]')
    g_num.add_argument('--k-max', type=float, default=10.0, help='Max k [Mpc^-1]')
    g_num.add_argument('--ktau-min', type=float, default=1e-4, help='Min k*tau')
    g_num.add_argument('--ktau-max', type=float, default=1e3, help='Max k*tau')
    g_num.add_argument('--weighting-gamma', type=float, default=0.25, help='Weighting gamma')
    g_out = parser.add_argument_group('Output Parameters')
    g_out.add_argument('--units', '-u', type=str, choices=['muK', 'K'], default='muK', help='Internal CMB units')
    g_out.add_argument('--output', '-o', type=str, default=None, help='Filename for final C_l plot')
    g_out.add_argument('--no-plot', action='store_true', help='Disable final C_l plot')
    g_phys.add_argument('--gmu', type=float, default=1.58e-7, help='String tension parameter G*mu (for plotting only)')
    g_out.add_argument('--plot-uetc', action='store_true', help='Plot UETC, eigenvectors and reconstruction comparison')
    g_out.add_argument('--save-correlators', action='store_true', help='Save the calculated correlator table to .npz')

    args = parser.parse_args()
    if not (args.scalar or args.vector or args.tensor):
        print("Warning: No perturbation types selected. Defaulting to all three (S+V+T).")
        args.scalar, args.vector, args.tensor = True, True, True

    t_start_total = time.time()
    print("--- Starting Unscaled String C_l Generation ---")
    print("Configuration:"); [print(f"  {k:<20}: {v}") for k,v in sorted(vars(args).items())]; print("-" * 40)

    ls_calc, cl_strings_all = run_string_simulation(args, verbose=True)
    plot_final_cls(ls_calc, cl_strings_all, args)

    print(f"\n--- Script Finished ---")
    print(f"Total execution time: {time.time() - t_start_total:.2f} seconds.")

if __name__ == "__main__":
    main()