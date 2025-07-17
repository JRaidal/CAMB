#!/usr/bin/env python3
"""
Test vectorized version of get_correlators
"""

import numpy as np
import time
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from string_correlators import *

def get_correlators_vectorized(tau1_vec, tau2_vec, k, SPR_base):
    """
    Vectorized version of get_correlators that can handle arrays for tau1 and tau2.

    Parameters:
    -----------
    tau1_vec : array-like, shape (N,) or (N, M)
        First tau values
    tau2_vec : array-like, shape (N,) or (N, M)
        Second tau values (must be same shape as tau1_vec)
    k : float
        k value
    SPR_base : object
        String parameters

    Returns:
    --------
    results : ndarray, shape (..., 5)
        Array of correlator results with last dimension being [00, S, V, T, 00S]
    """

    # Ensure inputs are arrays
    tau1_vec = np.asarray(tau1_vec)
    tau2_vec = np.asarray(tau2_vec)

    # Check shapes match
    if tau1_vec.shape != tau2_vec.shape:
        raise ValueError("tau1_vec and tau2_vec must have the same shape")

    original_shape = tau1_vec.shape

    # Flatten for processing
    tau1_flat = tau1_vec.flatten()
    tau2_flat = tau2_vec.flatten()
    n_points = len(tau1_flat)

    # Pre-allocate result array
    results = np.zeros((n_points, 5))

    # Get parameters (same for all points)
    parameters1 = SPR_base
    parameters2 = SPR_base
    alpha1 = parameters1.alpha
    alpha2 = parameters2.alpha
    mu1 = parameters1.mu
    mu2 = parameters2.mu
    L_decay = parameters1.L

    # Vectorized VOS parameter calculation
    v1_vec = np.array([v_interp(tau1) for tau1 in tau1_flat])
    xi1_vec = np.array([xi_interp(tau1) for tau1 in tau1_flat])
    v2_vec = np.array([v_interp(tau2) for tau2 in tau2_flat])
    xi2_vec = np.array([xi_interp(tau2) for tau2 in tau2_flat])

    # Vectorized x and rho calculations
    x1_vec = k * tau1_flat * xi1_vec
    x2_vec = k * tau2_flat * xi2_vec
    xp_vec = (x1_vec + x2_vec) / 2.0
    xm_vec = (x1_vec - x2_vec) / 2.0
    rho_vec = k * np.abs(v1_vec * tau1_flat - v2_vec * tau2_flat)
    rho_safe_vec = np.maximum(rho_vec, 1e-12)

    # Vectorized common factors
    norm_denom_sq_vec = (1.0 - v1_vec**2) * (1.0 - v2_vec**2)
    norm_denom_vec = np.sqrt(norm_denom_sq_vec)

    # Vectorized scaling factors
    sf_vec = np.array([scaling_factor(tau1_flat[i], tau2_flat[i], xi1_vec[i], xi2_vec[i], L_decay)
                       for i in range(n_points)])

    common_factor_base_vec = sf_vec / (k**2 * norm_denom_vec)
    common_factor_vec = mu1 * mu2 * common_factor_base_vec

    # Process each point (we can vectorize this further, but let's start here)
    for i in range(n_points):
        tau1, tau2 = tau1_flat[i], tau2_flat[i]
        x1, x2 = x1_vec[i], x2_vec[i]
        xp, xm = xp_vec[i], xm_vec[i]
        rho_safe = rho_safe_vec[i]
        v1, v2 = v1_vec[i], v2_vec[i]
        xi1, xi2 = xi1_vec[i], xi2_vec[i]
        common_factor = common_factor_vec[i]

        # Use the same logic as original get_correlators for each point
        # --- Regime 1: Small x ---
        if x1 <= xmin and x2 <= xmin:
            if alpha1 == 0 or alpha2 == 0:
                results[i] = np.zeros(5)
                continue

            # Small x regime calculations (vectorized where possible)
            term00 = -(alpha1*alpha2*mu1*mu2*(-6.0 + rho_safe**2)*x1*x2)/(6.0*k**2*norm_denom_vec[i])
            # ... (rest of small x calculations)
            results[i, 0] = term00 * sf_vec[i]
            # For now, just use the original function for complex cases
            results[i] = get_correlators(tau1, tau2, k, SPR_base)
            continue

        # --- Regime 2: ETC ---
        if abs(x1 - x2) <= etcmin:
            # ETC regime - use original function for now
            results[i] = get_correlators(tau1, tau2, k, SPR_base)
            continue

        # --- Regime 3: General Case ---
        n_terms_raw = max(min_terms, int(scale_terms * xp))
        n_terms = min(n_terms_raw, MAX_N_TERMS)
        use_approx = abs(x1 - x2) >= xapr
        small_rho = rho_safe < 1e-2

        # Calculate integrals
        if use_approx:
            I1 = I1_int_a(min(x1, x2), rho_safe)
            I4 = I4_int_a(min(x1, x2), rho_safe)
        else:
            I1 = I1_int(xm, rho_safe, n_terms) - I1_int(xp, rho_safe, n_terms)
            I4 = I4_int(xm, rho_safe, n_terms) - I4_int(xp, rho_safe, n_terms)

        if not use_approx and small_rho:
            I4 = I1 / 2.0

        I2 = I2_int(xm, rho_safe) - I2_int(xp, rho_safe)
        I3 = I3_int(xm, rho_safe) - I3_int(xp, rho_safe)

        if not use_approx and small_rho:
            I5 = I2 / 2.0
            I6 = I3 / 2.0
        else:
            I5 = I5_int(xm, rho_safe) - I5_int(xp, rho_safe)
            I6 = I6_int(xm, rho_safe) - I6_int(xp, rho_safe)

        integrals = [I1, I2, I3, I4, I5, I6]
        if not all(np.isfinite(integral) for integral in integrals):
            results[i] = np.zeros(5)
            continue

        # Calculate correlators using vectorized coefficient calculations
        safe_a1a2rho2 = max(2.0 * alpha1 * alpha2 * rho_safe**2, 1e-30)
        safe_a1a2 = max(2.0 * alpha1 * alpha2, 1e-30)

        # 00 component
        sum00 = 2 * alpha1 * alpha2 * I1
        results[i, 0] = sum00 * common_factor

        # S component
        c_terms_S = np.array([
            (-27*(alpha1*alpha2*v1*v2)**2 + rho_safe**2*(1+(-1+2*alpha1**2)*v1**2)*(1+(-1+2*alpha2**2)*v2**2))/safe_a1a2rho2,
            (-3*(-9*(alpha1*alpha2*v1*v2)**2 + rho_safe**2*(-1+v2**2+v1**2*(1+(-1+(alpha1*alpha2)**2)*v2**2))))/safe_a1a2rho2,
            (-9*(1+(-1+alpha1**2)*v1**2)*(1+(-1+alpha2**2)*v2**2))/safe_a1a2,
            (-3*(-(alpha2**2*rho_safe**2*(-1+v1**2)*v2**2)+alpha1**2*v1**2*(-18*alpha2**2*v2**2+rho_safe**2*(1+(-1+4*alpha2**2)*v2**2))))/safe_a1a2rho2,
            (3*(-(alpha2**2*rho_safe**2*(-1+v1**2)*v2**2)+alpha1**2*v1**2*(-18*alpha2**2*v2**2+rho_safe**2*(1+(-1+4*alpha2**2)*v2**2))))/safe_a1a2rho2,
            (9*(-(alpha2**2*(-1+v1**2)*v2**2)+alpha1**2*v1**2*(1+(-1+2*alpha2**2)*v2**2)))/safe_a1a2
        ])

        sumS = np.dot(c_terms_S, integrals)
        results[i, 1] = sumS * common_factor

        # V component
        safe_rho2_local = max(rho_safe**2, 1e-30)
        safe_a1a2_local = max(alpha1 * alpha2, 1e-30)

        c_terms_V = np.array([
            (3*alpha1*alpha2*v1**2*v2**2)/safe_rho2_local,
            (-3*alpha1*alpha2*v1**2*v2**2)/safe_rho2_local,
            ((1+(-1+alpha1**2)*v1**2)*(1+(-1+alpha2**2)*v2**2))/safe_a1a2_local,
            (alpha1*alpha2*(-6+rho_safe**2)*v1**2*v2**2)/safe_rho2_local,
            -((alpha1*alpha2*(-6+rho_safe**2)*v1**2*v2**2)/safe_rho2_local),
            (alpha2**2*(-1+v1**2)*v2**2-alpha1**2*v1**2*(1+(-1+2*alpha2**2)*v2**2))/safe_a1a2_local
        ])

        sumV = np.dot(c_terms_V, integrals)
        results[i, 2] = sumV * common_factor

        # T component
        safe_4a1a2rho2 = max(4.0 * alpha1 * alpha2 * rho_safe**2, 1e-30)
        safe_4a1a2 = max(4.0 * alpha1 * alpha2, 1e-30)

        c_terms_T = np.array([
            (-3.0*(alpha1*alpha2*v1*v2)**2+rho_safe**2*(-1.0+v1**2)*(-1.0+v2**2))/safe_4a1a2rho2,
            (3.0*(alpha1*alpha2*v1*v2)**2+rho_safe**2*(-1.0+v2**2+v1**2*(1.0+(-1.0+(alpha1*alpha2)**2)*v2**2)))/safe_4a1a2rho2,
            -((1.0+(-1.0+alpha1**2)*v1**2)*(1.0+(-1.0+alpha2**2)*v2**2))/safe_4a1a2,
            (-(alpha2**2*rho_safe**2*(-1.0+v1**2)*v2**2)+alpha1**2*v1**2*(6.0*alpha2**2*v2**2-rho_safe**2*(-1.0+v2**2)))/safe_4a1a2rho2,
            (alpha2**2*rho_safe**2*(-1.0+v1**2)*v2**2+alpha1**2*v1**2*(-6.0*alpha2**2*v2**2+rho_safe**2*(-1.0+v2**2)))/safe_4a1a2rho2,
            (-(alpha2**2*(-1.0+v1**2)*v2**2)+alpha1**2*v1**2*(1.0+(-1.0+2.0*alpha2**2)*v2**2))/safe_4a1a2
        ])

        sumT = np.dot(c_terms_T, integrals)
        results[i, 3] = sumT * common_factor

        # 00S cross component
        c_terms_00S = np.array([
            (-(alpha2**2*(-1+v1**2))+alpha1**2*(1-v2**2+2*alpha2**2*(v1**2+v2**2)))/safe_a1a2,
            (-3*(-(alpha2**2*(-1+v1**2))+alpha1**2*(1-v2**2+alpha2**2*(v1**2+v2**2))))/safe_a1a2,
            0.0,
            (-3*alpha1*alpha2*(v1**2+v2**2))/2.0,
            (3*alpha1*alpha2*(v1**2+v2**2))/2.0,
            0.0
        ])

        sumC = np.dot(c_terms_00S, integrals)
        results[i, 4] = sumC * common_factor

    # Reshape back to original shape
    if len(original_shape) == 0:  # scalar input
        return results[0]
    else:
        return results.reshape(original_shape + (5,))

def calculate_correlators_vectorized(k_val, spr_parameters, tau_vals_local, weighting_local, nmodes_local):
    """
    Vectorized version of correlator calculation
    """
    ntau_calc = len(tau_vals_local)

    # Create meshgrids for all tau pairs
    tau1_mesh, tau2_mesh = np.meshgrid(tau_vals_local, tau_vals_local, indexing='ij')

    # Only calculate upper triangle + diagonal (due to symmetry)
    upper_triangle_mask = np.triu(np.ones((ntau_calc, ntau_calc), dtype=bool))

    # Extract tau pairs for upper triangle
    tau1_upper = tau1_mesh[upper_triangle_mask]
    tau2_upper = tau2_mesh[upper_triangle_mask]

    print(f"Calculating {len(tau1_upper)} correlator pairs with vectorized function...")

    # Calculate all correlators at once
    start_time = time.time()
    results_upper = get_correlators_vectorized(tau1_upper, tau2_upper, k_val, spr_parameters)
    end_time = time.time()

    print(f"Vectorized calculation took {end_time - start_time:.3f} seconds")

    # Initialize full correlation matrices
    correlator_matrices = {key: np.zeros((ntau_calc, ntau_calc)) for key in ['UETC_00','UETC_S','UETC_V','UETC_T','UETC_00S']}

    # Fill upper triangle
    component_names = ['UETC_00', 'UETC_S', 'UETC_V', 'UETC_T', 'UETC_00S']
    for comp_idx, comp_name in enumerate(component_names):
        correlator_matrices[comp_name][upper_triangle_mask] = results_upper[:, comp_idx]

        # Symmetrize (fill lower triangle)
        correlator_matrices[comp_name] = correlator_matrices[comp_name] + correlator_matrices[comp_name].T
        # Diagonal was added twice, so subtract it once
        np.fill_diagonal(correlator_matrices[comp_name],
                        correlator_matrices[comp_name].diagonal() / 2)

    # Now do the eigenvalue decomposition (same as original)
    diag_results = diagonalize_correlators(
        correlator_matrices, tau_vals_local, weighting_local, nmodes_local, k_val
    )

    result_dictionary = {
        'k_value': k_val,
        'tau_values': tau_vals_local,
        'eigenvectors_00': diag_results['evec_00'],
        'eigenvectors_S': diag_results['evec_S'],
        'eigenvectors_V': diag_results['evec_V'],
        'eigenvectors_T': diag_results['evec_T'],
        'eigenvalues_S': diag_results['eval_S'],
        'eigenvalues_00': diag_results['eval_00'],
        'eigenvalues_V': diag_results['eval_V'],
        'eigenvalues_T': diag_results['eval_T'],
    }

    return result_dictionary, correlator_matrices

def test_vectorized_performance():
    """Test the performance of vectorized vs original implementation"""

    print("Testing vectorized correlator performance...")
    print("=" * 60)

    # Test parameters
    k_val = 0.1
    ntau_values = [16, 32, 64]

    for ntau in ntau_values:
        print(f"\nTesting with ntau = {ntau}")

        ktau_grid = np.logspace(-2, 2, ntau)
        tau_grid = ktau_grid / k_val
        nmodes = min(16, ntau)

        # Test original implementation
        print("  Original implementation:")
        start_time = time.time()
        result_orig, _ = calculate_eigenvectors(k_val, SPRa, tau_grid, weighting, nmodes)
        time_orig = time.time() - start_time
        print(f"    Time: {time_orig:.3f}s")

        # Test vectorized implementation
        print("  Vectorized implementation:")
        start_time = time.time()
        result_vec, _ = calculate_correlators_vectorized(k_val, SPRa, tau_grid, weighting, nmodes)
        time_vec = time.time() - start_time
        print(f"    Time: {time_vec:.3f}s")

        # Compare results
        speedup = time_orig / time_vec if time_vec > 0 else float('inf')
        print(f"    Speedup: {speedup:.2f}x")

        # Quick accuracy check
        orig_evals = result_orig['eigenvalues_S']
        vec_evals = result_vec['eigenvalues_S']
        if orig_evals is not None and vec_evals is not None:
            max_diff = np.max(np.abs(orig_evals - vec_evals))
            print(f"    Max eigenvalue difference: {max_diff:.2e}")

if __name__ == "__main__":
    test_vectorized_performance()