#!/usr/bin/env python3
"""
Fully vectorized version of get_correlators
"""

import numpy as np
import time
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from string_correlators import *

def get_correlators_fully_vectorized(tau1_vec, tau2_vec, k, SPR_base):
    """
    Fully vectorized version of get_correlators - processes all points simultaneously.
    """

    # Ensure inputs are arrays
    tau1_vec = np.asarray(tau1_vec)
    tau2_vec = np.asarray(tau2_vec)

    if tau1_vec.shape != tau2_vec.shape:
        raise ValueError("tau1_vec and tau2_vec must have the same shape")

    original_shape = tau1_vec.shape
    tau1_flat = tau1_vec.flatten()
    tau2_flat = tau2_vec.flatten()
    n_points = len(tau1_flat)

    # Get parameters
    alpha1 = SPR_base.alpha
    alpha2 = SPR_base.alpha
    mu1 = SPR_base.mu
    mu2 = SPR_base.mu
    L_decay = SPR_base.L

    # Vectorized VOS parameter calculation
    v1_vec = np.array([v_interp(tau1) for tau1 in tau1_flat])
    xi1_vec = np.array([xi_interp(tau1) for tau1 in tau1_flat])
    v2_vec = np.array([v_interp(tau2) for tau2 in tau2_flat])
    xi2_vec = np.array([xi_interp(tau2) for tau2 in tau2_flat])

    # Vectorized calculations
    x1_vec = k * tau1_flat * xi1_vec
    x2_vec = k * tau2_flat * xi2_vec
    xp_vec = (x1_vec + x2_vec) / 2.0
    xm_vec = (x1_vec - x2_vec) / 2.0
    rho_vec = k * np.abs(v1_vec * tau1_flat - v2_vec * tau2_flat)
    rho_safe_vec = np.maximum(rho_vec, 1e-12)

    # Common factors
    norm_denom_sq_vec = (1.0 - v1_vec**2) * (1.0 - v2_vec**2)
    norm_denom_vec = np.sqrt(norm_denom_sq_vec)

    # Scaling factors - vectorized
    if scaling_option == 1:
        sf_vec = np.ones(n_points)
    elif scaling_option == 2:
        denom_vec = np.maximum(np.maximum(xi1_vec * tau1_flat, xi2_vec * tau2_flat), 1e-30)
        sf_vec = 1.0 / (denom_vec**3)
    else:
        sf_vec = np.ones(n_points)

    common_factor_base_vec = sf_vec / (k**2 * norm_denom_vec)
    common_factor_vec = mu1 * mu2 * common_factor_base_vec

    # Determine which regime each point belongs to
    small_x_mask = (x1_vec <= xmin) & (x2_vec <= xmin)
    etc_mask = np.abs(x1_vec - x2_vec) <= etcmin
    general_mask = ~(small_x_mask | etc_mask)

    # Initialize results
    results = np.zeros((n_points, 5))

    # Handle points that need individual processing (small_x and etc regimes)
    individual_mask = small_x_mask | etc_mask
    if np.any(individual_mask):
        for i in np.where(individual_mask)[0]:
            results[i] = get_correlators(tau1_flat[i], tau2_flat[i], k, SPR_base)

    # Process general case points vectorized
    if np.any(general_mask):
        # Extract general case points
        gen_indices = np.where(general_mask)[0]
        n_gen = len(gen_indices)

        if n_gen > 0:
            # Extract vectors for general case
            x1_gen = x1_vec[gen_indices]
            x2_gen = x2_vec[gen_indices]
            xp_gen = xp_vec[gen_indices]
            xm_gen = xm_vec[gen_indices]
            rho_safe_gen = rho_safe_vec[gen_indices]
            v1_gen = v1_vec[gen_indices]
            v2_gen = v2_vec[gen_indices]
            common_factor_gen = common_factor_vec[gen_indices]

            # Vectorized n_terms calculation
            n_terms_raw_gen = np.maximum(min_terms, (scale_terms * xp_gen).astype(int))
            n_terms_gen = np.minimum(n_terms_raw_gen, MAX_N_TERMS)

            # Vectorized regime determination
            use_approx_gen = np.abs(x1_gen - x2_gen) >= xapr
            small_rho_gen = rho_safe_gen < 1e-2

            # Vectorized integral calculations
            I1_gen = np.zeros(n_gen)
            I2_gen = np.zeros(n_gen)
            I3_gen = np.zeros(n_gen)
            I4_gen = np.zeros(n_gen)
            I5_gen = np.zeros(n_gen)
            I6_gen = np.zeros(n_gen)

            # Process approximation cases
            approx_mask = use_approx_gen
            if np.any(approx_mask):
                approx_indices = np.where(approx_mask)[0]
                for i in approx_indices:
                    idx = gen_indices[i]
                    I1_gen[i] = I1_int_a(min(x1_gen[i], x2_gen[i]), rho_safe_gen[i])
                    I4_gen[i] = I4_int_a(min(x1_gen[i], x2_gen[i]), rho_safe_gen[i])

            # Process exact integral cases
            exact_mask = ~approx_mask
            if np.any(exact_mask):
                exact_indices = np.where(exact_mask)[0]
                for i in exact_indices:
                    idx = gen_indices[i]
                    n_terms = int(n_terms_gen[i])
                    I1_gen[i] = I1_int(xm_gen[i], rho_safe_gen[i], n_terms) - I1_int(xp_gen[i], rho_safe_gen[i], n_terms)
                    I4_gen[i] = I4_int(xm_gen[i], rho_safe_gen[i], n_terms) - I4_int(xp_gen[i], rho_safe_gen[i], n_terms)

            # Handle small rho cases
            small_rho_exact_mask = (~approx_mask) & small_rho_gen
            if np.any(small_rho_exact_mask):
                I4_gen[small_rho_exact_mask] = I1_gen[small_rho_exact_mask] / 2.0

            # Calculate I2, I3, I5, I6 for all general case points
            for i in range(n_gen):
                I2_gen[i] = I2_int(xm_gen[i], rho_safe_gen[i]) - I2_int(xp_gen[i], rho_safe_gen[i])
                I3_gen[i] = I3_int(xm_gen[i], rho_safe_gen[i]) - I3_int(xp_gen[i], rho_safe_gen[i])

                if not use_approx_gen[i] and small_rho_gen[i]:
                    I5_gen[i] = I2_gen[i] / 2.0
                    I6_gen[i] = I3_gen[i] / 2.0
                else:
                    I5_gen[i] = I5_int(xm_gen[i], rho_safe_gen[i]) - I5_int(xp_gen[i], rho_safe_gen[i])
                    I6_gen[i] = I6_int(xm_gen[i], rho_safe_gen[i]) - I6_int(xp_gen[i], rho_safe_gen[i])

            # Check for finite integrals
            integrals_gen = np.column_stack([I1_gen, I2_gen, I3_gen, I4_gen, I5_gen, I6_gen])
            finite_mask = np.all(np.isfinite(integrals_gen), axis=1)

            # Only process finite cases
            if np.any(finite_mask):
                finite_indices = np.where(finite_mask)[0]

                # Extract finite cases
                I1_fin = I1_gen[finite_indices]
                I2_fin = I2_gen[finite_indices]
                I3_fin = I3_gen[finite_indices]
                I4_fin = I4_gen[finite_indices]
                I5_fin = I5_gen[finite_indices]
                I6_fin = I6_gen[finite_indices]

                v1_fin = v1_gen[finite_indices]
                v2_fin = v2_gen[finite_indices]
                rho_safe_fin = rho_safe_gen[finite_indices]
                common_factor_fin = common_factor_gen[finite_indices]

                # Vectorized coefficient calculations
                safe_a1a2rho2_fin = np.maximum(2.0 * alpha1 * alpha2 * rho_safe_fin**2, 1e-30)
                safe_a1a2_fin = np.maximum(2.0 * alpha1 * alpha2, 1e-30)

                # 00 component - vectorized
                sum00_fin = 2 * alpha1 * alpha2 * I1_fin
                results_00 = sum00_fin * common_factor_fin

                # S component - vectorized
                c1_S = (-27*(alpha1*alpha2*v1_fin*v2_fin)**2 + rho_safe_fin**2*(1+(-1+2*alpha1**2)*v1_fin**2)*(1+(-1+2*alpha2**2)*v2_fin**2))/safe_a1a2rho2_fin
                c2_S = (-3*(-9*(alpha1*alpha2*v1_fin*v2_fin)**2 + rho_safe_fin**2*(-1+v2_fin**2+v1_fin**2*(1+(-1+(alpha1*alpha2)**2)*v2_fin**2))))/safe_a1a2rho2_fin
                c3_S = (-9*(1+(-1+alpha1**2)*v1_fin**2)*(1+(-1+alpha2**2)*v2_fin**2))/safe_a1a2_fin
                c4_S = (-3*(-(alpha2**2*rho_safe_fin**2*(-1+v1_fin**2)*v2_fin**2)+alpha1**2*v1_fin**2*(-18*alpha2**2*v2_fin**2+rho_safe_fin**2*(1+(-1+4*alpha2**2)*v2_fin**2))))/safe_a1a2rho2_fin
                c5_S = (3*(-(alpha2**2*rho_safe_fin**2*(-1+v1_fin**2)*v2_fin**2)+alpha1**2*v1_fin**2*(-18*alpha2**2*v2_fin**2+rho_safe_fin**2*(1+(-1+4*alpha2**2)*v2_fin**2))))/safe_a1a2rho2_fin
                c6_S = (9*(-(alpha2**2*(-1+v1_fin**2)*v2_fin**2)+alpha1**2*v1_fin**2*(1+(-1+2*alpha2**2)*v2_fin**2)))/safe_a1a2_fin

                sumS_fin = c1_S*I1_fin + c2_S*I2_fin + c3_S*I3_fin + c4_S*I4_fin + c5_S*I5_fin + c6_S*I6_fin
                results_S = sumS_fin * common_factor_fin

                # V component - vectorized
                safe_rho2_local_fin = np.maximum(rho_safe_fin**2, 1e-30)
                safe_a1a2_local_fin = np.maximum(alpha1 * alpha2, 1e-30)

                c1_V = (3*alpha1*alpha2*v1_fin**2*v2_fin**2)/safe_rho2_local_fin
                c2_V = (-3*alpha1*alpha2*v1_fin**2*v2_fin**2)/safe_rho2_local_fin
                c3_V = ((1+(-1+alpha1**2)*v1_fin**2)*(1+(-1+alpha2**2)*v2_fin**2))/safe_a1a2_local_fin
                c4_V = (alpha1*alpha2*(-6+rho_safe_fin**2)*v1_fin**2*v2_fin**2)/safe_rho2_local_fin
                c5_V = -((alpha1*alpha2*(-6+rho_safe_fin**2)*v1_fin**2*v2_fin**2)/safe_rho2_local_fin)
                c6_V = (alpha2**2*(-1+v1_fin**2)*v2_fin**2-alpha1**2*v1_fin**2*(1+(-1+2*alpha2**2)*v2_fin**2))/safe_a1a2_local_fin

                sumV_fin = c1_V*I1_fin + c2_V*I2_fin + c3_V*I3_fin + c4_V*I4_fin + c5_V*I5_fin + c6_V*I6_fin
                results_V = sumV_fin * common_factor_fin

                # T component - vectorized
                safe_4a1a2rho2_fin = np.maximum(4.0 * alpha1 * alpha2 * rho_safe_fin**2, 1e-30)
                safe_4a1a2_fin = np.maximum(4.0 * alpha1 * alpha2, 1e-30)

                c1_T = (-3.0*(alpha1*alpha2*v1_fin*v2_fin)**2+rho_safe_fin**2*(-1.0+v1_fin**2)*(-1.0+v2_fin**2))/safe_4a1a2rho2_fin
                c2_T = (3.0*(alpha1*alpha2*v1_fin*v2_fin)**2+rho_safe_fin**2*(-1.0+v2_fin**2+v1_fin**2*(1.0+(-1.0+(alpha1*alpha2)**2)*v2_fin**2)))/safe_4a1a2rho2_fin
                c3_T = -((1.0+(-1.0+alpha1**2)*v1_fin**2)*(1.0+(-1.0+alpha2**2)*v2_fin**2))/safe_4a1a2_fin
                c4_T = (-(alpha2**2*rho_safe_fin**2*(-1.0+v1_fin**2)*v2_fin**2)+alpha1**2*v1_fin**2*(6.0*alpha2**2*v2_fin**2-rho_safe_fin**2*(-1.0+v2_fin**2)))/safe_4a1a2rho2_fin
                c5_T = (alpha2**2*rho_safe_fin**2*(-1.0+v1_fin**2)*v2_fin**2+alpha1**2*v1_fin**2*(-6.0*alpha2**2*v2_fin**2+rho_safe_fin**2*(-1.0+v2_fin**2)))/safe_4a1a2rho2_fin
                c6_T = (-(alpha2**2*(-1.0+v1_fin**2)*v2_fin**2)+alpha1**2*v1_fin**2*(1.0+(-1.0+2.0*alpha2**2)*v2_fin**2))/safe_4a1a2_fin

                sumT_fin = c1_T*I1_fin + c2_T*I2_fin + c3_T*I3_fin + c4_T*I4_fin + c5_T*I5_fin + c6_T*I6_fin
                results_T = sumT_fin * common_factor_fin

                # 00S cross component - vectorized
                c1_00S = (-(alpha2**2*(-1+v1_fin**2))+alpha1**2*(1-v2_fin**2+2*alpha2**2*(v1_fin**2+v2_fin**2)))/safe_a1a2_fin
                c2_00S = (-3*(-(alpha2**2*(-1+v1_fin**2))+alpha1**2*(1-v2_fin**2+alpha2**2*(v1_fin**2+v2_fin**2))))/safe_a1a2_fin
                c3_00S = np.zeros_like(v1_fin)
                c4_00S = (-3*alpha1*alpha2*(v1_fin**2+v2_fin**2))/2.0
                c5_00S = (3*alpha1*alpha2*(v1_fin**2+v2_fin**2))/2.0
                c6_00S = np.zeros_like(v1_fin)

                sumC_fin = c1_00S*I1_fin + c2_00S*I2_fin + c3_00S*I3_fin + c4_00S*I4_fin + c5_00S*I5_fin + c6_00S*I6_fin
                results_00S = sumC_fin * common_factor_fin

                # Store results
                for i, fin_idx in enumerate(finite_indices):
                    gen_idx = gen_indices[fin_idx]
                    results[gen_idx, 0] = results_00[i]
                    results[gen_idx, 1] = results_S[i]
                    results[gen_idx, 2] = results_V[i]
                    results[gen_idx, 3] = results_T[i]
                    results[gen_idx, 4] = results_00S[i]

    # Reshape back to original shape
    if len(original_shape) == 0:
        return results[0]
    else:
        return results.reshape(original_shape + (5,))

def test_full_vectorization():
    """Test the fully vectorized implementation"""

    print("Testing fully vectorized correlator performance...")
    print("=" * 60)

    # Test parameters
    k_val = 0.1
    ntau_values = [16, 32, 64]

    for ntau in ntau_values:
        print(f"\nTesting with ntau = {ntau}")

        ktau_grid = np.logspace(-2, 2, ntau)
        tau_grid = ktau_grid / k_val
        nmodes = min(16, ntau)

        # Create tau pairs for upper triangle
        tau1_mesh, tau2_mesh = np.meshgrid(tau_grid, tau_grid, indexing='ij')
        upper_triangle_mask = np.triu(np.ones((ntau, ntau), dtype=bool))
        tau1_upper = tau1_mesh[upper_triangle_mask]
        tau2_upper = tau2_mesh[upper_triangle_mask]

        n_pairs = len(tau1_upper)
        print(f"  Number of tau pairs: {n_pairs}")

        # Test original implementation (individual calls)
        print("  Original (individual calls):")
        start_time = time.time()
        results_orig = np.zeros((n_pairs, 5))
        for i in range(n_pairs):
            results_orig[i] = get_correlators(tau1_upper[i], tau2_upper[i], k_val, SPRa)
        time_orig = time.time() - start_time
        print(f"    Time: {time_orig:.3f}s")

        # Test fully vectorized implementation
        print("  Fully vectorized implementation:")
        start_time = time.time()
        results_vec = get_correlators_fully_vectorized(tau1_upper, tau2_upper, k_val, SPRa)
        time_vec = time.time() - start_time
        print(f"    Time: {time_vec:.3f}s")

        # Compare results
        speedup = time_orig / time_vec if time_vec > 0 else float('inf')
        print(f"    Speedup: {speedup:.2f}x")

        # Accuracy check
        max_diff = np.max(np.abs(results_orig - results_vec))
        print(f"    Max difference: {max_diff:.2e}")

if __name__ == "__main__":
    test_full_vectorization()