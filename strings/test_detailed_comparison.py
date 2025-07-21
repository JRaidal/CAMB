#!/usr/bin/env python3
"""
Detailed diagnostic script to identify sources of differences between implementations
"""

import numpy as np
import sys
import os

# Add the strings directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from string_correlators_fast import _get_correlator_pair, _TAU, _XI, _V, _lookup, build_uetc_mats, mu, alpha, L
from string_correlators import get_correlators, SPRa, xi_interp, v_interp

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available - plotting functions disabled")

def compare_vos_lookups(tau_values):
    """
    Compare VOS parameter lookups between implementations
    """
    print("Comparing VOS parameter lookups...")
    print("τ [Mpc]      | ξ (fast)   | ξ (orig)   | v (fast)   | v (orig)   | Δξ         | Δv")
    print("-" * 80)

    for tau in tau_values:
        if tau < _TAU[0] or tau > _TAU[-1]:
            continue

        # Fast implementation lookups
        xi_fast = _lookup(_XI, tau)
        v_fast = _lookup(_V, tau)

        # Original implementation lookups
        xi_orig = xi_interp(tau)
        v_orig = v_interp(tau)

        diff_xi = abs(xi_fast - xi_orig)
        diff_v = abs(v_fast - v_orig)

        print(f"{tau:.2e} | {xi_fast:.6f} | {xi_orig:.6f} | {v_fast:.6f} | {v_orig:.6f} | {diff_xi:.2e} | {diff_v:.2e}")

def detailed_correlator_comparison(tau1, tau2, k):
    """
    Detailed step-by-step comparison of correlator calculation
    """
    print(f"\nDetailed comparison for τ₁={tau1:.2e}, τ₂={tau2:.2e}, k={k:.2e}")
    print("=" * 70)

    # String parameters
    mu1, mu2 = SPRa.mu, SPRa.mu
    alpha1, alpha2 = SPRa.alpha, SPRa.alpha
    L = SPRa.L

    # VOS parameters - fast
    v1_fast = _lookup(_V, tau1)
    xi1_fast = _lookup(_XI, tau1)
    v2_fast = _lookup(_V, tau2)
    xi2_fast = _lookup(_XI, tau2)

    # VOS parameters - original
    v1_orig = v_interp(tau1)
    xi1_orig = xi_interp(tau1)
    v2_orig = v_interp(tau2)
    xi2_orig = xi_interp(tau2)

    print(f"VOS parameters:")
    print(f"  v₁: fast={v1_fast:.8f}, orig={v1_orig:.8f}, diff={abs(v1_fast-v1_orig):.2e}")
    print(f"  ξ₁: fast={xi1_fast:.8f}, orig={xi1_orig:.8f}, diff={abs(xi1_fast-xi1_orig):.2e}")
    print(f"  v₂: fast={v2_fast:.8f}, orig={v2_orig:.8f}, diff={abs(v2_fast-v2_orig):.2e}")
    print(f"  ξ₂: fast={xi2_fast:.8f}, orig={xi2_orig:.8f}, diff={abs(xi2_fast-xi2_orig):.2e}")

    # Calculate x and rho
    x1_fast = k * tau1 * xi1_fast
    x2_fast = k * tau2 * xi2_fast
    x1_orig = k * tau1 * xi1_orig
    x2_orig = k * tau2 * xi2_orig

    rho_fast = k * abs(v1_fast * tau1 - v2_fast * tau2)
    rho_orig = k * abs(v1_orig * tau1 - v2_orig * tau2)

    print(f"\nDimensionless parameters:")
    print(f"  x₁: fast={x1_fast:.8f}, orig={x1_orig:.8f}, diff={abs(x1_fast-x1_orig):.2e}")
    print(f"  x₂: fast={x2_fast:.8f}, orig={x2_orig:.8f}, diff={abs(x2_fast-x2_orig):.2e}")
    print(f"  ρ:  fast={rho_fast:.8f}, orig={rho_orig:.8f}, diff={abs(rho_fast-rho_orig):.2e}")

    # Identify regime
    xmin = 0.15
    etcmin = 1e-3

    regime_fast = "unknown"
    if x1_fast <= xmin and x2_fast <= xmin:
        regime_fast = "small x"
    elif abs(x1_fast - x2_fast) <= etcmin:
        regime_fast = "ETC"
    else:
        regime_fast = "general case"

    regime_orig = "unknown"
    if x1_orig <= xmin and x2_orig <= xmin:
        regime_orig = "small x"
    elif abs(x1_orig - x2_orig) <= etcmin:
        regime_orig = "ETC"
    else:
        regime_orig = "general case"

    print(f"\nRegime identification:")
    print(f"  Fast: {regime_fast}")
    print(f"  Orig: {regime_orig}")

    # Get final results
    result_fast = _get_correlator_pair(tau1, tau2, k, mu1, alpha1, mu2, alpha2, L)
    result_orig = get_correlators(tau1, tau2, k, SPRa)

    print(f"\nFinal results:")
    component_names = ["00", "S", "V", "T", "00S"]
    for i, name in enumerate(component_names):
        fast_val = result_fast[i]
        orig_val = result_orig[i]
        rel_err = abs(fast_val - orig_val) / (abs(orig_val) + 1e-20) if orig_val != 0 else abs(fast_val)
        print(f"  {name:3s}: fast={fast_val:.8e}, orig={orig_val:.8e}, rel_err={rel_err:.2e}")

def test_scaling_factor():
    """
    Test the scaling factor implementation
    """
    print("\nTesting scaling factor...")

    from string_correlators_fast import _scaling_factor
    from string_correlators import scaling_factor

    test_cases = [
        (1e0, 1e0, 0.1, 0.1, 0.95),
        (1e1, 5e1, 0.13, 0.14, 0.95),
        (1e-1, 1e-1, 0.12, 0.12, 0.95),
    ]

    for tau1, tau2, xi1, xi2, L in test_cases:
        fast_sf = _scaling_factor(tau1, tau2, xi1, xi2, L)
        orig_sf = scaling_factor(tau1, tau2, xi1, xi2, L)
        diff = abs(fast_sf - orig_sf)
        print(f"  τ₁={tau1:.1e}, τ₂={tau2:.1e}: fast={fast_sf:.6e}, orig={orig_sf:.6e}, diff={diff:.2e}")

def plot_uetc_comparison(k=1.0, nktau=128, plot_filename="uetc_comparison.png"):
    """
    Plot UETC matrices side by side with difference plots
    5 rows (one for each UETC type), 3 columns (fast, original, difference)
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not available")
        return

    print(f"\nGenerating UETC comparison plots for k = {k:.1f} Mpc⁻¹...")

    # Create tau grid
    ktau_min, ktau_max = 1e-4, 1e3
    ktau_grid = np.logspace(np.log10(ktau_min), np.log10(ktau_max), nktau)
    tau_vec = ktau_grid / k

    print(f"Using {nktau} tau points from {tau_vec[0]:.2e} to {tau_vec[-1]:.2e} Mpc")

    # Build matrices using fast implementation
    print("Building matrices with fast implementation...")
    mats_fast = build_uetc_mats(tau_vec, k, mu, alpha, L)
    m00_fast, mS_fast, mV_fast, mT_fast, m00S_fast = mats_fast

    # Build matrices using original implementation
    print("Building matrices with original implementation...")
    n = len(tau_vec)
    m00_orig = np.zeros((n, n))
    mS_orig = np.zeros((n, n))
    mV_orig = np.zeros((n, n))
    mT_orig = np.zeros((n, n))
    m00S_orig = np.zeros((n, n))

    for i in range(n):
        tau1 = tau_vec[i]
        if i % 20 == 0:  # Progress indicator
            print(f"  Progress: {i}/{n} ({100*i/n:.1f}%)")
        for j in range(i, n):  # Only compute upper triangle, use symmetry
            tau2 = tau_vec[j]
            uetc_orig = get_correlators(tau1, tau2, k, SPRa)

            m00_orig[i, j] = uetc_orig[0]
            mS_orig[i, j] = uetc_orig[1]
            mV_orig[i, j] = uetc_orig[2]
            mT_orig[i, j] = uetc_orig[3]
            m00S_orig[i, j] = uetc_orig[4]

            if i != j:  # Fill lower triangle
                m00_orig[j, i] = uetc_orig[0]
                mS_orig[j, i] = uetc_orig[1]
                mV_orig[j, i] = uetc_orig[2]
                mT_orig[j, i] = uetc_orig[3]
                m00S_orig[j, i] = uetc_orig[4]

    print("Matrices built. Creating plots...")

    # Apply display scaling (like in the plotting functions)
    mu_sq = mu**2
    def apply_scaling(mat):
        mat_scaled = np.zeros_like(mat)
        for i_tau in range(n):
            tau1 = tau_vec[i_tau]
            for j_tau in range(n):
                tau2 = tau_vec[j_tau]
                plot_display_scaling = (tau1 * tau2)**0.5 / mu_sq if mu_sq != 0 else 0
                mat_scaled[i_tau, j_tau] = mat[i_tau, j_tau] * plot_display_scaling
        return mat_scaled

    # Scale all matrices
    matrices_fast = [
        apply_scaling(m00_fast),
        apply_scaling(mS_fast),
        apply_scaling(mV_fast),
        apply_scaling(mT_fast),
        apply_scaling(m00S_fast)
    ]

    matrices_orig = [
        apply_scaling(m00_orig),
        apply_scaling(mS_orig),
        apply_scaling(mV_orig),
        apply_scaling(mT_orig),
        apply_scaling(m00S_orig)
    ]

    # Create log axes for contour plots
    log_tau_axis = np.log10(tau_vec)
    log_tau_grid_x, log_tau_grid_y = np.meshgrid(log_tau_axis, log_tau_axis, indexing='ij')

    # Matrix names
    matrix_names = ['C₀₀', 'Cₛ', 'Cᵥ', 'Cₜ', 'C₀₀,ₛ']

    # Create the figure
    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    fig.suptitle(f'UETC Matrix Comparison: k = {k:.1f} Mpc⁻¹', fontsize=16)

    n_levels = 50

    for row in range(5):  # 5 UETC types
        mat_fast = matrices_fast[row]
        mat_orig = matrices_orig[row]
        mat_diff = mat_fast - mat_orig

        matrices_row = [mat_fast, mat_orig, mat_diff]
        titles = [f'{matrix_names[row]} (Fast)', f'{matrix_names[row]} (Original)', f'{matrix_names[row]} (Difference)']
        cmaps = ['jet', 'jet', 'RdBu_r']

        for col in range(3):  # 3 columns: fast, orig, diff
            ax = axes[row, col]
            mat = matrices_row[col]

            # Mask invalid values
            data_plot = np.ma.masked_invalid(mat)

            if not data_plot.mask.all():
                if col == 2:  # Difference plot - use symmetric range
                    vmax = np.max(np.abs(data_plot))
                    vmin = -vmax
                else:  # Regular plots
                    vmin, vmax = np.nanmin(data_plot), np.nanmax(data_plot)
                    if vmin == vmax:
                        vmin -= abs(vmin * 0.1) if vmin != 0 else 0.1
                        vmax += abs(vmax * 0.1) if vmax != 0 else 0.1
                    if vmin == vmax:
                        vmin, vmax = vmin - 0.1, vmax + 0.1

                if vmax > vmin:
                    levels = np.linspace(vmin, vmax, n_levels)
                    contour = ax.contourf(log_tau_grid_x, log_tau_grid_y, data_plot,
                                         levels=levels, cmap=cmaps[col], vmin=vmin, vmax=vmax, extend='both')
                    plt.colorbar(contour, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                else:
                    ax.contourf(log_tau_grid_x, log_tau_grid_y, np.zeros_like(mat),
                               levels=1, cmap='gray')
            else:
                ax.contourf(log_tau_grid_x, log_tau_grid_y, np.zeros_like(mat),
                           levels=1, cmap='gray')

            ax.set_xlabel(r'$\log_{10}(\tau_1)$ [Mpc]')
            ax.set_ylabel(r'$\log_{10}(\tau_2)$ [Mpc]')
            ax.set_title(titles[col], fontsize=10)
            ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_filename}")

    # Print some statistics
    print(f"\nMatrix comparison statistics:")
    for i, name in enumerate(matrix_names):
        mat_fast = matrices_fast[i]
        mat_orig = matrices_orig[i]
        mat_diff = mat_fast - mat_orig

        # Compute relative error
        rel_err = np.abs(mat_diff) / (np.abs(mat_orig) + 1e-20)

        print(f"  {name:4s}: max_abs_diff={np.max(np.abs(mat_diff)):.2e}, "
              f"mean_rel_err={np.mean(rel_err):.2e}, max_rel_err={np.max(rel_err):.2e}")

    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Detailed comparison of UETC implementations')
    parser.add_argument('--plot', action='store_true',
                       help='Generate UETC comparison plots')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only generate plots, skip other tests')
    parser.add_argument('--k', type=float, default=1.0,
                       help='k value for plotting (default: 1.0 Mpc⁻¹)')
    parser.add_argument('--nktau', type=int, default=64,
                       help='Number of ktau points for plotting (default: 64)')

    args = parser.parse_args()

    print("DETAILED CORRELATOR COMPARISON")
    print("=" * 70)

    if args.plot_only:
        # Only generate plots
        plot_uetc_comparison(k=args.k, nktau=args.nktau)
    else:
        # Run all tests
        # Test VOS lookups
        tau_test_values = [1e-2, 1e-1, 1e0, 1e1, 1e2]
        compare_vos_lookups(tau_test_values)

        # Test scaling factor
        test_scaling_factor()

        # Detailed comparison for a few cases
        test_cases = [
            (1e-2, 1e-2, 1e-4),   # Small x regime
            (1e-1, 1e-1, 1e-3),   # ETC regime
            (1e0, 2e0, 1e-2),     # General case
        ]

        for tau1, tau2, k in test_cases:
            if tau1 >= _TAU[0] and tau1 <= _TAU[-1] and tau2 >= _TAU[0] and tau2 <= _TAU[-1]:
                detailed_correlator_comparison(tau1, tau2, k)

        # Generate plots if requested
        if args.plot:
            plot_uetc_comparison(k=args.k, nktau=args.nktau)