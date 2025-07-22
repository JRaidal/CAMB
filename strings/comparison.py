#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def compare_npz_files(file_old, file_new, plot_output_dir='comparison_plots'):
    """
    Compares two .npz files containing UETC correlator table data.

    Args:
        file_old (str): Path to the .npz file from the old script.
        file_new (str): Path to the .npz file from the new/fast script.
        plot_output_dir (str): Directory to save comparison plots.
    """
    print(f"--- Comparing UETC Tables ---")
    print(f"Old (Baseline): {file_old}")
    print(f"New (Optimized): {file_new}\n")

    try:
        data_old = np.load(file_old)
        data_new = np.load(file_new)
    except FileNotFoundError as e:
        print(f"Error: Could not open one of the files. {e}")
        return

    # --- 1. Key and Shape Comparison ---
    print("\n--- 1. Structure and Shape Sanity Checks ---")
    keys_old = set(data_old.keys())
    keys_new = set(data_new.keys())

    if keys_old != keys_new:
        print("Warning: Keys do not match!")
        print(f"  Keys only in old file: {keys_old - keys_new}")
        print(f"  Keys only in new file: {keys_new - keys_old}")

    common_keys = sorted(list(keys_old.intersection(keys_new)))
    print(f"Found {len(common_keys)} common keys to compare.")

    all_shapes_match = True
    for key in common_keys:
        shape_old = data_old[key].shape
        shape_new = data_new[key].shape
        if shape_old != shape_new:
            print(f"  Shape mismatch for key '{key}': Old={shape_old}, New={shape_new}")
            all_shapes_match = False

    if all_shapes_match:
        print("All common arrays have matching shapes.")

    # --- 2. Numerical Comparison ---
    print("\n--- 2. Numerical Difference Statistics ---")
    for key in common_keys:
        arr_old = data_old[key]
        arr_new = data_new[key]

        if not np.issubdtype(arr_old.dtype, np.number):
            if np.array_equal(arr_old, arr_new):
                print(f"'{key}': Values are identical.")
            else:
                print(f"'{key}': Values differ. Old='{arr_old}', New='{arr_new}'")
            continue

        valid_mask = ~np.isnan(arr_old) & ~np.isnan(arr_new)
        if not np.any(valid_mask):
            print(f"'{key}': No valid (non-NaN) data to compare.")
            continue

        abs_diff = np.abs(arr_new[valid_mask] - arr_old[valid_mask])

        nonzero_mask = np.abs(arr_old[valid_mask]) > 1e-12
        rel_diff = np.zeros_like(abs_diff)
        if np.any(nonzero_mask):
            rel_diff[nonzero_mask] = abs_diff[nonzero_mask] / np.abs(arr_old[valid_mask][nonzero_mask])

        print(f"\nComparing array: '{key}' (shape: {arr_old.shape})")
        print(f"  Max Absolute Difference: {np.max(abs_diff):.3e}")
        print(f"  Mean Absolute Difference: {np.mean(abs_diff):.3e}")
        if np.any(nonzero_mask):
            print(f"  Max Relative Difference: {np.max(rel_diff):.3e}")
            print(f"  Mean Relative Difference: {np.mean(rel_diff):.3e}")
        else:
            print(f"  Relative Difference: (all old values are near zero)")

    # --- 3. Visual Comparison ---
    print(f"\n--- 3. Generating Visual Comparison Plots ---")
    if not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)
        print(f"Created directory for plots: {plot_output_dir}")

    k_grid = data_old['k_grid']
    ktau_grid = data_old['ktau_grid']
    nk, nmodes_saved = data_old['eigenvalues_S'].shape

    k_indices = [0, nk // 4, nk // 2, nk - 1]
    mode_indices = [0, 1, 3]

    # --- Plot 1: Eigenvalues ---
    fig_evals, axes_evals = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig_evals.suptitle('Eigenvalue Comparison', fontsize=16)

    eval_keys = [('eigenvalues_S', 'Scalar'), ('eigenvalues_V', 'Vector'), ('eigenvalues_T', 'Tensor')]

    for i, (key, title) in enumerate(eval_keys):
        ax_abs = axes_evals[0, i]
        ax_abs.loglog(k_grid, np.abs(data_old[key][:, mode_indices]), 'k--', alpha=0.6, label=[f'Old M{j+1}' for j in mode_indices])
        ax_abs.loglog(k_grid, np.abs(data_new[key][:, mode_indices]), 'r-', alpha=0.6, label=[f'New M{j+1}' for j in mode_indices])
        ax_abs.set_title(f'|{title} Eigenvalues|')
        ax_abs.set_xlabel('k [Mpc⁻¹]')
        ax_abs.set_ylabel(r'$|\lambda|$')
        ax_abs.legend(fontsize='small')
        ax_abs.grid(True, which='both', linestyle=':', alpha=0.6)

        ax_rel = axes_evals[1, i]
        rel_diff_evals = np.abs(data_new[key] - data_old[key]) / np.abs(data_old[key])
        ax_rel.loglog(k_grid, rel_diff_evals[:, mode_indices])
        ax_rel.set_title(f'Relative Difference in {title} Eigenvalues')
        ax_rel.set_xlabel('k [Mpc⁻¹]')
        ax_rel.set_ylabel(r'$|\lambda_{new} - \lambda_{old}| / |\lambda_{old}|$')
        ax_rel.grid(True, which='both', linestyle=':', alpha=0.6)

    plot_path = os.path.join(plot_output_dir, 'eigenvalue_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved eigenvalue comparison plot to: {plot_path}")
    plt.close(fig_evals)

    # --- Plot 2: Eigenvectors and their derivatives ---
    efuncs_old = data_old['eigenfunctions']
    efuncs_new = data_new['eigenfunctions']
    derivs_old = data_old.get('eigenfunctions_d_dlogkt')
    derivs_new = data_new.get('eigenfunctions_d_dlogkt')

    type_names = ['00', 'S', 'V', 'T']

    for k_idx in k_indices:
        k_val = k_grid[k_idx]
        fig_evecs, axes_evecs = plt.subplots(4, 3, figsize=(18, 20), constrained_layout=True)
        fig_evecs.suptitle(f'Eigenfunction & Derivative Comparison at k = {k_val:.2e} Mpc⁻¹', fontsize=16)

        for type_idx in range(4):
            for mode_idx_plot, mode_idx in enumerate(mode_indices):
                ax = axes_evecs[type_idx, mode_idx_plot]

                ax.semilogx(ktau_grid, efuncs_old[k_idx, type_idx, mode_idx, :], 'k--', label='Old u(k,τ)')
                ax.semilogx(ktau_grid, efuncs_new[k_idx, type_idx, mode_idx, :], 'r-', alpha=0.7, label='New u(k,τ)')

                if derivs_old is not None and derivs_new is not None and type_idx < 2:
                    ax.semilogx(ktau_grid, derivs_old[k_idx, type_idx, mode_idx, :], 'b--', label='Old du/dlog(kτ)')
                    ax.semilogx(ktau_grid, derivs_new[k_idx, type_idx, mode_idx, :], 'g-', alpha=0.7, label='New du/dlog(kτ)')

                ax.set_title(f'Type: {type_names[type_idx]}, Mode: {mode_idx+1}')
                ax.set_xlabel('kτ')
                ax.grid(True, linestyle=':', alpha=0.6)
                if type_idx == 0 and mode_idx_plot == 0:
                    ax.legend(fontsize='small')

        plot_path = os.path.join(plot_output_dir, f'eigenfunction_comparison_k_idx_{k_idx}.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Saved eigenfunction comparison plot to: {plot_path}")
        plt.close(fig_evecs)

    print("\n--- Comparison Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two UETC correlator .npz files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'file_old',
        default='correlator_table.npz',
        nargs='?',
        help='Path to the .npz file from the original/slow script.'
    )
    parser.add_argument(
        'file_new',
        default='correlator_table_fast.npz',
        nargs='?',
        help='Path to the .npz file from the optimized/fast script.'
    )
    parser.add_argument(
        '--outdir',
        default='comparison_plots',
        help='Directory to save the output comparison plots.'
    )
    args = parser.parse_args()

    if not os.path.exists(args.file_old) or not os.path.exists(args.file_new):
        print("Error: One or both of the default input files were not found.")
        print("Please specify the paths explicitly, e.g.:")
        print(f"python {os.path.basename(__file__)} path/to/old_file.npz path/to/new_file.npz")
    else:
        compare_npz_files(args.file_old, args.file_new, args.outdir)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparison Script for UETC Eigenvectors
=======================================
This script loads the output tables generated by the "slow" and "fast"
correlator scripts and plots a comparison of their eigenvectors as a
function of the wavenumber k.

This allows for a direct visual confirmation that the sign-alignment
procedure in the fast code works correctly and that the numerical
results from both scripts are identical.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
SLOW_FILE = 'correlator_table.npz'
FAST_FILE = 'correlator_table_fast.npz'

# Select which modes to plot (e.g., 0, 1, 4 for modes 1, 2, and 5)
MODES_TO_PLOT = [0, 1, 4]

# Select which k*tau indices to sample for the plot.
# We pick one early, one middle, and one late time.
KTAU_INDICES_TO_PLOT = [30, 128, 220]

EIGEN_TYPE_LABELS = [
    "Scalar (00 component)",
    "Scalar (S component)",
    "Vector",
    "Tensor"
]

# --- Main Plotting Function ---

def plot_eigenvector_vs_k_comparison(slow_data, fast_data):
    """
    Generates plots comparing eigenvectors from the two data files.

    Each plot shows u_i(k, τ) vs. k for a fixed mode i and several fixed τ values.
    """
    k_grid = slow_data['k_grid']
    ktau_grid = slow_data['ktau_grid']
    nk = k_grid.size

    # Check for consistency between the files
    if not np.allclose(k_grid, fast_data['k_grid']):
        print("Error: k-grids do not match between the two files.")
        return

    # Create a figure with subplots: 4 rows (types) x N columns (modes)
    n_types = len(EIGEN_TYPE_LABELS)
    n_modes_plot = len(MODES_TO_PLOT)
    fig, axes = plt.subplots(
        n_types,
        n_modes_plot,
        figsize=(5 * n_modes_plot, 4 * n_types),
        squeeze=False,
        sharex=True
    )
    fig.suptitle("Comparison of Eigenvectors vs. Wavenumber (k)\n(Slow Script vs. Fast Script)", fontsize=16)

    # Extract eigenvector data
    efuncs_slow = slow_data['eigenfunctions']
    efuncs_fast = fast_data['eigenfunctions']

    # Loop over each type and mode to create the plots
    for type_idx in range(n_types):
        for plot_col_idx, mode_idx in enumerate(MODES_TO_PLOT):
            ax = axes[type_idx, plot_col_idx]

            # Plot a line for each selected k*tau index
            for ktau_idx in KTAU_INDICES_TO_PLOT:
                # Get the specific k*tau value for the legend
                ktau_val = ktau_grid[ktau_idx]

                # Extract the eigenvector values across all k for this mode and ktau_idx
                # Shape of slice will be (nk,)
                u_vs_k_slow = efuncs_slow[:, type_idx, mode_idx, ktau_idx]
                u_vs_k_fast = efuncs_fast[:, type_idx, mode_idx, ktau_idx]

                # Plot slow data as solid line, fast data as dashed
                ax.plot(
                    k_grid,
                    u_vs_k_slow,
                    label=f'kτ={ktau_val:.1f} (Slow)',
                    lw=2.5,
                    alpha=0.8
                )
                ax.plot(
                    k_grid,
                    u_vs_k_fast,
                    ls='--',
                    label=f'kτ={ktau_val:.1f} (Fast)',
                    lw=1.5,
                    alpha=1.0
                )

            # --- Formatting ---
            ax.set_xscale('log')
            ax.grid(True, which='both', linestyle=':', alpha=0.6)
            ax.set_title(f"{EIGEN_TYPE_LABELS[type_idx]}, Mode {mode_idx + 1}")

            if type_idx == n_types - 1:
                ax.set_xlabel("Wavenumber k [Mpc⁻¹]")
            if plot_col_idx == 0:
                ax.set_ylabel("Eigenvector value u(k,τ)")

            # Create a single combined legend for the last subplot in the row
            if plot_col_idx == n_modes_plot -1:
                 # Get handles and labels from one of the plots to build a clean legend
                handles, labels = axes[type_idx, 0].get_legend_handles_labels()
                # Manually create a clean legend
                clean_handles = handles[0::2] + handles[1::2] # Group by slow/fast
                clean_labels = [f'kτ={ktau_grid[i]:.1f}' for i in KTAU_INDICES_TO_PLOT] * 2
                # This is a bit complex; a simpler legend might be better
                ax.legend(handles, labels, fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')


    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.show()


# --- Main Execution ---

if __name__ == "__main__":
    print("--- Running Eigenvector Comparison Script ---")

    # Check if input files exist
    if not os.path.exists(SLOW_FILE) or not os.path.exists(FAST_FILE):
        print("\nError: Make sure both output files exist in this directory:")
        print(f"  - '{SLOW_FILE}' (from the slow script)")
        print(f"  - '{FAST_FILE}' (from the fast script)")
        exit()

    # Load the data from both files
    print(f"Loading data from '{SLOW_FILE}'...")
    data_slow = np.load(SLOW_FILE)

    print(f"Loading data from '{FAST_FILE}'...")
    data_fast = np.load(FAST_FILE)
    print("Data loaded successfully.")

    # Generate the comparison plots
    print("Generating plots...")
    plot_eigenvector_vs_k_comparison(data_slow, data_fast)

    print("\n--- Script Finished ---")