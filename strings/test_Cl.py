#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import argparse
import time

# Get the absolute path to the CAMB root directory
CAMB_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if CAMB_dir not in sys.path:
    sys.path.insert(0, CAMB_dir)

from camb.active_sources import ActiveSources
import camb
import matplotlib.pyplot as plt

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Calculate CMB power spectra with UETC string sources',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Individual mode flags - can be combined
    parser.add_argument(
        '--scalar', '-s',
        action='store_true',
        help='Enable scalar perturbations'
    )

    parser.add_argument(
        '--vector', '-v',
        action='store_true',
        help='Enable vector perturbations'
    )

    parser.add_argument(
        '--tensor', '-t',
        action='store_true',
        help='Enable tensor perturbations'
    )

    # Number of eigenmodes
    parser.add_argument(
        '--nmodes', '-n',
        type=int,
        default=32,
        help='Number of UETC eigenmodes to sum'
    )

    # Maximum multipole for plotting
    parser.add_argument(
        '--lmax', '-l',
        type=int,
        default=4000,
        help='Maximum multipole for plotting'
    )

    # CMB units
    parser.add_argument(
        '--units', '-u',
        type=str,
        choices=['muK', 'K'],
        default='muK',
        help='CMB units for output'
    )

    # Data file path
    parser.add_argument(
        '--datafile', '-d',
        type=str,
        default=None,
        help='Path to correlator data file (default: correlator_table.npz in script directory)'
    )

    # Output file
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output filename for plot (if not specified, plot is shown)'
    )

    # String tension parameter
    parser.add_argument(
        '--gmu',
        type=float,
        default=1.58e-7,
        help='String tension parameter (default: 2e-7)'
    )

    # Disable plotting
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plotting'
    )

    return parser.parse_args()

def setup_camb_params(args):
    """Setup CAMB parameters based on arguments"""
    # Validate that at least one mode is selected
    if not (args.scalar or args.vector or args.tensor):
        print("WARNING: No modes selected, defaulting to scalar mode")
        args.scalar = True

    enabled_modes = []
    if args.scalar:
        enabled_modes.append('scalar')
    if args.vector:
        enabled_modes.append('vector')
    if args.tensor:
        enabled_modes.append('tensor')

    print(f"Setting up CAMB parameters for {', '.join(enabled_modes)} mode(s)...")
    pars = camb.CAMBparams()

    # Have turned off massive neutrinos to avoid issues with the vector modes
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)

    # Set maximum l values for different modes
    if args.tensor:
        pars.max_l_tensor = 2500
    if args.scalar or args.vector:
        pars.max_l = 4000  # For scalar and vector modes

    # Set mode types based on arguments
    pars.WantScalars = args.scalar
    pars.WantVectors = args.vector
    pars.WantTensors = args.tensor

    if args.vector:
        print("  - Note: Massive neutrinos disabled for vector modes")

    pars.DoLensing = False

    return pars

def load_correlator_data(args):
    """Load UETC correlator data"""
    t_start_load = time.time()
    if args.datafile is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        npz_filename = os.path.join(script_dir, "correlator_table.npz")
    else:
        npz_filename = args.datafile

    print(f"Loading UETC data from: {npz_filename}...")
    correlator_data = np.load(npz_filename)

    print(f"Data loaded. (took {time.time() - t_start_load:.2f}s)")

    return {
        'k_grid': correlator_data['k_grid'],
        'tau_grid': correlator_data['ktau_grid'],
        'eigenfunctions': correlator_data['eigenfunctions'],
        'eigenfunctions_d_dlogkt': correlator_data['eigenfunctions_d_dlogkt'],
        'eigenvalues_S': correlator_data['eigenvalues_S'],
        'eigenvalues_00': correlator_data['eigenvalues_00'],
        'eigenvalues_V': correlator_data['eigenvalues_V'],
        'eigenvalues_T': correlator_data['eigenvalues_T'],
        'string_p_mu': correlator_data['string_params_mu'].item(),
        'nmodes_from_file': correlator_data['nmodes'].item(),
        'weighting_from_file': correlator_data['weighting_gamma'].item()
    }

def setup_active_sources(correlator_data, args):
    """Setup ActiveSources object with correlator data"""
    t_start_setup = time.time()
    print("Initializing custom Fortran object...")
    active_sources = ActiveSources()
    active_sources.set_correlator_table(
        k_grid=correlator_data['k_grid'],
        tau_grid=correlator_data['tau_grid'],
        eigenfunctions=correlator_data['eigenfunctions'],
        eigenfunctions_d_dlogkt=correlator_data['eigenfunctions_d_dlogkt'],
        eigenvalues_S=correlator_data['eigenvalues_S'],
        eigenvalues_00=correlator_data['eigenvalues_00'],
        eigenvalues_V=correlator_data['eigenvalues_V'],
        eigenvalues_T=correlator_data['eigenvalues_T'],
        string_params_mu=correlator_data['string_p_mu'],
        nmodes_param=correlator_data['nmodes_from_file'],
        weighting_param=correlator_data['weighting_from_file']
    )
    print(f"Fortran data transfer successful. (took {time.time() - t_start_setup:.2f}s)")

    return active_sources

def calculate_power_spectra(pars, args):
    """Calculate power spectra for baseline and UETC sources for all polarization components"""

    # Calculate baseline (no UETC sources)
    t_start_baseline = time.time()
    print("\nCalculating baseline C_l (UETC sources OFF)...")
    pars.ActiveSources.set_active_eigenmode(0)
    results_baseline = camb.get_results(pars)

    power_spectra_baseline = results_baseline.get_cmb_power_spectra(pars, CMB_unit=args.units, raw_cl=False)
    cl_baseline_all = power_spectra_baseline['total']  # Shape: (lmax+1, 4) for TT, EE, BB, TE

    lmax_calc = cl_baseline_all.shape[0] - 1
    ls_calc = np.arange(lmax_calc + 1)

    print(f"Baseline C_l calculated up to LMAX={lmax_calc}. (took {time.time() - t_start_baseline:.2f}s)")

    # Calculate UETC sources by summing modes
    correlator_data = load_correlator_data(args)
    actual_n_modes_to_sum = min(args.nmodes, correlator_data['nmodes_from_file'])

    # Determine which modes are enabled
    enabled_modes = []
    if args.scalar:
        enabled_modes.append('scalar')
    if args.vector:
        enabled_modes.append('vector')
    if args.tensor:
        enabled_modes.append('tensor')

    print(f"\nCalculating UETC C_l by summing {actual_n_modes_to_sum} eigenmodes...")
    print(f"Using {', '.join(enabled_modes)} mode(s)")
    t_start_strings = time.time()

    cl_strings_sum_all = np.zeros_like(cl_baseline_all)

    for i_mode in range(1, actual_n_modes_to_sum + 1):
        t_start_mode = time.time()
        print(f"  Processing eigenmode {i_mode}/{actual_n_modes_to_sum}...", end='', flush=True)
        pars.ActiveSources.set_active_eigenmode(i_mode)

        results_mode_i = camb.get_results(pars)

        power_spectra_mode_i = results_mode_i.get_cmb_power_spectra(pars, CMB_unit=args.units, raw_cl=False)

        cl_mode_i_all = power_spectra_mode_i['total']  # Shape: (lmax+1, 4)

        min_len = min(len(cl_strings_sum_all), len(cl_mode_i_all))
        cl_strings_sum_all[:min_len] += cl_mode_i_all[:min_len]

        print(f" (took {time.time() - t_start_mode:.2f}s)")

    pars.ActiveSources.set_active_eigenmode(0)
    print(f"UETC C_l calculation finished. (took {time.time() - t_start_strings:.2f}s)")

    # Return all polarization components
    return ls_calc, cl_baseline_all, cl_strings_sum_all

def plot_results(ls_calc, cl_baseline_all, cl_strings_all, args):
    """Plot the results for all polarization components"""
    if args.no_plot:
        return

    t_start_plot = time.time()
    print("\nPlotting results...")

    # Use a more compatible matplotlib style
    try:
        plt.style.use('seaborn-v0_8-colorblind')
    except:
        try:
            plt.style.use('seaborn-colorblind')
        except:
            pass  # Use default style if seaborn not available

    # Create 4 rows x 2 columns subplot
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))

    plot_mask = (ls_calc >= 1) & (ls_calc <= args.lmax)
    ls_plot = ls_calc[plot_mask]

    # Determine which modes are enabled for plot titles
    enabled_modes = []
    if args.scalar:
        enabled_modes.append('Scalar')
    if args.vector:
        enabled_modes.append('Vector')
    if args.tensor:
        enabled_modes.append('Tensor')
    mode_str = '+'.join(enabled_modes)

    # Polarization components and their indices
    pol_components = ['TT', 'EE', 'TE', 'BB']
    pol_indices = [0, 1, 3, 2]

    units_label = r'\mu K^2' if args.units == 'muK' else r'K^2'

    # Plot each polarization component
    for i, (pol_name, pol_idx) in enumerate(zip(pol_components, pol_indices)):
        # Left column: Strings
        cl_strings_pol = cl_strings_all[plot_mask, pol_idx]
        axs[i, 0].plot(ls_plot, cl_strings_pol*(args.gmu)**2, label=f'Strings ({mode_str})', color='C1', linestyle='-')
        axs[i, 0].set_xlabel(r'$\ell$')
        axs[i, 0].set_ylabel(rf'$\ell(\ell+1)C_\ell^{{{pol_name}}}/2\pi \, [{units_label}]$')
        axs[i, 0].set_xlim([2, args.lmax])
        axs[i, 0].set_xscale('log')
        if pol_name in ['EE', 'BB']:
            axs[i, 0].set_yscale('log')
        axs[i, 0].legend()
        axs[i, 0].grid(True, which="both", ls="-", alpha=0.5)

        # Right column: Baseline
        cl_baseline_pol = cl_baseline_all[plot_mask, pol_idx]
        axs[i, 1].plot(ls_plot, cl_baseline_pol, label=f'Baseline ({mode_str})', color='C0', linestyle='-')
        axs[i, 1].set_xlabel(r'$\ell$')
        axs[i, 1].set_ylabel(rf'$\ell(\ell+1)C_\ell^{{{pol_name}}}/2\pi \, [{units_label}]$')
        axs[i, 1].set_xlim([2, args.lmax])
        if pol_name in ['EE', 'BB']:
            axs[i, 1].set_yscale('log')
        else:
            axs[i, 1].set_ylim(ymin=0)
        axs[i, 1].set_xscale('log')
        axs[i, 1].legend()
        axs[i, 1].grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output)
        print(f"Plot saved to {args.output}")
    else:
        plt.show()
    print(f"Plotting complete. (took {time.time() - t_start_plot:.2f}s)")

def main():
    """Main function"""
    overall_start_time = time.time()
    args = parse_arguments()

    # Determine which modes are enabled
    enabled_modes = []
    if args.scalar:
        enabled_modes.append('scalar')
    if args.vector:
        enabled_modes.append('vector')
    if args.tensor:
        enabled_modes.append('tensor')

    # Default to scalar if no modes specified
    if not enabled_modes:
        print("WARNING: No modes specified, defaulting to scalar mode")
        args.scalar = True
        enabled_modes = ['scalar']

    print(f"Configuration:")
    print(f"  Enabled modes: {', '.join(enabled_modes)}")
    print(f"  Number of modes: {args.nmodes}")
    print(f"  String tension: {args.gmu}")
    print(f"  Max multipole: {args.lmax}")
    print(f"  Units: {args.units}")
    print()

    t_start_setup = time.time()
    print("Setting up CAMB parameters...")

    # Setup CAMB parameters
    pars = setup_camb_params(args)
    print(f"CAMB parameter setup complete. (took {time.time() - t_start_setup:.2f}s)")


    # Load correlator data
    correlator_data = load_correlator_data(args)

    # Setup ActiveSources object
    active_sources = setup_active_sources(correlator_data, args)
    pars.ActiveSources = active_sources

    # Calculate power spectra
    t_start_calc = time.time()
    ls_calc, cl_baseline_all, cl_strings_all = calculate_power_spectra(pars, args)
    print(f"\nTotal power spectra calculation finished. (took {time.time() - t_start_calc:.2f}s)")

    # Plot results
    plot_results(ls_calc, cl_baseline_all, cl_strings_all, args)
    print(f"\nTotal script execution time: {time.time() - overall_start_time:.2f} seconds.")
    print("--- Script Finished ---")

if __name__ == "__main__":
    main()