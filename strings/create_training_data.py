# # create_training_data.py (Corrected)

# import numpy as np
# import time
# from argparse import Namespace
# from tqdm import tqdm

# # Attempt to import pyDOE2, guiding the user if it's not found
# try:
#     import pyDOE2 as pyDOE
# except ImportError:
#     print("Error: pyDOE2 is not installed. Please install it to use LHS.")
#     print("Install using: pip install pyDOE2")
#     exit(1)

# # Import the core simulation function from our other script
# from string_simulation import run_string_simulation

# # ===================================================================== #
# # 1. DEFINE THE PARAMETER SPACE
# # ===================================================================== #

# # --- FIX 1: Correct the parameter sampling strategy ---
# # We now sample Omega_matter and CALCULATE Omega_lambda to ensure a flat universe.
# param_names = ['alpha', 'cr', 'H0', 'Omega_matter']

# # Define the prior ranges [min, max] for each parameter
# param_priors = {
#     'alpha':        [.1, 10],
#     'cr':           [0.05, 1.0],
#     'H0':           [66.4, 68.4],
#     'Omega_matter': [0.301, 0.329],
# }

# # Define the number of training samples to generate
# N_SAMPLES = 1000 # Increased for a more meaningful run, can be set back to 10

# # Fixed numerical/physical parameters that are not sampled
# FIXED_ARGS = {
#     'Omega_rad': 9.24e-5,
#     'tau': 0.06,
#     'scalar': True,
#     'vector': True,
#     'tensor': True,
#     'nmodes': 32,
#     'lmax': 4000,
#     'nk': 100,
#     'nktau': 512,
#     'k_min': 1e-6,
#     'k_max': 10.0,
#     'ktau_min': 1e-4,
#     'ktau_max': 1e3,
#     'weighting_gamma': 0.25,
#     'units': 'muK',
# }

# # ===================================================================== #
# # 2. GENERATE THE LATIN HYPERCUBE
# # ===================================================================== #

# print(f"Generating {N_SAMPLES} samples using Latin Hypercube Sampling...")

# n_params = len(param_names)
# lhd = pyDOE.lhs(n_params, samples=N_SAMPLES, criterion='c')

# # Scale the Latin Hypercube to the defined parameter ranges
# param_samples_dict = {}
# for i, param in enumerate(param_names):
#     min_val, max_val = param_priors[param]
#     param_samples_dict[param] = lhd[:, i] * (max_val - min_val) + min_val

# # --- FIX 1 (continued): Calculate Omega_lambda ---
# # Enforce the flatness constraint: Omega_lambda = 1 - Omega_matter - Omega_rad
# param_samples_dict['Omega_lambda'] = 1.0 - param_samples_dict['Omega_matter'] - FIXED_ARGS['Omega_rad']

# print("LHS grid generated for a flat universe.")

# # ===================================================================== #
# # 3. RUN SIMULATIONS AND COLLECT DATA
# # ===================================================================== #

# successful_params = []
# successful_spectra = []
# l_values = None

# print(f"\nStarting simulation for {N_SAMPLES} parameter sets...")
# start_time = time.time()

# for i in tqdm(range(N_SAMPLES), desc="Running Simulations"):
#     # Combine sampled and fixed parameters into a single Namespace object
#     current_params = {p_name: param_samples_dict[p_name][i] for p_name in param_names + ['Omega_lambda']}
#     args = Namespace(**FIXED_ARGS, **current_params)

#     try:
#         ls_calc, cl_strings_all = run_string_simulation(args, verbose=False)

#         if l_values is None:
#             l_values = ls_calc

#         # --- FIX 2: Add a safety net before taking log10 ---
#         # Add a small epsilon to prevent log(0) = -inf for EE and BB spectra.
#         # Use np.maximum to also guard against small negative numerical noise.
#         epsilon = 1e-30
#         cl_tt = cl_strings_all[:, 0]
#         cl_ee = cl_strings_all[:, 1]
#         cl_bb = cl_strings_all[:, 2]
#         cl_te = cl_strings_all[:, 3]

#         feature_vector = np.concatenate((cl_tt, cl_ee, cl_bb, cl_te))

#         if not np.isfinite(feature_vector).all():
#             # This check will now only trigger on true NaN failures from CAMB
#             print(f"Warning: Sample {i} resulted in hard NaN failure. Skipping.")
#             continue

#         # If successful, append the original sampled parameters and the resulting spectra
#         successful_params.append([param_samples_dict[p][i] for p in param_names])
#         successful_spectra.append(feature_vector)

#     except Exception as e:
#         print(f"\nWarning: Simulation {i} crashed with error: {e}. Skipping.")
#         continue

# end_time = time.time()
# print(f"\nFinished simulations in {end_time - start_time:.2f} seconds.")
# print(f"Successfully generated {len(successful_params)} / {N_SAMPLES} samples.")

# # ===================================================================== #
# # 4. SAVE DATA IN COSMOPOWER FORMAT
# # ===================================================================== #

# if len(successful_params) > 0:
#     final_params_array = np.array(successful_params)
#     final_spectra_array = np.array(successful_spectra)

#     # CosmoPower expects a dictionary of named parameter arrays
#     params_dict = {param_names[i]: final_params_array[:, i] for i in range(n_params)}

#     # The 'modes' for C_ls are the l-values repeated for each spectrum type (TT,EE,BB,TE)
#     l_modes = np.concatenate([l_values] * 4)
#     spectra_dict = {
#         'modes': l_modes,
#         'features': final_spectra_array
#     }

#     params_filename = 'string_training_parameters_high_res.npz'
#     spectra_filename = 'string_training_spectra_high_res.npz'

#     np.savez(params_filename, **params_dict)
#     np.savez(spectra_filename, **spectra_dict)

#     print("\nTraining data successfully saved in CosmoPower format:")
#     print(f"  - Parameters: {params_filename}")
#     print(f"  - Spectra:    {spectra_filename}")
# else:
#     print("\nNo simulations completed successfully. No data was saved.")


# create_training_data_curved.py

import numpy as np
import time
from argparse import Namespace
from tqdm import tqdm

# Attempt to import pyDOE2
try:
    import pyDOE2 as pyDOE
except ImportError:
    print("Error: pyDOE2 is not installed. Please install it to use LHS.")
    print("Install using: pip install pyDOE2")
    exit(1)

# Import the core simulation function
# Ensure 'string_simulation_par.py' or 'string_simulation.py' is available

from string_simulation import run_string_simulation



# ===================================================================== #
# 1. DEFINE THE EXPANDED PARAMETER SPACE
# ===================================================================== #

# <-- CHANGE: Added Omega_rad and Omega_k to the list of sampled parameters -->
param_names = ['alpha', 'cr', 'H0', 'Omega_matter', 'Omega_rad', 'Omega_k']

# Define the prior ranges [min, max] for each parameter
param_priors = {
    'alpha':        [0.05, 10.0],
    'cr':           [0.05, 1.0],
    'H0':           [64.9, 69.9],         # 67.4 +/- (5 * 0.5)
    'Omega_matter': [0.280, 0.350],        # 0.315 +/- (5 * 0.007)
    'Omega_rad':    [8.0e-5, 1.05e-4],      # Corresponds to a wide 5-sigma range on Neff
    'Omega_k':      [-0.01, 0.01],         # Covers the 5-sigma range for curvature
}

# Define the number of training samples to generate
N_SAMPLES = 1200 # Can be adjusted as needed

# Fixed numerical/physical parameters
FIXED_ARGS = {
    # <-- CHANGE: Omega_rad is no longer fixed -->
    'tau': 0.06,
    'scalar': True,
    'vector': True,
    'tensor': True,
    'nmodes': 128,
    'lmax': 4300,
    'nk': 100,
    'nktau': 1024,
    'k_min': 1e-6,
    'k_max': 10.0,
    'ktau_min': 1e-4,
    'ktau_max': 1e3,
    'weighting_gamma': 0.25,
    'units': 'muK',
    # For the parallel version, let's use all available cores
    'num_cores': 16
}

# ===================================================================== #
# 2. GENERATE THE LATIN HYPERCUBE
# ===================================================================== #

print(f"Generating {N_SAMPLES} samples for {len(param_names)} parameters using LHS...")

n_params = len(param_names)
lhd = pyDOE.lhs(n_params, samples=N_SAMPLES, criterion='c')

# Scale the Latin Hypercube to the defined parameter ranges
param_samples_dict = {}
for i, param in enumerate(param_names):
    min_val, max_val = param_priors[param]
    param_samples_dict[param] = lhd[:, i] * (max_val - min_val) + min_val

# <-- CHANGE: Update Omega_lambda calculation for non-flat cosmologies -->
# Enforce the Friedmann constraint: Omega_lambda = 1 - Omega_matter - Omega_rad - Omega_k
param_samples_dict['Omega_lambda'] = (1.0 - param_samples_dict['Omega_matter']
                                      - param_samples_dict['Omega_rad']
                                      - param_samples_dict['Omega_k'])

print("LHS grid generated for variable curvature universes.")

# ===================================================================== #
# 3. RUN SIMULATIONS AND COLLECT DATA
# ===================================================================== #

successful_params = []
successful_spectra = []
l_values = None

print(f"\nStarting simulation for {N_SAMPLES} parameter sets...")
start_time = time.time()

for i in tqdm(range(N_SAMPLES), desc="Running Simulations"):
    # Combine sampled and fixed parameters into a single Namespace object
    current_params = {p_name: param_samples_dict[p_name][i] for p_name in param_names + ['Omega_lambda']}
    args = Namespace(**FIXED_ARGS, **current_params)

    try:
        # Run simulation with verbose=False to keep the progress bar clean
        ls_calc, cl_strings_all = run_string_simulation(args, verbose=False)

        if l_values is None:
            l_values = ls_calc

        # We need to ensure the output C_l array has the expected length.
        # It can sometimes be off by one, so we will align it.
        expected_len = args.lmax + 1
        if len(cl_strings_all) < expected_len:
             # Pad with zeros if it's too short
            padding = np.zeros((expected_len - len(cl_strings_all), cl_strings_all.shape[1]))
            cl_strings_all = np.vstack([cl_strings_all, padding])
        elif len(cl_strings_all) > expected_len:
            # Truncate if it's too long
            cl_strings_all = cl_strings_all[:expected_len]

        # Concatenate all polarization modes into a single feature vector
        feature_vector = cl_strings_all.flatten(order='F') # Flattens column by column (TT, then EE, etc.)

        if not np.isfinite(feature_vector).all():
            print(f"Warning: Sample {i} resulted in NaN/inf failure. Skipping.")
            continue

        # If successful, append the original SAMPLED parameters and the resulting spectra
        successful_params.append([param_samples_dict[p][i] for p in param_names])
        successful_spectra.append(feature_vector)

    except Exception as e:
        print(f"\nWarning: Simulation {i} crashed with error: {e}. Skipping.")
        continue

end_time = time.time()
print(f"\nFinished simulations in {end_time - start_time:.2f} seconds.")
print(f"Successfully generated {len(successful_params)} / {N_SAMPLES} samples.")

# ===================================================================== #
# 4. SAVE DATA IN COSMOPOWER FORMAT
# ===================================================================== #

if len(successful_params) > 0:
    final_params_array = np.array(successful_params)
    final_spectra_array = np.array(successful_spectra)

    # CosmoPower expects a dictionary of named parameter arrays
    params_dict = {param_names[i]: final_params_array[:, i] for i in range(n_params)}

    # The 'modes' for C_ls are the l-values repeated for each spectrum type
    l_modes = np.concatenate([l_values] * 4)
    spectra_dict = {
        'modes': l_modes,
        'features': final_spectra_array
    }

    # Use a new name for the output files
    params_filename = 'string_training_params_final_1.npz'
    spectra_filename = 'string_training_spectra_final_1.npz'

    np.savez(params_filename, **params_dict)
    np.savez(spectra_filename, **spectra_dict)

    print("\nTraining data successfully saved in CosmoPower format:")
    print(f"  - Parameters: {params_filename}")
    print(f"  - Spectra:    {spectra_filename}")
else:
    print("\nNo simulations completed successfully. No data was saved.")