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
from string_simulation import run_string_simulation

# ===================================================================== #
# 1. DEFINE THE EXPANDED PARAMETER SPACE
# ===================================================================== #

param_names = ['alpha', 'cr', 'H0', 'Omega_matter', 'Omega_rad', 'Omega_k']

# Define the prior ranges [min, max] for each parameter
param_priors = {
    'alpha':        [0.05, 10.0],
    'cr':           [0.05, 1.0],
    'H0':           [64.9, 69.9],         
    'Omega_matter': [0.280, 0.350],        
    'Omega_rad':    [8.0e-5, 1.05e-4],     
    'Omega_k':      [-0.01, 0.01],        
}

# Define the number of training samples to generate
N_SAMPLES = 1200

# Fixed numerical/physical parameters
FIXED_ARGS = {
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
        ls_calc, cl_strings_all = run_string_simulation(args, verbose=False)

        if l_values is None:
            l_values = ls_calc

        expected_len = args.lmax + 1
        if len(cl_strings_all) < expected_len:
            padding = np.zeros((expected_len - len(cl_strings_all), cl_strings_all.shape[1]))
            cl_strings_all = np.vstack([cl_strings_all, padding])
        elif len(cl_strings_all) > expected_len:
            cl_strings_all = cl_strings_all[:expected_len]

        # Concatenate all polarization modes into a single feature vector
        feature_vector = cl_strings_all.flatten(order='F')

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

    params_dict = {param_names[i]: final_params_array[:, i] for i in range(n_params)}

    l_modes = np.concatenate([l_values] * 4)
    spectra_dict = {
        'modes': l_modes,
        'features': final_spectra_array
    }

    # Use a new name for the output files
    params_filename = 'string_training_params.npz'
    spectra_filename = 'string_training_spectra.npz'

    np.savez(params_filename, **params_dict)
    np.savez(spectra_filename, **spectra_dict)

    print("\nTraining data successfully saved in:")
    print(f"  - Parameters: {params_filename}")
    print(f"  - Spectra:    {spectra_filename}")
else:
    print("\nNo simulations completed successfully. No data was saved.")