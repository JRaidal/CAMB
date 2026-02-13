# Cosmic String Implementation

This directory contains a high-performance pipeline for computing the contribution of cosmic string networks to CMB power spectra. It utilizes analytic Unequal Time Correlator (UETC) approximations (arXiv:1603.01275v2), whose eigenvectors are used as source terms for Einstein-Boltzmann equations in **CAMB** to generate accurate $C_\ell$ spectra.

## Directory Structure

- **`string_simulation.py`**: The primary script and physics engine. It handles background cosmology, the VOS model solver, UETC matrix computation, eigenmode diagonalization, and the CAMB calculations.
- **`integrals.py`**: Contains numerically stable, Numba-accelerated implementations of the UETC integrals ($I_1$ to $I_6$) and spherical Bessel functions.
- **`create_training_data.py`**: A script that  generates large datasets of power spectra (using string_simulation.py) for emulator training.
- **`emulator.py`**: A PyTorch-based multi-head neural network script for training and running emulators that predict C_l spectra.

## Usage

### Run a single simulation to test
To calculate and plot CMB power spectra for a specific string model (Default includes Scalar, Vector, and Tensor modes):
```bash
python string_simulation.py --alpha 1.9 --cr 0.23 --lmax 4000 --gmu 1.58e-7

### Create training data
Modify the correlator parameters inside create_training_data and run

### Train emulator
Train the emulator using emulator.py. This can then be directly used in an MCMC code (like cobaya) to accurately predict the string anisotropy spectra.