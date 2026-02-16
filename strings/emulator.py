"""
Neural Network Emulator for Cosmic String CMB Power Spectra

This script trains a neural network to predict C_l power spectra
as a function of cosmological and string parameters.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import json
from datetime import datetime


# ===================================================================== #
# --- CONFIGURATION ---
# ===================================================================== #

CONFIG = {
    # Data paths
    'params_file': 'string_training_parameters.npz',
    'spectra_file': 'string_training_spectra.npz',

    # Train/val/test split
    'train_frac': 0.8,
    'val_frac': 0.1,
    'test_frac': 0.1,

    # Network architecture
    'hidden_layers': [512, 512, 256],
    'activation': 'silu',  # Options: 'relu', 'silu', 'gelu', 'tanh'
    'dropout': 0.0,

    # Training parameters
    'batch_size': 64,
    'learning_rate': 5e-4,
    'weight_decay': 1e-1,
    'epochs': 2000,
    'patience': 100,

    # Log transform parameters
    'log_floor': 1e-30,  # Floor value for log transform

    # Output
    'model_dir': 'trained_models',
    'random_seed': 42,
}

MODES = ['TT', 'EE', 'BB', 'TE']
PARAM_NAMES = ['alpha', 'cr', 'H0', 'Omega_matter', 'Omega_rad', 'Omega_k']


# ===================================================================== #
# --- DATA HANDLING ---
# ===================================================================== #

class LogTransform:
    """
    Handles log transformation of power spectra.

    For all spectra (including TE): log10(|C_l| + floor)
    TE sign is stored separately and predicted by a separate output head.

    This provides cleaner separation of magnitude and sign for TE.
    """

    def __init__(self, floor=1e-30):
        self.floor = floor

    def forward(self, spectra, mode_indices):
        """Transform spectra to log space (magnitude only)."""
        # Always take log of absolute value
        transformed = np.log10(np.abs(spectra) + self.floor)
        return transformed

    def get_te_signs(self, spectra, mode_indices):
        """Extract signs for TE spectrum (1 for positive, 0 for negative)."""
        te_idx = MODES.index('TE')
        start_idx = mode_indices[te_idx]
        end_idx = mode_indices[te_idx + 1]
        te_data = spectra[:, start_idx:end_idx]
        # Return 1 for positive/zero, 0 for negative
        return (te_data >= 0).astype(np.float32)

    def inverse(self, log_spectra, mode_indices, te_signs=None):
        """Transform from log space back to linear."""
        spectra = 10 ** log_spectra - self.floor

        # Apply TE signs if provided
        if te_signs is not None:
            te_idx = MODES.index('TE')
            start_idx = mode_indices[te_idx]
            end_idx = mode_indices[te_idx + 1]
            # Convert from [0,1] to [-1,1]
            sign_multiplier = 2 * te_signs - 1
            spectra[:, start_idx:end_idx] *= sign_multiplier

        return spectra


class Normalizer:
    """Handles standardization of inputs and outputs."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """Compute mean and std from training data."""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero
        self.std[self.std < 1e-10] = 1.0
        return self

    def transform(self, data):
        """Standardize data."""
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """Inverse standardization."""
        return data * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}

    def load_state_dict(self, state):
        self.mean = np.array(state['mean'])
        self.std = np.array(state['std'])


class SpectraDataset(Dataset):
    """PyTorch dataset for cosmic string spectra with separate TE signs."""

    def __init__(self, params, spectra, te_signs):
        self.params = torch.FloatTensor(params)
        self.spectra = torch.FloatTensor(spectra)
        self.te_signs = torch.FloatTensor(te_signs)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx], self.spectra[idx], self.te_signs[idx]


def load_data(config):
    """Load and preprocess the training data."""
    print("Loading data...")

    # Load files
    params_data = np.load(config['params_file'])
    spectra_data = np.load(config['spectra_file'])

    # Extract parameters in consistent order
    params = np.column_stack([params_data[name] for name in PARAM_NAMES])
    spectra = spectra_data['features']
    l_values = np.unique(spectra_data['modes'])
    n_ell = len(l_values)

    print(f"  Loaded {len(params)} samples")
    print(f"  Parameters: {PARAM_NAMES}")
    print(f"  Spectra shape: {spectra.shape}")
    print(f"  l-values: {n_ell} (l={l_values.min()} to l={l_values.max()})")

    # Mode indices for separating TT, EE, BB, TE
    mode_indices = [i * n_ell for i in range(len(MODES) + 1)]

    return params, spectra, l_values, mode_indices


# ===================================================================== #
# --- NEURAL NETWORK MODEL ---
# ===================================================================== #

class SpectraEmulator(nn.Module):
    """
    Neural network emulator for CMB power spectra.

    Architecture: MLP with configurable hidden layers and activation.
    Has a separate head for predicting TE sign (binary classification).
    """

    def __init__(self, input_dim, output_dim, te_sign_dim, hidden_layers, activation='silu', dropout=0.0):
        super().__init__()

        # Activation function
        act_fn = {
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
        }[activation]

        # Build shared backbone
        backbone_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers[:-1]:
            backbone_layers.append(nn.Linear(prev_dim, hidden_dim))
            backbone_layers.append(act_fn())
            if dropout > 0:
                backbone_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*backbone_layers)

        # Magnitude head (for log spectra)
        self.magnitude_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_layers[-1]),
            act_fn(),
            nn.Linear(hidden_layers[-1], output_dim)
        )

        # TE sign head (binary classification per l-value)
        self.te_sign_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_layers[-1]),
            act_fn(),
            nn.Linear(hidden_layers[-1], te_sign_dim)
        )

        self.te_sign_dim = te_sign_dim

    def forward(self, x):
        features = self.backbone(x)
        magnitude = self.magnitude_head(features)
        te_sign_logits = self.te_sign_head(features)
        return magnitude, te_sign_logits

    def predict_with_signs(self, x):
        """Predict magnitude and apply sign classification."""
        magnitude, te_sign_logits = self.forward(x)
        te_signs = torch.sigmoid(te_sign_logits)
        return magnitude, te_signs


# ===================================================================== #
# --- TRAINING ---
# ===================================================================== #

def train_epoch(model, dataloader, mse_criterion, bce_criterion, optimizer, device, sign_loss_weight=0.1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_mag_loss = 0
    total_sign_loss = 0

    for params, spectra, te_signs in dataloader:
        params = params.to(device)
        spectra = spectra.to(device)
        te_signs = te_signs.to(device)

        optimizer.zero_grad()
        mag_pred, sign_logits = model(params)

        # Magnitude loss (MSE on log spectra)
        mag_loss = mse_criterion(mag_pred, spectra)

        # Sign loss (BCE for TE sign classification)
        sign_loss = bce_criterion(sign_logits, te_signs)

        # Combined loss
        loss = mag_loss + sign_loss_weight * sign_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(params)
        total_mag_loss += mag_loss.item() * len(params)
        total_sign_loss += sign_loss.item() * len(params)

    n = len(dataloader.dataset)
    return total_loss / n, total_mag_loss / n, total_sign_loss / n


def validate(model, dataloader, mse_criterion, bce_criterion, device, sign_loss_weight=0.1):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_mag_loss = 0
    total_sign_loss = 0

    with torch.no_grad():
        for params, spectra, te_signs in dataloader:
            params = params.to(device)
            spectra = spectra.to(device)
            te_signs = te_signs.to(device)

            mag_pred, sign_logits = model(params)

            mag_loss = mse_criterion(mag_pred, spectra)
            sign_loss = bce_criterion(sign_logits, te_signs)
            loss = mag_loss + sign_loss_weight * sign_loss

            total_loss += loss.item() * len(params)
            total_mag_loss += mag_loss.item() * len(params)
            total_sign_loss += sign_loss.item() * len(params)

    n = len(dataloader.dataset)
    return total_loss / n, total_mag_loss / n, total_sign_loss / n


def train_model(config):
    """Main training function."""

    # Set random seeds
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    params, spectra, l_values, mode_indices = load_data(config)
    n_samples = len(params)
    n_ell = len(l_values)

    # Apply log transformation
    print("\nApplying log transformation to spectra...")
    log_transform = LogTransform(floor=config['log_floor'])
    log_spectra = log_transform.forward(spectra, mode_indices)

    # Extract TE signs separately
    print("Extracting TE signs...")
    te_signs = log_transform.get_te_signs(spectra, mode_indices)
    print(f"  TE sign shape: {te_signs.shape}")
    print(f"  Positive fraction: {te_signs.mean():.3f}")

    # Normalize inputs
    print("Normalizing parameters...")
    param_normalizer = Normalizer()
    param_normalizer.fit(params)
    params_normalized = param_normalizer.transform(params)

    # Normalize log-transformed outputs
    print("Normalizing log-spectra...")
    spectra_normalizer = Normalizer()
    spectra_normalizer.fit(log_spectra)
    log_spectra_normalized = spectra_normalizer.transform(log_spectra)

    # Create dataset with TE signs
    dataset = SpectraDataset(params_normalized, log_spectra_normalized, te_signs)

    # Split data
    n_train = int(config['train_frac'] * n_samples)
    n_val = int(config['val_frac'] * n_samples)
    n_test = n_samples - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config['random_seed'])
    )

    print(f"\nDataset split:")
    print(f"  Training:   {n_train}")
    print(f"  Validation: {n_val}")
    print(f"  Test:       {n_test}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # Create model
    input_dim = len(PARAM_NAMES)
    output_dim = log_spectra.shape[1]
    te_sign_dim = n_ell  # One sign per l-value for TE

    model = SpectraEmulator(
        input_dim=input_dim,
        output_dim=output_dim,
        te_sign_dim=te_sign_dim,
        hidden_layers=config['hidden_layers'],
        activation=config['activation'],
        dropout=config['dropout']
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel architecture:")
    print(f"  Input dim:  {input_dim}")
    print(f"  Output dim: {output_dim} (log spectra)")
    print(f"  TE sign dim: {te_sign_dim}")
    print(f"  Hidden layers: {config['hidden_layers']}")
    print(f"  Total parameters: {n_params:,}")

    # Loss functions
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_mag_loss': [], 'val_mag_loss': [],
               'train_sign_loss': [], 'val_sign_loss': []}

    # Create output directory
    model_dir = Path(config['model_dir'])
    model_dir.mkdir(exist_ok=True)

    sign_loss_weight = 0.1

    for epoch in range(config['epochs']):
        train_loss, train_mag, train_sign = train_epoch(
            model, train_loader, mse_criterion, bce_criterion, optimizer, device, sign_loss_weight
        )
        val_loss, val_mag, val_sign = validate(
            model, val_loader, mse_criterion, bce_criterion, device, sign_loss_weight
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mag_loss'].append(train_mag)
        history['val_mag_loss'].append(val_mag)
        history['train_sign_loss'].append(train_sign)
        history['val_sign_loss'].append(val_sign)

        scheduler.step(val_loss)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_dir / 'best_model.pt')
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:4d}/{config['epochs']}  "
                  f"Loss: {train_loss:.5f}/{val_loss:.5f}  "
                  f"Mag: {train_mag:.5f}/{val_mag:.5f}  "
                  f"Sign: {train_sign:.4f}/{val_sign:.4f}  "
                  f"LR: {current_lr:.1e}  "
                  f"Pat: {patience_counter}/{config['patience']}")

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model for final evaluation
    checkpoint = torch.load(model_dir / 'best_model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test evaluation
    test_loss, test_mag, test_sign = validate(model, test_loader, mse_criterion, bce_criterion, device, sign_loss_weight)
    print(f"\n{'=' * 60}")
    print(f"Final Test Loss: {test_loss:.6f} (Mag: {test_mag:.6f}, Sign: {test_sign:.6f})")
    print(f"{'=' * 60}")

    # Compute relative errors on original scale
    print("\nComputing relative errors on original scale...")
    model.eval()

    all_true_mag = []
    all_pred_mag = []
    all_true_signs = []
    all_pred_signs = []

    with torch.no_grad():
        for params_batch, spectra_batch, signs_batch in test_loader:
            params_batch = params_batch.to(device)
            mag_pred, sign_pred = model.predict_with_signs(params_batch)
            all_true_mag.append(spectra_batch.numpy())
            all_pred_mag.append(mag_pred.cpu().numpy())
            all_true_signs.append(signs_batch.numpy())
            all_pred_signs.append(sign_pred.cpu().numpy())

    all_true_mag = np.concatenate(all_true_mag, axis=0)
    all_pred_mag = np.concatenate(all_pred_mag, axis=0)
    all_true_signs = np.concatenate(all_true_signs, axis=0)
    all_pred_signs = np.concatenate(all_pred_signs, axis=0)

    # Inverse transform
    all_true_log = spectra_normalizer.inverse_transform(all_true_mag)
    all_pred_log = spectra_normalizer.inverse_transform(all_pred_mag)

    # Apply sign prediction (threshold at 0.5)
    all_pred_signs_binary = (all_pred_signs > 0.5).astype(np.float32)

    all_true_linear = log_transform.inverse(all_true_log, mode_indices, all_true_signs)
    all_pred_linear = log_transform.inverse(all_pred_log, mode_indices, all_pred_signs_binary)

    # Compute TE sign accuracy
    te_sign_accuracy = (all_pred_signs_binary == all_true_signs).mean() * 100
    print(f"\nTE sign prediction accuracy: {te_sign_accuracy:.2f}%")

    # Compute relative errors per mode
    print("\nRelative errors by mode (on |C_l|):")
    for i, mode in enumerate(MODES):
        start_idx = mode_indices[i]
        end_idx = mode_indices[i + 1]

        true_mode = np.abs(all_true_linear[:, start_idx:end_idx])
        pred_mode = np.abs(all_pred_linear[:, start_idx:end_idx])

        # Relative error (avoid division by very small values)
        mask = true_mode > 1e-15
        rel_error = np.abs(pred_mode[mask] - true_mode[mask]) / true_mode[mask]

        mean_rel_err = np.mean(rel_error) * 100
        median_rel_err = np.median(rel_error) * 100
        p95_rel_err = np.percentile(rel_error, 95) * 100

        print(f"  {mode}: Mean={mean_rel_err:.2f}%  Median={median_rel_err:.2f}%  95th={p95_rel_err:.2f}%")

    # Save everything needed for inference
    print("\nSaving model and preprocessing info...")

    # Save full checkpoint
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'param_normalizer': param_normalizer.state_dict(),
        'spectra_normalizer': spectra_normalizer.state_dict(),
        'l_values': l_values.tolist(),
        'mode_indices': mode_indices,
        'param_names': PARAM_NAMES,
        'modes': MODES,
        'log_floor': config['log_floor'],
        'history': history,
        'test_loss': test_loss,
        'te_sign_dim': te_sign_dim,
    }

    torch.save(save_dict, model_dir / 'emulator_full.pt')

    # Also save config as JSON for reference
    with open(model_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nModel saved to {model_dir}/")
    print(f"  - emulator_full.pt (complete model + preprocessing)")
    print(f"  - best_model.pt (model weights only)")
    print(f"  - config.json (configuration)")

    return model, history


# ===================================================================== #
# --- INFERENCE HELPER ---
# ===================================================================== #

class TrainedEmulator:
    """
    Wrapper class for easy inference with the trained emulator.

    Usage:
        emulator = TrainedEmulator.load('trained_models/emulator_full.pt')
        params = {'alpha': 5.0, 'cr': 0.5, 'H0': 67.4,
                  'Omega_matter': 0.315, 'Omega_rad': 9.2e-5, 'Omega_k': 0.0}
        spectra = emulator.predict(params)
    """

    def __init__(self, model, param_normalizer, spectra_normalizer,
                 log_transform, mode_indices, l_values, param_names, device):
        self.model = model
        self.param_normalizer = param_normalizer
        self.spectra_normalizer = spectra_normalizer
        self.log_transform = log_transform
        self.mode_indices = mode_indices
        self.l_values = l_values
        self.param_names = param_names
        self.device = device
        self.model.eval()

    @classmethod
    def load(cls, path, device=None):
        """Load a trained emulator from file."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Reconstruct model
        config = checkpoint['config']
        n_ell = len(checkpoint['l_values'])
        model = SpectraEmulator(
            input_dim=len(checkpoint['param_names']),
            output_dim=n_ell * len(checkpoint['modes']),
            te_sign_dim=checkpoint.get('te_sign_dim', n_ell),
            hidden_layers=config['hidden_layers'],
            activation=config['activation'],
            dropout=config['dropout']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Reconstruct normalizers
        param_normalizer = Normalizer()
        param_normalizer.load_state_dict(checkpoint['param_normalizer'])

        spectra_normalizer = Normalizer()
        spectra_normalizer.load_state_dict(checkpoint['spectra_normalizer'])

        # Log transform
        log_transform = LogTransform(floor=checkpoint['log_floor'])

        return cls(
            model=model,
            param_normalizer=param_normalizer,
            spectra_normalizer=spectra_normalizer,
            log_transform=log_transform,
            mode_indices=checkpoint['mode_indices'],
            l_values=np.array(checkpoint['l_values']),
            param_names=checkpoint['param_names'],
            device=device
        )

    def predict(self, params):
        """
        Predict power spectra for given parameters.

        Args:
            params: dict with parameter names as keys, or array of shape (6,) or (N, 6)

        Returns:
            dict with keys 'l', 'TT', 'EE', 'BB', 'TE' containing the spectra
        """
        # Convert dict to array
        if isinstance(params, dict):
            params = np.array([[params[name] for name in self.param_names]])
        elif params.ndim == 1:
            params = params.reshape(1, -1)

        # Normalize
        params_norm = self.param_normalizer.transform(params)

        # Predict
        with torch.no_grad():
            params_tensor = torch.FloatTensor(params_norm).to(self.device)
            mag_pred, sign_pred = self.model.predict_with_signs(params_tensor)
            mag_pred = mag_pred.cpu().numpy()
            sign_pred = sign_pred.cpu().numpy()

        # Inverse transform
        pred_log = self.spectra_normalizer.inverse_transform(mag_pred)
        sign_binary = (sign_pred > 0.5).astype(np.float32)
        pred_linear = self.log_transform.inverse(pred_log, self.mode_indices, sign_binary)

        # Reshape to separate modes
        n_ell = len(self.l_values)
        result = {'l': self.l_values}
        for i, mode in enumerate(MODES):
            start_idx = self.mode_indices[i]
            end_idx = self.mode_indices[i + 1]
            result[mode] = pred_linear[:, start_idx:end_idx].squeeze()

        return result


# ===================================================================== #
# --- MAIN ---
# ===================================================================== #

if __name__ == '__main__':
    print("=" * 60)
    print("Cosmic String CMB Power Spectra Emulator")
    print("=" * 60)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model, history = train_model(CONFIG)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Quick test of inference
    print("\n" + "=" * 60)
    print("Testing inference...")
    print("=" * 60)

    emulator = TrainedEmulator.load('trained_models/emulator_full.pt')

    # Test with middle-of-range parameters
    test_params = {
        'alpha': 5.0,
        'cr': 0.5,
        'H0': 67.4,
        'Omega_matter': 0.315,
        'Omega_rad': 9.2e-5,
        'Omega_k': 0.0
    }

    spectra = emulator.predict(test_params)
    print(f"\nTest prediction for params: {test_params}")
    print(f"  l range: {spectra['l'].min()} to {spectra['l'].max()}")
    for mode in MODES:
        print(f"  {mode}: min={spectra[mode].min():.2e}, max={spectra[mode].max():.2e}")

    print("\nDone!")
