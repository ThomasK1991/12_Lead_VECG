import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import os

from model.encoder import Encoder
from model.decoder import Decoder

# --- Set parameters ---
LATENT_DIM = 16
MODEL_PATH = '/users/newc6477/VAE/12_Lead_VECG/results/improved_encoder/test_is_split1/I/2025-03-28_18-21-16/model_best.keras'
os.environ['TFDS_DATA_DIR'] = r"/data/newc6477/VAE/Single_Beat/5_percent_Physionet/"
T_WAVE_DIAG_IDX = 25  # 26th index (0-based)

# --- Load model components ---
encoder = Encoder(LATENT_DIM)
decoder = Decoder(LATENT_DIM)

# Create dummy inputs so models can build
dummy_input = tf.zeros((1, 500))
encoder(dummy_input)
z = tf.zeros((1, LATENT_DIM))
decoder(z)

# Load model weights (shared .keras file)
encoder.encoder.load_weights(MODEL_PATH)
decoder.decoder.load_weights(MODEL_PATH)

# --- Load dataset ---
dataset_name = "physionet"  # Change this if yours has a custom name
split = "split1"  # or "validation" / "test"
ds = tfds.load(dataset_name, split=split)

# Convert to numpy for filtering
ecgs = []
diagnoses = []

for sample in tfds.as_numpy(ds):
    ecgs.append(sample['ecg'])  # shape: (500,)
    diagnoses.append(sample['diagnosis'])  # shape: (num_classes,)

ecgs = np.stack(ecgs)
diagnoses = np.stack(diagnoses)

# --- Filter for T-wave inversion ---
t_wave_idx = np.where(diagnoses[:, T_WAVE_DIAG_IDX] == 1)[0]
print(f"Found {len(t_wave_idx)} T-wave inversion samples.")

N = min(5, len(t_wave_idx))
selected_samples = ecgs[t_wave_idx[:N]]
selected_samples = tf.convert_to_tensor(selected_samples, dtype=tf.float32)

# --- Inference ---
z_mean, z_log_var = encoder(selected_samples)
eps = tf.random.normal(shape=z_mean.shape)
z = z_mean + tf.exp(0.5 * z_log_var) * eps
recon = decoder(z).numpy()

# --- Plotting ---
for i in range(N):
    plt.figure(figsize=(10, 4))
    plt.plot(selected_samples[i].numpy(), label="Original", linewidth=2)
    plt.plot(recon[i], label="Reconstructed", linewidth=2)
    plt.title(f"T-Wave Inversion Sample #{i}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
