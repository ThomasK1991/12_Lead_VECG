import pandas as pd
import matplotlib.pyplot as plt
import os
# Load training log CSV
csv_path = '/users/newc6477/VAE/12_Lead_VECG/results/Hope/test_is_split1/I/2025-03-25_22-44-31/training/'
save_path = os.path.join(csv_path,'kl_loss')
csv_path =os.path.join(csv_path,'training_progress.csv')
df = pd.read_csv(csv_path)

# Plot KL divergence components and reconstruction loss over epochs
plt.figure(figsize=(12, 8))
plt.plot(df['epoch'], df['kl_loss'], label='Total KL Loss')
plt.plot(df['epoch'], df['mi'], label='Mutual Information (MI)')
plt.plot(df['epoch'], df['tc'], label='Total Correlation (TC)')
plt.plot(df['epoch'], df['dw_kl'], label='Dimension-wise KL')
plt.plot(df['epoch'], df['recon'], label='Reconstruction Loss', linestyle='--', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss Component Value')
plt.title('KL Divergence Components and Reconstruction Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(save_path)
