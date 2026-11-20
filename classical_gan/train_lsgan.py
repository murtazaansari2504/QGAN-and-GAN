
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance, entropy

# ==========================================
# 1. SETUP
# ==========================================

torch.manual_seed(42)
np.random.seed(42)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

from generator import Generator, LATENT_DIM
from critic import Critic

BATCH_SIZE = 32
LR = 0.001
EPOCHS = 1000

# ==========================================
# 2. CSV DATASET LOADING
# ==========================================

possible_paths = [
    "/content/QGAN-and-GAN/LR dataset/linear_dataset.csv",
    os.path.join(parent_dir, "LR dataset", "linear_dataset.csv"),
    os.path.join(os.getcwd(), "LR dataset", "linear_dataset.csv"),
    "/content/LR dataset/linear_dataset.csv"
]

csv_path = None

for p in possible_paths:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    print("Error: Could not find 'linear_dataset.csv'!")
    sys.exit(1)

try:
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    real_data = torch.tensor(df.values, dtype=torch.float32)

except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    sys.exit(1)

# ==========================================
# 3. INITIALIZE LSGAN
# ==========================================

netG = Generator()
netC = Critic()

# LSGAN uses Mean Squared Error instead of BCE
criterion = nn.MSELoss()

optimizerG = optim.Adam(netG.parameters(), lr=LR)
optimizerC = optim.Adam(netC.parameters(), lr=LR)

lossC_history = []
lossG_history = []

print("\nStarting LSGAN Training...")

# ==========================================
# 4. TRAINING
# ==========================================

for epoch in range(EPOCHS):

    permutation = torch.randperm(real_data.size(0))
    shuffled_data = real_data[permutation]

    epoch_loss_c = 0.0
    epoch_loss_g = 0.0
    num_batches = 0

    for i in range(0, len(shuffled_data), BATCH_SIZE):

        batch_real = shuffled_data[i:i + BATCH_SIZE]
        curr_batch_size = batch_real.size(0)

        real_labels = torch.ones(curr_batch_size, 1)
        fake_labels = torch.zeros(curr_batch_size, 1)

        # ------------------------------
        # Train Critic
        # ------------------------------

        optimizerC.zero_grad()

        lossC_real = criterion(
            netC(batch_real), real_labels
        )

        noise = torch.randn(curr_batch_size, LATENT_DIM)
        batch_fake = netG(noise)

        lossC_fake = criterion(
            netC(batch_fake.detach()), fake_labels
        )

        lossC = 0.5 * (lossC_real + lossC_fake)

        lossC.backward()
        optimizerC.step()

        # ------------------------------
        # Train Generator
        # ------------------------------

        optimizerG.zero_grad()

        # Generator tries to fool Critic
        lossG = criterion(
            netC(batch_fake), real_labels
        )

        lossG.backward()
        optimizerG.step()

        epoch_loss_c += lossC.item()
        epoch_loss_g += lossG.item()

        num_batches += 1

    lossC_history.append(epoch_loss_c / num_batches)
    lossG_history.append(epoch_loss_g / num_batches)

    if epoch % 100 == 0 or epoch == EPOCHS - 1:

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"Loss C: {lossC_history[-1]:.4f} | "
            f"Loss G: {lossG_history[-1]:.4f}"
        )

# ==========================================
# 5. GENERATE FAKE DATA
# ==========================================

netG.eval()

with torch.no_grad():
    noise = torch.randn(200, LATENT_DIM)
    generated_points = netG(noise).numpy()

real_points = real_data.numpy()

# ==========================================
# 6. EVALUATION METRICS
# ==========================================

# Flatten data for distribution-based metrics
real_flat = real_points.flatten()
fake_flat = generated_points.flatten()

# MAE
mae = np.mean(np.abs(real_flat[:min(len(real_flat), len(fake_flat))]
                     - fake_flat[:min(len(real_flat), len(fake_flat))]))

# RMSE
rmse = np.sqrt(
    np.mean(
        (real_flat[:min(len(real_flat), len(fake_flat))]
         - fake_flat[:min(len(real_flat), len(fake_flat))]) ** 2
    )
)

# KL Divergence
# Using common histogram bins to estimate probability distributions
hist_range = (
    min(real_flat.min(), fake_flat.min()),
    max(real_flat.max(), fake_flat.max())
)

real_hist, bin_edges = np.histogram(
    real_flat, bins=50, range=hist_range, density=False
)

fake_hist, _ = np.histogram(
    fake_flat, bins=50, range=hist_range, density=False
)

epsilon = 1e-10

real_prob = real_hist / real_hist.sum()
fake_prob = fake_hist / fake_hist.sum()

kl_div = entropy(
    real_prob + epsilon,
    fake_prob + epsilon
)

# Wasserstein Distance
wasserstein_dist = wasserstein_distance(
    real_flat, fake_flat
)

# Histogram Overlap
real_prob = real_prob / real_prob.sum()
fake_prob = fake_prob / fake_prob.sum()

histogram_overlap = np.sum(
    np.minimum(real_prob, fake_prob)
)

# Distribution Variance Difference
variance_difference = abs(
    np.var(real_flat) - np.var(fake_flat)
)

# ==========================================
# 7. DISPLAY METRICS
# ==========================================

print("\n" + "=" * 45)
print("          LSGAN EVALUATION METRICS")
print("=" * 45)

print(f"MAE:                          {mae:.6f}")
print(f"RMSE:                         {rmse:.6f}")
print(f"KL Divergence:                {kl_div:.6f}")
print(f"Wasserstein Distance:         {wasserstein_dist:.6f}")
print(f"Histogram Overlap:            {histogram_overlap:.6f}")
print(f"Distribution Variance Difference: {variance_difference:.6f}")

print("=" * 45)

# ==========================================
# 8. VISUALIZATION
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    real_points[:, 0],
    real_points[:, 1],
    alpha=0.6,
    color='blue',
    label='Real Data'
)

plt.scatter(
    generated_points[:, 0],
    generated_points[:, 1],
    alpha=0.6,
    marker='x',
    color='red',
    label='Generated Data'
)

plt.title("LSGAN Performance")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    os.path.join(current_dir, 'lsgan_performance.png')
)

plt.show()

print("\nSaved LSGAN Plot!")
