
import sys
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance
from scipy.special import rel_entr

# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

torch.manual_seed(42)
np.random.seed(42)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

from generator import Generator, LATENT_DIM
from critic import Critic

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
# 3. HYPERPARAMETERS
# ==========================================

BATCH_SIZE = 32
LR = 0.00005
EPOCHS = 1000
CLIP_VALUE = 0.01

# Evaluation settings (same as LSGAN)
NUM_GENERATED = 200
NUM_BINS = 50
NUM_QUANTILES = 1000

# ==========================================
# 4. INITIALIZATION
# ==========================================

netG = Generator()
netC = Critic()

optimizerG = optim.RMSprop(netG.parameters(), lr=LR)
optimizerC = optim.RMSprop(netC.parameters(), lr=LR)

lossC_history = []
lossG_history = []

# ==========================================
# 5. WGAN TRAINING
# ==========================================

print("Starting WGAN Training with CSV dataset...")

for epoch in range(EPOCHS):

    permutation = torch.randperm(real_data.size(0))
    shuffled_data = real_data[permutation]

    epoch_loss_c = 0.0
    epoch_loss_g = 0.0
    num_batches = 0

    for i in range(0, len(shuffled_data), BATCH_SIZE):

        batch_real = shuffled_data[i:i + BATCH_SIZE]
        current_batch_size = batch_real.size(0)

        # --------------------------
        # Train Critic
        # --------------------------

        optimizerC.zero_grad()

        noise = torch.randn(current_batch_size, LATENT_DIM)
        batch_fake = netG(noise).detach()

        lossC = -(
            torch.mean(netC(batch_real))
            - torch.mean(netC(batch_fake))
        )

        lossC.backward()
        optimizerC.step()

        # Weight clipping
        for p in netC.parameters():
            p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

        # --------------------------
        # Train Generator
        # --------------------------

        optimizerG.zero_grad()

        noise = torch.randn(current_batch_size, LATENT_DIM)
        batch_fake = netG(noise)

        lossG = -torch.mean(netC(batch_fake))

        lossG.backward()
        optimizerG.step()

        epoch_loss_c += lossC.item()
        epoch_loss_g += lossG.item()
        num_batches += 1

    lossC_history.append(epoch_loss_c / num_batches)
    lossG_history.append(epoch_loss_g / num_batches)

    if epoch % 50 == 0 or epoch == EPOCHS - 1:

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Loss C: {lossC_history[-1]:.4f} | "
            f"Loss G: {lossG_history[-1]:.4f}"
        )

# ==========================================
# 6. GENERATE SAMPLES
# ==========================================

print("\nTraining complete. Evaluating WGAN...")

netG.eval()

with torch.no_grad():

    test_noise = torch.randn(NUM_GENERATED, LATENT_DIM)
    generated_points = netG(test_noise).numpy()

real_points = real_data.numpy()

# ==========================================
# 7. EVALUATION METRICS
# ==========================================

print("\n========== WGAN EVALUATION METRICS ==========")

# A. MAE and RMSE (Empirical Quantiles)

real_quantiles = np.quantile(
    real_points,
    np.linspace(0, 1, NUM_QUANTILES),
    axis=0
)

generated_quantiles = np.quantile(
    generated_points,
    np.linspace(0, 1, NUM_QUANTILES),
    axis=0
)

quantile_difference = real_quantiles - generated_quantiles

MAE = np.mean(np.abs(quantile_difference))

RMSE = np.sqrt(
    np.mean(quantile_difference ** 2)
)

print(f"MAE: {MAE:.6f}")
print(f"RMSE: {RMSE:.6f}")


# B. 2D Histograms (50 x 50 bins)

combined_data = np.vstack((real_points, generated_points))

hist_range = [
    (combined_data[:, 0].min(), combined_data[:, 0].max()),
    (combined_data[:, 1].min(), combined_data[:, 1].max())
]

real_hist, _ = np.histogramdd(
    real_points,
    bins=NUM_BINS,
    range=hist_range
)

generated_hist, _ = np.histogramdd(
    generated_points,
    bins=NUM_BINS,
    range=hist_range
)

# Normalize histograms

epsilon = 1e-10

P = real_hist / real_hist.sum()
Q = generated_hist / generated_hist.sum()


# C. KL Divergence (Real || Generated)

P_smooth = P + epsilon
Q_smooth = Q + epsilon

P_smooth /= P_smooth.sum()
Q_smooth /= Q_smooth.sum()

KL_Divergence = np.sum(
    rel_entr(P_smooth, Q_smooth)
)

print(f"KL Divergence: {KL_Divergence:.6f}")


# D. Wasserstein Distance

wasserstein_x = wasserstein_distance(
    real_points[:, 0],
    generated_points[:, 0]
)

wasserstein_y = wasserstein_distance(
    real_points[:, 1],
    generated_points[:, 1]
)

Wasserstein_Distance = (
    wasserstein_x + wasserstein_y
) / 2

print(f"Wasserstein Distance: {Wasserstein_Distance:.6f}")


# E. Histogram Overlap

Histogram_Overlap = np.sum(
    np.minimum(P, Q)
)

print(f"Histogram Overlap: {Histogram_Overlap:.6f}")
print(f"Histogram Overlap (%): {Histogram_Overlap * 100:.2f}%")


# F. Distribution Variance Difference

real_variance = np.var(real_points, axis=0)
generated_variance = np.var(generated_points, axis=0)

Variance_Difference = np.mean(
    np.abs(real_variance - generated_variance)
)

print(
    f"Distribution Variance Difference: "
    f"{Variance_Difference:.6f}"
)

print("==============================================")


# ==========================================
# 8. VISUALIZATION & SAVING
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    real_points[:, 0],
    real_points[:, 1],
    label="Real CSV Data",
    alpha=0.6,
    color="blue"
)

plt.scatter(
    generated_points[:, 0],
    generated_points[:, 1],
    label="WGAN Generated Data",
    alpha=0.6,
    marker='x',
    color="red"
)

plt.title("WGAN Performance")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.legend()
plt.grid(True)

save_path = os.path.join(
    current_dir,
    'wgan_performance.png'
)

plt.savefig(save_path)
plt.show()

print(f"Plot successfully saved to: {save_path}")
