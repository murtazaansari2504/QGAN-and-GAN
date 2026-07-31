import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Added pandas for CSV loading

# Get the directory where train.py is currently located (.../classical_gan)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the main repository root folder (.../QGAN-and-GAN)
parent_dir = os.path.dirname(current_dir)

from generator import Generator, LATENT_DIM
from discriminator import Discriminator

# --- NEW DATA LOADING LOGIC ---
csv_path = "/content/QGAN-and-GAN/LR dataset/linear_dataset.csv"

try:
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract the values and convert them to a PyTorch tensor
    # We assume the CSV has two columns representing your X and Y coordinates.
    # df.values returns a 2D numpy array which we convert to float32.
    real_data = torch.tensor(df.values, dtype=torch.float32)
    
except FileNotFoundError:
    print(f"Error: Could not find the dataset at {csv_path}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    sys.exit(1)
# ------------------------------

# HYPERPARAMETERS
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 1000

# INITIALIZATION
netG = Generator()
netD = Discriminator()

criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=LR)
optimizerD = optim.Adam(netD.parameters(), lr=LR)

# TRAINING LOOP
print("Starting Classical GAN Training with CSV dataset...")

for epoch in range(EPOCHS):
    permutation = torch.randperm(real_data.size(0))
    shuffled_data = real_data[permutation]
    
    for i in range(0, len(shuffled_data), BATCH_SIZE):
        batch_real = shuffled_data[i:i+BATCH_SIZE]
        current_batch_size = batch_real.size(0)
        
        real_labels = torch.ones(current_batch_size, 1)
        fake_labels = torch.zeros(current_batch_size, 1)
        
        # Train Discriminator
        optimizerD.zero_grad()
        output_real = netD(batch_real)
        lossD_real = criterion(output_real, real_labels)
        
        noise = torch.randn(current_batch_size, LATENT_DIM)
        batch_fake = netG(noise)
        output_fake = netD(batch_fake.detach()) 
        lossD_fake = criterion(output_fake, fake_labels)
        
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()
        
        # Train Generator
        optimizerG.zero_grad()
        output_fake_for_G = netD(batch_fake)
        lossG = criterion(output_fake_for_G, real_labels)
        lossG.backward()
        optimizerG.step()
        
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch [{epoch}/{EPOCHS}] | Loss D: {lossD.item():.4f} | Loss G: {lossG.item():.4f}")

# VISUALIZATION & SAVING
print("\nTraining complete. Saving evaluation plot...")
with torch.no_grad():
    test_noise = torch.randn(200, LATENT_DIM)
    generated_points = netG(test_noise).numpy()

real_points = real_data.numpy()

plt.figure(figsize=(8, 5))
plt.scatter(real_points[:, 0], real_points[:, 1], label="Real CSV Data", alpha=0.6, color="blue")
plt.scatter(generated_points[:, 0], generated_points[:, 1], label="GAN Generated Data", alpha=0.6, marker='x', color="red")
plt.title("Classical GAN Performance")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.legend()
plt.grid(True)

# Save the image dynamically in the same directory as train.py
save_path = os.path.join(current_dir, 'gan_performance.png')
plt.savefig(save_path)
print(f"Plot successfully saved to: {save_path}")

# ADDED DIAGNOSTIC: DISCRIMINATOR DECISION HEATMAP
print("Generating decision boundary heatmap...")

# Create a dense grid of 2D coordinates to query the Discriminator
# You may need to adjust the min/max values (-0.2, 1.2) based on your CSV's actual data range
x_grid_vals = np.linspace(-0.2, 1.2, 100)
y_grid_vals = np.linspace(0.0, 1.2, 100)
X_mesh, Y_mesh = np.meshgrid(x_grid_vals, y_grid_vals)
grid_points = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

# Pass the coordinates mesh grid through the trained Discriminator
with torch.no_grad():
    predictions_prob = netD(grid_tensor).numpy().reshape(X_mesh.shape)

plt.figure(figsize=(10, 6))

# Render continuous decision contours (Red is high probability, Blue is low)
contour_map = plt.contourf(X_mesh, Y_mesh, predictions_prob, levels=50, cmap='RdYlBu_r', alpha=0.8)
color_bar = plt.colorbar(contour_map)
color_bar.set_label("Discriminator Prediction Confidence (Real Probability)", rotation=270, labelpad=15)

# Superimpose the underlying ground truth points and fake generated points
plt.scatter(real_points[:, 0], real_points[:, 1], color='blue', alpha=0.4, s=25, label="Real CSV Data")
plt.scatter(generated_points[:, 0], generated_points[:, 1], color='lime', marker='x', s=55, linewidths=2.0, alpha=0.9, label="GAN Generated Data")

plt.title("Discriminator Decision Space Heatmap")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.legend(loc="upper left")
plt.grid(True, linestyle=":", alpha=0.5)

heatmap_save_path = os.path.join(current_dir, 'discriminator_heatmap.png')
plt.savefig(heatmap_save_path)
print(f"Discriminator heatmap saved to: {heatmap_save_path}")
