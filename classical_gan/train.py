import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Get the directory where train.py is currently located (.../classical_gan)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the main repository root folder (.../QGAN-and-GAN)
parent_dir = os.path.dirname(current_dir)

# Point directly to the folder containing the dataset script
dataset_dir = os.path.join(parent_dir, 'LR dataset')

# Add the dataset folder to Python's system path
sys.path.append(dataset_dir)

# Now Python can successfully find and import the data and your local modules
try:
    from generationds import X, y  
except ImportError:
    print("Error: Could not find 'generationds.py' in the 'LR dataset' folder.")
    sys.exit(1)

from generator import Generator, LATENT_DIM
from discriminator import Discriminator

# Combine X and y into [x, y] coordinate pairs
real_data = torch.cat((X, y), dim=1)

# HYPERPARAMETERS
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 300

# INITIALIZATION
netG = Generator()
netD = Discriminator()

criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=LR)
optimizerD = optim.Adam(netD.parameters(), lr=LR)

# TRAINING LOOP
print("Starting Classical GAN Training from custom folder...")

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
plt.scatter(real_points[:, 0], real_points[:, 1], label="Real Data (y = 0.7x + 0.3)", alpha=0.6, color="blue")
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