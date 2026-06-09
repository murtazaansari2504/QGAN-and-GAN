import torch

# Create known parameters
weight = 0.7
bias = 0.3

# Generate 500 data points
start = 0
end = 1
step = 0.002  # (end - start) / 500

# Create input features (X)
X = torch.arange(start, end, step).unsqueeze(dim=1)

# Create target labels (y) using the equation y = wx + b
y = weight * X + bias

# Display the first 400 samples
print("First 400 values of X:")
print(X[:400])

print("\nFirst 400 values of y:")
print(y[:400])

# Display tensor shapes
print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

# Display total number of samples
print("\nNumber of X samples:", len(X))
print("Number of y samples:", len(y))