import tensorflow as tf

print("Downloading and Loading MNIST dataset...")

# Step 1: Loading the dataset (Pehli baar download hoga, fir cache se load hoga)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Step 2: Verification of data shape
print(f"Successfully Loaded!")
print(f"Training images: {x_train.shape[0]}") # Output should be 60,000
print(f"Testing images: {x_test.shape[0]}")   # Output should be 10,000
print(f"Image Resolution: {x_train.shape[1]}x{x_train.shape[2]}") # Output: 28x28