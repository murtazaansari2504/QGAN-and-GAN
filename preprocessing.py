import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_and_preprocess_data():
    """
    Step 1, 2 & 3: Loading, Normalizing and Flattening.
    Goal: Prepare data for Quantum processing. 
    """
    print("--- Starting Data Preprocessing ---")
    
    # 1. Loading Dataset [cite: 709]
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(f"✅ Data Loaded! Training images: {x_train.shape[0]}")

    # 2. Normalization (Range 0 to 1) [cite: 710, 714]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    print("✅ Normalization Complete!")

    # 3. Flattening (28x28 -> 784) [cite: 678]
    x_train_flat = x_train.reshape((len(x_train), 784))
    x_test_flat = x_test.reshape((len(x_test), 784))
    print(f"✅ Flattening Complete! Shape: {x_train_flat.shape}")

    return (x_train, y_train), (x_train_flat, x_test_flat)

def apply_pca(flat_data, n_components=16):
    """
    Step 4: Dimensionality Reduction (PCA).
    Why? To fit large images into limited Quantum Qubits. [cite: 741, 742]
    """
    print(f"--- Applying PCA (Reducing to {n_components} components) ---")
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(flat_data)
    
    print(f"✅ PCA Complete! New Data Shape: {pca_data.shape}")
    return pca_data, pca

def visualize_data(images, labels, index=0):
    """
    Visualization to verify original data quality. [cite: 711]
    """
    plt.imshow(images[index], cmap='gray')
    plt.title(f"Processed Image - Label: {labels[index]}")
    plt.show()
    print("✅ Visualization Successful!")

# --- Main Execution ---
if __name__ == "__main__":
    # Get preprocessed data
    (original_images, labels), (flat_train, flat_test) = load_and_preprocess_data()
    
    # Apply PCA (Reducing 784 pixels to 16 important features) 
    train_pca, pca_model = apply_pca(flat_train, n_components=16)
    
    # Visualize a sample to verify
    visualize_data(original_images, labels, index=10000)