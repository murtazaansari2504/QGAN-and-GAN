import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

# --- STEP 1: CLASSICAL PREPROCESSING (Shivam's Core Role) ---
def get_quantum_ready_data(n_components=16):
    print("--- Step 1: Loading & Normalizing Data ---")
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    
    # Normalization (0 se 1 ke beech)
    x_train = x_train.astype('float32') / 255.0
    
    # Flattening (28x28 -> 784)
    x_train_flat = x_train.reshape((len(x_train), 784))
    
    print(f"--- Step 2: Applying PCA (Reducing to {n_components} features) ---")
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(x_train_flat)
    
    print(f"✅ Data Ready! Shape: {train_pca.shape}")
    return train_pca, y_train

# --- STEP 2: QUANTUM MODEL CONSTRUCTION ---
def build_quantum_circuit(qubits):
    print(f"--- Step 3: Building {qubits}-Qubit Quantum Circuit ---")
    
    # Encoding Layer (ZZFeatureMap) - Data ko qubits me feed karne ke liye
    feature_map = ZZFeatureMap(feature_dimension=qubits, reps=1)
    
    # Trainable Layer (Ansatz) - Model isi ko learn karega
    ansatz = RealAmplitudes(qubits, reps=1)
    
    # Combining both
    circuit = feature_map.compose(ansatz)
    
    return circuit

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Get processed data (16 features)
    num_features = 16
    data, labels = get_quantum_ready_data(n_components=num_features)
    
    # 2. Create the Circuit
    q_circuit = build_quantum_circuit(qubits=num_features)
    
    print("\n--- Quantum Model Summary ---")
    print(f"Total Qubits in use: {q_circuit.num_qubits}")
    print(f"Trainable Parameters: {q_circuit.num_parameters}")
    
    # 3. Visualize the result
    print("\nDrawing the Circuit... (Closing the window will end the script)")
    q_circuit.decompose().draw(output='mpl')
    plt.show()
    
    print("✅ Pipeline executed successfully!")