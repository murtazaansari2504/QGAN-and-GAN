import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler

# --- STEP 1: DATA PREPARATION (Built-in) ---
def get_data(n_components=16):
    print("--- Step 1: Loading & Reducing Data ---")
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_train_flat = x_train.reshape((len(x_train), 784))
    pca = PCA(n_components=n_components)
    return pca.fit_transform(x_train_flat), y_train

# --- STEP 2: QUANTUM CIRCUIT (Built-in) ---
def get_circuit(qubits):
    print("--- Step 2: Designing Quantum Circuit ---")
    f_map = ZZFeatureMap(feature_dimension=qubits, reps=1)
    ansatz = RealAmplitudes(qubits, reps=1)
    return f_map, ansatz

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Get Data
    data, labels = get_data(n_components=16)
    train_data, train_labels = data[:20], labels[:20] # 20 samples for speed

    # 2. Get Circuit Parts
    feature_map, ansatz = get_circuit(qubits=16)

    # 3. Setup VQC
    print("--- Step 3: Initializing VQC Engine ---")
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=10),
        sampler=Sampler()
    )

    # 4. Start Training
    print("--- Step 4: Starting Hybrid Training (Please Wait...) ---")
    try:
        vqc.fit(train_data, train_labels)
        print("\n✅ SUCCESS: Training Complete!")
        print(f"📊 Training Score: {vqc.score(train_data, train_labels)*100:.2f}%")
    except Exception as e:
        print(f"❌ Error: {e}")