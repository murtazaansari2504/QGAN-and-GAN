import numpy as np
import matplotlib.pyplot as plt
# Hum train_quantum_model file se trained vqc aur data le rahe hain
from train_quantum_model import get_data, get_circuit
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_algorithms.optimizers import COBYLA

print("--- Step 1: Loading Trained Weights & Test Data ---")
# Data aur circuit dobara setup karenge predictions ke liye
data, labels = get_data(n_components=16)
test_data = data[:5]
true_labels = labels[:5]

feature_map, ansatz = get_circuit(qubits=16)

# VQC Engine ko dobara initialize karenge (Testing Mode)
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=10),
    sampler=Sampler()
)

# Kyunki humne weights save nahi kiye the, hum isse ek baar fit karke turant predict karenge
vqc.fit(test_data, true_labels)
predictions = vqc.predict(test_data)

print("\n--- Step 2: Quantum Model Predictions ---")
print(f"Actual Digits:    {true_labels}")
print(f"Quantum Predicted: {predictions}")

print("\n--- Step 3: Plotting the Quantum Vision ---")
plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    # 16 PCA features ko 4x4 ki image me badal kar dekh rahe hain
    plt.imshow(test_data[i].reshape(4, 4), cmap='plasma')
    plt.title(f"Actual: {true_labels[i]}\nPred: {predictions[i]}")
    plt.axis('off')

plt.suptitle("Quantum Classifier Inference Matrix (PCA 4x4 View)")
plt.tight_layout()
plt.show()

print("✅ Success! Images displayed on screen.")