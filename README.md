# QGAN-and-GAN

A research-oriented project exploring the capabilities of **Classical Generative Adversarial Networks (GANs)** and **Quantum Generative Adversarial Networks (QGANs)** for learning and generating data distributions.

This project begins with a synthetic linear-regression dataset, trains a classical GAN, implements a variational quantum generator using Qiskit, and compares the performance of both approaches.

---

## Overview

Generative Adversarial Networks (GANs) consist of two competing models:

- **Generator** – learns to produce synthetic samples.
- **Discriminator** – learns to distinguish between real and generated samples.

Quantum GANs (QGANs) replace the classical generator with a **Variational Quantum Circuit (VQC)**, enabling data generation through quantum states and measurements.

This repository investigates whether a quantum generator can approximate a target distribution and how its performance compares to a classical GAN.

---

## Dataset Generation

A synthetic dataset is generated using the linear equation:

\[
y = wx + b
\]

where:

- Weight (**w**) = 0.7
- Bias (**b**) = 0.3

### Dataset Parameters


 Start = 0 
 End = 1 
 Step Size = 0.002 
 Samples = 500 

Implemented using PyTorch:

```python
y = 0.7 * X + 0.3
```


---

## Classical GAN

### Architecture

#### Generator

A feed-forward neural network that transforms random noise into synthetic samples.

#### Discriminator

A neural network that classifies samples as:

- Real → 1
- Fake → 0

### Training Components

- Binary Cross Entropy Loss (BCELoss)
- Adam Optimizer
- Adversarial Training

### Workflow

```text
Random Noise
      ↓
Generator
      ↓
Fake Samples
      ↓
Discriminator
      ↓
Real / Fake Classification
```

---

## Quantum GAN (QGAN)

### Quantum Generator

The QGAN generator is implemented using **Qiskit** and consists of a parameterized quantum circuit.

### Technologies Used

- Qiskit
- Qiskit Aer Simulator
- Variational Quantum Circuits (VQC)
- Parameterized Rotation Gates
- Quantum Entanglement

### Circuit Architecture

```text
RY Layer
    ↓
CNOT Entanglement Layer
    ↓
RY Layer
    ↓
Measurement
```

### Configuration


Qubits = 4
Trainable Parameters = 8
Shots = 1000
Backend = AerSimulator

### Example Circuit

```python
for i in range(4):
    qc.ry(theta[i], i)

for i in range(3):
    qc.cx(i, i + 1)

for i in range(4):
    qc.ry(theta[i + 4], i)
```

---

## Quantum Sample Generation

After executing the quantum circuit:

1. Measurement counts are collected.
2. Bitstrings are converted into integers.
3. Integers are scaled into numerical values.
4. Generated values form the synthetic dataset.

Example:

```text
Bitstring: 1011

Binary → Decimal
1011 → 11

Scaled into range [-1,1]
```

---

## QGAN Optimization Strategy

The current implementation uses a **Random Parameter Search** approach.

### Procedure

For each trial:

1. Generate random quantum parameters.
2. Execute the quantum circuit.
3. Generate fake samples.
4. Evaluate samples using the discriminator.
5. Retain the best-performing parameter set.

Workflow:

```text
Quantum Parameters
        ↓
Quantum Circuit
        ↓
Generated Samples
        ↓
Discriminator Score
        ↓
Best Parameters Selected
```

---

## Evaluation Metrics

### 1. Discriminator Score

Measures how realistic the generated samples appear.

Interpretation:


if score = 0.0 ---> Completely Fake 
if score = 0.5 ---> Discriminator Confused 
if score = 1.0 ---> Indistinguishable from Real

---

### 2. Mean Squared Error (MSE)

Used to compare generated and target distributions.

```text
MSE = (Real Mean − Generated Mean)²
```

Lower values indicate closer agreement between distributions.

---

## Experimental Results

### QGAN Results


Qubits = 4
Parameters = 8
Trials = 100
Best Discriminator Score = 0.5707
Real Mean = 0.4990 
Generated Mean = 0.7113
MSE = 0.0451

### Observations

- The quantum generator successfully learned important characteristics of the target distribution.
- The discriminator achieved a confidence score greater than 0.5 on generated samples.
- The generated distribution partially matched the real distribution.
- The low MSE demonstrates reasonable approximation quality despite using a small quantum circuit.

---

## Project Structure

```text
QGAN-and-GAN/
│
├── LR dataset/
│   ├── generationds.py
│   └── requirements.txt
│
├── classical_gan/
│   ├── generator.py
│   ├── discriminator.py
│   ├── train.py
│   └── gan_performance.png
│
├── qgan/
│   ├── quantum_generator.py
│   ├── discriminator.py
│   ├── qgan_training.py
│   └── evaluation.py
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/murtazaansari2504/QGAN-and-GAN.git

cd QGAN-and-GAN
```

Install dependencies:

```bash
pip install torch matplotlib numpy scikit-learn qiskit qiskit-aer
```

---

## Running Dataset Generation

```bash
python "LR dataset/generationds.py"
```

---

## Running Classical GAN

```bash
python classical_gan/train.py
```

---

## Running QGAN

```bash
python qgan/qgan_training.py
```

---

## Research Objectives

- Study the fundamentals of GANs and QGANs.
- Implement a variational quantum generator.
- Compare classical and quantum generative approaches.
- Evaluate generated distributions using statistical metrics.
- Explore the potential of quantum machine learning in generative modeling.

---

## Future Work

- Gradient-based QGAN optimization
- Parameter Shift Rule implementation
- SPSA and COBYLA optimizers
- Hybrid Quantum-Classical Adversarial Training
- MNIST Dataset Experiments
- Image Generation using QGANs
- IBM Quantum Hardware Execution
- Comprehensive GAN vs QGAN Benchmarking

---

## Authors

### Murtaza Ansari

GitHub: https://github.com/murtazaansari2504

### Devansh Rathore

GitHub: https://github.com/devansh025

---

## License

This project is intended for academic, educational, and research purposes.
