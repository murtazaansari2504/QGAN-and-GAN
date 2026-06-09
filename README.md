# QGAN-and-GAN


This repository contains experiments, implementations, and supporting code related to **Generative Adversarial Networks (GANs)** and **Quantum Generative Adversarial Networks (QGANs)**.

## Current Contents

### 1. Linear Regression Dataset Generation using PyTorch

A simple PyTorch script that generates synthetic data following the linear equation:

[
y = wx + b
]

where:

* **Weight (w):** 0.7
* **Bias (b):** 0.3

The generated dataset consists of **500 samples** and can be used for learning and experimenting with basic machine learning models.

---

## Dataset Generation

The dataset is created using PyTorch tensors.

### Parameters

* Start value: `0`
* End value: `1`
* Step size: `0.002`
* Number of samples: `500`

The target values are computed as:

```python
y = 0.7 * X + 0.3
```

---

## Project Structure

```
QGAN-and-GAN/
│
├── LR dataset/
│   ├── generationds.py
│   └── requirements.txt
│
└── README.md
```

---

## Requirements

* Python 3.11+
* PyTorch

Install dependencies using:

```bash
pip install -r "LR dataset/requirements.txt"
```

or directly:

```bash
pip install torch
```

---

## Running the Script

Navigate to the project directory and run:

```bash
python "LR dataset/generationds.py"
```

The script outputs:

* The first 400 values of `X`
* The first 400 values of `y`
* Shapes of both tensors
* Total number of samples

---

## Example Output

```text
Shape of X: torch.Size([500, 1])
Shape of y: torch.Size([500, 1])

Number of X samples: 500
Number of y samples: 500
```

---

## Future Work

* Classical GAN implementation using PyTorch
* Quantum GAN (QGAN) implementation using PennyLane/Qiskit
* Comparative analysis between GAN and QGAN performance
* Visualization of generated distributions
* Research-oriented experiments and documentation

---

## Author

**Murtaza Ansari**

GitHub: https://github.com/murtazaansari2504
