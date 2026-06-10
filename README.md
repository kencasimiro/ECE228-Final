# Physics-Informed Constraints and Generative Models for Structural Health Monitoring

**Author:** Kenneth Casimiro (Team 59)  
**Institution:** University of California, San Diego (UCSD), Department of Electrical and Computer Engineering  

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
Automated Structural Health Monitoring (SHM) systems using standard deep learning are highly brittle to environmental noise (shadows, stains) and geometric distractors (the "Spalling Paradox"). 

This repository introduces a **Scientific Machine Learning (SciML)** framework that bridges statistical anomaly detection with deterministic solid mechanics. By training an unsupervised Convolutional $\beta$-Variational Autoencoder ($\beta$-VAE) to establish a healthy structural prior, anomalies are isolated via Mean Squared Error ($\mathcal{E}_{MSE}$). Crucially, this pipeline implements a post-reconstruction deterministic physics filter that mathematically evaluates visual anomalies against:
1. **Analytical Solid Mechanics** (Airy Stress Fields)
2. **Topological Curvature** (Hessian Matrix Eigenvalues)
3. **Optical Cavity Physics** (Variance-based light trapping)

This approach mathematically erases out-of-distribution noise and verifies true structural defects, bypassing the need for exhaustive pixel-level annotations.

final.ipynb is the organized final notebook that showcases my work. 
z_exploratory_analysis.ipynb is a notebook used to run different scenarios/change of parameters.

---

## Features
* **Baseline Formulation:** Supervised ResNet-50 for comparative performance metrics.
* **Unsupervised Generative Prior:** Custom Convolutional $\beta$-VAE mapping complex concrete textures into a continuous latent distribution.
* **The SciML Master Equation:** Custom PyTorch tensor operations integrating:
  * Scharr directional gradient extraction.
  * Analytical Solid Mechanics (Airy Stress Field)
  * Topological Curvature (Hessian Matrix Analysis)
  * Optical Cavity Variance
* **Environmental Stress Injection:** Stochastic pipeline to synthetically degrade test imagery with Gaussian blur and severe geometric shadowing.

---

## 🛠 Prerequisites and Installation

This project is built using Python and PyTorch. To run the code, you will need a machine with CUDA support (recommended for faster training) or a standard CPU.

### 1. Install Required Libraries
You can install all necessary dependencies using `pip`:

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow
```

### Datasets
Files can be downloaded at the following location: 
https://apps.peer.berkeley.edu/phi-net/?page_id=1304

### File setup
```
├── dataset/
│   ├── task7_X_train.npy
│   ├── task7_y_train.npy
│   ├── task7_X_test.npy
│   ├── task7_y_test.npy
│   ├── task8_X_train.npy
│   ├── task8_y_train.npy
│   ├── task8_X_test.npy
│   └── task8_y_test.npy
├── final.ipynb
├── analysis.ipynb
└── README.md
```
