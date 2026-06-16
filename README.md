# Ultra-Lightweight and Interpretable Intrusion Detection in SDN
**A Consensus XAI-Driven Feature Reduction Framework**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Author:** Mohammad Javad Akbari

## 📖 Abstract
Modern network architectures increasingly rely on Software-Defined Networking (SDN) to decouple the control logic from data forwarding planes. However, this centralization introduces severe security risks. Existing Network Intrusion Detection Systems (NIDS) frequently deploy heavy Deep Learning (DL) architectures that induce high computational overhead and unacceptable latency. 

This repository implements an ultra-lightweight, high-throughput, and fully interpretable NIDS pipeline. By leveraging a **Dual-XAI (SHAP + LIME) consensus mechanism** for dimensionality reduction, I systematically eliminate non-informative flow features and deploy classical machine learning classifiers directly optimized for resource-constrained SDN edge controllers.

## ✨ Key Features
* **Zero-Leakage Data Pipeline:** Strict 70/15/15 stratified splitting ensuring zero mathematical leakage during scaling or XAI feature selection.
* **Dual-XAI Consensus:** Utilizes both Global (SHAP) and Local (LIME) attributions, verified via a Jaccard Similarity Index.
* **Dynamic Dimensionality Cutoff:** Uses a 95% Cumulative Variance "Elbow Method" to prevent arbitrary feature thresholding.
* **Hardware-Constrained Micro-Benchmarking:** Includes a Dockerized edge-controller simulation hard-capped to 1 CPU core and 512MB RAM to prove real-world viability against heavy DL architectures (e.g., CNN-LSTM).

## 📊 Hardware-Constrained Edge Benchmarks
The following empirical metrics were evaluated within the Docker container simulation environments under strict resource quotas (**1.0 CPU Core Limit | 512MB RAM Limit**):

| Model Architecture | Feature Subspace Phase | Latency / Sample (µs) | Throughput (PPS) | Accuracy | Macro-$F_1$ | Model Size (KB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Tuned)** | Reduced (Consensus XAI) | **37.20** | **26,880** | 0.9973 | **0.9030** | **319.1** |
| **LightGBM (Tuned)** | Reduced (Consensus XAI) | 127.73 | 7,828 | 0.9974 | 0.8621 | 10,543.3 |
| **XGBoost (Tuned)** | Reduced (Consensus XAI) | 135.91 | 7,357 | 0.9979 | 0.8979 | 3,915.3 |
| **Random Forest (Tuned)** | Reduced (Consensus XAI) | 137.58 | 7,268 | 0.9977 | 0.9024 | 14,260.0 |
| **CNN-LSTM (PyTorch)** | Full Baseline (Heavy) | 631.34 | 1,583 | 0.9966 | 0.8131 | 443.4 |
| **MLP (DL Baseline)** | Full Baseline (Heavy) | 639.98 | 1,562 | 0.9972 | 0.8059 | 4,831.7 |

### 📈 Key Analytics
* **Speedup & Throughput Acceleration:** The Consensus XAI-driven Decision Tree runs **16.9x faster per sample** than the hybrid CNN-LSTM network, processing an extra **25,297 packets per second** under identical core limitations.
* **Metric Integrity Protection:** This massive latency mitigation was achieved with *no cost to accuracy*, actually expanding the macro-$F_1$ boundary from **0.8131 (CNN-LSTM)** to **0.9030 (Decision Tree)** due to the elimination of noisy, non-informative flow attributes.

## 🗂️ Repository Structure
```text
.
├── datasets/                 # Place InSDN and CSE-CIC-IDS2018 CSVs here
├── artifacts/                # Output models, figures, and benchmark CSVs
├── NIDS_Pipeline.ipynb       # Main training and feature reduction pipeline
├── edge_benchmark.py         # Throttled micro-batch streaming script
├── requirements.txt          # Python dependencies
├── Dockerfile                # Edge-controller simulation environment
└── docker-compose.yml        # Hardware constraint configurations