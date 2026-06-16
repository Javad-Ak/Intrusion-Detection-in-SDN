# Ultra-Lightweight and Interpretable Intrusion Detection in SDN
**A Consensus XAI-Driven Feature Reduction Framework**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Author:** Mohammad Javad Akbari

## 📖 Abstract
Modern network architectures increasingly rely on Software-Defined Networking (SDN) to decouple the control logic from data forwarding planes. However, this centralization introduces severe security risks. Existing Network Intrusion Detection Systems (NIDS) frequently deploy heavy Deep Learning (DL) architectures that induce high computational overhead and unacceptable latency. 

This repository implements an ultra-lightweight, high-throughput, and fully interpretable NIDS pipeline. By leveraging a **Dual-XAI (SHAP + LIME) consensus mechanism** for dimensionality reduction, we systematically eliminate non-informative flow features and deploy classical machine learning classifiers directly optimized for resource-constrained SDN edge controllers.

## ✨ Key Features
* **Zero-Leakage Data Pipeline:** Strict 70/15/15 stratified splitting ensuring zero mathematical leakage during scaling or XAI feature selection.
* **Dual-XAI Consensus:** Utilizes both Global (SHAP) and Local (LIME) attributions, verified via a Jaccard Similarity Index.
* **Dynamic Dimensionality Cutoff:** Uses a 95% Cumulative Variance "Elbow Method" to prevent arbitrary feature thresholding.
* **Hardware-Constrained Micro-Benchmarking:** Includes a Dockerized edge-controller simulation hard-capped to 1 CPU core and 512MB RAM to prove real-world viability against heavy DL architectures (e.g., CNN-LSTM).

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