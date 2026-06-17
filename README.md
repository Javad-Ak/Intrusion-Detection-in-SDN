# Lightweight and Interpretable Intrusion Detection in SDN

**Author:** Mohammad Javad Akbari

> An offline feature-space reduction framework using consensus values from SHAP and LIME to significantly decrease computing overhead, latency, and memory residency for Intrusion Detection Systems (IDS) in Software-Defined Networks (SDN).

## 📌 Overview

Software-Defined Networking (SDN) isolates control logic from the forwarding plane, shifting the processing burden to a centralized controller. Running complex deep learning architectures for packet inspection on this plane introduces notable processing bottlenecks. This project solves this by combining global **SHAP** parameters with localized **LIME** samples to optimize classical Machine Learning structures and a position-agnostic Tabular Fully Convolutional Network (FCN), forcing a pragmatic trade-off between slight accuracy degradation and massive throughput gains.

## ⚙️ Methodology Pipeline

1. **Chronological Data Ingestion:** Merges InSDN and CSE-CIC-IDS2018 datasets, sorted strictly temporally (no shuffling) to prevent temporal data leakage.
2. **Dual-XAI Consensus Mapping:** Extracts global behaviors via SHAP and local boundaries via LIME (isolated strictly to the validation fold) to generate a unified consensus vector for feature selection.
3. **Reduced Space Execution:** Symmetrically retrains the models strictly on the reduced feature subspace to ensure a purely feature-based evaluation.
4. **Hardware-Constrained Benchmarking:** Evaluates operational viability using a simulated, resource-constrained Docker micro-batching environment.

## 📊 Key Results

The consensus feature reduction framework achieved up to a **~91% increase in throughput** and a **~47% reduction in processing latency** for heavier architectures, with only minor, acceptable degradations in detection accuracy.

### Performance vs. Resource Trade-off (Full vs. Reduced)

| Model | Phase | Accuracy | Macro-F1 | Throughput (PPS) | Latency (ms/batch) |
| --- | --- | --- | --- | --- | --- |
| **Tabular FCN PyTorch** | Full | 0.9939 | 0.8022 | 11,333.0 | 88.23 |
| **Tabular FCN PyTorch** | **Reduced** | 0.9622 | 0.7374 | **21,732.0** | **46.01** |
| **XGBoost** | Full | 0.9988 | 0.8938 | 4,963.0 | 201.47 |
| **XGBoost** | **Reduced** | 0.9956 | 0.8978 | **9,519.0** | **105.05** |
| **Decision Tree** | Full | 0.9982 | 0.8562 | 22,143.0 | 45.16 |
| **Decision Tree** | **Reduced** | 0.9947 | 0.8797 | **26,520.0** | **37.71** |

### Relative Hardware Impact (Consensus Feature Reduction)

| Model | Latency Change | Throughput Change |
| --- | --- | --- |
| **Tabular FCN PyTorch** | **-47.85%** | **+91.76%** |
| **XGBoost** | **-47.86%** | **+91.80%** |
| **MLP DL Baseline** | -19.37% | +24.05% |
| **Decision Tree** | -16.50% | +19.77% |
| **LightGBM** | +3.54% | -3.42% |
| **Random Forest** | +0.25% | -0.26% |
