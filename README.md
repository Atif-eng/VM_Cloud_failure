# ☁️ VM Cloud Failure Prediction

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Deep Learning pipeline designed to predict Virtual Machine (VM) failures in cloud infrastructure environments using time-series metrics from the **Alibaba Cluster Dataset**.

---

## 📌 Project Overview

Proactive failure detection in cloud infrastructure allows resource managers to migrate workloads prior to node crashes, preventing downtime and minimizing SLA penalties. This project implements and compares four distinct neural network architectures to classify machine failure states:

*   **ANN (Artificial Neural Network):** Baseline dense network for 2D tabular feature sets.
*   **LSTM (Long Short-Term Memory):** Recurrent architecture optimized for capturing long-range temporal dependencies.
*   **GRU (Gated Recurrent Unit):** Efficient recurrent model with fewer parameters than standard LSTMs.
*   **Hybrid GRU-LSTM:** Dual-layer recurrent architecture combining sequence representation features.

---

## 🏗️ Repository Structure

```text
vm-cloud-failure-prediction/
│
├── .gitignore               # Ignores cached binaries, datasets, and virtualenvs
├── README.md                # Project documentation
├── requirements.txt         # Pinned python dependencies
├── preprocessing.py         # Data loading, imputation, balancing, and scaling
└── train.py                 # Training pipeline and model evaluation runner
