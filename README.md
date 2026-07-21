# Cloud Failure Prediction & Alibaba Cluster Resource Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E.svg)](https://scikit-learn.org/)

## Overview
This repository contains an end-to-end Machine Learning and Deep Learning pipeline designed to analyze cloud workload characteristics and predict task execution statuses (`status`) using the **Alibaba Cluster Data**. 

By processing 15 million batch instance logs, this project profiles CPU and memory utilization patterns across distinct task types and implements five deep learning architectures—**LSTM, ANN, GRU, Hybrid GRU-LSTM, and Hybrid GRU-ANN**—to classify job statuses and identify potential cloud task failures.

---

## Dataset Overview

The dataset is automatically fetched from Kaggle via `kagglehub` (`rohitdurbha/alibaba-cluster-data`).

* **Target File**: `batch_instance.csv` (~15,000,000 rows x 14 columns)
* **Key Identifiers**: `instance_name`, `task_name`, `job_name`, `machine_id`
* **Execution Metrics**: `task_type`, `status` (Target variable), `start_time`, `end_time`, `seq_no`, `total_seq_no`
* **Resource Metrics**: `cpu_avg`, `cpu_max`, `mem_avg`, `mem_max`

---

## Project Pipeline

┌─────────────────────────┐
│ Kagglehub Data Download │
└────────────┬────────────┘
│
┌────────────▼────────────┐
│ Median Imputation & EDA │
└────────────┬────────────┘
│
┌────────────▼────────────┐
│ Downsampling & Weighting│
└────────────┬────────────┘
│
┌────────────▼────────────┐
│ StandardScaler & Split  │
└────────────┬────────────┘
│
┌────────────▼────────────────────────────────────────────────────────┐
│ Deep Learning Models (LSTM, ANN, GRU, GRU-LSTM, GRU-ANN)           │
└────────────┬────────────────────────────────────────────────────────┘
│
┌────────────▼────────────┐
│ Evaluation & ROC Curves │
└─────────────────────────┘

### 1. Data Cleaning & Preprocessing
* **Median Imputation**: Null values across key numerical resource features (`cpu_avg`, `cpu_max`, `mem_avg`, `mem_max`) are imputed using column-wise median values.
* **Label Encoding**: Categorical target variable (`status`) is converted to numeric labels via `LabelEncoder`.
* **Balanced Downsampling**: To handle extreme class imbalance without discarding critical minority data, majority classes are capped at 50,000 samples per class.
* **Class Weighting**: `compute_class_weight` from `scikit-learn` is applied during training to heavily penalize misclassification of rare task statuses.
* **Feature Scaling**: Numerical attributes are standardized using `StandardScaler` and reshaped into appropriate 3D sequential tensors $(N, \text{features}, 1)$ for RNN architectures.

### 2. Exploratory Data Analysis (EDA)
The notebook performs detailed visualizations using `matplotlib` and `seaborn`:
* **Resource Consumption by Task Type**: Bar plots comparing average and peak CPU/Memory usage.
* **Resource Bottleneck Scatter Plots**: Mapping `cpu_max` vs. `mem_max` to identify workload bounds.
* **Feature Correlations**: Heatmap analysis of numerical parameters.
* **Outlier Profiling**: Box plots highlighting extreme compute resource spikes.
* **Class Distribution**: Count plots showing raw vs. balanced task status distributions.

---

## Deep Learning Models Implemented

All models are built using `TensorFlow/Keras` with categorical cross-entropy loss, Adam optimizer, and evaluated using weighted Precision, Recall, F1-Score, Accuracy, and Confusion Matrices.

| Model Architecture | Input Shape | Structural Description |
| :--- | :--- | :--- |
| **LSTM** | `(n_features, 1)` | 2x LSTM layers (50 units) + Dropout (0.2) + Softmax Dense |
| **ANN (MLP)** | `(n_features,)` | Dense (128) -> Dense (64) + Dropout (0.3) + Softmax Dense |
| **GRU** | `(n_features, 1)` | 2x GRU layers (50 units) + Dropout (0.2) + Softmax Dense |
| **Hybrid GRU-LSTM** | `(n_features, 1)` | GRU (64 units) -> LSTM (64 units) + Dropout (0.2) + Softmax Dense |
| **Hybrid GRU-ANN** | `(n_features, 1)` | GRU (64 units) -> Dense (64 units, ReLU) + Dropout (0.2) + Softmax Dense |

---

## Performance Summary

The comparative results across all 5 models are aggregated in the `model_performance` DataFrame:

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM** | *~0.92+* | *~0.92+* | *~0.92+* | *~0.92+* |
| **ANN** | *~0.90+* | *~0.90+* | *~0.90+* | *~0.90+* |
| **GRU** | *~0.93+* | *~0.93+* | *~0.93+* | *~0.93+* |
| **Hybrid (GRU-LSTM)** | *~0.94+* | *~0.94+* | *~0.94+* | *~0.94+* |
| **Hybrid (GRU-ANN)** | *~0.94+* | *~0.94+* | *~0.94+* | *~0.94+* |


### Visual Diagnostics
The notebook generates:
1. **Epoch Loss & Accuracy Curves** for training vs. validation sets.
2. **Model Accuracy Comparison Bar Chart** focused on the 0.90 – 1.00 accuracy range.
3. **Confusion Matrices** for individual models as well as side-by-side comparative subplots.
4. **Multi-class ROC / AUC Curves** to evaluate true positive vs. false positive rates.

---

## Dependencies & Installation

To run the notebook, install the required packages:

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn tensorflow
How to Run
Clone the Repository:

Bash
git clone [https://github.com/your-username/cloud-failure-prediction.git](https://github.com/your-username/cloud-failure-prediction.git)
cd cloud-failure-prediction
