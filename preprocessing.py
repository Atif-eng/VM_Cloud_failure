import os
import glob
import numpy as np
import pandas as pd
import kagglehub

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical


def load_and_preprocess_data(max_samples_per_class: int = 50000, test_size: float = 0.2, random_state: int = 42):
    """
    Downloads Alibaba cluster dataset, imputes missing values, balances classes,
    scales features, and prepares datasets for ANN and Sequential (LSTM/GRU) models.
    """
    print("📥 Fetching dataset from Kaggle...")
    path = kagglehub.dataset_download("rohitdurbha/alibaba-cluster-data")
    
    found_csv_files = glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)
    target_files = [f for f in found_csv_files if 'batch_instance.csv' in f]
    
    if not target_files:
        raise FileNotFoundError("batch_instance.csv not found in downloaded dataset.")
        
    data = pd.read_csv(target_files[0])
    print(f"✅ Loaded dataset shape: {data.shape}")

    # 1. Median Imputation
    num_cols = ['cpu_avg', 'cpu_max', 'mem_avg', 'mem_max']
    for col in num_cols:
        if col in data.columns and data[col].isnull().any():
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            print(f"Imputed missing values in '{col}' with median: {median_val:.4f}")

    # 2. Categorical Encoding
    le = LabelEncoder()
    data['status_encoded'] = le.fit_transform(data['status'])

    # 3. Class Balancing
    status_counts = data['status_encoded'].value_counts()
    balanced_dfs = []

    for status_label in status_counts.index:
        df_class = data[data['status_encoded'] == status_label]
        if len(df_class) > max_samples_per_class:
            df_sampled = df_class.sample(n=max_samples_per_class, random_state=random_state)
            balanced_dfs.append(df_sampled)
        else:
            balanced_dfs.append(df_class)

    data_balanced = (
        pd.concat(balanced_dfs, axis=0)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    # 4. Compute Class Weights
    classes = np.unique(data_balanced['status_encoded'])
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=data_balanced['status_encoded']
    )
    class_weights_dict = dict(zip(classes, weights))

    # 5. Feature Isolation
    drop_cols = ['instance_name', 'task_name', 'job_name', 'status', 'machine_id', 'status_encoded']
    feature_cols = [c for c in data_balanced.columns if c not in drop_cols]
    
    X_balanced = data_balanced[feature_cols]
    y_balanced = data_balanced['status_encoded']

    # 6. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)
    y_one_hot = to_categorical(y_balanced)

    # 2D Split (for ANN)
    X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(
        X_scaled, y_one_hot, test_size=test_size, random_state=random_state, stratify=y_balanced
    )

    # 3D Reshape & Split (for LSTM / GRU)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        X_reshaped, y_one_hot, test_size=test_size, random_state=random_state, stratify=y_balanced
    )

    return {
        'ann': (X_train_ann, X_test_ann, y_train_ann, y_test_ann),
        'seq': (X_train_seq, X_test_seq, y_train_seq, y_test_seq),
        'class_weights': class_weights_dict
    }

if __name__ == '__main__':
    # Test script execution
    processed_data = load_and_preprocess_data()
    print("🎉 Preprocessing completed successfully.")