import os
import time
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

ARTIFACTS_DIR = Path('/app/artifacts')
MODELS_DIR = ARTIFACTS_DIR / 'models'
MICRO_BATCH_SIZE = 16

print("==================================================")
print("Executing Isolated Local Controller Evaluation...")
print("Resource Parameters: 1.0 Allocated CPU Core | 512 MB RAM Limit")
print("==================================================\n")

class TabularFCN(nn.Module):
    """
    Unified Tabular FCN architecture.
    Both baseline and reduced variants must inherit this exact structure
    to isolate the feature reduction as the sole independent variable.
    """
    def __init__(self, input_dim, num_classes):
        super(TabularFCN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Treat input features as a sequence of length L with 1 channel
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_avg_pool(x).squeeze(-1) 
        x = self.fc(x)
        return x

def load_data():
    print("Extracting targets from test data splits...")
    data_path = ARTIFACTS_DIR / 'test_splits.pkl'
    if not data_path.exists():
        raise FileNotFoundError("Target pkl metrics data file not resolved within artifacts.")
    
    X_test_red, X_test_scaled, y_test = joblib.load(data_path)
    return X_test_red, X_test_scaled, y_test

def streaming_benchmark(name, model, X_test_df, y_test_arr, phase_name):
    print(f"Tracking evaluation metrics for: {name} [{phase_name}]")
    preds = []
    
    X_test_arr = X_test_df.values
    total_samples = len(X_test_arr)
    total_batches = np.ceil(total_samples / MICRO_BATCH_SIZE)
    
    start_infer = time.perf_counter()
    
    if 'PyTorch' in name:
        model.eval()
        with torch.no_grad():
            for i in range(0, total_samples, MICRO_BATCH_SIZE):
                batch = torch.tensor(X_test_arr[i:i+MICRO_BATCH_SIZE], dtype=torch.float32)
                out = model(batch)
                preds.extend(torch.argmax(out, dim=1).numpy())
    else:
        for i in range(0, total_samples, MICRO_BATCH_SIZE):
            batch = X_test_arr[i:i+MICRO_BATCH_SIZE]
            preds.extend(model.predict(batch))
            
    infer_time = time.perf_counter() - start_infer
    
    # Compute metrics mapped to expected notebook plot variables
    latency_ms_per_batch = (infer_time * 1000) / total_batches
    pps = total_samples / infer_time
    acc = float(accuracy_score(y_test_arr, preds))
    macro_f1 = float(f1_score(y_test_arr, preds, average='macro'))
    
    print(f"   Execution Delay: {latency_ms_per_batch:.2f} ms/batch | Processing Threshold: {int(pps)} PPS | Unweighted Macro F1: {macro_f1:.4f}\n")
    
    return {
        'Model': name,
        'Phase': phase_name,
        'Latency_ms_per_batch': round(latency_ms_per_batch, 2),
        'Throughput_PPS': int(pps),
        'Accuracy': round(acc, 4),
        'Macro-F1': round(macro_f1, 4)
    }

def main():
    try:
        X_test_red, X_test_scaled, y_test = load_data()
    except Exception as e:
        print(e)
        return

    results = []
    target_classifiers = ['Decision_Tree', 'Random_Forest', 'XGBoost', 'LightGBM', 'MLP_DL_Baseline']
    num_classes = len(np.unique(y_test))
    
    # 1. Evaluate Full Baseline
    for name in target_classifiers:
        model_path = MODELS_DIR / f"{name}_full.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            res = streaming_benchmark(f"{name.replace('_', ' ')}", model, X_test_scaled, y_test, 'Full_Baseline')
            res['Model Size (KB)'] = round(model_path.stat().st_size / 1024, 1)
            results.append(res)

    torch_full_path = MODELS_DIR / "Tabular_FCN_PyTorch_full.pth"
    if torch_full_path.exists():
        torch_full = TabularFCN(input_dim=X_test_scaled.shape[1], num_classes=num_classes)
        torch_full.load_state_dict(torch.load(torch_full_path, weights_only=True))
        res = streaming_benchmark('Tabular FCN PyTorch', torch_full, X_test_scaled, y_test, 'Full_Baseline')
        res['Model Size (KB)'] = round(torch_full_path.stat().st_size / 1024, 1)
        results.append(res)
        
    # 2. Evaluate Reduced Feature Space
    for name in target_classifiers:
        model_path = MODELS_DIR / f"{name}_reduced.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            res = streaming_benchmark(f"{name.replace('_', ' ')}", model, X_test_red, y_test, 'Reduced_Space')
            res['Model Size (KB)'] = round(model_path.stat().st_size / 1024, 1)
            results.append(res)

    torch_red_path = MODELS_DIR / "Tabular_FCN_PyTorch_reduced.pth"
    if torch_red_path.exists():
        torch_red = TabularFCN(input_dim=X_test_red.shape[1], num_classes=num_classes)
        torch_red.load_state_dict(torch.load(torch_red_path, weights_only=True))
        res = streaming_benchmark('Tabular FCN PyTorch', torch_red, X_test_red, y_test, 'Reduced_Space')
        res['Model Size (KB)'] = round(torch_red_path.stat().st_size / 1024, 1)
        results.append(res)

    df_results = pd.DataFrame(results)
    df_results.to_csv(ARTIFACTS_DIR / 'docker_stress_test_results.csv', index=False)
    print("\nTelemetry Tracking Complete. Output dataset saved to artifacts/docker_stress_test_results.csv")

if __name__ == "__main__":
    main()
