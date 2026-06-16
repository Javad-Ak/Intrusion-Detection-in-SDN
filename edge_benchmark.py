import os
import time
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

ARTIFACTS_DIR = Path('/app/artifacts')
MODELS_DIR = ARTIFACTS_DIR / 'models'
MICRO_BATCH_SIZE = 16

print("==================================================")
print("Initiating Hardware-Constrained Edge Simulation...")
print("Constraints: 1.0 CPU Core | 512 MB RAM")
print("==================================================\n")

class Deep1DCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(Deep1DCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def load_data():
    print("Loading serialized test splits...")
    data_path = ARTIFACTS_DIR / 'test_splits.pkl'
    if not data_path.exists():
        raise FileNotFoundError("test_splits.pkl not found! Run the Jupyter Notebook Phase 5 first.")
    
    # Load the exact test sets used in the main notebook
    X_test_red, X_test_scaled, y_test = joblib.load(data_path)
    return X_test_red, X_test_scaled, y_test

def streaming_benchmark(name, model, X_test_df, y_test_arr, phase_name):
    print(f"Benchmarking: {name} [{phase_name}]")
    preds = []
    start_infer = time.perf_counter()
    
    if 'CNN' in name:
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X_test_df), MICRO_BATCH_SIZE):
                batch = torch.tensor(X_test_df.iloc[i:i+MICRO_BATCH_SIZE].values, dtype=torch.float32).unsqueeze(1)
                out = model(batch)
                preds.extend(torch.argmax(out, dim=1).numpy())
    else:
        for i in range(0, len(X_test_df), MICRO_BATCH_SIZE):
            batch = X_test_df.iloc[i:i+MICRO_BATCH_SIZE]
            preds.extend(model.predict(batch))
            
    infer_time = time.perf_counter() - start_infer
    
    latency_us = (infer_time / len(X_test_df)) * 1e6
    pps = len(X_test_df) / infer_time
    acc = accuracy_score(y_test_arr, preds)
    macro_f1 = f1_score(y_test_arr, preds, average='macro')
    
    print(f"  -> Latency: {latency_us:.2f} us/sample | Throughput: {int(pps)} PPS | F1: {macro_f1:.4f}\n")
    
    return {
        'Model': name,
        'Phase': phase_name,
        'Latency/Sample (µs)': round(latency_us, 2),
        'Throughput (PPS)': int(pps),
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
    
    # Define the models we expect to find
    classical_models = ['Decision_Tree', 'Random_Forest', 'XGBoost', 'LightGBM']
    
    # Benchmark Reduced Models
    for name in classical_models:
        model_path = MODELS_DIR / f"{name}_reduced.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            res = streaming_benchmark(f"{name.replace('_', ' ')} (Tuned)", model, X_test_red, y_test, 'Reduced (Consensus XAI)')
            
            # Add size dynamically
            res['Model Size (KB)'] = round(os.path.getsize(model_path) / 1024, 1)
            results.append(res)
            
    # Benchmark DL Baselines
    mlp_path = MODELS_DIR / "MLP_DL_Baseline.pkl"
    if mlp_path.exists():
        mlp = joblib.load(mlp_path)
        res = streaming_benchmark('MLP (DL Baseline)', mlp, X_test_scaled, y_test, 'Full Baseline (Heavy)')
        res['Model Size (KB)'] = round(os.path.getsize(mlp_path) / 1024, 1)
        results.append(res)
        
    cnn_path = MODELS_DIR / "1D_CNN_PyTorch.pth"
    if cnn_path.exists():
        cnn = Deep1DCNN(num_classes=len(np.unique(y_test)))
        cnn.load_state_dict(torch.load(cnn_path, weights_only=True))
        res = streaming_benchmark('1D CNN (PyTorch)', cnn, X_test_scaled, y_test, 'Full Baseline (Heavy)')
        res['Model Size (KB)'] = round(os.path.getsize(cnn_path) / 1024, 1)
        results.append(res)

    # Save final stress-test metrics
    df_results = pd.DataFrame(results)
    df_results.to_csv(ARTIFACTS_DIR / 'docker_stress_test_results.csv', index=False)
    print("\nSimulation Complete. Results saved to artifacts/docker_stress_test_results.csv")

if __name__ == "__main__":
    main()