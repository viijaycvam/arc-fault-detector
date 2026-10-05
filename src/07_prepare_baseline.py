import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path

MAT_FILE = Path(r"data\raw\arc_fault_dataset\Datasets\dataset_14classes_20ms_50kHz.mat")
OUTPUT_FILE = Path(r"reports\baseline_features.csv")

def extract_features(X: np.ndarray) -> pd.DataFrame:
    """Extract AC-coupled time-domain features per waveform window using vectorized NumPy."""
    X = X.astype(np.float64)
    # AC coupling: remove DC offset per window to eliminate baseline drift
    ac = X - X.mean(axis=1, keepdims=True)
    
    rms = np.sqrt((ac ** 2).mean(axis=1))
    peak = np.abs(ac).max(axis=1)
    p2p = np.ptp(ac, axis=1)
    std = ac.std(axis=1)
    crest = np.divide(peak, rms, out=np.zeros_like(rms), where=rms > 0)
    
    return pd.DataFrame({
        "mean": X.mean(axis=1),
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": p2p,
        "crest_factor": crest,
    })

def main():
    if not MAT_FILE.exists():
        print(f"Error: MAT dataset not found at {MAT_FILE}")
        return

    # 1. Load 14-class MAT dataset
    data = scipy.io.loadmat(MAT_FILE)
    X_train, y_train_det = data["yTrain"], data["labelsTrain"].ravel().astype(int)
    X_test, y_test_det = data["yTest"], data["labelsTest"].ravel().astype(int)

    # 2. Extract features using vectorized AC-coupling
    df_train_feats = extract_features(X_train)
    df_test_feats = extract_features(X_test)

    # 3. Build structured DataFrames with identifiers and labels
    df_train = pd.DataFrame({
        "record_id": range(len(X_train)),
        "split": "train",
        "detailed_label": y_train_det,
        "binary_label": (y_train_det > 0).astype(int),
    }).join(df_train_feats)

    df_test = pd.DataFrame({
        "record_id": range(len(X_train), len(X_train) + len(X_test)),
        "split": "test",
        "detailed_label": y_test_det,
        "binary_label": (y_test_det > 0).astype(int),
    }).join(df_test_feats)

    # 4. Combine train and test splits
    df = pd.concat([df_train, df_test], ignore_index=True)

    # 5. Save to CSV
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # 6. Print Summary
    print("=" * 70)
    print("B2.1 — BASELINE FEATURE DATASET GENERATOR")
    print("=" * 70)
    print(f"Total Records : {len(df)} (Train: {len(df_train)}, Test: {len(df_test)})")
    print(f"Normal Samples: {(df.binary_label == 0).sum()}")
    print(f"Arc Samples   : {(df.binary_label == 1).sum()}")
    print(f"Feature Count : 6 (mean, std, rms, peak, peak_to_peak, crest_factor)")
    print("-" * 70)
    print(f"Saved dataset : {OUTPUT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()