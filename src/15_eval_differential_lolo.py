import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

CSV_FILE = Path(r"reports\dsp_features.csv")

def main():
    if not CSV_FILE.exists():
        print(f"Error: DSP Feature CSV not found at {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)

    # Calculate baseline normal statistics across steady-state non-arcing data
    normal_df = df[df["binary_label"] == 0]
    
    mean_hf = normal_df["power_high"].mean()
    std_hf  = normal_df["power_high"].std() + 1e-9
    
    mean_kf = normal_df["kurtosis"].mean()
    std_kf  = normal_df["kurtosis"].std() + 1e-9

    # Generate relative & baseline differential metrics
    df["delta_power_high"] = (df["power_high"] - mean_hf) / std_hf
    df["delta_kurtosis"]   = (df["kurtosis"] - mean_kf) / std_kf
    df["hf_ratio_norm"]    = df["hf_ratio"] / (df["hf_ratio"].mean() + 1e-9)

    # 17 Feature Matrix: 14 Core DSP Features + 3 Differential Deltas
    feature_cols = [
        "mean", "std", "rms", "peak", "peak_to_peak", "crest_factor",
        "skewness", "kurtosis", "total_power", "power_low",
        "power_mid", "power_high", "hf_ratio", "spectral_centroid",
        "delta_power_high", "delta_kurtosis", "hf_ratio_norm"
    ]

    print("=" * 80)
    print("STEP 15 — OPTIMIZED FULL DSP + DIFFERENTIAL DELTA EVALUATION (LOLO)")
    print("=" * 80)
    print(f"Features Used ({len(feature_cols)}): Combined 14 Raw DSP + 3 Baseline Deltas")
    print("-" * 80)

    lolo_results = []

    for test_class in range(1, 14):
        train_mask = ~(df["detailed_label"] == test_class)
        test_mask = (df["detailed_label"] == test_class)

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "binary_label"].values

        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, "binary_label"].values

        if len(X_test) == 0:
            continue

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        mlp = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.01,
            max_iter=500,
            random_state=42
        )
        mlp.fit(X_tr_scaled, y_train)

        preds = mlp.predict(X_te_scaled)
        det_rate = (preds == 1).mean() * 100

        lolo_results.append(det_rate)
        print(f"Held-out Class {test_class:<2d} | Samples: {len(X_test):<4} | Detection Rate: {det_rate:6.2f}% ({preds.sum()}/{len(X_test)})")

    print("-" * 80)
    print(f"Overall Average Detection Rate: {np.mean(lolo_results):.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()