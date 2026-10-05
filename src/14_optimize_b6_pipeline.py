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

    # Standard 14 DSP feature set (Preserves both magnitude and frequency data)
    feature_cols = [
        "mean", "std", "rms", "peak", "peak_to_peak", "crest_factor",
        "skewness", "kurtosis", "total_power", "power_low",
        "power_mid", "power_high", "hf_ratio", "spectral_centroid"
    ]

    print("=" * 80)
    print("B6 — STABILIZED 14 DSP FEATURE PIPELINE (LOLO TEST)")
    print("=" * 80)
    print(f"Features Used ({len(feature_cols)}): Standard DSP & High-Frequency Power Matrix")
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