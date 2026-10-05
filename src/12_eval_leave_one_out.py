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

    # Compute normalized relative features (invariant to load current magnitude)
    df["rel_power_high"] = df["power_high"] / (df["total_power"] + 1e-9)
    df["rel_power_mid"] = df["power_mid"] / (df["total_power"] + 1e-9)
    df["rel_power_low"] = df["power_low"] / (df["total_power"] + 1e-9)

    # Relative feature set
    feature_cols = [
        "crest_factor", "skewness", "kurtosis", 
        "rel_power_high", "rel_power_mid", "rel_power_low", 
        "hf_ratio", "spectral_centroid"
    ]

    print("=" * 70)
    print("B5 — LOLO EVALUATION (RELATIVE FEATURE NORMALIZATION)")
    print("=" * 70)
    print(f"Features Used ({len(feature_cols)}): {feature_cols}")
    print("-" * 70)

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
        det_count = preds.sum()
        total_count = len(preds)

        lolo_results.append({
            "held_out_class": test_class,
            "total_samples": total_count,
            "detected_arcs": det_count,
            "detection_rate": det_rate
        })

        print(f"Held-out Appliance Class {test_class:2d} : {det_rate:6.2f}% detected ({det_count}/{total_count})")

    results_df = pd.DataFrame(lolo_results)
    avg_det_rate = results_df["detection_rate"].mean()

    print("-" * 70)
    print(f"Average Unseen Appliance Detection Rate: {avg_det_rate:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()