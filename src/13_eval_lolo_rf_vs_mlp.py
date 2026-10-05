import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

CSV_FILE = Path(r"reports\dsp_features.csv")

def main():
    if not CSV_FILE.exists():
        print(f"Error: DSP Feature CSV not found at {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)

    # Combined Absolute + Relative Features
    df["rel_power_high"] = df["power_high"] / (df["total_power"] + 1e-9)
    df["rel_power_mid"] = df["power_mid"] / (df["total_power"] + 1e-9)

    feature_cols = [
        "mean", "std", "rms", "peak", "peak_to_peak", "crest_factor",
        "skewness", "kurtosis", "total_power", "power_low",
        "power_mid", "power_high", "hf_ratio", "spectral_centroid",
        "rel_power_high", "rel_power_mid"
    ]

    print("=" * 80)
    print("B5 — LOLO HEAD-TO-HEAD: RANDOM FOREST vs LIGHTWEIGHT MLP")
    print("=" * 80)
    print(f"{'Class':<8} | {'Samples':<8} | {'Random Forest Det %':<22} | {'MLP Det %':<15}")
    print("-" * 80)

    rf_results = []
    mlp_results = []

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

        # 1. Random Forest
        rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_det = (rf_preds == 1).mean() * 100

        # 2. Lightweight MLP
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
        mlp_preds = mlp.predict(X_te_scaled)
        mlp_det = (mlp_preds == 1).mean() * 100

        rf_results.append(rf_det)
        mlp_results.append(mlp_det)

        print(f"Class {test_class:<2d} | {len(X_test):<8} | {rf_det:6.2f}% ({rf_preds.sum()}/{len(X_test)}){'':<6} | {mlp_det:6.2f}% ({mlp_preds.sum()}/{len(X_test)})")

    print("-" * 80)
    print(f"Average Detection Rate  | RF: {np.mean(rf_results):.2f}%               | MLP: {np.mean(mlp_results):.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()