import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

CSV_FILE = Path(r"reports\dsp_features.csv")

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    if not CSV_FILE.exists():
        print(f"Error: DSP Feature CSV not found at {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    test_df = df[df["split"] == "test"]
    train_df = df[df["split"] == "train"]

    feature_cols = [
        "mean", "std", "rms", "peak", "peak_to_peak", "crest_factor",
        "skewness", "kurtosis", "total_power", "power_low",
        "power_mid", "power_high", "hf_ratio", "spectral_centroid"
    ]

    X_tr = train_df[feature_cols].values
    y_tr = train_df["binary_label"].values
    X_te = test_df[feature_cols].values
    y_te = test_df["binary_label"].values

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_tr_scaled, y_tr)

    skl_preds = mlp.predict(X_te_scaled)

    c_preds = []
    for raw_sample in X_te:
        x_norm = (raw_sample - scaler.mean_) / scaler.scale_
        h1 = relu(np.dot(x_norm, mlp.coefs_[0]) + mlp.intercepts_[0])
        h2 = relu(np.dot(h1, mlp.coefs_[1]) + mlp.intercepts_[1])
        out = np.dot(h2, mlp.coefs_[2]) + mlp.intercepts_[2]
        prob = sigmoid(out[0])
        c_preds.append(1 if prob >= 0.5 else 0)

    c_preds = np.array(c_preds)
    mismatches = np.sum(skl_preds != c_preds)

    print("=" * 80)
    print("STEP 17 — C FORWARD PASS LOGIC VERIFICATION")
    print("=" * 80)
    print(f"Test Samples Evaluated : {len(y_te)}")
    print(f"Scikit-Learn Accuracy  : {(skl_preds == y_te).mean() * 100:.2f}%")
    print(f"C Forward Pass Accuracy: {(c_preds == y_te).mean() * 100:.2f}%")
    print(f"Prediction Mismatches  : {mismatches}")
    print("=" * 80)

if __name__ == "__main__":
    main()