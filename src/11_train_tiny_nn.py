import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

CSV_FILE = Path(r"reports\dsp_features.csv")

def main():
    if not CSV_FILE.exists():
        print(f"Error: DSP Feature CSV not found at {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    # Use all 14 DSP & Time-domain features for complete representation
    feature_cols = [
        "mean", "std", "rms", "peak", "peak_to_peak", "crest_factor",
        "skewness", "kurtosis", "total_power", "power_low",
        "power_mid", "power_high", "hf_ratio", "spectral_centroid"
    ]

    X_tr = train_df[feature_cols].values
    y_tr = train_df["binary_label"].values
    X_te = test_df[feature_cols].values
    y_te = test_df["binary_label"].values

    # Standardize features (essential for neural network convergence)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    print("=" * 70)
    print("B4 — OPTIMIZED LIGHTWEIGHT NEURAL NETWORK (14 -> 16 -> 8 -> 1)")
    print("=" * 70)
    print(f"Input Features: {len(feature_cols)}")
    print(f"Architecture  : 14 Inputs -> Dense(16) -> Dense(8) -> Output(1)")
    print("-" * 70)

    # Train a lightweight 2-layer MLP with early stopping and adaptive learning
    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        alpha=0.001,             # L2 regularization to prevent overfitting
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_tr_scaled, y_tr)

    # Predict on Test Set
    preds = mlp.predict(X_te_scaled)
    acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds)
    cm = confusion_matrix(y_te, preds)
    false_trip = (preds[y_te == 0].mean()) * 100

    print("\n[1] Overall Metrics")
    print(f"  Accuracy       : {acc * 100:.2f}%")
    print(f"  F1-Score       : {f1:.4f}")
    print(f"  False-Trip Rate: {false_trip:.2f}% ({preds[y_te == 0].sum()}/{(y_te == 0).sum()} normal misclassified)")

    print("\n[2] Confusion Matrix")
    print("               Pred NORMAL  Pred ARC")
    print(f"  Actual NORMAL    {cm[0, 0]:<11} {cm[0, 1]}")
    print(f"  Actual ARC       {cm[1, 0]:<11} {cm[1, 1]}")

    print("\n[3] Classification Report")
    print(classification_report(y_te, preds, target_names=["NORMAL", "ARC"], digits=4))

if __name__ == "__main__":
    main()