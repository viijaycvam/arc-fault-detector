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

    # Filter standard training dataset split
    train_df = df[df["split"] == "train"]

    # Production 14 DSP Feature Matrix
    feature_cols = [
        "mean", "std", "rms", "peak", "peak_to_peak", "crest_factor",
        "skewness", "kurtosis", "total_power", "power_low",
        "power_mid", "power_high", "hf_ratio", "spectral_centroid"
    ]

    X_tr = train_df[feature_cols].values
    y_tr = train_df["binary_label"].values

    # Fit feature scaling
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)

    # Train production B4 MLP (14 -> 16 -> 8 -> 1)
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

    print("=" * 80)
    print("STEP 16 — EXPORTING B4 TINYML MODEL TO C HEADER (model_weights.h)")
    print("=" * 80)

    header_path = Path(r"reports\model_weights.h")
    with open(header_path, "w") as f:
        f.write("// Auto-generated B4 TinyML Arc Detection Model Weights\n")
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")

        # Scaler parameters
        f.write(f"const float SCALER_MEAN[14] = {{{', '.join(map(str, scaler.mean_))}}};\n")
        f.write(f"const float SCALER_SCALE[14] = {{{', '.join(map(str, scaler.scale_))}}};\n\n")

        # Layer 1 Weights (14x16) and Biases (16)
        f.write("const float W1[14][16] = {\n")
        for row in mlp.coefs_[0]:
            f.write("  {" + ", ".join(map(str, row)) + "},\n")
        f.write("};\n")
        f.write(f"const float B1[16] = {{{', '.join(map(str, mlp.intercepts_[0]))}}};\n\n")

        # Layer 2 Weights (16x8) and Biases (8)
        f.write("const float W2[16][8] = {\n")
        for row in mlp.coefs_[1]:
            f.write("  {" + ", ".join(map(str, row)) + "},\n")
        f.write("};\n")
        f.write(f"const float B3[8] = {{{', '.join(map(str, mlp.intercepts_[1]))}}};\n\n")

        # Layer 3 Weights (8x1) and Biases (1)
        f.write("const float W3[8][1] = {\n")
        for row in mlp.coefs_[2]:
            f.write("  {" + ", ".join(map(str, row)) + "},\n")
        f.write("};\n")
        f.write(f"const float B3_OUT[1] = {{{', '.join(map(str, mlp.intercepts_[2]))}}};\n\n")

        f.write("#endif // MODEL_WEIGHTS_H\n")

    print(f"Successfully generated C header file at: {header_path.resolve()}")

if __name__ == "__main__":
    main()