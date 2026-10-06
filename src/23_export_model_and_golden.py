from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
MAT_FILE = Path(r"data\raw\arc_fault_dataset\Datasets\dataset_14classes_20ms_50kHz.mat")
FEAT_CSV = Path(r"reports\features_b.csv")

OUT_WEIGHTS = Path(r"src\model_weights_b.h")
OUT_SCALER = Path(r"src\scaler_b.h")
OUT_GOLDEN = Path(r"src\golden_cycles.h")

FEATURES = [
    "crest",
    "skew",
    "kurt",
    "d1_ratio",
    "d2_ratio",
    "peak_asym",
    "energy_asym",
    "frac_small",
]

# -----------------------------------------------------------------------------
# C-Header Code Generation Helpers
# -----------------------------------------------------------------------------
def write_c_array_1d(f, name, arr, precision=8):
    f.write(f"static const float {name}[{len(arr)}] = {{\n  ")
    formatted = [f"{float(v):.{precision}f}f" for v in arr]
    f.write(", ".join(formatted))
    f.write("\n};\n\n")


def write_c_array_2d(f, name, arr, precision=8):
    rows, cols = arr.shape
    f.write(f"static const float {name}[{rows}][{cols}] = {{\n")
    row_strings = []
    for r in range(rows):
        formatted = [f"{float(arr[r, c]):.{precision}f}f" for c in range(cols)]
        row_strings.append("  {" + ", ".join(formatted) + "}")
    f.write(",\n".join(row_strings))
    f.write("\n};\n\n")


# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------
def main():
    if not FEAT_CSV.exists():
        print(f"ERROR: Feature file not found at {FEAT_CSV}")
        return

    df = pd.read_csv(FEAT_CSV)

    print("=" * 70)
    print("EDGE-AI ARC FAULT DETECTOR — MLP TRAINING & C EXPORT PIPELINE")
    print("=" * 70)

    # 1. Dataset Preprocessing
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()

    X_train = train[FEATURES].values.astype(np.float64)
    y_train = train["binary_label"].values.astype(np.int32)
    X_test = test[FEATURES].values.astype(np.float64)
    y_test = test["binary_label"].values.astype(np.int32)

    print(f"Train records : {len(train)}")
    print(f"Test records  : {len(test)}")
    print(f"Feature count : {len(FEATURES)}")

    # 2. StandardScaler Fitting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model Training (8 -> 16 -> 8 -> 1 MLP)
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # 4. Evaluation & Metrics
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    false_trip_rate = (fp / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    arc_detection = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0

    print("\n--- CONFUSION MATRIX ---")
    print(f"TN: {tn:<5} FP: {fp:<5}")
    print(f"FN: {fn:<5} TP: {tp:<5}")
    print(f"\nAccuracy        : {acc * 100.0:.2f}%")
    print(f"False-Trip Rate : {false_trip_rate:.2f}%")
    print(f"Arc Detection   : {arc_detection:.2f}%")

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "ARC"], digits=4))

    # 5. NumPy Forward Pass Verification
    W1, B1 = model.coefs_[0], model.intercepts_[0]
    W2, B2 = model.coefs_[1], model.intercepts_[1]
    W3, B3 = model.coefs_[2], model.intercepts_[2]

    h1 = np.maximum(0.0, X_test_scaled @ W1 + B1)
    h2 = np.maximum(0.0, h1 @ W2 + B2)
    out_raw = h2 @ W3 + B3
    manual_prob = 1.0 / (1.0 + np.exp(-out_raw.ravel()))
    manual_pred = (manual_prob >= 0.5).astype(np.int32)

    mismatches = np.sum(manual_pred != y_pred)
    max_prob_err = np.max(np.abs(manual_prob - y_prob))

    print("--- FORWARD PASS VERIFICATION ---")
    print(f"Prediction Mismatches : {mismatches}")
    print(f"Max Probability Error : {max_prob_err:.10e}")

    # 6. Export C Header: model_weights_b.h
    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_WEIGHTS, "w", encoding="utf-8") as f:
        f.write("#ifndef MODEL_WEIGHTS_B_H\n#define MODEL_WEIGHTS_B_H\n\n")
        f.write("#define B_NUM_FEATURES 8\n#define B_H1_SIZE 16\n#define B_H2_SIZE 8\n\n")
        
        write_c_array_1d(f, "SCALER_MEAN_B", scaler.mean_)
        write_c_array_1d(f, "SCALER_SCALE_B", scaler.scale_)
        write_c_array_2d(f, "W1_B", W1)
        write_c_array_1d(f, "B1_B", B1)
        write_c_array_2d(f, "W2_B", W2)
        write_c_array_1d(f, "B2_B", B2)
        write_c_array_2d(f, "W3_B", W3)
        write_c_array_1d(f, "B3_B", B3)
        f.write("#endif // MODEL_WEIGHTS_B_H\n")

    # 7. Export C Header: scaler_b.h
    with open(OUT_SCALER, "w", encoding="utf-8") as f:
        f.write("#ifndef SCALER_B_H\n#define SCALER_B_H\n\n")
        write_c_array_1d(f, "SCALER_MEAN_B", scaler.mean_)
        write_c_array_1d(f, "SCALER_SCALE_B", scaler.scale_)
        f.write("#endif // SCALER_B_H\n")

    # 8. Export Golden Cycles: golden_cycles.h (if MAT file exists)
    if MAT_FILE.exists():
        mat_data = scipy.io.loadmat(MAT_FILE)
        X_raw_all = np.vstack([mat_data["yTrain"], mat_data["yTest"]]).astype(np.float64)

        norm_indices = test[test.binary_label == 0].head(50)["record_id"].values
        arc_indices = test[test.binary_label == 1].head(50)["record_id"].values
        golden_indices = np.concatenate([norm_indices, arc_indices])

        with open(OUT_GOLDEN, "w", encoding="utf-8") as f:
            f.write("#ifndef GOLDEN_CYCLES_H\n#define GOLDEN_CYCLES_H\n\n")
            f.write("#define NUM_GOLDEN_CYCLES 100\n#define CYCLE_SAMPLES 1000\n\n")
            f.write("typedef struct {\n")
            f.write("    int record_id;\n")
            f.write("    int expected_label;\n")
            f.write("    float expected_features[8];\n")
            f.write("    float expected_prob;\n")
            f.write("    float raw_samples[1000];\n")
            f.write("} GoldenCycle;\n\n")
            f.write("static const GoldenCycle GOLDEN_CYCLES[100] = {\n")

            entries = []
            for idx in golden_indices:
                raw_signal = X_raw_all[idx]
                feats = df.loc[df.record_id == idx, FEATURES].values[0]
                feats_scaled = scaler.transform([feats])
                prob = model.predict_proba(feats_scaled)[0][1]
                label = int(df.loc[df.record_id == idx, "binary_label"].values[0])

                feats_str = "{" + ", ".join([f"{x:.8f}f" for x in feats]) + "}"
                samples_str = "{" + ", ".join([f"{s:.6f}f" for s in raw_signal]) + "}"

                entry = f"  {{\n    {idx}, {label}, {feats_str}, {prob:.8f}f,\n    {samples_str}\n  }}"
                entries.append(entry)

            f.write(",\n".join(entries))
            f.write("\n};\n\n#endif // GOLDEN_CYCLES_H\n")

        print(f"\nSuccessfully generated golden cycles: {OUT_GOLDEN}")

    print(f"Successfully exported model weights to : {OUT_WEIGHTS}")
    print(f"Successfully exported scaler parameters to: {OUT_SCALER}")


if __name__ == "__main__":
    main()