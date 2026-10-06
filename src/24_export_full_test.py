import numpy as np
import pandas as pd
import scipy.io

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


MAT_FILE = Path(
    r"data\raw\arc_fault_dataset\Datasets\dataset_14classes_20ms_50kHz.mat"
)

FEAT_CSV = Path(r"reports\features_b.csv")

RAW_BIN = Path(r"reports\full_test_raw.bin")
FEAT_BIN = Path(r"reports\full_test_features.bin")
PROB_BIN = Path(r"reports\full_test_prob.bin")
LABEL_BIN = Path(r"reports\full_test_labels.bin")
PRED_BIN = Path(r"reports\full_test_pred.bin")

PY_REPORT = Path(r"reports\full_test_python_reference.txt")


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


def main():

    print("=" * 70)
    print("FULL OFFICIAL TEST SET REFERENCE EXPORT")
    print("=" * 70)

    if not MAT_FILE.exists():
        print(f"ERROR: dataset not found: {MAT_FILE}")
        return

    if not FEAT_CSV.exists():
        print(f"ERROR: feature CSV not found: {FEAT_CSV}")
        return

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    mat = scipy.io.loadmat(MAT_FILE)

    X_train_raw = mat["yTrain"].astype(np.float64)
    X_test_raw = mat["yTest"].astype(np.float64)

    y_train = (
        mat["labelsTrain"]
        .ravel()
        .astype(np.int32)
    )

    y_test = (
        mat["labelsTest"]
        .ravel()
        .astype(np.int32)
    )

    y_train_bin = (y_train > 0).astype(np.int32)
    y_test_bin = (y_test > 0).astype(np.int32)

    # ---------------------------------------------------------
    # Load exact Option-B features
    # ---------------------------------------------------------
    df = pd.read_csv(FEAT_CSV)

    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()

    if len(train) != len(X_train_raw):
        raise RuntimeError(
            f"Train size mismatch: CSV={len(train)} "
            f"MAT={len(X_train_raw)}"
        )

    if len(test) != len(X_test_raw):
        raise RuntimeError(
            f"Test size mismatch: CSV={len(test)} "
            f"MAT={len(X_test_raw)}"
        )

    X_train = train[FEATURES].values.astype(np.float64)
    X_test = test[FEATURES].values.astype(np.float64)

    # ---------------------------------------------------------
    # Verify labels agree with MAT file
    # ---------------------------------------------------------
    csv_train_labels = train["binary_label"].values.astype(np.int32)
    csv_test_labels = test["binary_label"].values.astype(np.int32)

    if not np.array_equal(csv_train_labels, y_train_bin):
        raise RuntimeError("Train labels in CSV do not match MAT file.")

    if not np.array_equal(csv_test_labels, y_test_bin):
        raise RuntimeError("Test labels in CSV do not match MAT file.")

    # ---------------------------------------------------------
    # Exact same StandardScaler / MLP configuration
    # as src\23_export_model_and_golden.py
    # ---------------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train_bin,
    )

    # ---------------------------------------------------------
    # Python reference prediction
    # ---------------------------------------------------------
    y_pred = model.predict(X_test_scaled).astype(np.uint8)
    y_prob = model.predict_proba(X_test_scaled)[:, 1].astype(np.float32)

    # ---------------------------------------------------------
    # Verify manually implemented Python forward pass
    # ---------------------------------------------------------
    W1, B1 = model.coefs_[0], model.intercepts_[0]
    W2, B2 = model.coefs_[1], model.intercepts_[1]
    W3, B3 = model.coefs_[2], model.intercepts_[2]

    h1 = np.maximum(
        0.0,
        X_test_scaled @ W1 + B1,
    )

    h2 = np.maximum(
        0.0,
        h1 @ W2 + B2,
    )

    out_raw = h2 @ W3 + B3

    manual_prob = (
        1.0 /
        (1.0 + np.exp(-out_raw.ravel()))
    )

    manual_pred = (
        manual_prob >= 0.5
    ).astype(np.uint8)

    manual_prediction_mismatches = int(
        np.sum(manual_pred != y_pred)
    )

    max_manual_prob_error = float(
        np.max(np.abs(manual_prob - y_prob))
    )

    # ---------------------------------------------------------
    # Python metrics
    # ---------------------------------------------------------
    tn = int(np.sum((y_test_bin == 0) & (y_pred == 0)))
    fp = int(np.sum((y_test_bin == 0) & (y_pred == 1)))
    fn = int(np.sum((y_test_bin == 1) & (y_pred == 0)))
    tp = int(np.sum((y_test_bin == 1) & (y_pred == 1)))

    total = len(y_test_bin)

    accuracy = (
        100.0 * (tn + tp) / total
    )

    false_trip_rate = (
        100.0 * fp / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    arc_detection = (
        100.0 * tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # Export raw test samples as float32
    # ---------------------------------------------------------
    X_test_raw.astype(
        np.float32
    ).tofile(RAW_BIN)

    # ---------------------------------------------------------
    # Export Python-computed features as float32
    # ---------------------------------------------------------
    X_test.astype(
        np.float32
    ).tofile(FEAT_BIN)

    # ---------------------------------------------------------
    # Export probabilities / labels / predictions
    # ---------------------------------------------------------
    y_prob.astype(
        np.float32
    ).tofile(PROB_BIN)

    y_test_bin.astype(
        np.uint8
    ).tofile(LABEL_BIN)

    y_pred.astype(
        np.uint8
    ).tofile(PRED_BIN)

    # ---------------------------------------------------------
    # Human-readable report
    # ---------------------------------------------------------
    with open(PY_REPORT, "w", encoding="utf-8") as f:

        f.write(
            "FULL OFFICIAL TEST SET — PYTHON REFERENCE\n"
        )
        f.write("=" * 70 + "\n")

        f.write(
            f"Test samples       : {total}\n"
        )

        f.write(
            f"Normal samples     : {int(np.sum(y_test_bin == 0))}\n"
        )

        f.write(
            f"Arc samples        : {int(np.sum(y_test_bin == 1))}\n"
        )

        f.write(
            f"TN                 : {tn}\n"
        )

        f.write(
            f"FP                 : {fp}\n"
        )

        f.write(
            f"FN                 : {fn}\n"
        )

        f.write(
            f"TP                 : {tp}\n"
        )

        f.write(
            f"Accuracy           : {accuracy:.4f}%\n"
        )

        f.write(
            f"False-trip rate    : {false_trip_rate:.4f}%\n"
        )

        f.write(
            f"Arc detection      : {arc_detection:.4f}%\n"
        )

        f.write(
            f"Manual Python prediction mismatches : "
            f"{manual_prediction_mismatches}\n"
        )

        f.write(
            f"Manual Python max probability error : "
            f"{max_manual_prob_error:.10e}\n"
        )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------
    print(f"Test samples       : {total}")
    print(
        f"Normal samples     : "
        f"{int(np.sum(y_test_bin == 0))}"
    )
    print(
        f"Arc samples        : "
        f"{int(np.sum(y_test_bin == 1))}"
    )

    print()
    print("PYTHON CONFUSION MATRIX")
    print(f"TN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TP = {tp}")

    print()
    print(f"Accuracy        : {accuracy:.4f}%")
    print(f"False-trip rate : {false_trip_rate:.4f}%")
    print(f"Arc detection   : {arc_detection:.4f}%")

    print()
    print(
        "Python forward-pass prediction mismatches:",
        manual_prediction_mismatches,
    )

    print(
        "Max Python probability error:",
        f"{max_manual_prob_error:.10e}",
    )

    print()
    print("Exported:")
    print(f"  {RAW_BIN}")
    print(f"  {FEAT_BIN}")
    print(f"  {PROB_BIN}")
    print(f"  {LABEL_BIN}")
    print(f"  {PRED_BIN}")
    print(f"  {PY_REPORT}")


if __name__ == "__main__":
    main()