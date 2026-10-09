"""31_noise_robustness.py - how much does the analog path hurt the DEPLOYED model?

Run from the repo root:  python src\31_noise_robustness.py [threshold]
Reads : data/raw/arc_fault_dataset/Datasets/dataset_14classes_20ms_50kHz.mat,
        src/model_weights_b.h (including scaler arrays, or optional src/scaler_b.h)
Writes: reports/noise_robustness.txt (also printed)

Each official-test cycle is rescaled to a chosen peak amplitude in ADC counts, optionally
quantised like a DAC, given Gaussian noise, rounded to integer ADC counts, and then run
through the same 8 features and the deployed MLP. Default threshold is 0.60.
"""
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io

MAT = Path("data") / "raw" / "arc_fault_dataset" / "Datasets" / "dataset_14classes_20ms_50kHz.mat"
MODEL_H = Path("src") / "model_weights_b.h"
SCALER_H = Path("src") / "scaler_b.h"  # optional; scaler may be in MODEL_H
OUT = Path("reports") / "noise_robustness.txt"

THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.60
COUNTS_PER_VOLT = 1050.0            # approx. ESP32 ADC gain at 12 dB (about 4095 counts / 3.9 V)
PEAKS = [1100, 550, 275]            # signal peak in ADC counts (1100 is about +-1.05 V)
NOISES = [0.0, 2.6, 5.24, 10.0]     # Gaussian noise, counts (5.24 = your measured DC std)
DACS = [None, 12, 8]                # None = ideal source, else DAC resolution in bits
rng = np.random.default_rng(0)
lines = []


def log(s=""):
    print(s)
    lines.append(s)


def read_arrays(path):
    txt = Path(path).read_text()
    txt = re.sub(r"//.*", "", txt)
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    out = []
    pat = r"float\s+(\w+)\s*((?:\[[^\]]*\]\s*)+)=\s*\{(.*?)\}\s*;"
    for m in re.finditer(pat, txt, re.S):
        nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", m.group(3))
        out.append((m.group(1), np.array(nums, dtype=np.float64)))
    return out


def load_params():
    """Load named arrays, supporting either bundled or separate scaler headers."""
    if not MODEL_H.exists():
        sys.exit("Missing model header: %s" % MODEL_H)

    model_arrays = dict(read_arrays(MODEL_H))
    required = ("W1_B", "B1_B", "W2_B", "B2_B", "W3_B", "B3_B")
    missing = [name for name in required if name not in model_arrays]
    if missing:
        sys.exit(
            "Could not find expected model arrays %s in %s. Arrays found: %s"
            % (missing, MODEL_H, sorted(model_arrays))
        )

    W1 = model_arrays["W1_B"]
    B1 = model_arrays["B1_B"]
    W2 = model_arrays["W2_B"]
    B2 = model_arrays["B2_B"]
    W3 = model_arrays["W3_B"]
    B3 = model_arrays["B3_B"]

    scaler_arrays = {}
    if SCALER_H.exists():
        scaler_arrays.update(dict(read_arrays(SCALER_H)))
    scaler_arrays.update({
        name: values for name, values in model_arrays.items()
        if "MEAN" in name.upper() or "SCALE" in name.upper()
    })

    # Important: "SCALER_MEAN_B" also contains the substring "SCALE".
    # Exclude MEAN names when selecting the scale vector, or mean will be
    # accidentally used for both arrays.
    mean = next((v for n, v in scaler_arrays.items() if "MEAN" in n.upper()), None)
    scale = next((v for n, v in scaler_arrays.items()
                  if "SCALE" in n.upper() and "MEAN" not in n.upper()), None)
    if mean is None or scale is None:
        sys.exit(
            "Could not find scaler mean/scale arrays. Expected names containing "
            "MEAN and SCALE in %s or %s. Arrays found in model header: %s"
            % (MODEL_H, SCALER_H, sorted(model_arrays))
        )

    expected_sizes = {
        "W1_B": (W1.size, 128), "B1_B": (B1.size, 16),
        "W2_B": (W2.size, 128), "B2_B": (B2.size, 8),
        "W3_B": (W3.size, 8), "B3_B": (B3.size, 1),
    }
    bad = {k: got for k, (got, want) in expected_sizes.items() if got != want}
    if bad or mean.size != 8 or scale.size != 8:
        sys.exit(
            "Unexpected model/scaler array sizes. Model sizes=%s, mean=%d, scale=%d"
            % ({k: v[0] for k, v in expected_sizes.items()}, mean.size, scale.size)
        )

    return (W1.reshape(8, 16), B1, W2.reshape(16, 8), B2,
            W3.reshape(8), float(B3[0]), mean, scale)


def predict(X, params):
    W1, B1, W2, B2, W3, B3, mean, scale = params
    z = (X - mean) / scale
    h1 = np.maximum(z @ W1 + B1, 0.0)
    h2 = np.maximum(h1 @ W2 + B2, 0.0)
    return 1.0 / (1.0 + np.exp(-(h2 @ W3 + B3)))


def features_b(X):
    X = np.asarray(X, dtype=np.float64)
    a = X - X.mean(axis=1, keepdims=True)
    e = (a ** 2).sum(axis=1)
    rms = np.sqrt(e / a.shape[1])
    safe_e = np.where(e > 1e-12, e, 1.0)
    safe_rms = np.where(rms > 1e-9, rms, 1.0)
    peak = np.abs(a).max(axis=1)
    crest = peak / safe_rms
    skew = (a ** 3).mean(axis=1) / safe_rms ** 3
    kurt = (a ** 4).mean(axis=1) / safe_rms ** 4 - 3.0
    d1 = np.diff(a, axis=1)
    d2 = a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]
    d1_ratio = (d1 ** 2).sum(axis=1) / safe_e
    d2_ratio = (d2 ** 2).sum(axis=1) / safe_e
    pmax, pmin = a.max(axis=1), a.min(axis=1)
    span = np.where((pmax - pmin) > 1e-12, pmax - pmin, 1.0)
    peak_asym = (pmax + pmin) / span
    pos_e = np.where(a > 0, a ** 2, 0.0).sum(axis=1)
    neg_e = np.where(a < 0, a ** 2, 0.0).sum(axis=1)
    energy_asym = (pos_e - neg_e) / safe_e
    frac_small = (np.abs(a) < 0.1 * peak[:, None]).mean(axis=1)
    return np.column_stack([crest, skew, kurt, d1_ratio, d2_ratio,
                            peak_asym, energy_asym, frac_small])


def analog_path(X, peak_counts, noise_counts, dac_bits):
    a = X - X.mean(axis=1, keepdims=True)
    pk = np.abs(a).max(axis=1, keepdims=True)
    y = a / np.where(pk > 0, pk, 1.0) * peak_counts
    if dac_bits:
        step = 3.3 / (2 ** dac_bits - 1) * COUNTS_PER_VOLT
        y = np.round(y / step) * step
    if noise_counts > 0:
        y = y + rng.normal(0.0, noise_counts, size=y.shape)
    return np.round(y)


def metrics(pred, y):
    tn = int(((~pred) & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum()); tp = int((pred & (y == 1)).sum())
    return tn, fp, fn, tp


def main():
    if not MAT.exists():
        sys.exit("Missing dataset: %s\nRun this script from the project root and check the dataset path." % MAT)
    params = load_params()
    d = scipy.io.loadmat(MAT)
    X = d["yTest"].astype(np.float64)
    y = (d["labelsTest"].ravel().astype(int) > 0).astype(int)

    p0 = predict(features_b(X), params)
    tn, fp, fn, tp = metrics(p0 >= 0.5, y)
    log("Clean features, deployed model, threshold 0.50: TN/FP/FN/TP = %d %d %d %d"
        % (tn, fp, fn, tp))
    log("(expected 850 11 81 392)")
    if (tn, fp, fn, tp) != (850, 11, 81, 392):
        log("WARNING: feature definitions differ from the deployed pipeline - send me this output.")
    ideal = p0 >= THR
    tn, fp, fn, tp = metrics(ideal, y)
    log("")
    log("Ideal signal at threshold %.2f: false-trip %.2f%%  detection %.2f%%  accuracy %.2f%%"
        % (THR, 100 * fp / (tn + fp), 100 * tp / (tp + fn), 100 * (tn + tp) / len(y)))
    log("")
    log("%-6s %-6s %-6s %-9s %-9s %-9s %s"
        % ("DAC", "peak", "noise", "FTR %", "det %", "acc %", "flips vs ideal"))
    for dac in DACS:
        for peak in PEAKS:
            for noise in NOISES:
                pred = predict(features_b(analog_path(X, peak, noise, dac)), params) >= THR
                tn, fp, fn, tp = metrics(pred, y)
                log("%-6s %-6d %-6.2f %-9.2f %-9.2f %-9.2f %d"
                    % ("ideal" if dac is None else "%d-bit" % dac, peak, noise,
                       100 * fp / (tn + fp), 100 * tp / (tp + fn),
                       100 * (tn + tp) / len(y), int((pred != ideal).sum())))
        log("")
    log("Notes: counts are ESP32 ADC counts (about 1050 counts per volt, approximate).")
    log("Noise is independent Gaussian; real noise is not exactly white. This is a model")
    log("of the analog path, not a measurement - confirm with the hardware replay.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
