"""32_export_replay.py - export dataset cycles as 8-bit DAC codes for the ESP32 replay test.

Run from the repo root:  python src\32_export_replay.py
Reads : data/raw/arc_fault_dataset/Datasets/dataset_14classes_20ms_50kHz.mat,
        src/model_weights_b.h (scaler arrays may be in this same header or src/scaler_b.h)
Writes: src/replay_cycles.h, reports/replay_manifest.csv (also prints a summary)

Each selected official-test cycle is (1) rotated to start at a rising zero crossing,
(2) scaled to 8-bit DAC codes 22..178 (about 0.28-2.30 V, inside the ESP32 ADC's accurate
window), and (3) stored with its label and the deployed model's expected probability both
for the ideal cycle and for the quantised codes.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

MAT = Path("data") / "raw" / "arc_fault_dataset" / "Datasets" / "dataset_14classes_20ms_50kHz.mat"
MODEL_H = Path("src") / "model_weights_b.h"
SCALER_H = Path("src") / "scaler_b.h"  # optional if scaler arrays are in model_weights_b.h
OUT_H = Path("src") / "replay_cycles.h"
OUT_CSV = Path("reports") / "replay_manifest.csv"

THR = 0.60
CENTER, AMP = 100, 78               # DAC codes: 100 +- 78 -> 22..178
N_NORMAL, N_PER_ARC_CLASS = 30, 3
FEATS = ["crest", "skew", "kurt", "d1_ratio", "d2_ratio", "peak_asym", "energy_asym", "frac_small"]
rng = np.random.default_rng(0)


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
    # The project's model_weights_b.h may bundle scaler arrays together
    # with the weights. Read arrays by symbol name, not declaration order.
    arrays = {}
    for path in (MODEL_H, SCALER_H):
        if path.exists():
            arrays.update(dict(read_arrays(path)))

    required = ("W1_B", "B1_B", "W2_B", "B2_B", "W3_B", "B3_B")
    missing = [name for name in required if name not in arrays]
    if missing:
        sys.exit(
            "Missing model array(s) %s. Parsed from existing headers: %s"
            % (missing, {str(p): [n for n, _ in read_arrays(p)]
                         for p in (MODEL_H, SCALER_H) if p.exists()})
        )

    mean_names = ("SCALER_MEAN_B", "SCALER_MEAN", "MEAN_B")
    scale_names = ("SCALER_SCALE_B", "SCALER_SCALE", "SCALE_B")
    mean_name = next((name for name in mean_names if name in arrays), None)
    scale_name = next((name for name in scale_names if name in arrays), None)
    if mean_name is None or scale_name is None:
        sys.exit(
            "Could not find scaler mean/scale arrays in %s or %s. Parsed arrays: %s"
            % (MODEL_H, SCALER_H, sorted(arrays))
        )

    W1, B1 = arrays["W1_B"], arrays["B1_B"]
    W2, B2 = arrays["W2_B"], arrays["B2_B"]
    W3, B3 = arrays["W3_B"], arrays["B3_B"]
    mean, scale = arrays[mean_name], arrays[scale_name]

    expected = {
        "W1_B": 128, "B1_B": 16, "W2_B": 128, "B2_B": 8,
        "W3_B": 8, "B3_B": 1, mean_name: 8, scale_name: 8
    }
    actual = {
        "W1_B": W1.size, "B1_B": B1.size, "W2_B": W2.size, "B2_B": B2.size,
        "W3_B": W3.size, "B3_B": B3.size, mean_name: mean.size, scale_name: scale.size
    }
    bad = {name: (expected[name], size) for name, size in actual.items()
           if size != expected[name]}
    if bad:
        sys.exit("Unexpected model/scaler array sizes (expected, actual): %s" % bad)

    return (W1.reshape(8, 16), B1, W2.reshape(16, 8), B2,
            W3.reshape(8), B3[0], mean, scale)


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


def rotate_to_rising_crossing(x, smooth=51, guard=100):
    a = x - x.mean()
    n = len(a)
    k = np.ones(smooth) / smooth
    s = np.convolve(np.r_[a[-smooth:], a, a[:smooth]], k, mode="same")[smooth:-smooth]
    for i in range(1, n):
        if s[i - 1] < 0 <= s[i]:
            before = s[i - guard:i] if i >= guard else np.r_[s[i - guard:], s[:i]]
            after = np.r_[s[i:], s[:i]][:guard]
            if before.mean() < 0 < after.mean():
                return np.roll(x, -i)
    return x


def to_codes(x):
    a = x - x.mean()
    pk = np.abs(a).max()
    c = CENTER + a * (AMP / pk if pk > 0 else 0.0)
    return np.clip(np.round(c), 0, 255).astype(np.uint8)


def arr(vals, fmt):
    return ", ".join(fmt % v for v in vals)


def main():
    params = load_params()
    d = scipy.io.loadmat(MAT)
    X = d["yTest"].astype(np.float64)
    det = d["labelsTest"].ravel().astype(int)

    pick = list(rng.choice(np.where(det == 0)[0], size=N_NORMAL, replace=False))
    for k in range(1, 14):
        idx = np.where(det == k)[0]
        if len(idx):
            pick += list(rng.choice(idx, size=min(N_PER_ARC_CLASS, len(idx)), replace=False))
    pick = np.array(pick)

    rot = np.array([rotate_to_rising_crossing(X[i]) for i in pick])
    codes = np.array([to_codes(r) for r in rot])
    f_clean = features_b(rot)
    f_quant = features_b(codes.astype(np.float64))
    p_orig = predict(features_b(X[pick]), params)
    p_clean = predict(f_clean, params)
    p_quant = predict(f_quant, params)
    label = (det[pick] > 0).astype(int)

    flips_rot = int(((p_orig >= THR) != (p_clean >= THR)).sum())
    flips_q = int(((p_clean >= THR) != (p_quant >= THR)).sum())
    print("cycles exported: %d (normal %d, arc %d)" % (len(pick), int((label == 0).sum()), int(label.sum())))
    print("prediction flips from rotation alone      : %d" % flips_rot)
    print("prediction flips from 8-bit quantisation  : %d  (threshold %.2f)" % (flips_q, THR))
    print("mean |p_quant - p_clean| = %.4f   max = %.4f"
          % (float(np.abs(p_quant - p_clean).mean()), float(np.abs(p_quant - p_clean).max())))
    spread = f_clean.std(axis=0) + 1e-9
    chg = np.median(np.abs(f_quant - f_clean), axis=0) / spread
    print("median feature change from quantisation, as a fraction of the feature's spread:")
    for name, v in zip(FEATS, chg):
        print("   %-12s %.3f" % (name, v))
    print("accuracy on this subset at %.2f: ideal %.1f%%, quantised %.1f%%"
          % (THR, 100 * ((p_clean >= THR) == label).mean(), 100 * ((p_quant >= THR) == label).mean()))

    OUT_H.parent.mkdir(parents=True, exist_ok=True)
    rows = ",\n".join("  { " + arr(c, "%d") + " }" for c in codes)
    OUT_H.write_text(
        "// Generated by 32_export_replay.py - do not edit by hand.\n"
        "// 8-bit DAC codes (center %d, +-%d), one 20 ms cycle at 50 kHz per row.\n"
        "#ifndef REPLAY_CYCLES_H\n#define REPLAY_CYCLES_H\n#include <stdint.h>\n"
        "#define REPLAY_N %d\n#define REPLAY_LEN 1000\n"
        "static const uint8_t REPLAY_CODES[REPLAY_N][REPLAY_LEN] = {\n%s\n};\n"
        "static const uint8_t REPLAY_LABEL[REPLAY_N] = { %s };\n"
        "static const uint8_t REPLAY_CLASS[REPLAY_N] = { %s };\n"
        "static const uint16_t REPLAY_SRC_INDEX[REPLAY_N] = { %s };\n"
        "static const float REPLAY_P_CLEAN[REPLAY_N] = { %s };\n"
        "static const float REPLAY_P_QUANT[REPLAY_N] = { %s };\n#endif\n"
        % (CENTER, AMP, len(pick), rows, arr(label, "%d"), arr(det[pick], "%d"),
           arr(pick, "%d"), arr(p_clean, "%.8ef"), arr(p_quant, "%.8ef")))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row": range(len(pick)), "test_index": pick, "label": label,
                  "class": det[pick], "p_clean": p_clean, "p_quant": p_quant}).to_csv(OUT_CSV, index=False)
    print("wrote:", OUT_H, "and", OUT_CSV)


if __name__ == "__main__":
    main()
