import numpy as np, pandas as pd, scipy.io
from pathlib import Path

MAT_FILE = Path(r"data\raw\arc_fault_dataset\Datasets\dataset_14classes_20ms_50kHz.mat")
OUT_FILE = Path(r"reports\features_b.csv")
FEATURE_NAMES = ["crest", "skew", "kurt", "d1_ratio", "d2_ratio",
                 "peak_asym", "energy_asym", "frac_small"]

def features_b(X):
    X = np.asarray(X, dtype=np.float64)
    a = X - X.mean(axis=1, keepdims=True)             # remove DC
    e = (a ** 2).sum(axis=1)                          # AC energy
    rms = np.sqrt(e / a.shape[1])
    safe_e = np.where(e > 1e-12, e, 1.0)
    safe_rms = np.where(rms > 1e-9, rms, 1.0)
    peak = np.abs(a).max(axis=1)
    crest = peak / safe_rms
    skew = (a ** 3).mean(axis=1) / safe_rms ** 3
    kurt = (a ** 4).mean(axis=1) / safe_rms ** 4 - 3.0
    d1 = np.diff(a, axis=1)
    d2 = a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]
    d1_ratio = (d1 ** 2).sum(axis=1) / safe_e         # high-frequency energy proxy
    d2_ratio = (d2 ** 2).sum(axis=1) / safe_e
    pmax, pmin = a.max(axis=1), a.min(axis=1)
    span = np.where((pmax - pmin) > 1e-12, pmax - pmin, 1.0)
    peak_asym = (pmax + pmin) / span                  # positive vs negative peak
    pos_e = np.where(a > 0, a ** 2, 0.0).sum(axis=1)
    neg_e = np.where(a < 0, a ** 2, 0.0).sum(axis=1)
    energy_asym = (pos_e - neg_e) / safe_e
    frac_small = (np.abs(a) < 0.1 * peak[:, None]).mean(axis=1)   # flat "shoulder" near zero
    return np.column_stack([crest, skew, kurt, d1_ratio, d2_ratio,
                            peak_asym, energy_asym, frac_small])

def main():
    d = scipy.io.loadmat(MAT_FILE)
    parts = []
    for split, xk, lk in [("train", "yTrain", "labelsTrain"), ("test", "yTest", "labelsTest")]:
        lab = d[lk].ravel().astype(int)
        df = pd.DataFrame(features_b(d[xk]), columns=FEATURE_NAMES)
        df.insert(0, "binary_label", (lab > 0).astype(int))
        df.insert(0, "detailed_label", lab)
        df.insert(0, "split", split)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "record_id", range(len(df)))
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print("records:", len(df), "| train:", (df.split == "train").sum(), "| test:", (df.split == "test").sum())
    print("normal:", (df.binary_label == 0).sum(), "| arc:", (df.binary_label == 1).sum())
    print(df.groupby("binary_label")[FEATURE_NAMES].mean().round(4).T)
    print("saved:", OUT_FILE)

if __name__ == "__main__":
    main()