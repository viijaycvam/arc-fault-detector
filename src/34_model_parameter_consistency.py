"""Compare the live parameter-loading and predictions in voting vs noise scripts.

Run from project root:
    python src\\34_model_parameter_consistency.py

Read-only: does not alter the model, CSV, dataset, or reports.
"""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CSV = ROOT / "reports" / "features_b.csv"
EXPECTED = (850, 11, 81, 392)


def load_module(name: str, path: Path):
    if not path.exists():
        print(f"MISSING: {path}")
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        print(f"ERROR: could not load module from {path}")
        return None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SRC))
    spec.loader.exec_module(module)
    return module


def confusion(pred, y):
    pred = np.asarray(pred, dtype=bool)
    y = np.asarray(y, dtype=int)
    tn = int(np.count_nonzero((~pred) & (y == 0)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero((~pred) & (y == 1)))
    tp = int(np.count_nonzero(pred & (y == 1)))
    return tn, fp, fn, tp


def main() -> int:
    voter_path = SRC / "30_voting_eval.py"
    noise_path = SRC / "31_noise_robustness.py"
    missing = [p for p in (voter_path, noise_path, CSV) if not p.exists()]
    if missing:
        print("Missing required file(s):")
        for p in missing:
            print("  ", p)
        print("Run this command from the project root and check filenames.")
        return 2

    voter = load_module("voting_eval_live", voter_path)
    noise = load_module("noise_robustness_live", noise_path)
    if voter is None or noise is None:
        return 2

    try:
        pv = voter.load_params()
        pn = noise.load_params()
    except Exception as exc:
        print("PARAMETER LOAD FAILED:", repr(exc))
        return 2

    names = ["W1", "B1", "W2", "B2", "W3", "B3", "SCALER_MEAN", "SCALER_SCALE"]
    print("Parameter comparison (loaded from the live scripts):")
    all_equal = True
    for name, a, b in zip(names, pv, pn):
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape != bb.shape:
            print(f"  {name:13s} SHAPE MISMATCH: voting={aa.shape}, noise={bb.shape}")
            all_equal = False
            continue
        diff = float(np.max(np.abs(aa - bb))) if aa.size else 0.0
        print(f"  {name:13s} shape={str(aa.shape):12s} max_abs_diff={diff:.12g}")
        if diff > 1e-12:
            all_equal = False
    print("PARAMETERS IDENTICAL:", "YES" if all_equal else "NO")

    df = pd.read_csv(CSV)
    te = df[df["split"].astype(str).str.lower() == "test"].copy()
    required = ["crest", "skew", "kurt", "d1_ratio", "d2_ratio", "peak_asym", "energy_asym", "frac_small", "binary_label"]
    absent = [c for c in required if c not in te.columns]
    if absent:
        print("CSV missing columns:", absent)
        return 2
    X = te[required[:8]].to_numpy(dtype=np.float64)
    y = te["binary_label"].to_numpy(dtype=int)

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        p_v = np.asarray(voter.predict(X, pv), dtype=np.float64)
        p_n = np.asarray(noise.predict(X, pn), dtype=np.float64)

    print("\nPrediction comparison on reports/features_b.csv test rows:")
    if p_v.shape != p_n.shape:
        print("  prediction shape mismatch:", p_v.shape, p_n.shape)
        return 2
    pdiff = np.abs(p_v - p_n)
    print("  rows:", len(y))
    print("  max probability difference:", float(np.nanmax(pdiff)))
    print("  rows with probability difference > 1e-7:", int(np.count_nonzero(pdiff > 1e-7)))
    print("  threshold-0.50 prediction disagreements:", int(np.count_nonzero((p_v >= .5) != (p_n >= .5))))
    print("\nConfusion matrices at threshold 0.50 (TN FP FN TP):")
    cm_v = confusion(p_v >= .5, y)
    cm_n = confusion(p_n >= .5, y)
    print("  voting script:", cm_v)
    print("  noise script :", cm_n)
    print("  expected     :", EXPECTED)

    if all_equal and np.nanmax(pdiff) <= 1e-7 and cm_v == cm_n == EXPECTED:
        print("\nPASS: both scripts load the same parameters and reproduce the expected baseline.")
        return 0
    if not all_equal or np.nanmax(pdiff) > 1e-7 or cm_v != cm_n:
        print("\nFAIL: the live scripts do not load/produce equivalent predictions. Do not trust the noise report yet.")
        return 1
    print("\nFAIL: both scripts agree, but their shared baseline does not match the authoritative confusion matrix.")
    print("This points to a model/scaler version or baseline provenance mismatch, not feature CSV mismatch.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
