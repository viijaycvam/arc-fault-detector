"""30_voting_eval.py - M-of-N voting evaluation for the DEPLOYED 8-feature model.

Run from the repo root:  python src\30_voting_eval.py
Reads : reports/features_b.csv, src/model_weights_b.h, src/scaler_b.h
Writes: reports/voting_eval.txt (also printed)

The deployed weights are read from your C headers, so no retraining happens.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATS = ["crest", "skew", "kurt", "d1_ratio", "d2_ratio",
         "peak_asym", "energy_asym", "frac_small"]
CSV = Path("reports") / "features_b.csv"
MODEL_H = Path("src") / "model_weights_b.h"
SCALER_H = Path("src") / "scaler_b.h"
OUT = Path("reports") / "voting_eval.txt"

CYCLES_PER_SEC = 50
NORMAL_CYCLES = 5_000_000          # about 27.8 hours of simulated normal operation
SEQ_PER_CLASS, SEQ_LEN = 2000, 100  # arc sequences per class, cycles per sequence
FAST = 25                           # "fast" detection = within 25 cycles = 0.5 s
RULES = [(3, 3), (5, 3), (5, 4), (8, 5), (8, 6), (10, 7), (10, 8)]  # (window N, votes M)
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
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
    m = read_arrays(MODEL_H)
    md = {name: arr for name, arr in m}
    required = ['SCALER_MEAN_B', 'SCALER_SCALE_B', 'W1_B', 'B1_B', 'W2_B', 'B2_B', 'W3_B', 'B3_B']
    missing = [name for name in required if name not in md]
    if missing:
        sys.exit(f'Missing arrays in {MODEL_H}: {missing}')

    s = read_arrays(SCALER_H)
    sd = {name: arr for name, arr in s}
    mean = sd.get('SCALER_MEAN_B', md['SCALER_MEAN_B'])
    scale = sd.get('SCALER_SCALE_B', md['SCALER_SCALE_B'])

    if mean.size != 8 or scale.size != 8:
        sys.exit(f'Unexpected scaler sizes: mean={mean.size}, scale={scale.size}')

    if md['W1_B'].size != 128 or md['B1_B'].size != 16 or md['W2_B'].size != 128 or md['B2_B'].size != 8 or md['W3_B'].size != 8 or md['B3_B'].size != 1:
        sys.exit('Unexpected deployed model array sizes.')

    return (md['W1_B'].reshape(8, 16), md['B1_B'], md['W2_B'].reshape(16, 8), md['B2_B'], md['W3_B'].reshape(8), md['B3_B'][0], mean, scale)

def predict(X, params):
    W1, B1, W2, B2, W3, B3, mean, scale = params
    z = (X - mean) / scale
    h1 = np.maximum(z @ W1 + B1, 0.0)
    h2 = np.maximum(h1 @ W2 + B2, 0.0)
    return 1.0 / (1.0 + np.exp(-(h2 @ W3 + B3)))


def false_trips_per_hour(samples, thr, n_win, m_votes):
    bits = (samples >= thr).astype(np.int64)
    c = np.cumsum(bits)
    w = c.copy()
    w[n_win:] = c[n_win:] - c[:-n_win]
    trip = w >= m_votes
    events = int(np.count_nonzero(trip[1:] & ~trip[:-1])) + int(trip[0])
    hours = len(samples) / CYCLES_PER_SEC / 3600.0
    return events, hours


def detection(pc, thr, n_win, m_votes):
    bits = (rng.choice(pc, size=(SEQ_PER_CLASS, SEQ_LEN)) >= thr).astype(np.int64)
    c = np.cumsum(bits, axis=1)
    w = c.copy()
    w[:, n_win:] = c[:, n_win:] - c[:, :-n_win]
    trip = w >= m_votes
    hit = trip.any(axis=1)
    first = np.where(hit, trip.argmax(axis=1) + 1, SEQ_LEN + 1)
    within = float((first <= FAST).mean())
    delay = float(np.median(first[hit])) if hit.any() else float("nan")
    return within, delay


def main():
    params = load_params()
    df = pd.read_csv(CSV)
    te = df[df["split"] == "test"]
    p = predict(te[FEATS].to_numpy(dtype=np.float64), params)
    y = te["binary_label"].to_numpy()
    cls = te["detailed_label"].to_numpy()

    pred = p >= 0.5
    tn = int(((~pred) & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum()); tp = int((pred & (y == 1)).sum())
    log("DEPLOYED MODEL, official test set, threshold 0.5")
    log("TN/FP/FN/TP: %d %d %d %d   (expected 850 11 81 392)" % (tn, fp, fn, tp))
    if (tn, fp, fn, tp) != (850, 11, 81, 392):
        log("WARNING: does not match the authoritative result - stop and send me this output.")
    log("")

    pn = p[y == 0]
    classes = sorted(int(k) for k in np.unique(cls) if k > 0)
    normal_stream = rng.choice(pn, size=NORMAL_CYCLES)

    log("Voting evaluation (normal cycles and arc cycles resampled independently)")
    log("false trips: %d simulated normal cycles; detect = within %d cycles (%.1f s)"
        % (NORMAL_CYCLES, FAST, FAST / CYCLES_PER_SEC))
    log("%-6s %-5s %-9s %-8s %-9s %-8s %-18s %s"
        % ("rule", "thr", "cycFTR%", "cycDet%", "FT/hour", "meanDet", "worst class (P)", "median delay"))
    for thr in THRESHOLDS:
        cyc_ftr = 100.0 * float((pn >= thr).mean())
        cyc_det = 100.0 * float((p[y == 1] >= thr).mean())
        for n_win, m_votes in RULES:
            ev, hours = false_trips_per_hour(normal_stream, thr, n_win, m_votes)
            ft = ("<%.2f" % (3.0 / hours)) if ev == 0 else "%.2f" % (ev / hours)
            res = [detection(p[cls == k], thr, n_win, m_votes) for k in classes]
            dets = np.array([r[0] for r in res]); dly = np.array([r[1] for r in res])
            worst = int(np.argmin(dets))
            log("%-6s %-5.2f %-9.2f %-8.2f %-9s %-8.3f class %-2d (%.2f)    %.1f cycles"
                % ("%d/%d" % (m_votes, n_win), thr, cyc_ftr, cyc_det, ft,
                   float(dets.mean()), classes[worst], float(dets[worst]),
                   float(np.nanmedian(dly))))
        log("")
    log("Notes:")
    log("- FT/hour: '<x' means zero false trips were seen; x is the rule-of-three upper bound.")
    log("- Normal cycles are resampled independently, so real false trips from a persistent")
    log("  nuisance load could be MORE frequent than shown. Arc cycles are resampled within")
    log("  each class, so hard classes stay hard.")
    log("- Sequences are simulated from dataset cycles; this is not field performance.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("saved:", OUT)


if __name__ == "__main__":
    main()