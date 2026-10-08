"""
31_tune_voting_train.py

Training-only temporal voting tuner for the DEPLOYED 8-feature model.

Reads:
    reports/features_b.csv
    src/model_weights_b.h

Uses:
    ONLY split == "train"

Does NOT retrain the model.

Purpose:
    Tune threshold + M-of-N voting policy on training data only,
    while preserving short-range correlation through block bootstrap.

Run from repo root:
    python src\31_tune_voting_train.py

Writes:
    reports/voting_tune_train.txt
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FEATS = [
    "crest",
    "skew",
    "kurt",
    "d1_ratio",
    "d2_ratio",
    "peak_asym",
    "energy_asym",
    "frac_small",
]

CSV = Path("reports") / "features_b.csv"
MODEL_H = Path("src") / "model_weights_b.h"
OUT = Path("reports") / "voting_tune_train.txt"

CYCLES_PER_SEC = 50

# Training-only simulation sizes.
NORMAL_CYCLES = 5_000_000

SEQ_PER_CLASS = 2_000
SEQ_LEN = 100

FAST_CYCLES = 25

BLOCK_LEN = 5

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
]

# (votes, window)
RULES = [
    (3, 3),
    (3, 5),
    (4, 5),
    (5, 8),
    (6, 8),
    (7, 10),
    (8, 10),
]

RANDOM_SEED = 12345

rng = np.random.default_rng(RANDOM_SEED)

lines = []


def log(s=""):
    print(s)
    lines.append(s)


def read_named_array(text, name):
    pattern = (
        r"static\s+const\s+float\s+"
        + re.escape(name)
        + r"(?:\[[^\]]+\])+"
        r"\s*=\s*\{(.*?)\};"
    )

    m = re.search(pattern, text, re.S)

    if not m:
        raise RuntimeError(
            f"Could not find array {name} in {MODEL_H}"
        )

    values = re.findall(
        r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?",
        m.group(1),
    )

    return np.asarray(values, dtype=np.float64)


def load_model():
    text = MODEL_H.read_text(encoding="utf-8")

    mean = read_named_array(text, "SCALER_MEAN_B")
    scale = read_named_array(text, "SCALER_SCALE_B")

    W1 = read_named_array(text, "W1_B")
    B1 = read_named_array(text, "B1_B")

    W2 = read_named_array(text, "W2_B")
    B2 = read_named_array(text, "B2_B")

    W3 = read_named_array(text, "W3_B")
    B3 = read_named_array(text, "B3_B")

    if mean.size != 8:
        raise RuntimeError("Unexpected scaler mean size.")

    if scale.size != 8:
        raise RuntimeError("Unexpected scaler scale size.")

    if W1.size != 128:
        raise RuntimeError("Unexpected W1 size.")

    if B1.size != 16:
        raise RuntimeError("Unexpected B1 size.")

    if W2.size != 128:
        raise RuntimeError("Unexpected W2 size.")

    if B2.size != 8:
        raise RuntimeError("Unexpected B2 size.")

    if W3.size != 8:
        raise RuntimeError("Unexpected W3 size.")

    if B3.size != 1:
        raise RuntimeError("Unexpected B3 size.")

    return (
        W1.reshape(8, 16),
        B1,
        W2.reshape(16, 8),
        B2,
        W3.reshape(8),
        B3[0],
        mean,
        scale,
    )


def predict(X, params):
    W1, B1, W2, B2, W3, B3, mean, scale = params

    z = (X - mean) / scale

    h1 = np.maximum(
        z @ W1 + B1,
        0.0,
    )

    h2 = np.maximum(
        h1 @ W2 + B2,
        0.0,
    )

    out = h2 @ W3 + B3

    return 1.0 / (1.0 + np.exp(-out))


def make_runs(df, probability):
    """
    Build contiguous runs using record_id adjacency and detailed_label.

    A run never crosses:
      - an appliance/class boundary
      - a gap in record_id
    """

    order = df.sort_values("record_id").reset_index(drop=True)

    runs = []

    start = 0

    for i in range(1, len(order) + 1):

        boundary = False

        if i == len(order):
            boundary = True
        else:
            prev_id = int(order.loc[i - 1, "record_id"])
            cur_id = int(order.loc[i, "record_id"])

            prev_label = int(
                order.loc[i - 1, "detailed_label"]
            )

            cur_label = int(
                order.loc[i, "detailed_label"]
            )

            if cur_id != prev_id + 1:
                boundary = True

            if cur_label != prev_label:
                boundary = True

        if boundary:

            part = probability[start:i]

            label = int(
                order.loc[start, "detailed_label"]
            )

            runs.append(
                {
                    "label": label,
                    "start_id": int(
                        order.loc[start, "record_id"]
                    ),
                    "end_id": int(
                        order.loc[i - 1, "record_id"]
                    ),
                    "prob": part.copy(),
                }
            )

            start = i

    return runs


def bootstrap_sequence(
    runs,
    length,
    block_len,
    rng_local,
):
    """
    Build a synthetic sequence using contiguous blocks.

    Each block comes from one real contiguous training run.
    Blocks never cross run boundaries.
    """

    if not runs:
        raise RuntimeError("No runs available.")

    out = []

    while len(out) < length:

        eligible = [
            r for r in runs
            if len(r["prob"]) >= 1
        ]

        r = eligible[
            int(
                rng_local.integers(
                    0,
                    len(eligible)
                )
            )
        ]

        values = r["prob"]

        take = min(
            block_len,
            length - len(out),
        )

        if len(values) <= take:

            start = 0

        else:

            start = int(
                rng_local.integers(
                    0,
                    len(values) - take + 1,
                )
            )

        out.extend(
            values[
                start:start + take
            ]
        )

    return np.asarray(
        out,
        dtype=np.float64,
    )


def first_trip(
    positive,
    votes,
    window,
):
    """
    Return first cycle index where the M-of-N rule trips.
    Returns None when no trip occurs.
    """

    if window <= 0:
        raise ValueError("Window must be positive.")

    if votes <= 0 or votes > window:
        raise ValueError(
            "Invalid voting rule."
        )

    running = 0

    for i, flag in enumerate(positive):

        if flag:
            running += 1

        if i >= window:

            if positive[
                i - window
            ]:

                running -= 1

        if i + 1 >= window:

            if running >= votes:

                return i

    return None


def false_trip_stream(
    normal_runs,
    threshold,
    votes,
    window,
    rng_local,
):
    """
    Generate 5,000,000 synthetic normal cycles using
    correlated block bootstrap and count rising trip events.
    """

    remaining = NORMAL_CYCLES

    history = []

    events = 0
    previous_trip = False

    while remaining > 0:

        take = min(
            100,
            remaining,
        )

        seq = bootstrap_sequence(
            normal_runs,
            take,
            BLOCK_LEN,
            rng_local,
        )

        history.extend(
            (seq >= threshold).astype(
                np.uint8
            ).tolist()
        )

        remaining -= take

    positive = np.asarray(
        history,
        dtype=np.uint8,
    )

    running = 0

    for i, flag in enumerate(positive):

        if flag:
            running += 1

        if i >= window:

            if positive[
                i - window
            ]:

                running -= 1

        trip = (
            i + 1 >= window
            and running >= votes
        )

        if trip and not previous_trip:
            events += 1

        previous_trip = trip

    hours = (
        len(positive)
        / CYCLES_PER_SEC
        / 3600.0
    )

    return events, hours


def arc_detection(
    class_runs,
    threshold,
    votes,
    window,
    rng_local,
):
    """
    Build 2,000 synthetic 100-cycle arc sequences
    for one class using correlated block bootstrap.

    Detection means first trip occurs within 25 cycles.
    """

    hits = 0
    delays = []

    for _ in range(SEQ_PER_CLASS):

        seq = bootstrap_sequence(
            class_runs,
            SEQ_LEN,
            BLOCK_LEN,
            rng_local,
        )

        positive = (
            seq >= threshold
        )

        trip = first_trip(
            positive,
            votes,
            window,
        )

        if trip is not None:

            delay = trip + 1

            if delay <= FAST_CYCLES:

                hits += 1
                delays.append(
                    delay
                )

    detection_rate = (
        hits / SEQ_PER_CLASS
    )

    if delays:

        median_delay = float(
            np.median(
                np.asarray(
                    delays,
                    dtype=np.float64,
                )
            )
        )

    else:

        median_delay = float(
            "nan"
        )

    return (
        detection_rate,
        median_delay,
    )


def main():

    if not CSV.exists():
        raise FileNotFoundError(
            f"Missing {CSV}"
        )

    if not MODEL_H.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_H}"
        )

    params = load_model()

    df = pd.read_csv(CSV)

    train = (
        df[
            df["split"] == "train"
        ]
        .sort_values("record_id")
        .reset_index(drop=True)
    )

    log("=" * 72)
    log("TRAINING-ONLY TEMPORAL VOTING TUNER")
    log("=" * 72)

    log(
        f"Training records : {len(train)}"
    )

    log(
        f"Normal records   : "
        f"{int((train.binary_label == 0).sum())}"
    )

    log(
        f"Arc records      : "
        f"{int((train.binary_label == 1).sum())}"
    )

    log(
        f"Block bootstrap  : {BLOCK_LEN} cycles"
    )

    log(
        f"Normal simulation: "
        f"{NORMAL_CYCLES:,} cycles"
    )

    log(
        f"Arc sequences/class: "
        f"{SEQ_PER_CLASS:,}"
    )

    log(
        f"Detection limit  : "
        f"{FAST_CYCLES} cycles "
        f"({FAST_CYCLES / CYCLES_PER_SEC:.1f} s)"
    )

    log(
        f"Random seed      : {RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # Predict the exact deployed model on training records.
    # --------------------------------------------------------

    X_train = train[
        FEATS
    ].to_numpy(
        dtype=np.float64
    )

    probabilities = predict(
        X_train,
        params,
    )

    train["probability"] = probabilities

    # --------------------------------------------------------
    # Build contiguous runs.
    # --------------------------------------------------------

    runs = make_runs(
        train,
        probabilities,
    )

    normal_runs = [
        r for r in runs
        if r["label"] == 0
    ]

    arc_runs = {
        k: [
            r for r in runs
            if r["label"] == k
        ]
        for k in range(1, 14)
    }

    log("")
    log(
        f"Normal contiguous runs: "
        f"{len(normal_runs)}"
    )

    for k in range(1, 14):

        count = sum(
            len(r["prob"])
            for r in arc_runs[k]
        )

        log(
            f"Arc class {k:2d}: "
            f"{len(arc_runs[k])} run(s), "
            f"{count} cycles"
        )

    # --------------------------------------------------------
    # Candidate evaluation.
    # --------------------------------------------------------

    log("")
    log(
        "RULE RESULTS"
    )
    log(
        "-" * 72
    )

    log(
        f"{'rule':<7}"
        f"{'thr':<7}"
        f"{'FT/hour':<12}"
        f"{'meanDet':<12}"
        f"{'worstCls':<10}"
        f"{'worstDet':<12}"
        f"{'medDelay':<10}"
    )

    all_results = []

    for threshold in THRESHOLDS:

        for votes, window in RULES:

            normal_rng = np.random.default_rng(
                RANDOM_SEED
                + int(threshold * 100)
                + votes * 100
                + window
            )

            events, hours = (
                false_trip_stream(
                    normal_runs,
                    threshold,
                    votes,
                    window,
                    normal_rng,
                )
            )

            if events == 0:

                ft_hour = (
                    3.0 / hours
                )

            else:

                ft_hour = (
                    events / hours
                )

            class_results = []

            for k in range(1, 14):

                class_rng = (
                    np.random.default_rng(
                        RANDOM_SEED
                        + 1000 * k
                        + int(threshold * 100)
                        + votes * 100
                        + window
                    )
                )

                det, delay = (
                    arc_detection(
                        arc_runs[k],
                        threshold,
                        votes,
                        window,
                        class_rng,
                    )
                )

                class_results.append(
                    (
                        k,
                        det,
                        delay,
                    )
                )

            dets = np.asarray(
                [
                    x[1]
                    for x in class_results
                ],
                dtype=np.float64,
            )

            delays = np.asarray(
                [
                    x[2]
                    for x in class_results
                    if np.isfinite(x[2])
                ],
                dtype=np.float64,
            )

            worst_idx = int(
                np.argmin(dets)
            )

            worst_class = (
                class_results[worst_idx][0]
            )

            worst_det = float(
                dets[worst_idx]
            )

            mean_det = float(
                np.mean(dets)
            )

            median_delay = (
                float(np.median(delays))
                if len(delays)
                else float("nan")
            )

            row = {
                "votes": votes,
                "window": window,
                "threshold": threshold,
                "ft_hour": ft_hour,
                "mean_det": mean_det,
                "worst_class": worst_class,
                "worst_det": worst_det,
                "median_delay": median_delay,
                "class_results": class_results,
            }

            all_results.append(row)

            log(
                f"{votes}/{window:<4}"
                f"{threshold:<7.2f}"
                f"{ft_hour:<12.3f}"
                f"{mean_det:<12.3f}"
                f"class {worst_class:<5}"
                f"{worst_det:<12.3f}"
                f"{median_delay:<10.1f}"
            )

    # --------------------------------------------------------
    # Ranking.
    #
    # Primary:
    #   maximize worst-class detection
    #
    # Secondary:
    #   minimize false trips/hour
    #
    # Tertiary:
    #   maximize mean detection
    #
    # Final:
    #   minimize median delay
    # --------------------------------------------------------

    ranked = sorted(
        all_results,
        key=lambda r: (
            -r["worst_det"],
            r["ft_hour"],
            -r["mean_det"],
            r["median_delay"]
            if np.isfinite(
                r["median_delay"]
            )
            else 1e9,
        ),
    )

    log("")
    log(
        "TOP TRAINING-SET CANDIDATES"
    )
    log(
        "-" * 72
    )

    for i, r in enumerate(
        ranked[:10],
        start=1,
    ):

        log(
            f"{i:2d}. "
            f"{r['votes']}/{r['window']} "
            f"thr={r['threshold']:.2f} "
            f"FT/hour={r['ft_hour']:.3f} "
            f"meanDet={r['mean_det']:.3f} "
            f"worst=class {r['worst_class']} "
            f"{r['worst_det']:.3f} "
            f"median={r['median_delay']:.1f}"
        )

    log("")
    log(
        "IMPORTANT LIMITATIONS"
    )
    log(
        "- Training data only; no test data was used for tuning."
    )
    log(
        "- Record adjacency is used as a proxy for local correlation."
    )
    log(
        "- No explicit appliance/session identifier exists in features_b.csv."
    )
    log(
        "- Block bootstrap preserves short-range score correlation but is not"
    )
    log(
        "  a measurement of real appliance behavior."
    )
    log(
        "- False-trip estimates are simulation estimates, not field rates."
    )
    log(
        "- The selected rule MUST be frozen before test-set evaluation."
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    log("")
    log(
        f"Saved: {OUT}"
    )


if __name__ == "__main__":
    main()