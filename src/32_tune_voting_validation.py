"""
32_tune_voting_validation.py

POST-HOC VALIDATION OF THE FROZEN DEPLOYED MODEL

Important:
    model_weights_b.h is the already-trained deployed model.
    This script does NOT retrain the neural network.

Data:
    reports/features_b.csv

Only:
    split == "train"

The official test split is NEVER used.

Validation construction:
    For every contiguous training run, the final 20% of that run
    is reserved as a post-hoc validation block.

Temporal simulation:
    5-cycle block bootstrap is used to preserve short-range
    score correlation.

Outputs:
    reports/voting_tune_validation.txt

Run from repo root:
    python src\32_tune_voting_validation.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

CSV = Path(r"reports\features_b.csv")
MODEL_H = Path(r"src\model_weights_b.h")
OUT = Path(r"reports\voting_tune_validation.txt")

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

CYCLES_PER_SEC = 50

# Validation holdout from each contiguous run.
VALIDATION_FRACTION = 0.20

# Correlation-preserving block size.
BLOCK_LEN = 5

# Simulated normal stream.
NORMAL_CYCLES = 1_000_000

# Arc sequence simulation.
SEQ_PER_CLASS = 2000
SEQ_LEN = 100

# "Fast" detection target.
FAST_CYCLES = 25

# Candidate probability thresholds.
THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
]

# Candidate voting rules.
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

RANDOM_SEED = 24680


# ============================================================
# REPORT BUFFER
# ============================================================

lines = []


def log(text=""):
    print(text)
    lines.append(text)


# ============================================================
# C HEADER ARRAY READER
# ============================================================

def read_named_array(text, name):
    """
    Read one static const float C array by name.
    """

    pattern = (
        r"static\s+const\s+float\s+"
        + re.escape(name)
        + r"(?:\[[^\]]+\])+\s*=\s*\{(.*?)\};"
    )

    match = re.search(
        pattern,
        text,
        flags=re.S,
    )

    if match is None:
        raise RuntimeError(
            f"Missing array {name} in {MODEL_H}"
        )

    numbers = re.findall(
        r"[-+]?(?:\d+\.?\d*|\.\d+)"
        r"(?:[eE][-+]?\d+)?",
        match.group(1),
    )

    return np.asarray(
        numbers,
        dtype=np.float64,
    )


# ============================================================
# LOAD EXACT DEPLOYED MODEL
# ============================================================

def load_model():
    """
    Load the exact model exported for ESP32.

    Architecture:
        8 -> 16 -> 8 -> 1
    """

    text = MODEL_H.read_text(
        encoding="utf-8"
    )

    mean = read_named_array(
        text,
        "SCALER_MEAN_B",
    )

    scale = read_named_array(
        text,
        "SCALER_SCALE_B",
    )

    W1 = read_named_array(
        text,
        "W1_B",
    )

    B1 = read_named_array(
        text,
        "B1_B",
    )

    W2 = read_named_array(
        text,
        "W2_B",
    )

    B2 = read_named_array(
        text,
        "B2_B",
    )

    W3 = read_named_array(
        text,
        "W3_B",
    )

    B3 = read_named_array(
        text,
        "B3_B",
    )

    sizes = [
        mean.size,
        scale.size,
        W1.size,
        B1.size,
        W2.size,
        B2.size,
        W3.size,
        B3.size,
    ]

    expected = [
        8,
        8,
        128,
        16,
        128,
        8,
        8,
        1,
    ]

    if sizes != expected:
        raise RuntimeError(
            "Unexpected deployed model array sizes: "
            f"{sizes}"
        )

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


# ============================================================
# EXACT DEPLOYED FORWARD PASS
# ============================================================

def predict(X, params):
    """
    Exact Python equivalent of the deployed C MLP.
    """

    (
        W1,
        B1,
        W2,
        B2,
        W3,
        B3,
        mean,
        scale,
    ) = params

    z = (
        X - mean
    ) / scale

    h1 = np.maximum(
        z @ W1 + B1,
        0.0,
    )

    h2 = np.maximum(
        h1 @ W2 + B2,
        0.0,
    )

    out = (
        h2 @ W3 + B3
    )

    return 1.0 / (
        1.0 + np.exp(-out)
    )


# ============================================================
# FIND CONTIGUOUS RUNS
# ============================================================

def make_runs(df):
    """
    Split rows into contiguous runs.

    A new run starts when:
        record_id is not consecutive
        OR detailed_label changes.
    """

    d = (
        df.sort_values("record_id")
        .reset_index(drop=True)
    )

    runs = []

    if len(d) == 0:
        return runs

    start = 0

    for i in range(1, len(d) + 1):

        boundary = False

        if i == len(d):

            boundary = True

        else:

            prev_id = int(
                d.loc[
                    i - 1,
                    "record_id",
                ]
            )

            cur_id = int(
                d.loc[
                    i,
                    "record_id",
                ]
            )

            prev_label = int(
                d.loc[
                    i - 1,
                    "detailed_label",
                ]
            )

            cur_label = int(
                d.loc[
                    i,
                    "detailed_label",
                ]
            )

            if cur_id != prev_id + 1:
                boundary = True

            if cur_label != prev_label:
                boundary = True

        if boundary:

            runs.append(
                d.iloc[
                    start:i
                ].copy()
            )

            start = i

    return runs


# ============================================================
# BUILD POST-HOC VALIDATION SET
# ============================================================

def build_validation_set(train):
    """
    Reserve the final 20% of every contiguous run.

    This does NOT retrain the model.

    It is therefore post-hoc validation of the frozen
    model, not an independent training/validation split.
    """

    tune_parts = []
    validation_parts = []

    for run in make_runs(train):

        n = len(run)

        holdout = max(
            BLOCK_LEN,
            int(
                np.ceil(
                    n * VALIDATION_FRACTION
                )
            ),
        )

        # Never consume the entire run if a tuning portion exists.
        if holdout >= n and n > 1:
            holdout = n // 2

        split_at = n - holdout

        tune_part = run.iloc[
            :split_at
        ].copy()

        validation_part = run.iloc[
            split_at:
        ].copy()

        if len(tune_part) > 0:
            tune_parts.append(
                tune_part
            )

        if len(validation_part) > 0:
            validation_parts.append(
                validation_part
            )

    tune = (
        pd.concat(
            tune_parts,
            ignore_index=True,
        )
        if tune_parts
        else pd.DataFrame()
    )

    validation = (
        pd.concat(
            validation_parts,
            ignore_index=True,
        )
        if validation_parts
        else pd.DataFrame()
    )

    return (
        tune.sort_values(
            "record_id"
        ).reset_index(drop=True),
        validation.sort_values(
            "record_id"
        ).reset_index(drop=True),
    )


# ============================================================
# MAKE FIXED-SIZE CORRELATION BLOCKS
# ============================================================

def make_blocks_from_runs(df):
    """
    Convert validation runs into exact 5-cycle blocks.

    Any incomplete remainder is discarded.
    """

    result = {}

    for label in range(14):
        result[label] = []

    for run in make_runs(df):

        label = int(
            run.iloc[0][
                "detailed_label"
            ]
        )

        values = (
            run[
                "probability"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        count = (
            len(values)
            // BLOCK_LEN
        )

        if count == 0:
            continue

        usable = (
            values[
                : count * BLOCK_LEN
            ]
        )

        reshaped = usable.reshape(
            count,
            BLOCK_LEN,
        )

        for block in reshaped:
            result[label].append(
                block.copy()
            )

    return result


# ============================================================
# BUILD BOOTSTRAPPED SEQUENCES
# ============================================================

def bootstrap_sequences(
    block_array,
    n_sequences,
    sequence_length,
    rng,
):
    """
    Build sequences from complete 5-cycle blocks.
    """

    if block_array.size == 0:
        raise RuntimeError(
            "No bootstrap blocks available."
        )

    blocks_needed = int(
        np.ceil(
            sequence_length
            / BLOCK_LEN
        )
    )

    indices = rng.integers(
        0,
        len(block_array),
        size=(
            n_sequences,
            blocks_needed,
        ),
    )

    selected = block_array[
        indices
    ]

    sequences = selected.reshape(
        n_sequences,
        blocks_needed * BLOCK_LEN,
    )

    return sequences[
        :,
        :sequence_length,
    ]


# ============================================================
# FIRST M-OF-N TRIP
# ============================================================

def first_trip_indices(
    positive,
    votes,
    window,
):
    """
    For each sequence, return the first cycle index
    where the M-of-N rule trips.

    Returns:
        zero-based index
        -1 if no trip exists
    """

    n_sequences, length = (
        positive.shape
    )

    if window > length:
        return np.full(
            n_sequences,
            -1,
            dtype=np.int32,
        )

    cumulative = np.cumsum(
        positive.astype(
            np.int32
        ),
        axis=1,
    )

    current = cumulative[
        :,
        window - 1:
    ]

    if window == 1:

        previous = np.zeros(
            current.shape,
            dtype=np.int32,
        )

    else:

        previous = np.concatenate(
            [
                np.zeros(
                    (
                        n_sequences,
                        1,
                    ),
                    dtype=np.int32,
                ),
                cumulative[
                    :,
                    :-window,
                ],
            ],
            axis=1,
        )

    counts = (
        current - previous
    )

    trip = (
        counts >= votes
    )

    found = trip.any(
        axis=1
    )

    first_window = (
        trip.argmax(
            axis=1
        )
    )

    result = np.full(
        n_sequences,
        -1,
        dtype=np.int32,
    )

    result[found] = (
        first_window[found]
        + window
    )

    return result


# ============================================================
# NORMAL FALSE-TRIP SIMULATION
# ============================================================

def simulate_normal_stream(
    normal_blocks,
    threshold,
    votes,
    window,
    rng,
):
    """
    Simulate a long normal stream using 5-cycle block
    bootstrap.

    Counts rising edges of the temporal trip state.
    """

    block_array = np.asarray(
        normal_blocks,
        dtype=np.float64,
    )

    if (
        block_array.ndim != 2
        or block_array.shape[1]
        != BLOCK_LEN
    ):
        raise RuntimeError(
            "Invalid normal block array."
        )

    blocks_needed = int(
        np.ceil(
            NORMAL_CYCLES
            / BLOCK_LEN
        )
    )

    indices = rng.integers(
        0,
        len(block_array),
        size=blocks_needed,
    )

    stream = block_array[
        indices
    ].reshape(-1)[
        :NORMAL_CYCLES
    ]

    positive = (
        stream >= threshold
    )

    cumulative = np.cumsum(
        positive.astype(
            np.int32
        )
    )

    current = cumulative[
        window - 1:
    ]

    if window == 1:

        previous = np.zeros(
            len(current),
            dtype=np.int32,
        )

    else:

        previous = np.concatenate(
            [
                np.zeros(
                    1,
                    dtype=np.int32,
                ),
                cumulative[
                    :-window
                ],
            ]
        )

    counts = (
        current - previous
    )

    trip = (
        counts >= votes
    )

    if len(trip) == 0:
        events = 0

    else:

        events = int(
            trip[0]
        )

        if len(trip) > 1:

            events += int(
                np.sum(
                    trip[1:]
                    & ~trip[:-1]
                )
            )

    hours = (
        NORMAL_CYCLES
        / CYCLES_PER_SEC
        / 3600.0
    )

    return events, hours


# ============================================================
# ARC DETECTION SIMULATION
# ============================================================

def simulate_arc_class(
    block_array,
    threshold,
    votes,
    window,
    rng,
):
    """
    Build synthetic arc sequences for one detailed class.

    Detection:
        first temporal trip <= FAST_CYCLES.

    Delay:
        first trip cycle number, one-based.
    """

    if len(block_array) == 0:
        return (
            np.nan,
            np.nan,
        )

    sequences = bootstrap_sequences(
        block_array,
        SEQ_PER_CLASS,
        SEQ_LEN,
        rng,
    )

    positive = (
        sequences >= threshold
    )

    first = first_trip_indices(
        positive,
        votes,
        window,
    )

    hit = (
        (first >= 0)
        & (first + 1 <= FAST_CYCLES)
    )

    detection_rate = float(
        np.mean(hit)
    )

    delays = (
        first[hit] + 1
    )

    if len(delays) == 0:

        median_delay = np.nan

    else:

        median_delay = float(
            np.median(
                delays
            )
        )

    return (
        detection_rate,
        median_delay,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # File checks
    # --------------------------------------------------------

    if not CSV.exists():
        raise FileNotFoundError(
            f"Missing {CSV}"
        )

    if not MODEL_H.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_H}"
        )

    # --------------------------------------------------------
    # Load exact deployed model
    # --------------------------------------------------------

    params = load_model()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        CSV
    )

    required_columns = (
        [
            "record_id",
            "split",
            "detailed_label",
            "binary_label",
        ]
        + FEATURES
    )

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing CSV columns: "
            + ", ".join(missing)
        )

    train = (
        df[
            df["split"] == "train"
        ]
        .copy()
        .sort_values(
            "record_id"
        )
        .reset_index(drop=True)
    )

    test = df[
        df["split"] == "test"
    ].copy()

    # --------------------------------------------------------
    # Compute exact frozen-model probabilities
    # --------------------------------------------------------

    train["probability"] = predict(
        train[
            FEATURES
        ].to_numpy(
            dtype=np.float64
        ),
        params,
    )

    test["probability"] = predict(
        test[
            FEATURES
        ].to_numpy(
            dtype=np.float64
        ),
        params,
    )

    # --------------------------------------------------------
    # Sanity-check the official deployed model.
    # --------------------------------------------------------

    test_pred = (
        test[
            "probability"
        ].to_numpy()
        >= 0.5
    )

    test_y = (
        test[
            "binary_label"
        ].to_numpy()
    )

    tn = int(
        np.sum(
            (~test_pred)
            & (test_y == 0)
        )
    )

    fp = int(
        np.sum(
            test_pred
            & (test_y == 0)
        )
    )

    fn = int(
        np.sum(
            (~test_pred)
            & (test_y == 1)
        )
    )

    tp = int(
        np.sum(
            test_pred
            & (test_y == 1)
        )
    )

    # --------------------------------------------------------
    # Build post-hoc validation split.
    # --------------------------------------------------------

    _, validation = (
        build_validation_set(
            train
        )
    )

    if len(validation) == 0:
        raise RuntimeError(
            "Validation set is empty."
        )

    validation_classes = sorted(
        int(x)
        for x in validation[
            "detailed_label"
        ].unique()
    )

    validation_blocks = (
        make_blocks_from_runs(
            validation
        )
    )

    normal_blocks = validation_blocks[
        0
    ]

    arc_blocks = {
        k: validation_blocks[k]
        for k in range(1, 14)
    }

    normal_block_array = np.asarray(
        normal_blocks,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    log("=" * 72)
    log(
        "POST-HOC VALIDATION OF FROZEN DEPLOYED MODEL"
    )
    log("=" * 72)

    log(
        f"Full training records : {len(train)}"
    )

    log(
        f"Official test records  : {len(test)}"
    )

    log(
        f"Post-hoc validation    : {len(validation)}"
    )

    log(
        "Official test split is NOT used for tuning."
    )

    log(
        "Neural-network weights are fixed; no retraining occurs."
    )

    log(
        f"Validation fraction/run: "
        f"{VALIDATION_FRACTION:.0%}"
    )

    log(
        f"Bootstrap block size   : "
        f"{BLOCK_LEN} cycles"
    )

    log(
        f"Normal simulation      : "
        f"{NORMAL_CYCLES:,} cycles"
    )

    log(
        f"Arc sequences/class    : "
        f"{SEQ_PER_CLASS:,}"
    )

    log(
        f"Detection limit        : "
        f"{FAST_CYCLES} cycles "
        f"({FAST_CYCLES / CYCLES_PER_SEC:.2f} s)"
    )

    log(
        f"Random seed            : "
        f"{RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # Official model sanity check
    # --------------------------------------------------------

    log("")
    log(
        "FROZEN MODEL SANITY CHECK"
    )
    log(
        "-" * 72
    )

    log(
        f"TN/FP/FN/TP = "
        f"{tn}/{fp}/{fn}/{tp}"
    )

    if (
        tn,
        fp,
        fn,
        tp,
    ) == (
        850,
        11,
        81,
        392,
    ):

        log(
            "Sanity check: PASS"
        )

    else:

        log(
            "Sanity check: FAIL"
        )

        raise RuntimeError(
            "Frozen model does not reproduce "
            "authoritative 850/11/81/392."
        )

    # --------------------------------------------------------
    # Validation set composition
    # --------------------------------------------------------

    log("")
    log(
        "VALIDATION DATA"
    )
    log(
        "-" * 72
    )

    log(
        "Class 0 normal cycles : "
        + str(
            int(
                (
                    validation[
                        "detailed_label"
                    ] == 0
                ).sum()
            )
        )
    )

    for k in range(1, 14):

        count = int(
            (
                validation[
                    "detailed_label"
                ] == k
            ).sum()
        )

        block_count = len(
            arc_blocks[k]
        )

        log(
            f"Class {k:2d} arc cycles : "
            f"{count:3d} "
            f"({block_count} blocks)"
        )

    if len(
        normal_block_array
    ) == 0:

        raise RuntimeError(
            "No complete normal validation blocks."
        )

    # --------------------------------------------------------
    # Validation per-cycle reference metrics
    # --------------------------------------------------------

    log("")
    log(
        "VALIDATION PER-CYCLE RESULTS"
    )
    log(
        "-" * 72
    )

    val_prob = (
        validation[
            "probability"
        ].to_numpy(
            dtype=np.float64
        )
    )

    val_y = (
        validation[
            "binary_label"
        ].to_numpy()
    )

    for threshold in THRESHOLDS:

        pred = (
            val_prob
            >= threshold
        )

        normal = (
            val_y == 0
        )

        arc = (
            val_y == 1
        )

        vtn = int(
            np.sum(
                normal & ~pred
            )
        )

        vfp = int(
            np.sum(
                normal & pred
            )
        )

        vfn = int(
            np.sum(
                arc & ~pred
            )
        )

        vtp = int(
            np.sum(
                arc & pred
            )
        )

        ftr = (
            vfp / (vfp + vtn)
            if vfp + vtn
            else 0.0
        )

        det = (
            vtp / (vtp + vfn)
            if vtp + vfn
            else 0.0
        )

        log(
            f"thr={threshold:.2f} "
            f"TN={vtn} FP={vfp} "
            f"FN={vfn} TP={vtp} "
            f"FTR={ftr:.3%} "
            f"Det={det:.3%}"
        )

    # --------------------------------------------------------
    # Temporal rule sweep
    # --------------------------------------------------------

    log("")
    log(
        "TEMPORAL RULE RESULTS"
    )
    log(
        "-" * 72
    )

    log(
        "rule   thr    FT/hour    meanDet    "
        "worstClass  worstDet    medianDelay"
    )

    results = []

    for threshold in THRESHOLDS:

        for votes, window in RULES:

            seed_base = (
                RANDOM_SEED
                + int(
                    threshold * 100
                )
                + votes * 100
                + window
            )

            normal_rng = (
                np.random.default_rng(
                    seed_base
                    + 1
                )
            )

            events, hours = (
                simulate_normal_stream(
                    normal_block_array,
                    threshold,
                    votes,
                    window,
                    normal_rng,
                )
            )

            if events == 0:

                # Rule-of-three upper bound.
                ft_per_hour = (
                    3.0 / hours
                )

                ft_display = (
                    f"<{ft_per_hour:.3f}"
                )

            else:

                ft_per_hour = (
                    events / hours
                )

                ft_display = (
                    f"{ft_per_hour:.3f}"
                )

            class_results = []

            for k in range(1, 14):

                block_array = (
                    np.asarray(
                        arc_blocks[k],
                        dtype=np.float64,
                    )
                )

                if len(block_array) == 0:

                    class_results.append(
                        (
                            k,
                            np.nan,
                            np.nan,
                        )
                    )

                    continue

                rng = (
                    np.random.default_rng(
                        seed_base
                        + 1000 * k
                    )
                )

                detection_rate, median_delay = (
                    simulate_arc_class(
                        block_array,
                        threshold,
                        votes,
                        window,
                        rng,
                    )
                )

                class_results.append(
                    (
                        k,
                        detection_rate,
                        median_delay,
                    )
                )

            valid = [
                x
                for x in class_results
                if np.isfinite(x[1])
            ]

            detections = np.asarray(
                [
                    x[1]
                    for x in valid
                ],
                dtype=np.float64,
            )

            delays = np.asarray(
                [
                    x[2]
                    for x in valid
                    if np.isfinite(x[2])
                ],
                dtype=np.float64,
            )

            mean_detection = float(
                np.mean(
                    detections
                )
            )

            worst_index = int(
                np.argmin(
                    detections
                )
            )

            worst_class = int(
                valid[
                    worst_index
                ][0]
            )

            worst_detection = float(
                valid[
                    worst_index
                ][1]
            )

            median_delay = (
                float(
                    np.median(
                        delays
                    )
                )
                if len(delays)
                else np.nan
            )

            result = {
                "votes": votes,
                "window": window,
                "threshold": threshold,
                "ft_per_hour": ft_per_hour,
                "ft_display": ft_display,
                "mean_detection": mean_detection,
                "worst_class": worst_class,
                "worst_detection": worst_detection,
                "median_delay": median_delay,
            }

            results.append(
                result
            )

            delay_display = (
                f"{median_delay:.1f}"
                if np.isfinite(
                    median_delay
                )
                else "NA"
            )

            log(
                f"{votes}/{window:<4} "
                f"{threshold:.2f}   "
                f"{ft_display:<9} "
                f"{mean_detection:.3f}      "
                f"{worst_class:<10d} "
                f"{worst_detection:.3f}       "
                f"{delay_display}"
            )

    # --------------------------------------------------------
    # Rank candidates.
    #
    # Primary:
    #     higher mean detection.
    #
    # Secondary:
    #     lower false-trip rate.
    #
    # Tertiary:
    #     higher worst-class detection.
    #
    # Final:
    #     lower median delay.
    # --------------------------------------------------------

    def ranking_key(r):

        delay = (
            r["median_delay"]
            if np.isfinite(
                r["median_delay"]
            )
            else 1e9
        )

        return (
            -r["mean_detection"],
            r["ft_per_hour"],
            -r["worst_detection"],
            delay,
        )

    ranked = sorted(
        results,
        key=ranking_key,
    )

    # --------------------------------------------------------
    # Top candidates
    # --------------------------------------------------------

    log("")
    log(
        "TOP VALIDATION CANDIDATES"
    )
    log(
        "-" * 72
    )

    for index, r in enumerate(
        ranked[:10],
        start=1,
    ):

        delay = (
            f"{r['median_delay']:.1f}"
            if np.isfinite(
                r["median_delay"]
            )
            else "NA"
        )

        log(
            f"{index:2d}. "
            f"{r['votes']}/{r['window']} "
            f"thr={r['threshold']:.2f} "
            f"FT/hour={r['ft_display']} "
            f"meanDet={r['mean_detection']:.3f} "
            f"worst=class {r['worst_class']} "
            f"{r['worst_detection']:.3f} "
            f"median={delay}"
        )

    # --------------------------------------------------------
    # Important interpretation notes
    # --------------------------------------------------------

    log("")
    log(
        "IMPORTANT INTERPRETATION"
    )
    log(
        "-" * 72
    )

    log(
        "- The neural-network weights are fixed."
    )

    log(
        "- No official test records were used for tuning."
    )

    log(
        "- This is POST-HOC validation because the frozen model"
    )

    log(
        "  was originally trained on the full training split."
    )

    log(
        "- Therefore these results must not be described as an"
    )

    log(
        "  independent model-validation result."
    )

    log(
        "- Record adjacency is only a proxy for temporal correlation."
    )

    log(
        "- The 5-cycle block bootstrap is a simulation, not a"
    )

    log(
        "  measurement of real appliance persistence."
    )

    log(
        "- False trips/hour are simulation estimates, not field rates."
    )

    log(
        "- A zero observed false-trip count means an upper bound,"
    )

    log(
        "  not a true zero probability."
    )

    log(
        "- Hard arc classes remain visible in the class-wise results."
    )

    log(
        "- The final voting rule must be frozen before the official"
    )

    log(
        "  test-set evaluation."
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    log("")
    log(
        f"Saved: {OUT}"
    )


if __name__ == "__main__":
    main()