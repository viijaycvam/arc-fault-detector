import os
import numpy as np


# ============================================================
# TEMPORAL / PERSISTENCE LOGIC SIMULATION
#
# Inputs:
#   reports/full_test_prob.bin
#   reports/full_test_labels.bin
#
# Important:
# The official test set contains individual cycle windows.
# Their ordering must NOT be treated as a real chronological
# electrical event sequence.
#
# Therefore this script:
#   1. Verifies the official per-cycle scores.
#   2. Builds synthetic normal -> arc -> normal sequences.
#   3. Samples probabilities from the observed normal/arc
#      distributions with replacement.
#   4. Tests several temporal voting policies.
#
# This is a SOFTWARE SIMULATION, not a hardware measurement.
# ============================================================


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

N_TEST = 1334

PROB_FILE = r"reports\full_test_prob.bin"
LABEL_FILE = r"reports\full_test_labels.bin"

THRESHOLD = 0.50

# Synthetic sequence:
#
# 50 normal cycles
# 20 arc cycles
# 50 normal cycles
#
# Total = 120 cycles.
NORMAL_PRE = 50
ARC_LEN = 20
NORMAL_POST = 50

N_SEQUENCES = 2000

RANDOM_SEED = 42


# Policies to compare:
#
# consecutive K:
#   K consecutive positive cycle predictions
#
# N-of-M:
#   At least N positives inside the last M cycles
#
POLICIES = [
    ("2-consecutive", "consecutive", 2, 0),
    ("3-consecutive", "consecutive", 3, 0),
    ("2-of-3", "n_of_m", 2, 3),
    ("3-of-5", "n_of_m", 3, 5),
    ("4-of-5", "n_of_m", 4, 5),
]


# ------------------------------------------------------------
# Binary loader helpers
# ------------------------------------------------------------

def load_probabilities(path: str, expected_n: int) -> np.ndarray:
    size = os.path.getsize(path)

    if size == expected_n * 4:
        x = np.fromfile(path, dtype=np.float32)
    elif size == expected_n * 8:
        x = np.fromfile(path, dtype=np.float64)
    else:
        raise RuntimeError(
            f"Unexpected probability file size: {size} bytes. "
            f"Expected {expected_n * 4} or {expected_n * 8}."
        )

    if len(x) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} probabilities, got {len(x)}."
        )

    return x.astype(np.float64)


def load_labels(path: str, expected_n: int) -> np.ndarray:
    size = os.path.getsize(path)

    if size == expected_n:
        x = np.fromfile(path, dtype=np.uint8)
    elif size == expected_n * 2:
        x = np.fromfile(path, dtype=np.int16)
    elif size == expected_n * 4:
        x = np.fromfile(path, dtype=np.int32)
    else:
        raise RuntimeError(
            f"Unexpected label file size: {size} bytes."
        )

    if len(x) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} labels, got {len(x)}."
        )

    return x


# ------------------------------------------------------------
# Check official per-cycle performance
# ------------------------------------------------------------

def official_metrics(prob: np.ndarray,
                      labels: np.ndarray,
                      threshold: float):

    pred = (prob >= threshold).astype(np.uint8)

    normal = labels == 0
    arc = labels == 1

    if np.sum(normal) + np.sum(arc) != len(labels):
        raise RuntimeError(
            "Expected binary labels 0=normal, 1=arc."
        )

    tn = int(np.sum(normal & (pred == 0)))
    fp = int(np.sum(normal & (pred == 1)))
    fn = int(np.sum(arc & (pred == 0)))
    tp = int(np.sum(arc & (pred == 1)))

    accuracy = (
        (tn + tp) /
        (tn + fp + fn + tp)
    )

    fpr = (
        fp /
        (fp + tn)
    )

    tpr = (
        tp /
        (tp + fn)
    )

    return tn, fp, fn, tp, accuracy, fpr, tpr


# ------------------------------------------------------------
# Apply one temporal policy
# ------------------------------------------------------------

def policy_trip(
    positive_flags: np.ndarray,
    policy_type: str,
    n: int,
    m: int
):
    """
    Returns the index of the first trip,
    or None if the policy never trips.
    """

    if policy_type == "consecutive":

        count = 0

        for i, flag in enumerate(positive_flags):

            if flag:
                count += 1
            else:
                count = 0

            if count >= n:
                return i

        return None

    elif policy_type == "n_of_m":

        if m <= 0:
            raise ValueError("M must be > 0.")

        for i in range(len(positive_flags)):

            start = max(0, i - m + 1)

            window = positive_flags[start:i + 1]

            if len(window) >= m:
                if int(np.sum(window)) >= n:
                    return i

        return None

    else:
        raise ValueError(
            f"Unknown policy type: {policy_type}"
        )


# ------------------------------------------------------------
# Simulate one normal -> arc -> normal sequence
# ------------------------------------------------------------

def simulate_policy(
    normal_probs: np.ndarray,
    arc_probs: np.ndarray,
    rng: np.random.Generator,
    policy_type: str,
    n: int,
    m: int,
    threshold: float
):
    # --------------------------------------------
    # Bootstrap probabilities from observed data
    # --------------------------------------------

    normal_part_1 = rng.choice(
        normal_probs,
        size=NORMAL_PRE,
        replace=True
    )

    arc_part = rng.choice(
        arc_probs,
        size=ARC_LEN,
        replace=True
    )

    normal_part_2 = rng.choice(
        normal_probs,
        size=NORMAL_POST,
        replace=True
    )

    probs = np.concatenate(
        [
            normal_part_1,
            arc_part,
            normal_part_2
        ]
    )

    # Cycle-level ML decisions.
    positive = probs >= threshold

    # --------------------------------------------
    # First temporal trip
    # --------------------------------------------

    trip_idx = policy_trip(
        positive,
        policy_type,
        n,
        m
    )

    arc_start = NORMAL_PRE
    arc_end = NORMAL_PRE + ARC_LEN

    false_trip = False
    detected_during_arc = False
    detection_delay = None

    if trip_idx is not None:

        # Trip before arc begins.
        if trip_idx < arc_start:
            false_trip = True

        # Trip during the synthetic arc interval.
        elif trip_idx < arc_end:
            detected_during_arc = True

            detection_delay = (
                trip_idx - arc_start + 1
            )

    return (
        false_trip,
        detected_during_arc,
        detection_delay
    )


# ------------------------------------------------------------
# Run all simulations
# ------------------------------------------------------------

def run_simulation(
    name: str,
    policy_type: str,
    n: int,
    m: int,
    normal_probs: np.ndarray,
    arc_probs: np.ndarray,
    rng: np.random.Generator,
    threshold: float
):

    false_trip_count = 0
    detection_count = 0

    delays = []

    for _ in range(N_SEQUENCES):

        (
            false_trip,
            detected,
            delay
        ) = simulate_policy(
            normal_probs,
            arc_probs,
            rng,
            policy_type,
            n,
            m,
            threshold
        )

        if false_trip:
            false_trip_count += 1

        if detected:
            detection_count += 1

            if delay is not None:
                delays.append(delay)

    false_trip_rate = (
        false_trip_count /
        N_SEQUENCES
    )

    detection_rate = (
        detection_count /
        N_SEQUENCES
    )

    if len(delays) > 0:

        delays_np = np.asarray(
            delays,
            dtype=np.float64
        )

        mean_delay = float(
            np.mean(delays_np)
        )

        median_delay = float(
            np.median(delays_np)
        )

        p95_delay = float(
            np.percentile(
                delays_np,
                95
            )
        )

        max_delay = float(
            np.max(delays_np)
        )

    else:

        mean_delay = float("nan")
        median_delay = float("nan")
        p95_delay = float("nan")
        max_delay = float("nan")

    return {
        "name": name,
        "false_trip_rate": false_trip_rate,
        "detection_rate": detection_rate,
        "mean_delay": mean_delay,
        "median_delay": median_delay,
        "p95_delay": p95_delay,
        "max_delay": max_delay,
    }


# ------------------------------------------------------------
# Save text report
# ------------------------------------------------------------

def write_report(
    path: str,
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    accuracy: float,
    fpr: float,
    tpr: float,
    normal_probs: np.ndarray,
    arc_probs: np.ndarray,
    results
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "TEMPORAL VOTING / PERSISTENCE SIMULATION\n"
        )

        f.write(
            "============================================\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "Synthetic software simulation only.\n"
        )

        f.write(
            "The official test windows are not treated as a\n"
        )

        f.write(
            "chronological electrical event sequence.\n\n"
        )

        f.write(
            "Official per-cycle reference performance\n"
        )

        f.write(
            "--------------------------------------------\n"
        )

        f.write(
            f"TN = {tn}\n"
        )

        f.write(
            f"FP = {fp}\n"
        )

        f.write(
            f"FN = {fn}\n"
        )

        f.write(
            f"TP = {tp}\n"
        )

        f.write(
            f"Accuracy        = {accuracy * 100:.4f}%\n"
        )

        f.write(
            f"False-trip rate = {fpr * 100:.4f}%\n"
        )

        f.write(
            f"Arc detection   = {tpr * 100:.4f}%\n\n"
        )

        f.write(
            "Synthetic sequence\n"
        )

        f.write(
            "--------------------------------------------\n"
        )

        f.write(
            f"Normal pre      = {NORMAL_PRE} cycles\n"
        )

        f.write(
            f"Arc             = {ARC_LEN} cycles\n"
        )

        f.write(
            f"Normal post     = {NORMAL_POST} cycles\n"
        )

        f.write(
            f"Sequences       = {N_SEQUENCES}\n"
        )

        f.write(
            f"Threshold       = {THRESHOLD:.2f}\n"
        )

        f.write(
            f"Random seed     = {RANDOM_SEED}\n\n"
        )

        f.write(
            "Observed score distributions\n"
        )

        f.write(
            "--------------------------------------------\n"
        )

        f.write(
            f"Normal samples  = {len(normal_probs)}\n"
        )

        f.write(
            f"Arc samples     = {len(arc_probs)}\n"
        )

        f.write(
            f"Normal mean p   = {np.mean(normal_probs):.6f}\n"
        )

        f.write(
            f"Arc mean p      = {np.mean(arc_probs):.6f}\n\n"
        )

        f.write(
            "Temporal policy results\n"
        )

        f.write(
            "--------------------------------------------\n"
        )

        f.write(
            "Policy         "
            "FalseTrip%  "
            "Detect%  "
            "MeanDelay  "
            "Median  "
            "P95  "
            "Max\n"
        )

        for r in results:

            f.write(
                f"{r['name']:<14}"
                f"{r['false_trip_rate'] * 100:>9.3f}  "
                f"{r['detection_rate'] * 100:>7.3f}  "
                f"{r['mean_delay']:>9.3f}  "
                f"{r['median_delay']:>6.1f}  "
                f"{r['p95_delay']:>4.1f}  "
                f"{r['max_delay']:>4.1f}\n"
            )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("TEMPORAL / PERSISTENCE LOGIC SIMULATION")
    print("=" * 70)

    # --------------------------------------------
    # Load official files
    # --------------------------------------------

    if not os.path.exists(PROB_FILE):
        raise FileNotFoundError(
            f"Missing: {PROB_FILE}"
        )

    if not os.path.exists(LABEL_FILE):
        raise FileNotFoundError(
            f"Missing: {LABEL_FILE}"
        )

    prob = load_probabilities(
        PROB_FILE,
        N_TEST
    )

    labels = load_labels(
        LABEL_FILE,
        N_TEST
    )

    print(
        f"Loaded probabilities : {len(prob)}"
    )

    print(
        f"Loaded labels        : {len(labels)}"
    )

    print(
        f"Unique labels        : {np.unique(labels)}"
    )

    # --------------------------------------------
    # Validate labels
    # --------------------------------------------

    unique_labels = np.unique(labels)

    if not np.array_equal(
        unique_labels,
        np.array([0, 1], dtype=unique_labels.dtype)
    ):
        raise RuntimeError(
            "Expected binary labels exactly {0, 1}."
        )

    normal_probs = prob[labels == 0]
    arc_probs = prob[labels == 1]

    print(
        f"Normal probabilities : {len(normal_probs)}"
    )

    print(
        f"Arc probabilities    : {len(arc_probs)}"
    )

    # --------------------------------------------
    # Verify official per-cycle performance
    # --------------------------------------------

    (
        tn,
        fp,
        fn,
        tp,
        accuracy,
        fpr,
        tpr
    ) = official_metrics(
        prob,
        labels,
        THRESHOLD
    )

    print()
    print("OFFICIAL PER-CYCLE CHECK")
    print("-" * 70)

    print(
        f"TN = {tn}"
    )

    print(
        f"FP = {fp}"
    )

    print(
        f"FN = {fn}"
    )

    print(
        f"TP = {tp}"
    )

    print(
        f"Accuracy        : {accuracy * 100:.4f}%"
    )

    print(
        f"False-trip rate : {fpr * 100:.4f}%"
    )

    print(
        f"Arc detection   : {tpr * 100:.4f}%"
    )

    # --------------------------------------------
    # Synthetic simulation
    # --------------------------------------------

    print()
    print("SYNTHETIC TEMPORAL SIMULATION")
    print("-" * 70)

    print(
        f"Sequence: "
        f"{NORMAL_PRE} normal + "
        f"{ARC_LEN} arc + "
        f"{NORMAL_POST} normal"
    )

    print(
        f"Sequences: {N_SEQUENCES}"
    )

    print(
        f"Threshold: {THRESHOLD:.2f}"
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = []

    for (
        name,
        policy_type,
        n,
        m
    ) in POLICIES:

        # Use a fresh deterministic RNG stream for
        # reproducibility across policies.
        policy_rng = np.random.default_rng(
            RANDOM_SEED
        )

        result = run_simulation(
            name,
            policy_type,
            n,
            m,
            normal_probs,
            arc_probs,
            policy_rng,
            THRESHOLD
        )

        results.append(result)

    # --------------------------------------------
    # Print table
    # --------------------------------------------

    print()
    print(
        f"{'Policy':<16}"
        f"{'FalseTrip%':>12}"
        f"{'Detect%':>12}"
        f"{'MeanDelay':>12}"
        f"{'Median':>10}"
        f"{'P95':>8}"
        f"{'Max':>8}"
    )

    print("-" * 78)

    for r in results:

        print(
            f"{r['name']:<16}"
            f"{r['false_trip_rate'] * 100:>11.3f}"
            f"{r['detection_rate'] * 100:>11.3f}"
            f"{r['mean_delay']:>11.3f}"
            f"{r['median_delay']:>9.1f}"
            f"{r['p95_delay']:>7.1f}"
            f"{r['max_delay']:>7.1f}"
        )

    # --------------------------------------------
    # Save report
    # --------------------------------------------

    output_path = (
        r"reports\temporal_voting_results.txt"
    )

    write_report(
        output_path,
        tn,
        fp,
        fn,
        tp,
        accuracy,
        fpr,
        tpr,
        normal_probs,
        arc_probs,
        results
    )

    print()
    print(
        f"Saved: {output_path}"
    )

    print()
    print(
        "STATUS: TEMPORAL SIMULATION COMPLETE"
    )


if __name__ == "__main__":
    main()