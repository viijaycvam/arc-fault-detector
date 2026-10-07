import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# TEMPORAL VOTING / PERSISTENCE SIMULATION
#
# Uses the official 1334-cycle test-set probabilities exported
# by src/24_export_full_test.py.
#
# IMPORTANT:
# These sequences are SYNTHETIC.
# The official test set is not a time-ordered recording of
# appliance transitions, so this experiment must not be
# reported as real temporal/field performance.
# ============================================================

N_TEST = 1334
THRESHOLD = 0.50

TRIALS = 2000

# Synthetic transition sequence:
# normal -> arc -> normal
PRE_NORMAL = 30
ARC_LENGTH = 50
POST_NORMAL = 30

# Reproducible random generator.
RNG_SEED = 20261007

REPORT_PATH = Path("reports/voting_simulation.txt")
PLOT_PATH = Path("reports/voting_detection_delay.png")

PROB_PATH = Path("reports/full_test_prob.bin")
LABEL_PATH = Path("reports/full_test_labels.bin")


# ------------------------------------------------------------
# Load probabilities
# ------------------------------------------------------------

probs = np.fromfile(
    PROB_PATH,
    dtype=np.float32
)

if len(probs) != N_TEST:
    raise RuntimeError(
        f"Expected {N_TEST} probabilities, "
        f"got {len(probs)}"
    )


# ------------------------------------------------------------
# Load labels robustly.
#
# Accept common binary formats:
#   uint8
#   int32
#   int64
# ------------------------------------------------------------

def load_binary_labels(path, expected_n):
    candidates = [
        np.uint8,
        np.int32,
        np.int64,
    ]

    for dtype in candidates:
        values = np.fromfile(
            path,
            dtype=dtype
        )

        if len(values) != expected_n:
            continue

        unique = set(
            np.unique(values).tolist()
        )

        if unique.issubset({0, 1}):
            return values.astype(np.int8)

    raise RuntimeError(
        "Could not determine label format."
    )


labels = load_binary_labels(
    LABEL_PATH,
    N_TEST
)


# ------------------------------------------------------------
# Confirm official test-set counts
# ------------------------------------------------------------

normal_probs = probs[labels == 0]
arc_probs = probs[labels == 1]

normal_count = len(normal_probs)
arc_count = len(arc_probs)

if normal_count != 861:
    raise RuntimeError(
        f"Expected 861 normal cycles, "
        f"got {normal_count}"
    )

if arc_count != 473:
    raise RuntimeError(
        f"Expected 473 arc cycles, "
        f"got {arc_count}"
    )


# ------------------------------------------------------------
# Official single-cycle baseline
# ------------------------------------------------------------

official_pred = (
    probs >= THRESHOLD
)

official_normal_fp = np.sum(
    official_pred[labels == 0]
)

official_arc_tp = np.sum(
    official_pred[labels == 1]
)

official_fpr = (
    official_normal_fp /
    normal_count
)

official_tpr = (
    official_arc_tp /
    arc_count
)


# ------------------------------------------------------------
# Temporal voting function
#
# N-of-M rule:
#   trip when at least N of the latest M cycle
#   probabilities are >= threshold.
#
# Returns first trip index, or None.
# ------------------------------------------------------------

def first_trip_index(
    sequence,
    required_votes,
    window_size
):
    if window_size <= 0:
        raise ValueError(
            "window_size must be positive"
        )

    if required_votes <= 0:
        raise ValueError(
            "required_votes must be positive"
        )

    if required_votes > window_size:
        raise ValueError(
            "required_votes cannot exceed window_size"
        )

    if len(sequence) < window_size:
        return None

    decisions = (
        sequence >= THRESHOLD
    )

    for i in range(
        window_size - 1,
        len(sequence)
    ):
        window =
            decisions[
                i - window_size + 1 :
                i + 1
            ]

        if np.sum(window) >= required_votes:
            return i

    return None


# ------------------------------------------------------------
# Configurations to compare
# ------------------------------------------------------------

CONFIGS = [
    (1, 1),
    (2, 3),
    (3, 5),
    (4, 5),
]


# ------------------------------------------------------------
# Synthetic normal-only false-trip test
# ------------------------------------------------------------

def run_normal_only_trials(
    rng,
    required_votes,
    window_size
):
    false_trip_sequences = 0

    first_trip_cycles = []

    for _ in range(TRIALS):

        sequence = rng.choice(
            normal_probs,
            size=100,
            replace=True
        )

        trip_idx = first_trip_index(
            sequence,
            required_votes,
            window_size
        )

        if trip_idx is not None:
            false_trip_sequences += 1
            first_trip_cycles.append(
                trip_idx + 1
            )

    false_trip_rate = (
        false_trip_sequences /
        TRIALS
    )

    return (
        false_trip_rate,
        first_trip_cycles
    )


# ------------------------------------------------------------
# Synthetic normal -> arc -> normal trials
# ------------------------------------------------------------

def run_transition_trials(
    rng,
    required_votes,
    window_size
):
    detected = 0
    early_false_trip = 0
    late_trip = 0
    missed = 0

    delays = []

    for _ in range(TRIALS):

        pre = rng.choice(
            normal_probs,
            size=PRE_NORMAL,
            replace=True
        )

        arc = rng.choice(
            arc_probs,
            size=ARC_LENGTH,
            replace=True
        )

        post = rng.choice(
            normal_probs,
            size=POST_NORMAL,
            replace=True
        )

        sequence = np.concatenate(
            [pre, arc, post]
        )

        trip_idx = first_trip_index(
            sequence,
            required_votes,
            window_size
        )

        if trip_idx is None:
            missed += 1
            continue

        if trip_idx < PRE_NORMAL:
            early_false_trip += 1
            continue

        if trip_idx < PRE_NORMAL + ARC_LENGTH:

            detected += 1

            # Delay measured in cycles from arc onset.
            #
            # First arc cycle = delay 1.
            delay_cycles = (
                trip_idx -
                PRE_NORMAL +
                1
            )

            delays.append(
                delay_cycles
            )

            continue

        # Trip occurred after the synthetic arc interval.
        late_trip += 1

    detection_rate = (
        detected /
        TRIALS
    )

    early_false_trip_rate = (
        early_false_trip /
        TRIALS
    )

    late_trip_rate = (
        late_trip /
        TRIALS
    )

    if delays:
        mean_delay = float(
            np.mean(delays)
        )

        median_delay = float(
            np.median(delays)
        )

        p95_delay = float(
            np.percentile(
                delays,
                95
            )
        )

        max_delay = int(
            np.max(delays)
        )
    else:
        mean_delay = float("nan")
        median_delay = float("nan")
        p95_delay = float("nan")
        max_delay = 0

    return {
        "detection_rate":
            detection_rate,
        "early_false_trip_rate":
            early_false_trip_rate,
        "late_trip_rate":
            late_trip_rate,
        "miss_rate":
            missed / TRIALS,
        "delays":
            delays,
        "mean_delay":
            mean_delay,
        "median_delay":
            median_delay,
        "p95_delay":
            p95_delay,
        "max_delay":
            max_delay,
    }


# ------------------------------------------------------------
# Run all configurations
# ------------------------------------------------------------

results = []

rng = np.random.default_rng(
    RNG_SEED
)

for required_votes, window_size in CONFIGS:

    normal_false_rate, _ = (
        run_normal_only_trials(
            rng,
            required_votes,
            window_size
        )
    )

    transition = (
        run_transition_trials(
            rng,
            required_votes,
            window_size
        )
    )

    results.append({
        "N": required_votes,
        "M": window_size,
        "normal_false_trip_rate":
            normal_false_rate,
        **transition,
    })


# ------------------------------------------------------------
# Print report
# ------------------------------------------------------------

lines = []

lines.append(
    "=" * 72
)

lines.append(
    "TEMPORAL VOTING / PERSISTENCE SIMULATION"
)

lines.append(
    "=" * 72
)

lines.append(
    "IMPORTANT: Synthetic simulation only."
)

lines.append(
    "The official test set is not time ordered."
)

lines.append("")

lines.append(
    "OFFICIAL SINGLE-CYCLE BASELINE"
)

lines.append(
    f"Test cycles       : {N_TEST}"
)

lines.append(
    f"Normal cycles     : {normal_count}"
)

lines.append(
    f"Arc cycles        : {arc_count}"
)

lines.append(
    f"Threshold         : {THRESHOLD:.2f}"
)

lines.append(
    f"False-trip rate   : "
    f"{official_fpr * 100:.4f}%"
)

lines.append(
    f"Arc detection     : "
    f"{official_tpr * 100:.4f}%"
)

lines.append("")

lines.append(
    "SYNTHETIC SEQUENCE"
)

lines.append(
    f"Normal pre-roll   : {PRE_NORMAL} cycles"
)

lines.append(
    f"Arc interval      : {ARC_LENGTH} cycles"
)

lines.append(
    f"Normal post-roll  : {POST_NORMAL} cycles"
)

lines.append(
    f"Trials/config     : {TRIALS}"
)

lines.append("")

lines.append(
    "TEMPORAL RESULTS"
)

lines.append(
    "-" * 72
)

header = (
    "Rule     Normal FP    Arc detect    "
    "Mean delay    Median    P95    Miss"
)

lines.append(header)

lines.append(
    "-" * 72
)

for r in results:

    rule = (
        f"{r['N']}-of-{r['M']}"
    )

    lines.append(
        f"{rule:<8}"
        f"{r['normal_false_trip_rate'] * 100:>9.3f}%"
        f"{r['detection_rate'] * 100:>13.3f}%"
        f"{r['mean_delay']:>13.2f}"
        f"{r['median_delay']:>10.2f}"
        f"{r['p95_delay']:>8.2f}"
        f"{r['miss_rate'] * 100:>8.3f}%"
    )

lines.append(
    "-" * 72
)

lines.append("")

lines.append(
    "Interpretation:"
)

lines.append(
    "The temporal results are synthetic simulations "
    "using probability samples from the official test set."
)

lines.append(
    "They do not represent real appliance transitions "
    "or field detection latency."
)

lines.append(
    "Use the official single-cycle metrics as the "
    "primary model-performance numbers."
)

report_text = "\n".join(lines)

print(report_text)

REPORT_PATH.write_text(
    report_text,
    encoding="utf-8"
)


# ------------------------------------------------------------
# Detection-delay plot
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 5)
)

labels_plot = []
means = []
p95s = []

for r in results:

    labels_plot.append(
        f"{r['N']}-of-{r['M']}"
    )

    means.append(
        r["mean_delay"]
    )

    p95s.append(
        r["p95_delay"]
    )

x = np.arange(
    len(labels_plot)
)

width = 0.35

plt.bar(
    x - width / 2,
    means,
    width,
    label="Mean detection delay"
)

plt.bar(
    x + width / 2,
    p95s,
    width,
    label="95th percentile delay"
)

plt.xticks(
    x,
    labels_plot
)

plt.xlabel(
    "Temporal voting rule"
)

plt.ylabel(
    "Detection delay (cycles)"
)

plt.title(
    "Synthetic Temporal Voting Detection Delay"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=150
)

plt.show()

print()
print(
    f"Saved report: {REPORT_PATH}"
)

print(
    f"Saved plot  : {PLOT_PATH}"
)