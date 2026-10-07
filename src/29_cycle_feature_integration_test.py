import importlib.util
import os
import numpy as np


# ============================================================
# CYCLE SLICING + 8-FEATURE INTEGRATION TEST
#
# Pipeline:
#
#   50 kHz waveform
#        ↓
#   low-pass filtered copy
#        ↓
#   hysteresis zero-crossing
#        ↓
#   cycle boundaries
#        ↓
#   raw waveform cycle
#        ↓
#   resample to exactly 1000 samples
#        ↓
#   canonical features_b()
#
# SOFTWARE SIMULATION ONLY.
# ============================================================


FS = 50000.0

TARGET_CYCLE_LEN = 1000

N_CYCLES = 20

RANDOM_SEED = 42


FREQUENCIES = [
    49.5,
    50.0,
    50.5,
]


CASES = [
    {
        "name": "clean",
        "noise_rms": 0.00,
        "h3": 0.00,
        "h5": 0.00,
    },
    {
        "name": "5pct_noise",
        "noise_rms": 0.05,
        "h3": 0.00,
        "h5": 0.00,
    },
    {
        "name": "20pct_noise",
        "noise_rms": 0.20,
        "h3": 0.00,
        "h5": 0.00,
    },
    {
        "name": "10pct_h3_5pct_noise",
        "noise_rms": 0.05,
        "h3": 0.10,
        "h5": 0.00,
    },
    {
        "name": "20pct_h3_10pct_h5_5pct_noise",
        "noise_rms": 0.05,
        "h3": 0.20,
        "h5": 0.10,
    },
]


FEATURE_NAMES = [
    "crest",
    "skew",
    "kurt",
    "d1_ratio",
    "d2_ratio",
    "peak_asym",
    "energy_asym",
    "frac_small",
]


# ------------------------------------------------------------
# Load canonical Python modules without executing main().
# ------------------------------------------------------------

def load_module(
    module_name,
    file_path,
):
    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {file_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


# ------------------------------------------------------------
# Generate test waveform.
# ------------------------------------------------------------

def generate_waveform(
    frequency,
    noise_rms,
    h3,
    h5,
    rng,
):
    duration = (
        N_CYCLES /
        frequency
    )

    n_samples = int(
        np.ceil(
            duration *
            FS
        )
    )

    t = (
        np.arange(
            n_samples,
            dtype=np.float64,
        ) /
        FS
    )

    x = np.sin(
        2.0 *
        np.pi *
        frequency *
        t
    )

    if h3 != 0.0:
        x += (
            h3 *
            np.sin(
                2.0 *
                np.pi *
                3.0 *
                frequency *
                t
            )
        )

    if h5 != 0.0:
        x += (
            h5 *
            np.sin(
                2.0 *
                np.pi *
                5.0 *
                frequency *
                t
            )
        )

    if noise_rms != 0.0:
        x += rng.normal(
            0.0,
            noise_rms,
            size=len(x),
        )

    return x


# ------------------------------------------------------------
# Extract all raw cycles using robust crossings.
# ------------------------------------------------------------

def slice_cycles(
    raw_waveform,
    crossing_module,
    resample_module,
):
    filtered = (
        crossing_module.lowpass_filter(
            raw_waveform,
            crossing_module.LOWPASS_CUTOFF_HZ,
            FS,
        )
    )

    crossings = (
        crossing_module.rising_crossings_hysteresis(
            filtered,
            crossing_module.HYSTERESIS,
            crossing_module.MIN_CROSSING_INTERVAL_SAMPLES,
        )
    )

    cycles = []

    for i in range(
        len(crossings) - 1
    ):
        cycle = (
            resample_module.resample_cycle(
                raw_waveform,
                crossings[i],
                crossings[i + 1],
                TARGET_CYCLE_LEN,
            )
        )

        cycles.append(cycle)

    if not cycles:
        return (
            crossings,
            np.empty(
                (
                    0,
                    TARGET_CYCLE_LEN,
                ),
                dtype=np.float64,
            ),
        )

    return (
        crossings,
        np.asarray(
            cycles,
            dtype=np.float64,
        ),
    )


# ------------------------------------------------------------
# Main.
# ------------------------------------------------------------

def main():

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    base = os.path.dirname(
        os.path.abspath(__file__)
    )

    zero_crossing_module = load_module(
        "zero_crossing_hysteresis",
        os.path.join(
            base,
            "28_zero_crossing_hysteresis_test.py",
        ),
    )

    resample_module = load_module(
        "zero_crossing_reference",
        os.path.join(
            base,
            "26_zero_crossing_test.py",
        ),
    )

    features_module = load_module(
        "features_b_reference",
        os.path.join(
            base,
            "20_features_b.py",
        ),
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = []

    print("=" * 88)
    print(
        "CYCLE SLICING + 8-FEATURE INTEGRATION TEST"
    )
    print("=" * 88)

    print(
        f"Sampling rate : {FS:.0f} Hz"
    )

    print(
        f"Target length : {TARGET_CYCLE_LEN}"
    )

    print(
        "Feature order :"
    )

    print(
        "  " +
        ", ".join(FEATURE_NAMES)
    )

    print()

    for case in CASES:

        print("=" * 88)
        print(
            f"CASE: {case['name']}"
        )
        print("=" * 88)

        for frequency in FREQUENCIES:

            raw = generate_waveform(
                frequency,
                case["noise_rms"],
                case["h3"],
                case["h5"],
                rng,
            )

            crossings, cycles = slice_cycles(
                raw,
                zero_crossing_module,
                resample_module,
            )

            if len(cycles) == 0:
                raise RuntimeError(
                    f"No cycles detected for "
                    f"{case['name']} / {frequency} Hz"
                )

            # ----------------------------------------------
            # Validate cycle length.
            # ----------------------------------------------

            lengths = np.asarray(
                [
                    len(c)
                    for c in cycles
                ],
                dtype=int,
            )

            all_lengths_ok = bool(
                np.all(
                    lengths ==
                    TARGET_CYCLE_LEN
                )
            )

            # ----------------------------------------------
            # Canonical feature extractor.
            #
            # This is src/20_features_b.py.
            # Do not duplicate the feature equations here.
            # ----------------------------------------------

            feature_matrix = (
                features_module.features_b(
                    cycles
                )
            )

            # ----------------------------------------------
            # Numeric validity.
            # ----------------------------------------------

            finite_ok = bool(
                np.all(
                    np.isfinite(
                        feature_matrix
                    )
                )
            )

            # ----------------------------------------------
            # Print results.
            # ----------------------------------------------

            print()
            print(
                f"{frequency:.1f} Hz"
            )

            print(
                f"  Crossings       : "
                f"{len(crossings)}"
            )

            print(
                f"  Cycles           : "
                f"{len(cycles)}"
            )

            print(
                f"  Cycle length     : "
                f"{lengths.min()}.."
                f"{lengths.max()}"
            )

            print(
                f"  Features shape   : "
                f"{feature_matrix.shape}"
            )

            print(
                f"  All finite       : "
                f"{finite_ok}"
            )

            # Print mean feature vector.
            mean_features = (
                np.mean(
                    feature_matrix,
                    axis=0,
                )
            )

            print(
                "  Mean features:"
            )

            for name, value in zip(
                FEATURE_NAMES,
                mean_features,
            ):

                print(
                    f"    {name:<14} "
                    f"{value: .8f}"
                )

            results.append(
                {
                    "case": case["name"],
                    "frequency": frequency,
                    "crossings": len(crossings),
                    "cycles": len(cycles),
                    "min_length": int(
                        lengths.min()
                    ),
                    "max_length": int(
                        lengths.max()
                    ),
                    "finite": finite_ok,
                    "features": feature_matrix,
                }
            )

    # --------------------------------------------------------
    # Overall checks.
    # --------------------------------------------------------

    length_failures = []

    finite_failures = []

    feature_spans = []

    for result in results:

        if (
            result["min_length"] !=
            TARGET_CYCLE_LEN
            or
            result["max_length"] !=
            TARGET_CYCLE_LEN
        ):

            length_failures.append(
                (
                    result["case"],
                    result["frequency"],
                )
            )

        if not result["finite"]:

            finite_failures.append(
                (
                    result["case"],
                    result["frequency"],
                )
            )

        feature_min = np.min(
            result["features"],
            axis=0,
        )

        feature_max = np.max(
            result["features"],
            axis=0,
        )

        feature_spans.append(
            feature_max -
            feature_min
        )

    all_lengths_ok = (
        len(length_failures) == 0
    )

    all_finite_ok = (
        len(finite_failures) == 0
    )

    # --------------------------------------------------------
    # The primary purpose of this test is pipeline integrity:
    #
    #   robust slicing
    #       +
    #   exact 1000-point resampling
    #       +
    #   canonical 8-feature extraction
    #
    # We do NOT impose arbitrary feature thresholds here.
    # --------------------------------------------------------

    passed = (
        all_lengths_ok
        and
        all_finite_ok
    )

    print()
    print("=" * 88)

    if passed:

        print(
            "STATUS: CYCLE + FEATURE INTEGRATION TEST PASSED"
        )

    else:

        print(
            "STATUS: CYCLE + FEATURE INTEGRATION TEST FAILED"
        )

    print("=" * 88)

    if length_failures:

        print()
        print(
            "Cycle-length failures:"
        )

        for failure in length_failures:
            print(
                f"  {failure[0]} / "
                f"{failure[1]:.1f} Hz"
            )

    if finite_failures:

        print()
        print(
            "Non-finite feature failures:"
        )

        for failure in finite_failures:
            print(
                f"  {failure[0]} / "
                f"{failure[1]:.1f} Hz"
            )

    # --------------------------------------------------------
    # Save report.
    # --------------------------------------------------------

    output_file = (
        r"reports\cycle_feature_integration_results.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "CYCLE SLICING + 8-FEATURE INTEGRATION TEST\n"
        )

        f.write(
            "============================================\n\n"
        )

        f.write(
            "Software simulation only.\n"
        )

        f.write(
            "No hardware measurement.\n"
        )

        f.write(
            "No official dataset samples used.\n\n"
        )

        f.write(
            f"Sampling rate: {FS:.0f} Hz\n"
        )

        f.write(
            f"Target cycle length: "
            f"{TARGET_CYCLE_LEN}\n\n"
        )

        f.write(
            "Feature order:\n"
        )

        for i, name in enumerate(
            FEATURE_NAMES
        ):

            f.write(
                f"  {i}: {name}\n"
            )

        f.write("\n")

        for result in results:

            f.write(
                f"Case: {result['case']}\n"
            )

            f.write(
                f"Frequency: "
                f"{result['frequency']:.3f} Hz\n"
            )

            f.write(
                f"Crossings: "
                f"{result['crossings']}\n"
            )

            f.write(
                f"Cycles: "
                f"{result['cycles']}\n"
            )

            f.write(
                f"Cycle length: "
                f"{result['min_length']}.."
                f"{result['max_length']}\n"
            )

            f.write(
                f"Features finite: "
                f"{result['finite']}\n"
            )

            mean_features = (
                np.mean(
                    result["features"],
                    axis=0,
                )
            )

            f.write(
                "Mean features:\n"
            )

            for name, value in zip(
                FEATURE_NAMES,
                mean_features,
            ):

                f.write(
                    f"  {name}: "
                    f"{value:.10f}\n"
                )

            f.write("\n")

        f.write(
            "Overall status: "
            + (
                "PASSED\n"
                if passed
                else
                "FAILED\n"
            )
        )

    print()
    print(
        f"Saved: {output_file}"
    )


if __name__ == "__main__":
    main()