import os
import numpy as np


# ============================================================
# ROBUST ZERO-CROSSING TEST
#
# Software simulation only.
#
# Processing:
#   1. Generate 50 Hz-class waveform
#   2. Add noise / harmonics
#   3. Apply first-order low-pass filter
#   4. Detect rising zero crossing with hysteresis
#   5. Reject crossings that occur too close together
#   6. Estimate mains frequency
#
# Sampling rate:
#   50 kHz
#
# Test frequencies:
#   49.5, 50.0, 50.5 Hz
# ============================================================

FS = 50000.0

FREQUENCIES = [
    49.5,
    50.0,
    50.5,
]

N_CYCLES = 20

RANDOM_SEED = 42


# ------------------------------------------------------------
# Processing parameters
# ------------------------------------------------------------

# Low-pass filter cutoff.
#
# The goal is to suppress high-frequency noise while keeping
# the mains fundamental.
#
LOWPASS_CUTOFF_HZ = 100.0

# Hysteresis thresholds.
#
# The waveform amplitude is approximately +/-1.
#
# A rising crossing is accepted only after the filtered
# waveform has previously gone below -HYSTERESIS and then
# rises above +HYSTERESIS.
#
HYSTERESIS = 0.10

# Minimum allowed time between accepted rising crossings.
#
# 700 samples at 50 kHz = 14 ms.
#
# A 50 Hz waveform has a 20 ms period, so legitimate rising
# crossings are about 1000 samples apart.
#
MIN_CROSSING_INTERVAL_SAMPLES = 700


# ------------------------------------------------------------
# Noise / harmonic test cases
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Generate waveform
# ------------------------------------------------------------

def generate_waveform(
    frequency,
    noise_rms,
    h3,
    h5,
    rng,
):
    duration = N_CYCLES / frequency

    n_samples = int(
        np.ceil(duration * FS)
    )

    t = np.arange(
        n_samples,
        dtype=np.float64,
    ) / FS

    # Fundamental.
    x = np.sin(
        2.0 *
        np.pi *
        frequency *
        t
    )

    # 3rd harmonic.
    if h3 != 0.0:
        x += h3 * np.sin(
            2.0 *
            np.pi *
            3.0 *
            frequency *
            t
        )

    # 5th harmonic.
    if h5 != 0.0:
        x += h5 * np.sin(
            2.0 *
            np.pi *
            5.0 *
            frequency *
            t
        )

    # Add Gaussian noise.
    if noise_rms != 0.0:
        x += rng.normal(
            0.0,
            noise_rms,
            size=len(x),
        )

    return t, x


# ------------------------------------------------------------
# First-order low-pass filter
# ------------------------------------------------------------

def lowpass_filter(
    x,
    cutoff_hz,
    fs,
):
    dt = 1.0 / fs

    rc = 1.0 / (
        2.0 *
        np.pi *
        cutoff_hz
    )

    alpha = dt / (
        rc + dt
    )

    y = np.empty_like(x)

    y[0] = x[0]

    for i in range(
        1,
        len(x),
    ):
        y[i] = (
            y[i - 1] +
            alpha *
            (
                x[i] -
                y[i - 1]
            )
        )

    return y


# ------------------------------------------------------------
# Hysteresis rising zero-crossing detector
# ------------------------------------------------------------

def rising_crossings_hysteresis(
    x,
    hysteresis,
    min_interval_samples,
):
    crossings = []

    # State machine:
    #
    # 0 = waiting for waveform to go below -H
    # 1 = armed, waiting for waveform to rise above +H
    #
    state = 0

    last_crossing = None

    for i in range(
        1,
        len(x),
    ):

        previous = x[i - 1]
        current = x[i]

        # ----------------------------------------------------
        # Arm detector after entering negative region.
        # ----------------------------------------------------

        if state == 0:

            if current <= -hysteresis:
                state = 1

            continue

        # ----------------------------------------------------
        # We are armed.
        # Wait for transition from below +H to above +H.
        # ----------------------------------------------------

        if (
            previous < hysteresis
            and
            current >= hysteresis
        ):

            # Linear interpolation for crossing location.
            denominator = (
                current -
                previous
            )

            if abs(
                denominator
            ) < 1e-15:
                crossing = float(i)

            else:
                crossing = (
                    (i - 1) +
                    (
                        (
                            hysteresis -
                            previous
                        ) /
                        denominator
                    )
                )

            # ------------------------------------------------
            # Enforce minimum interval.
            # ------------------------------------------------

            if (
                last_crossing is None
                or
                (
                    crossing -
                    last_crossing
                ) >= min_interval_samples
            ):

                crossings.append(
                    crossing
                )

                last_crossing = crossing

                # Require waveform to go negative again
                # before accepting another crossing.
                state = 0

    return np.asarray(
        crossings,
        dtype=np.float64,
    )


# ------------------------------------------------------------
# Frequency analysis
# ------------------------------------------------------------

def analyse_waveform(
    x,
    expected_frequency,
):
    filtered = lowpass_filter(
        x,
        LOWPASS_CUTOFF_HZ,
        FS,
    )

    crossings = (
        rising_crossings_hysteresis(
            filtered,
            HYSTERESIS,
            MIN_CROSSING_INTERVAL_SAMPLES,
        )
    )

    if len(crossings) < 2:

        return {
            "crossings": len(crossings),
            "cycles": max(
                0,
                len(crossings) - 1,
            ),
            "mean_period": np.nan,
            "measured_frequency": np.nan,
            "frequency_error": np.nan,
            "period_std": np.nan,
        }

    periods = np.diff(
        crossings
    )

    mean_period = float(
        np.mean(periods)
    )

    measured_frequency = (
        FS /
        mean_period
    )

    frequency_error = (
        100.0 *
        (
            measured_frequency -
            expected_frequency
        ) /
        expected_frequency
    )

    period_std = float(
        np.std(periods)
    )

    return {
        "crossings": len(crossings),
        "cycles": len(periods),
        "mean_period": mean_period,
        "measured_frequency": measured_frequency,
        "frequency_error": frequency_error,
        "period_std": period_std,
    }


# ------------------------------------------------------------
# Main test
# ------------------------------------------------------------

def main():

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = []

    print("=" * 86)
    print(
        "ROBUST ZERO-CROSSING TEST"
    )
    print("=" * 86)

    print(
        f"Sampling rate       : {FS:.0f} Hz"
    )

    print(
        f"Low-pass cutoff     : "
        f"{LOWPASS_CUTOFF_HZ:.1f} Hz"
    )

    print(
        f"Hysteresis          : "
        f"{HYSTERESIS:.2f}"
    )

    print(
        f"Minimum crossing gap: "
        f"{MIN_CROSSING_INTERVAL_SAMPLES} samples "
        f"({MIN_CROSSING_INTERVAL_SAMPLES / FS * 1000:.1f} ms)"
    )

    print()

    for case in CASES:

        print("=" * 86)
        print(
            f"CASE: {case['name']}"
        )
        print("=" * 86)

        for frequency in FREQUENCIES:

            _, x = generate_waveform(
                frequency,
                case["noise_rms"],
                case["h3"],
                case["h5"],
                rng,
            )

            result = analyse_waveform(
                x,
                frequency,
            )

            result["case"] = case["name"]
            result["input_frequency"] = frequency

            results.append(result)

            print()
            print(
                f"{frequency:.1f} Hz"
            )

            print(
                f"  Crossings          : "
                f"{result['crossings']}"
            )

            print(
                f"  Detected cycles    : "
                f"{result['cycles']}"
            )

            print(
                f"  Mean period        : "
                f"{result['mean_period']:.4f} samples"
            )

            print(
                f"  Measured frequency : "
                f"{result['measured_frequency']:.6f} Hz"
            )

            print(
                f"  Frequency error    : "
                f"{result['frequency_error']:+.6f} %"
            )

            print(
                f"  Period std         : "
                f"{result['period_std']:.6f} samples"
            )

    # --------------------------------------------------------
    # Pass/fail
    #
    # We intentionally allow one boundary-cycle difference
    # because the generated waveform starts and ends at arbitrary
    # phases relative to the crossing detector.
    #
    # For frequency estimation we require <= 0.5% error.
    # --------------------------------------------------------

    failures = []

    for r in results:

        if r["cycles"] < 10:

            failures.append(
                (
                    r["case"],
                    r["input_frequency"],
                    "too few detected cycles",
                )
            )

            continue

        if not np.isfinite(
            r["frequency_error"]
        ):

            failures.append(
                (
                    r["case"],
                    r["input_frequency"],
                    "invalid frequency",
                )
            )

            continue

        if abs(
            r["frequency_error"]
        ) > 0.5:

            failures.append(
                (
                    r["case"],
                    r["input_frequency"],
                    "frequency error > 0.5%",
                )
            )

    print()
    print("=" * 86)

    if len(failures) == 0:

        print(
            "STATUS: ROBUST ZERO-CROSSING TEST PASSED"
        )

    else:

        print(
            "STATUS: ROBUST ZERO-CROSSING TEST FAILED"
        )

        print()
        print(
            "Failures:"
        )

        for failure in failures:

            print(
                f"  case={failure[0]}, "
                f"frequency={failure[1]:.1f} Hz, "
                f"reason={failure[2]}"
            )

    print("=" * 86)

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    output_file = (
        r"reports\zero_crossing_hysteresis_results.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "ROBUST ZERO-CROSSING TEST\n"
        )

        f.write(
            "========================================\n\n"
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
            f"Sampling rate: "
            f"{FS:.0f} Hz\n"
        )

        f.write(
            f"Low-pass cutoff: "
            f"{LOWPASS_CUTOFF_HZ:.1f} Hz\n"
        )

        f.write(
            f"Hysteresis: "
            f"{HYSTERESIS:.3f}\n"
        )

        f.write(
            f"Minimum crossing interval: "
            f"{MIN_CROSSING_INTERVAL_SAMPLES} samples\n\n"
        )

        for r in results:

            f.write(
                f"Case: {r['case']}\n"
            )

            f.write(
                f"Input frequency: "
                f"{r['input_frequency']:.3f} Hz\n"
            )

            f.write(
                f"Detected crossings: "
                f"{r['crossings']}\n"
            )

            f.write(
                f"Detected cycles: "
                f"{r['cycles']}\n"
            )

            f.write(
                f"Mean period: "
                f"{r['mean_period']:.6f} samples\n"
            )

            f.write(
                f"Measured frequency: "
                f"{r['measured_frequency']:.9f} Hz\n"
            )

            f.write(
                f"Frequency error: "
                f"{r['frequency_error']:+.9f} %\n"
            )

            f.write(
                f"Period std: "
                f"{r['period_std']:.9f} samples\n\n"
            )

        f.write(
            "Failures\n"
        )

        f.write(
            "========================================\n"
        )

        if len(failures) == 0:

            f.write(
                "None\n"
            )

        else:

            for failure in failures:

                f.write(
                    f"{failure}\n"
                )

    print()
    print(
        f"Saved: {output_file}"
    )


if __name__ == "__main__":
    main()