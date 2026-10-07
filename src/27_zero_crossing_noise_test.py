import os
import numpy as np


# ============================================================
# ZERO-CROSSING ROBUSTNESS TEST
#
# SOFTWARE SIMULATION ONLY
#
# Sampling rate:
#   50 kHz
#
# Frequencies:
#   49.5, 50.0, 50.5 Hz
#
# Distortion/noise cases:
#   1. Clean sine
#   2. 5% Gaussian noise
#   3. 20% Gaussian noise
#   4. 10% 3rd harmonic + 5% noise
#   5. 20% 3rd + 10% 5th harmonic + 5% noise
#
# Goal:
#   Detect rising zero crossings and verify that the measured
#   mains frequency remains accurate.
#
# IMPORTANT:
#   This does NOT use the official arc-fault dataset.
#   It does NOT represent field performance.
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
# Test cases
#
# noise_rms and harmonic amplitudes are relative to the
# fundamental amplitude.
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
# Generate distorted/noisy waveform
# ------------------------------------------------------------

def generate_waveform(
    frequency: float,
    noise_rms: float,
    h3: float,
    h5: float,
    rng: np.random.Generator,
):
    duration = N_CYCLES / frequency

    n_samples = int(
        np.ceil(duration * FS)
    )

    t = np.arange(
        n_samples,
        dtype=np.float64
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

    # Gaussian noise.
    if noise_rms != 0.0:
        noise = rng.normal(
            loc=0.0,
            scale=noise_rms,
            size=len(x)
        )

        x += noise

    return t, x


# ------------------------------------------------------------
# Rising zero crossings
#
# Linear interpolation between adjacent samples.
# ------------------------------------------------------------

def rising_zero_crossings(x: np.ndarray):
    idx = np.where(
        (x[:-1] < 0.0) &
        (x[1:] >= 0.0)
    )[0]

    if len(idx) < 2:
        return np.empty(
            0,
            dtype=np.float64
        )

    y0 = x[idx]
    y1 = x[idx + 1]

    denominator = y1 - y0

    valid = (
        np.abs(denominator) >
        1e-15
    )

    idx = idx[valid]

    y0 = x[idx]
    y1 = x[idx + 1]

    crossings = (
        idx.astype(np.float64) -
        y0 /
        (y1 - y0)
    )

    return crossings


# ------------------------------------------------------------
# Analyse one waveform
# ------------------------------------------------------------

def analyse_waveform(
    x: np.ndarray,
    expected_frequency: float,
):
    crossings = rising_zero_crossings(x)

    if len(crossings) < 3:
        return {
            "crossings": len(crossings),
            "cycles": max(
                0,
                len(crossings) - 1
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
# Main
# ------------------------------------------------------------

def main():

    os.makedirs(
        "reports",
        exist_ok=True
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = []

    print("=" * 86)
    print(
        "ZERO-CROSSING ROBUSTNESS TEST"
    )
    print("=" * 86)

    print(
        f"Sampling rate : {FS:.0f} Hz"
    )

    print(
        f"Reference cycles per test : {N_CYCLES}"
    )

    print(
        "Noise/harmonics are relative to the "
        "fundamental amplitude."
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

            results.append(
                result
            )

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
    # Determine pass/fail
    #
    # We use:
    #   - exactly N_CYCLES detected
    #   - frequency error <= 0.5%
    #
    # This is a software robustness criterion, not a field
    # requirement.
    # --------------------------------------------------------

    failures = []

    for r in results:

        expected_cycles = N_CYCLES

        if r["cycles"] != expected_cycles:
            failures.append(
                (
                    r["case"],
                    r["input_frequency"],
                    "wrong cycle count",
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
            "STATUS: ZERO-CROSSING ROBUSTNESS TEST PASSED"
        )

    else:

        print(
            "STATUS: ZERO-CROSSING ROBUSTNESS TEST FAILED"
        )

        print()
        print("Failures:")

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
        r"reports\zero_crossing_noise_results.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ZERO-CROSSING ROBUSTNESS TEST\n"
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
            f"Sampling rate: {FS:.0f} Hz\n"
        )

        f.write(
            f"Reference cycles: {N_CYCLES}\n\n"
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