import os
import numpy as np


# ============================================================
# ZERO-CROSSING / CYCLE-SLICING SOFTWARE VALIDATION
#
# This is a SOFTWARE SIMULATION.
# No ESP32 hardware is used.
#
# Sampling:
#   fs = 50 kHz
#
# Test frequencies:
#   49.5 Hz
#   50.0 Hz
#   50.5 Hz
#
# Goal:
#   1. Detect rising zero crossings.
#   2. Measure actual raw samples/cycle.
#   3. Slice each cycle using interpolated crossings.
#   4. Resample each cycle to exactly 1000 points.
#   5. Check that the resampled cycle remains sinusoidal.
# ============================================================


FS = 50000.0
FREQUENCIES = [49.5, 50.0, 50.5]

N_CYCLES = 12

TARGET_CYCLE_LEN = 1000

AMPLITUDE = 1.0

PHASE = 0.23


OUTPUT_FILE = (
    r"reports\zero_crossing_results.txt"
)


# ------------------------------------------------------------
# Generate synthetic sampled waveform
# ------------------------------------------------------------

def generate_waveform(
    frequency: float,
    fs: float,
    n_cycles: int,
    amplitude: float,
    phase: float
):
    duration = (
        n_cycles / frequency
    )

    n_samples = int(
        np.ceil(
            duration * fs
        )
    )

    t = (
        np.arange(n_samples) /
        fs
    )

    x = (
        amplitude *
        np.sin(
            2.0 *
            np.pi *
            frequency *
            t +
            phase
        )
    )

    return t, x


# ------------------------------------------------------------
# Rising zero-crossing detection
#
# Uses linear interpolation between samples.
# ------------------------------------------------------------

def rising_zero_crossings(
    x: np.ndarray
):
    i = np.where(
        (x[:-1] < 0.0) &
        (x[1:] >= 0.0)
    )[0]

    if len(i) < 2:
        return np.asarray(
            [],
            dtype=np.float64
        )

    y0 = x[i]
    y1 = x[i + 1]

    denominator = y1 - y0

    valid = (
        np.abs(denominator) >
        1e-15
    )

    i = i[valid]

    y0 = x[i]
    y1 = x[i + 1]

    crossings = (
        i -
        y0 /
        (y1 - y0)
    )

    return crossings.astype(
        np.float64
    )


# ------------------------------------------------------------
# Slice cycle between two interpolated crossings
#
# Returns a uniformly sampled 1000-point cycle.
# ------------------------------------------------------------

def resample_cycle(
    x: np.ndarray,
    start_crossing: float,
    end_crossing: float,
    target_len: int
):
    if (
        end_crossing <=
        start_crossing
    ):
        raise ValueError(
            "Invalid crossing order."
        )

    # Integer sample coordinates available
    # inside the cycle.
    first = int(
        np.floor(start_crossing)
    )

    last = int(
        np.ceil(end_crossing)
    )

    if first < 0:
        first = 0

    if last >= len(x):
        last = len(x) - 1

    raw_indices = np.arange(
        first,
        last + 1,
        dtype=np.float64
    )

    raw_values = x[
        first:last + 1
    ].astype(np.float64)

    # Include exact interpolated crossing
    # coordinates in the interpolation grid.
    source_positions = np.concatenate(
        (
            np.asarray(
                [start_crossing],
                dtype=np.float64
            ),
            raw_indices,
            np.asarray(
                [end_crossing],
                dtype=np.float64
            )
        )
    )

    source_values = np.interp(
        source_positions,
        np.arange(
            len(x),
            dtype=np.float64
        ),
        x
    )

    # Target coordinates span one complete
    # cycle but exclude the duplicated endpoint.
    target_positions = (
        start_crossing +
        (
            np.arange(
                target_len,
                dtype=np.float64
            ) /
            target_len
        ) *
        (
            end_crossing -
            start_crossing
        )
    )

    cycle = np.interp(
        target_positions,
        source_positions,
        source_values
    )

    return cycle


# ------------------------------------------------------------
# Compare the resampled cycle against an ideal sine.
#
# Since phase starts at the zero crossing,
# the expected waveform is approximately sin(2*pi*u).
# ------------------------------------------------------------

def normalized_cycle_error(
    cycle: np.ndarray
):
    u = (
        np.arange(
            len(cycle),
            dtype=np.float64
        ) /
        len(cycle)
    )

    # Remove any tiny DC component.
    y = cycle - np.mean(cycle)

    # Normalize amplitude.
    scale = np.sqrt(
        np.mean(y * y)
    )

    if scale <= 1e-15:
        return np.nan

    y = y / scale

    reference = np.sin(
        2.0 *
        np.pi *
        u
    )

    reference = (
        reference /
        np.sqrt(
            np.mean(
                reference *
                reference
            )
        )
    )

    rmse = np.sqrt(
        np.mean(
            (y - reference) ** 2
        )
    )

    return float(rmse)


# ------------------------------------------------------------
# Test one frequency
# ------------------------------------------------------------

def test_frequency(
    frequency: float
):
    t, x = generate_waveform(
        frequency,
        FS,
        N_CYCLES,
        AMPLITUDE,
        PHASE
    )

    crossings = (
        rising_zero_crossings(x)
    )

    if len(crossings) < 3:
        raise RuntimeError(
            f"{frequency} Hz: insufficient "
            f"zero crossings ({len(crossings)})."
        )

    periods = np.diff(
        crossings
    )

    mean_period = float(
        np.mean(periods)
    )

    min_period = float(
        np.min(periods)
    )

    max_period = float(
        np.max(periods)
    )

    measured_frequency = (
        FS /
        mean_period
    )

    frequency_error = (
    (
        measured_frequency -
        frequency
    ) /
    frequency
) * 100.0

    resampled_lengths = []

    cycle_errors = []

    for j in range(
        len(crossings) - 1
    ):
        cycle = resample_cycle(
            x,
            crossings[j],
            crossings[j + 1],
            TARGET_CYCLE_LEN
        )

        resampled_lengths.append(
            len(cycle)
        )

        cycle_errors.append(
            normalized_cycle_error(
                cycle
            )
        )

    resampled_lengths = np.asarray(
        resampled_lengths
    )

    cycle_errors = np.asarray(
        cycle_errors,
        dtype=np.float64
    )

    return {
        "input_frequency": frequency,
        "crossings": len(crossings),
        "mean_period": mean_period,
        "min_period": min_period,
        "max_period": max_period,
        "measured_frequency": measured_frequency,
        "frequency_error": frequency_error,
        "num_cycles": len(periods),
        "min_resampled_len": int(
            np.min(resampled_lengths)
        ),
        "max_resampled_len": int(
            np.max(resampled_lengths)
        ),
        "mean_cycle_rmse": float(
            np.mean(cycle_errors)
        ),
        "max_cycle_rmse": float(
            np.max(cycle_errors)
        ),
    }


# ------------------------------------------------------------
# Write report
# ------------------------------------------------------------

def write_report(
    results
):
    os.makedirs(
        "reports",
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ZERO-CROSSING / CYCLE-SLICING VALIDATION\n"
        )

        f.write(
            "===========================================\n\n"
        )

        f.write(
            "Software simulation only.\n"
        )

        f.write(
            "No hardware measurement is represented here.\n\n"
        )

        f.write(
            f"Sampling frequency = {FS:.0f} Hz\n"
        )

        f.write(
            f"Target cycle length = "
            f"{TARGET_CYCLE_LEN} samples\n"
        )

        f.write(
            f"Test cycles/frequency = {N_CYCLES}\n\n"
        )

        f.write(
            "Results\n"
        )

        f.write(
            "-------------------------------------------\n"
        )

        for r in results:

            f.write(
                f"Input frequency      : "
                f"{r['input_frequency']:.3f} Hz\n"
            )

            f.write(
                f"Detected crossings   : "
                f"{r['crossings']}\n"
            )

            f.write(
                f"Detected cycles      : "
                f"{r['num_cycles']}\n"
            )

            f.write(
                f"Mean raw period      : "
                f"{r['mean_period']:.4f} samples\n"
            )

            f.write(
                f"Min raw period       : "
                f"{r['min_period']:.4f} samples\n"
            )

            f.write(
                f"Max raw period       : "
                f"{r['max_period']:.4f} samples\n"
            )

            f.write(
                f"Measured frequency   : "
                f"{r['measured_frequency']:.6f} Hz\n"
            )

            f.write(
                f"Frequency error      : "
                f"{r['frequency_error']:+.6f} %\n"
            )

            f.write(
                f"Resampled length     : "
                f"{r['min_resampled_len']}.."
                f"{r['max_resampled_len']}\n"
            )

            f.write(
                f"Mean cycle RMSE      : "
                f"{r['mean_cycle_rmse']:.8f}\n"
            )

            f.write(
                f"Max cycle RMSE       : "
                f"{r['max_cycle_rmse']:.8f}\n\n"
            )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "ZERO-CROSSING / CYCLE-SLICING VALIDATION"
    )
    print("=" * 70)

    print(
        f"Sampling rate      : {FS:.0f} Hz"
    )

    print(
        f"Target cycle length: "
        f"{TARGET_CYCLE_LEN} samples"
    )

    results = []

    for frequency in FREQUENCIES:

        result = test_frequency(
            frequency
        )

        results.append(
            result
        )

        print()
        print(
            f"{frequency:.1f} Hz TEST"
        )

        print("-" * 70)

        print(
            f"Detected crossings : "
            f"{result['crossings']}"
        )

        print(
            f"Detected cycles    : "
            f"{result['num_cycles']}"
        )

        print(
            f"Mean raw period    : "
            f"{result['mean_period']:.4f} samples"
        )

        print(
            f"Measured frequency : "
            f"{result['measured_frequency']:.6f} Hz"
        )

        print(
            f"Frequency error    : "
            f"{result['frequency_error']:+.6f} %"
        )

        print(
            f"Resampled length   : "
            f"{result['min_resampled_len']}.."
            f"{result['max_resampled_len']}"
        )

        print(
            f"Mean cycle RMSE    : "
            f"{result['mean_cycle_rmse']:.8f}"
        )

        print(
            f"Max cycle RMSE     : "
            f"{result['max_cycle_rmse']:.8f}"
        )

    # --------------------------------------------------------
    # Overall pass/fail
    # --------------------------------------------------------

    all_lengths_ok = all(
        r["min_resampled_len"] ==
        TARGET_CYCLE_LEN and
        r["max_resampled_len"] ==
        TARGET_CYCLE_LEN
        for r in results
    )

    all_frequency_ok = all(
        abs(r["frequency_error"]) <
        0.05
        for r in results
    )

    if (
        all_lengths_ok and
        all_frequency_ok
    ):
        status = (
            "STATUS: CYCLE SLICING TEST PASSED"
        )
    else:
        status = (
            "STATUS: CYCLE SLICING TEST FAILED"
        )

    print()
    print("=" * 70)
    print(status)
    print("=" * 70)

    write_report(
        results
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()