import importlib.util
import os
import struct
import numpy as np


FS = 50000.0
TARGET_CYCLE_LEN = 1000
N_CYCLES = 20
RANDOM_SEED = 42

FREQUENCIES = [49.5, 50.0, 50.5]

CASES = [
    ("clean", 0.00, 0.00, 0.00),
    ("5pct_noise", 0.05, 0.00, 0.00),
    ("20pct_noise", 0.20, 0.00, 0.00),
    ("10pct_h3_5pct_noise", 0.05, 0.10, 0.00),
    ("20pct_h3_10pct_h5_5pct_noise", 0.05, 0.20, 0.10),
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


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

    t = (
        np.arange(
            n_samples,
            dtype=np.float64,
        )
        / FS
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


def main():

    base = os.path.dirname(
        os.path.abspath(__file__)
    )

    zero_crossing = load_module(
        "zero_crossing",
        os.path.join(
            base,
            "28_zero_crossing_hysteresis_test.py",
        ),
    )

    resampler = load_module(
        "resampler",
        os.path.join(
            base,
            "26_zero_crossing_test.py",
        ),
    )

    features_module = load_module(
        "features_b",
        os.path.join(
            base,
            "20_features_b.py",
        ),
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    records = []

    for case_name, noise_rms, h3, h5 in CASES:

        for frequency in FREQUENCIES:

            raw = generate_waveform(
                frequency,
                noise_rms,
                h3,
                h5,
                rng,
            )

            filtered = zero_crossing.lowpass_filter(
                raw,
                zero_crossing.LOWPASS_CUTOFF_HZ,
                FS,
            )

            crossings = (
                zero_crossing.rising_crossings_hysteresis(
                    filtered,
                    zero_crossing.HYSTERESIS,
                    zero_crossing.MIN_CROSSING_INTERVAL_SAMPLES,
                )
            )

            if len(crossings) < 2:
                raise RuntimeError(
                    f"No usable cycle for "
                    f"{case_name} / {frequency} Hz"
                )

            cycle = resampler.resample_cycle(
                raw,
                crossings[0],
                crossings[1],
                TARGET_CYCLE_LEN,
            )

            cycle = np.asarray(
                cycle,
                dtype=np.float32,
            )

            features = features_module.features_b(
                cycle.reshape(1, -1)
            )[0]

            records.append(
                {
                    "case": case_name,
                    "frequency": frequency,
                    "raw": cycle,
                    "features": np.asarray(
                        features,
                        dtype=np.float32,
                    ),
                }
            )

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    output_file = (
        r"reports\sliced_cycles_golden.bin"
    )

    with open(
        output_file,
        "wb",
    ) as f:

        # Magic
        f.write(b"AFSG")

        # Version
        f.write(
            struct.pack(
                "<I",
                1,
            )
        )

        # Number of records
        f.write(
            struct.pack(
                "<I",
                len(records),
            )
        )

        # Samples per cycle
        f.write(
            struct.pack(
                "<I",
                TARGET_CYCLE_LEN,
            )
        )

        # Features per record
        f.write(
            struct.pack(
                "<I",
                8,
            )
        )

        for record in records:

            f.write(
                record["raw"].astype(
                    "<f4"
                ).tobytes()
            )

            f.write(
                record["features"].astype(
                    "<f4"
                ).tobytes()
            )

    print("=" * 80)
    print(
        "SLICED-CYCLE GOLDEN EXPORT"
    )
    print("=" * 80)

    print(
        f"Records       : {len(records)}"
    )

    print(
        f"Samples/cycle : {TARGET_CYCLE_LEN}"
    )

    print(
        f"Features      : 8"
    )

    print(
        f"Saved         : {output_file}"
    )

    for i, record in enumerate(records):

        print(
            f"{i:2d}: "
            f"{record['case']:<35} "
            f"{record['frequency']:4.1f} Hz"
        )

    print(
        "\nSTATUS: SLICED-CYCLE GOLDEN EXPORT PASSED"
    )


if __name__ == "__main__":
    main()