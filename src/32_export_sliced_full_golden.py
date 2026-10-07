import importlib.util
import os
import struct
import numpy as np
import pandas as pd


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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_model():
    """
    Recreate the exact Option-B training/export model
    from src/23_export_model_and_golden.py.
    """

    sklearn_mlp = load_module(
        "model_export_reference",
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            "23_export_model_and_golden.py",
        ),
    )

    features_csv = (
        "reports/features_b.csv"
    )

    if not os.path.exists(
        features_csv
    ):
        raise RuntimeError(
            f"Missing {features_csv}"
        )

    df = pd.read_csv(
        features_csv
    )

    train = df[
        df["split"] == "train"
    ].copy()

    X_train = train[
        FEATURE_NAMES
    ].values.astype(
        np.float64
    )

    y_train = train[
        "binary_label"
    ].values.astype(
        np.int32
    )

    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    return scaler, model


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
            duration * FS
        )
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

    scaler, model = load_model()

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

            filtered = (
                zero_crossing.lowpass_filter(
                    raw,
                    zero_crossing.LOWPASS_CUTOFF_HZ,
                    FS,
                )
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

            cycle = (
                resampler.resample_cycle(
                    raw,
                    crossings[0],
                    crossings[1],
                    TARGET_CYCLE_LEN,
                )
            )

            cycle = np.asarray(
                cycle,
                dtype=np.float32,
            )

            features = (
                features_module.features_b(
                    cycle.reshape(
                        1,
                        -1,
                    )
                )[0]
            )

            features = np.asarray(
                features,
                dtype=np.float64,
            )

            scaled = (
                scaler.transform(
                    features.reshape(
                        1,
                        -1,
                    )
                )
            )

            expected_prob = float(
                model.predict_proba(
                    scaled
                )[0, 1]
            )

            records.append(
                {
                    "case": case_name,
                    "frequency": frequency,
                    "raw": cycle,
                    "features": features.astype(
                        np.float32
                    ),
                    "prob": expected_prob,
                }
            )

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    output_file = (
        r"reports\sliced_full_golden.bin"
    )

    with open(
        output_file,
        "wb",
    ) as f:

        f.write(b"AFGF")

        f.write(
            struct.pack(
                "<I",
                1,
            )
        )

        f.write(
            struct.pack(
                "<I",
                len(records),
            )
        )

        f.write(
            struct.pack(
                "<I",
                TARGET_CYCLE_LEN,
            )
        )

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

            f.write(
                struct.pack(
                    "<f",
                    record["prob"],
                )
            )

    print("=" * 80)
    print(
        "FULL SLICED-CYCLE GOLDEN EXPORT"
    )
    print("=" * 80)

    print(
        f"Records       : {len(records)}"
    )

    print(
        f"Samples/cycle : {TARGET_CYCLE_LEN}"
    )

    print(
        "Features      : 8"
    )

    print(
        "MLP output    : probability"
    )

    print(
        f"Saved         : {output_file}"
    )

    print()

    for i, record in enumerate(
        records
    ):

        print(
            f"{i:2d}: "
            f"{record['case']:<35} "
            f"{record['frequency']:4.1f} Hz "
            f"prob={record['prob']:.8f}"
        )

    print()
    print(
        "STATUS: FULL SLICED-CYCLE GOLDEN EXPORT PASSED"
    )


if __name__ == "__main__":
    main()