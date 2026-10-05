import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

DATA_DIR = Path(
    r"data\raw\arc_fault_dataset\Datasets"
)

OUTPUT_DIR = Path(r"reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

pattern = re.compile(
    r"dataset_(\d+)classes_(\d+)ms_(\d+)kHz\.mat$"
)

files = sorted(DATA_DIR.glob("dataset_*.mat"))

print("=" * 80)
print("B1 — DATASET INVENTORY")
print("=" * 80)
print(f"Found {len(files)} MAT files")
print()

for path in files:
    match = pattern.match(path.name)

    if not match:
        print(f"Skipping unexpected filename: {path.name}")
        continue

    classes = int(match.group(1))
    window_ms = int(match.group(2))
    sampling_khz = int(match.group(3))

    print(f"Inspecting: {path.name}")

    info = {
        "file": path.name,
        "classes": classes,
        "window_ms": window_ms,
        "sampling_khz": sampling_khz,
        "samples_per_window_expected": window_ms * sampling_khz,
        "size_bytes": path.stat().st_size,
    }

    try:
        mat = scipy.io.loadmat(path)

        y_train = mat.get("yTrain")
        y_test = mat.get("yTest")
        labels_train = mat.get("labelsTrain")
        labels_test = mat.get("labelsTest")

        info["yTrain_shape"] = str(
            np.shape(y_train)
        )
        info["yTest_shape"] = str(
            np.shape(y_test)
        )
        info["labelsTrain_shape"] = str(
            np.shape(labels_train)
        )
        info["labelsTest_shape"] = str(
            np.shape(labels_test)
        )

        if y_train is not None:
            info["train_records"] = int(
                y_train.shape[0]
            )
            info["samples_per_record"] = int(
                y_train.shape[1]
            )
        else:
            info["train_records"] = ""
            info["samples_per_record"] = ""

        if labels_train is not None:
            unique_train = np.unique(
                labels_train.ravel()
            )
            info["train_unique_labels"] = ",".join(
                str(int(x))
                for x in unique_train
            )
        else:
            info["train_unique_labels"] = ""

        if labels_test is not None:
            unique_test = np.unique(
                labels_test.ravel()
            )
            info["test_unique_labels"] = ",".join(
                str(int(x))
                for x in unique_test
            )
        else:
            info["test_unique_labels"] = ""

        info["labelappliances_present"] = (
            "labelappliances" in mat
        )

        print(
            f"  train={info['train_records']} "
            f"records, "
            f"samples={info['samples_per_record']}"
        )

    except Exception as exc:
        info["error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        print(f"  ERROR: {info['error']}")

    rows.append(info)

df = pd.DataFrame(rows)

output_file = (
    OUTPUT_DIR / "dataset_inventory.csv"
)

df.to_csv(
    output_file,
    index=False
)

print()
print("=" * 80)
print("INVENTORY COMPLETE")
print("=" * 80)
print(f"Saved: {output_file}")
print()

print(
    df[
        [
            "file",
            "classes",
            "window_ms",
            "sampling_khz",
            "samples_per_window_expected",
            "samples_per_record",
            "train_records",
        ]
    ].to_string(index=False)
)