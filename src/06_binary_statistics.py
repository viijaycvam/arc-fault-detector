import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path

MAT_FILE = Path(
    r"data\raw\arc_fault_dataset\Datasets\dataset_2classes_20ms_50kHz.mat"
)

OUTPUT_FILE = Path(
    r"reports\binary_statistics.csv"
)

data = scipy.io.loadmat(MAT_FILE)

y = data["yTrain"].astype(np.float64)
labels = data["labelsTrain"].ravel().astype(int)

rows = []

for i in range(len(y)):
    x = y[i]

    mean = np.mean(x)
    std = np.std(x)
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    peak_to_peak = np.ptp(x)

    if rms > 0:
        crest_factor = peak / rms
    else:
        crest_factor = 0.0

    rows.append({
        "index": i,
        "label": labels[i],
        "class": "NORMAL" if labels[i] == 0 else "ARC",
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor
    })

df = pd.DataFrame(rows)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("=" * 70)
print("B1 — NORMAL VS ARC STATISTICS")
print("=" * 70)

for class_name in ["NORMAL", "ARC"]:

    subset = df[df["class"] == class_name]

    print()
    print(class_name)
    print("-" * 50)

    for feature in [
        "mean",
        "std",
        "rms",
        "peak",
        "peak_to_peak",
        "crest_factor"
    ]:
        print(
            f"{feature:15s}: "
            f"mean={subset[feature].mean():.6f}  "
            f"std={subset[feature].std():.6f}"
        )

print()
print(f"Saved detailed results to: {OUTPUT_FILE}")
print()
print("B1 statistics complete.")