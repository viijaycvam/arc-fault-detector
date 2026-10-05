import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MAT_FILE = Path(
    r"data\raw\arc_fault_dataset\Datasets\dataset_2classes_20ms_50kHz.mat"
)

OUTPUT_DIR = Path(r"reports\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = scipy.io.loadmat(MAT_FILE)

y_train = data["yTrain"]
labels_train = data["labelsTrain"].ravel().astype(int)

print("=" * 70)
print("B1 — BINARY WAVEFORM INSPECTION")
print("=" * 70)

print(f"yTrain shape      : {y_train.shape}")
print(f"labelsTrain shape : {labels_train.shape}")

# Plot first 5 NORMAL examples
normal_indices = np.where(labels_train == 0)[0][:5]

for i, idx in enumerate(normal_indices, start=1):
    plt.figure(figsize=(10, 4))
    plt.plot(y_train[idx])
    plt.title(f"NORMAL — sample index {idx}")
    plt.xlabel("Sample")
    plt.ylabel("Current waveform value")
    plt.grid(True)
    plt.tight_layout()

    filename = OUTPUT_DIR / f"normal_{i}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

# Plot first 5 ARC examples
arc_indices = np.where(labels_train == 1)[0][:5]

for i, idx in enumerate(arc_indices, start=1):
    plt.figure(figsize=(10, 4))
    plt.plot(y_train[idx])
    plt.title(f"ARC — sample index {idx}")
    plt.xlabel("Sample")
    plt.ylabel("Current waveform value")
    plt.grid(True)
    plt.tight_layout()

    filename = OUTPUT_DIR / f"arc_{i}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

print()
print("Normal samples:", normal_indices.tolist())
print("Arc samples   :", arc_indices.tolist())
print()
print("Saved:")
print("  normal_1.png ... normal_5.png")
print("  arc_1.png    ... arc_5.png")
print()
print("B1 binary waveform inspection complete.")