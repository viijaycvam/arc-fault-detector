import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MAT_FILE = Path(
    r"data\raw\arc_fault_dataset\Datasets\dataset_14classes_20ms_50kHz.mat"
)

OUTPUT_DIR = Path(r"reports\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Published class mapping for the 14-class DFDI dataset
CLASS_NAMES = {
    0: "NOARC - Any appliance",
    1: "ARC - Electric Pump",
    2: "ARC - Dell Desktop Computer",
    3: "ARC - Dolce Gusto Espresso",
    4: "ARC - Titan Drill",
    5: "ARC - Peugeot Drill",
    6: "ARC - Mewal Halogen Lamp",
    7: "ARC - Blow Heater",
    8: "ARC - Severin 2200 Kettle",
    9: "ARC - Moulinex Mini Food Processor",
    10: "ARC - Multi-functional Printer",
    11: "ARC - Sabre Electric Jigsaw",
    12: "ARC - Bluesky Optimo Vacuum Cleaner",
    13: "ARC - Philips Vacuum Cleaner",
}

print("=" * 70)
print("B1 — PLOTTING ONE TRAINING WAVEFORM PER CLASS")
print("=" * 70)

data = scipy.io.loadmat(MAT_FILE)

y_train = data["yTrain"]
labels_train = data["labelsTrain"].ravel().astype(int)

print(f"yTrain shape      : {y_train.shape}")
print(f"labelsTrain shape : {labels_train.shape}")

for label in range(14):

    indices = np.where(labels_train == label)[0]

    if len(indices) == 0:
        print(f"WARNING: no samples found for label {label}")
        continue

    idx = indices[0]
    waveform = y_train[idx]

    plt.figure(figsize=(10, 4))
    plt.plot(waveform)
    plt.title(
        f"Label {label}: {CLASS_NAMES.get(label, 'Unknown')}"
    )
    plt.xlabel("Sample")
    plt.ylabel("Waveform value")
    plt.grid(True)
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"class_{label:02d}.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(
        f"Label {label:2d} | "
        f"index {idx:4d} | "
        f"{CLASS_NAMES.get(label, 'Unknown'):<45} | "
        f"{output_file}"
    )

print()
print("B1 plotting complete.")