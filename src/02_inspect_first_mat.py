import scipy.io
from pathlib import Path
import numpy as np

file_path = Path(
    r"data\raw\arc_fault_dataset\Datasets\dataset_14classes_20ms_50kHz.mat"
)

print("=" * 70)
print("INSPECTING MATLAB DATASET")
print("=" * 70)
print(f"File: {file_path}")
print()

# Check that the file exists
if not file_path.exists():
    print("ERROR: File not found.")
    raise SystemExit(1)

# Show MATLAB variables without loading the full arrays
print("MATLAB variables:")
print("-" * 70)

try:
    variables = scipy.io.whosmat(file_path)

    for name, shape, dtype in variables:
        print(f"Name : {name}")
        print(f"Shape: {shape}")
        print(f"Type : {dtype}")
        print("-" * 70)

except Exception as e:
    print("Could not inspect the MAT file with scipy.io.whosmat().")
    print("Error:", e)
    raise SystemExit(1)

# Load the file
print()
print("Loading variables...")
data = scipy.io.loadmat(file_path)

print()
print("Top-level variables:")
print("-" * 70)

for key, value in data.items():

    # Skip MATLAB metadata
    if key.startswith("__"):
        continue

    print(f"Variable: {key}")
    print(f"Shape   : {np.shape(value)}")
    print(f"Dtype   : {getattr(value, 'dtype', type(value))}")

    if isinstance(value, np.ndarray) and value.size > 0:
        if np.issubdtype(value.dtype, np.number):
            finite = np.isfinite(value)

            if np.any(finite):
                print(f"Min     : {np.min(value[finite])}")
                print(f"Max     : {np.max(value[finite])}")

    print("-" * 70)

print()
print("Inspection complete.")