import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score

# Load official train/test data split
X_train = np.load("data/processed/X_train.npy") if False else None # update paths if needed
# Load raw preprocessed numpy feature arrays / scaler / model
with open("reports/trained_mlp.pkl", "rb") as f:
    saved_data = pickle.load(f)

scaler = saved_data["scaler"]
model = saved_data["model"]
X_train = saved_data["X_train"]
y_train = saved_data["y_train"]
X_test = saved_data["X_test"]
y_test = saved_data["y_test"]

print("====================================================")
print("DATASET & SPLIT VERIFICATION")
print("====================================================")
print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"Train label counts (0=Normal, 1=Arc): {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Test label counts  (0=Normal, 1=Arc): {dict(zip(*np.unique(y_test, return_counts=True)))}")

# Scaler and Predict
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n====================================================")
print("UNNORMALIZED CONFUSION MATRIX (Official Test Set)")
print("====================================================")
print(f"True Negative  (TN - Normal correctly predicted) : {tn}")
print(f"False Positive (FP - Normal false-tripped as Arc): {fp}")
print(f"False Negative (FN - Arc missed as Normal)      : {fn}")
print(f"True Positive  (TP - Arc correctly detected)    : {tp}")
print(f"Total Test Cycles Check                         : {tn + fp + fn + tp} / {len(y_test)}")
print("----------------------------------------------------")
print(f"False-Trip Rate (FTR)      : {fp / (tn + fp) * 100:.2f}% ({fp}/{tn + fp})")
print(f"Arc Detection Rate (Recall): {tp / (tp + fn) * 100:.2f}% ({tp}/{tp + fn})")
print(f"Overall Accuracy           : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("====================================================")