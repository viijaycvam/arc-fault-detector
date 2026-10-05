import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

CSV_FILE = Path(r"reports\baseline_features.csv")

def main():
    if not CSV_FILE.exists():
        print(f"Error: Feature CSV not found at {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)

    # Separate train and test splits
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    feature_cols = ["mean", "std", "rms", "peak", "peak_to_peak", "crest_factor"]

    X_train = train_df[feature_cols]
    y_train = train_df["binary_label"]

    X_test = test_df[feature_cols]
    y_test = test_df["binary_label"]
    y_test_det = test_df["detailed_label"]

    print("=" * 70)
    print("B2.2 — RANDOM FOREST BASELINE MODEL RESULTS")
    print("=" * 70)
    print(f"Train samples : {len(X_train)} (Normal: {(y_train == 0).sum()}, Arc: {(y_train == 1).sum()})")
    print(f"Test samples  : {len(X_test)} (Normal: {(y_test == 0).sum()}, Arc: {(y_test == 1).sum()})")
    print("-" * 70)

    # Train Random Forest Baseline
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)

    # Predict on Test Set
    pred = rf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    false_trip_rate = (pred[y_test == 0].mean()) * 100

    print("\n[1] Overall Metrics")
    print(f"  Accuracy       : {acc * 100:.2f}%")
    print(f"  F1-Score       : {f1:.4f}")
    print(f"  False-Trip Rate: {false_trip_rate:.2f}% ({pred[y_test == 0].sum()}/{(y_test == 0).sum()} normal samples misclassified as arc)")

    print("\n[2] Confusion Matrix")
    print("               Pred NORMAL  Pred ARC")
    print(f"  Actual NORMAL    {cm[0, 0]:<11} {cm[0, 1]}")
    print(f"  Actual ARC       {cm[1, 0]:<11} {cm[1, 1]}")

    print("\n[3] Classification Report")
    print(classification_report(y_test, pred, target_names=["NORMAL", "ARC"], digits=4))

    print("[4] Per-Appliance Arc Detection Performance")
    for k in range(1, 14):
        mask = (y_test_det == k)
        if mask.any():
            det_count = pred[mask].sum()
            total_count = mask.sum()
            det_rate = (det_count / total_count) * 100
            print(f"  Appliance Class {k:2d}: {det_rate:6.2f}% detected ({det_count}/{total_count})")

    print("\n[5] Feature Importances")
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"  {feat:15s}: {imp:.4f}")

if __name__ == "__main__":
    main()