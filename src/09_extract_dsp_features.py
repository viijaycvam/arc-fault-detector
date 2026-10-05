import scipy.io
import scipy.stats
import numpy as np
import pandas as pd
from pathlib import Path

MAT_FILE = Path(r"data\raw\arc_fault_dataset\Datasets\dataset_14classes_20ms_50kHz.mat")
OUTPUT_FILE = Path(r"reports\dsp_features.csv")

def extract_dsp_features(X: np.ndarray, fs: float = 50000.0) -> pd.DataFrame:
    """Extract combined time-domain and FFT spectral features per 20ms waveform window."""
    X = X.astype(np.float64)
    ac = X - X.mean(axis=1, keepdims=True)
    
    # --- Time-Domain Features ---
    mean = X.mean(axis=1)
    std = ac.std(axis=1)
    rms = np.sqrt((ac ** 2).mean(axis=1))
    peak = np.abs(ac).max(axis=1)
    p2p = np.ptp(ac, axis=1)
    crest = np.divide(peak, rms, out=np.zeros_like(rms), where=rms > 0)
    
    # Statistical higher moments
    skewness = scipy.stats.skew(ac, axis=1)
    kurtosis = scipy.stats.kurtosis(ac, axis=1)
    
    # --- Frequency-Domain Features (FFT) ---
    N = ac.shape[1]
    fft_vals = np.abs(np.fft.rfft(ac, axis=1))  # Magnitude spectrum
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    
    # Total spectral power
    total_power = np.sum(fft_vals ** 2, axis=1)
    
    # Bandpass energy ratios (Low, Mid, High frequency power)
    low_band = (freqs >= 0) & (freqs < 1000)        # Fundamental & low harmonics (<1kHz)
    mid_band = (freqs >= 1000) & (freqs < 10000)    # Mid-frequency noise (1kHz - 10kHz)
    high_band = (freqs >= 10000) & (freqs <= 25000) # High-frequency arcing broadband noise (>10kHz)
    
    power_low = np.sum(fft_vals[:, low_band] ** 2, axis=1)
    power_mid = np.sum(fft_vals[:, mid_band] ** 2, axis=1)
    power_high = np.sum(fft_vals[:, high_band] ** 2, axis=1)
    
    # High-frequency energy ratio
    hf_ratio = np.divide(power_high, total_power, out=np.zeros_like(total_power), where=total_power > 0)
    
    # Spectral Centroid (Center of mass of spectrum)
    spectral_centroid = np.sum(freqs * fft_vals, axis=1) / np.maximum(np.sum(fft_vals, axis=1), 1e-12)
    
    return pd.DataFrame({
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": p2p,
        "crest_factor": crest,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "total_power": total_power,
        "power_low": power_low,
        "power_mid": power_mid,
        "power_high": power_high,
        "hf_ratio": hf_ratio,
        "spectral_centroid": spectral_centroid
    })

def main():
    if not MAT_FILE.exists():
        print(f"Error: MAT file not found at {MAT_FILE}")
        return

    data = scipy.io.loadmat(MAT_FILE)
    X_train, y_train_det = data["yTrain"], data["labelsTrain"].ravel().astype(int)
    X_test, y_test_det = data["yTest"], data["labelsTest"].ravel().astype(int)

    df_train_feats = extract_dsp_features(X_train)
    df_test_feats = extract_dsp_features(X_test)

    df_train = pd.DataFrame({
        "record_id": range(len(X_train)),
        "split": "train",
        "detailed_label": y_train_det,
        "binary_label": (y_train_det > 0).astype(int),
    }).join(df_train_feats)

    df_test = pd.DataFrame({
        "record_id": range(len(X_train), len(X_train) + len(X_test)),
        "split": "test",
        "detailed_label": y_test_det,
        "binary_label": (y_test_det > 0).astype(int),
    }).join(df_test_feats)

    df = pd.concat([df_train, df_test], ignore_index=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 70)
    print("B3 — DSP & SPECTRAL FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Total Records : {len(df)}")
    print(f"Features Extracted: {len(df_train_feats.columns)}")
    print("New DSP Features   : skewness, kurtosis, spectral_centroid, hf_ratio, power_low/mid/high")
    print(f"Saved to          : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()