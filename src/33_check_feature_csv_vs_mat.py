"""Read-only consistency check: recompute test features from the MAT file and
compare them to reports/features_b.csv. Does not modify project files.

Run from project root:
    python src\\33_check_feature_csv_vs_mat.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy.io

MAT = Path('data') / 'raw' / 'arc_fault_dataset' / 'Datasets' / 'dataset_14classes_20ms_50kHz.mat'
CSV = Path('reports') / 'features_b.csv'
FEATURES = ['crest', 'skew', 'kurt', 'd1_ratio', 'd2_ratio', 'peak_asym', 'energy_asym', 'frac_small']


def features_b(X):
    X = np.asarray(X, dtype=np.float64)
    a = X - X.mean(axis=1, keepdims=True)
    e = (a ** 2).sum(axis=1)
    rms = np.sqrt(e / a.shape[1])
    safe_e = np.where(e > 1e-12, e, 1.0)
    safe_rms = np.where(rms > 1e-9, rms, 1.0)
    peak = np.abs(a).max(axis=1)
    crest = peak / safe_rms
    skew = (a ** 3).mean(axis=1) / safe_rms ** 3
    kurt = (a ** 4).mean(axis=1) / safe_rms ** 4 - 3.0
    d1 = np.diff(a, axis=1)
    d2 = a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]
    d1_ratio = (d1 ** 2).sum(axis=1) / safe_e
    d2_ratio = (d2 ** 2).sum(axis=1) / safe_e
    pmax, pmin = a.max(axis=1), a.min(axis=1)
    span = np.where((pmax - pmin) > 1e-12, pmax - pmin, 1.0)
    peak_asym = (pmax + pmin) / span
    pos_e = np.where(a > 0, a ** 2, 0.0).sum(axis=1)
    neg_e = np.where(a < 0, a ** 2, 0.0).sum(axis=1)
    energy_asym = (pos_e - neg_e) / safe_e
    frac_small = (np.abs(a) < 0.1 * peak[:, None]).mean(axis=1)
    return np.column_stack([crest, skew, kurt, d1_ratio, d2_ratio,
                            peak_asym, energy_asym, frac_small])


def main():
    missing = [str(p) for p in (MAT, CSV) if not p.exists()]
    if missing:
        print('MISSING REQUIRED FILE(S):')
        for p in missing:
            print('  ' + p)
        print('Run this from the project root; do not edit or regenerate the CSV first.')
        return 2

    mat = scipy.io.loadmat(MAT)
    X = np.asarray(mat['yTest'], dtype=np.float64)
    labels = np.asarray(mat['labelsTest']).ravel().astype(int)
    direct = features_b(X)

    df = pd.read_csv(CSV)
    te = df[df['split'].astype(str).str.lower() == 'test'].copy().reset_index(drop=True)
    print('MAT test shape:', X.shape, '| MAT labels:', labels.shape)
    print('CSV test rows :', len(te))
    if len(te) != len(X):
        print('FAIL: MAT and CSV test row counts differ.')
        return 1
    if any(c not in te.columns for c in FEATURES + ['detailed_label', 'binary_label']):
        print('FAIL: CSV is missing expected feature/label columns.')
        return 1

    csv_feat = te[FEATURES].to_numpy(dtype=np.float64)
    diff = np.abs(direct - csv_feat)
    print('\nFeature comparison (recomputed MAT vs CSV):')
    for i, name in enumerate(FEATURES):
        print(f'  {name:12s} max_abs_diff={diff[:, i].max():.10g}  mean_abs_diff={diff[:, i].mean():.10g}')
    print('Rows with all 8 feature differences <= 1e-6:', int(np.all(diff <= 1e-6, axis=1).sum()), '/', len(te))

    csv_detailed = te['detailed_label'].to_numpy(dtype=int)
    csv_binary = te['binary_label'].to_numpy(dtype=int)
    mat_binary = (labels > 0).astype(int)
    print('\nLabel comparison:')
    print('  detailed labels equal:', int((csv_detailed == labels).sum()), '/', len(labels))
    print('  binary labels equal  :', int((csv_binary == mat_binary).sum()), '/', len(labels))
    print('  MAT class counts     :', dict(zip(*np.unique(labels, return_counts=True))))
    print('  CSV detailed counts  :', dict(zip(*np.unique(csv_detailed, return_counts=True))))

    if np.all(diff <= 1e-6) and np.array_equal(csv_detailed, labels) and np.array_equal(csv_binary, mat_binary):
        print('\nPASS: CSV test features and labels match the current MAT dataset.')
        print('If noise_robustness still gives a different confusion matrix, next compare its parsed model parameters against the voting script and golden inference.')
        return 0
    else:
        print('\nFAIL: CSV test rows do not match the current MAT dataset. This likely explains the inconsistent baselines; do not overwrite either file yet.')
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
