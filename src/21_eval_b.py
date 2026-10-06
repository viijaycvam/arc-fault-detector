import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

FEATS = ["crest", "skew", "kurt", "d1_ratio", "d2_ratio", "peak_asym", "energy_asym", "frac_small"]
df = pd.read_csv(r"reports\features_b.csv")
tr, te = df[df.split == "train"], df[df.split == "test"]

def fit(d, seed=0):
    sc = StandardScaler().fit(d[FEATS])
    clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=seed)
    clf.fit(sc.transform(d[FEATS]), d["binary_label"])
    return sc, clf

def predict(sc, clf, d):
    return clf.predict(sc.transform(d[FEATS]))

sc, clf = fit(tr)
p = predict(sc, clf, te)
tn, fp, fn, tp = confusion_matrix(te.binary_label, p, labels=[0, 1]).ravel()
print("OFFICIAL SPLIT  TN/FP/FN/TP:", tn, fp, fn, tp, "| total", len(te))
print(f"  false-trip rate {fp/(tn+fp):.2%}   arc detection {tp/(tp+fn):.2%}")

print("\nLEAVE-ONE-ARC-CLASS-OUT (held-out arc class never seen in training)")
rows = []
for k in range(1, 14):
    held = df[df.detailed_label == k]
    sc, clf = fit(tr[tr.detailed_label != k])
    det = predict(sc, clf, held).mean()
    ftr = predict(sc, clf, te[te.binary_label == 0]).mean()
    rows.append((det, ftr))
    print(f"class {k:2d}: n={len(held):3d}  detection {det:7.2%}   false-trip {ftr:6.2%}")
print(f"\naverage detection {np.mean([r[0] for r in rows]):.2%}   average false-trip {np.mean([r[1] for r in rows]):.2%}")
print("note: normal cycles are pooled, so false-trip is measured on normals of seen appliances")