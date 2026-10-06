import numpy as np, scipy.io, matplotlib.pyplot as plt
d = scipy.io.loadmat(r"data\raw\arc_fault_dataset\Datasets\dataset_14classes_20ms_50kHz.mat")
X = np.vstack([d["yTrain"], d["yTest"]]).astype(float)
y = np.concatenate([d["labelsTrain"].ravel(), d["labelsTest"].ravel()]).astype(int)
fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for c, k in enumerate([1, 6]):
    for r, (lab, name) in enumerate([(k, f"class {k} arc"), (0, "normal")]):
        for i in np.where(y == lab)[0][:5]:
            ax[r, c].plot(X[i] - X[i].mean(), lw=0.8)
        ax[r, c].set_title(name)
plt.tight_layout(); plt.savefig(r"reports\classes_1_6.png", dpi=120); plt.show()