import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
OUT_DIR  = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── load CSVs ────────────────────────────────────────────────────────────────
print("Loading CSVs...")
X_train_raw = pd.read_csv(f"{DATA_DIR}/csvTrainImages 60k x 784.csv", header=None).values
y_train_raw = pd.read_csv(f"{DATA_DIR}/csvTrainLabel 60k x 1.csv",   header=None).values.flatten()
X_test_raw  = pd.read_csv(f"{DATA_DIR}/csvTestImages 10k x 784.csv", header=None).values
y_test_raw  = pd.read_csv(f"{DATA_DIR}/csvTestLabel 10k x 1.csv",    header=None).values.flatten()

print(f"Train images : {X_train_raw.shape}")
print(f"Train labels : {y_train_raw.shape}  | classes: {np.unique(y_train_raw)}")
print(f"Test images  : {X_test_raw.shape}")
print(f"Test labels  : {y_test_raw.shape}")

# ── normalize ────────────────────────────────────────────────────────────────
X_train = X_train_raw.astype("float32") / 255.0
X_test  = X_test_raw.astype("float32")  / 255.0
print(f"\nPixel range after normalization: [{X_train.min():.1f}, {X_train.max():.1f}]")

# ── sample grid ──────────────────────────────────────────────────────────────
print("\nSaving sample grid...")
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for digit in range(10):
    idx = np.where(y_train_raw == digit)[0][0]
    for row, data in enumerate([X_train_raw, X_train]):
        ax = axes[row, digit]
        ax.imshow(data[idx].reshape(28, 28), cmap="gray")
        ax.set_title(str(digit), fontsize=9)
        ax.axis("off")
axes[0, 0].set_ylabel("Raw", fontsize=8)
axes[1, 0].set_ylabel("Normalized", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/samples.png", dpi=120)
plt.close()
print(f"Saved → {OUT_DIR}/samples.png")

# ── class distribution ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
unique, counts = np.unique(y_train_raw, return_counts=True)
ax.bar(unique, counts, color="steelblue", edgecolor="white")
ax.set_xlabel("Digit class")
ax.set_ylabel("Count")
ax.set_title("Class distribution — training set")
ax.set_xticks(range(10))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/class_distribution.png", dpi=120)
plt.close()
print(f"Saved → {OUT_DIR}/class_distribution.png")

print("\nStep 4 complete. Data loaded and verified successfully.")
from sklearn.model_selection import train_test_split

# ── split: 70% train / 15% val / 15% test ───────────────────────────────────
# combine original train+test first so we control the split ourselves
X_all = np.concatenate([X_train, X_test], axis=0)
y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)

X_tr, X_temp, y_tr, y_temp = train_test_split(X_all, y_all, test_size=0.30, random_state=42, stratify=y_all)
X_val, X_te, y_val, y_te   = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nSplit sizes:")
print(f"  Train : {X_tr.shape[0]}")
print(f"  Val   : {X_val.shape[0]}")
print(f"  Test  : {X_te.shape[0]}")

# ── reshape for CNN (N, 28, 28, 1) ──────────────────────────────────────────
X_tr_cnn  = X_tr.reshape(-1, 28, 28, 1)
X_val_cnn = X_val.reshape(-1, 28, 28, 1)
X_te_cnn  = X_te.reshape(-1, 28, 28, 1)

# ── save all arrays ──────────────────────────────────────────────────────────
np.save(f"{OUT_DIR}/X_train_cnn.npy", X_tr_cnn)
np.save(f"{OUT_DIR}/X_val_cnn.npy",   X_val_cnn)
np.save(f"{OUT_DIR}/X_test_cnn.npy",  X_te_cnn)
np.save(f"{OUT_DIR}/X_train_svm.npy", X_tr)
np.save(f"{OUT_DIR}/X_val_svm.npy",   X_val)
np.save(f"{OUT_DIR}/X_test_svm.npy",  X_te)
np.save(f"{OUT_DIR}/y_train.npy",     y_tr)
np.save(f"{OUT_DIR}/y_val.npy",       y_val)
np.save(f"{OUT_DIR}/y_test.npy",      y_te)

print(f"\nAll .npy files saved to {OUT_DIR}/")
print("Step 5 complete.")
from sklearn.decomposition import PCA
import joblib

# ── PCA (fit on train only, transform val+test) ──────────────────────────────
print("\nFitting PCA...")
pca = PCA(n_components=0.95, random_state=42)  # keep 95% of variance
X_tr_pca  = pca.fit_transform(X_tr)
X_val_pca = pca.transform(X_val)
X_te_pca  = pca.transform(X_te)

print(f"Components kept : {pca.n_components_}")
print(f"X_train_svm PCA shape: {X_tr_pca.shape}")

# overwrite svm arrays with PCA-reduced versions
np.save(f"{OUT_DIR}/X_train_svm.npy", X_tr_pca)
np.save(f"{OUT_DIR}/X_val_svm.npy",   X_val_pca)
np.save(f"{OUT_DIR}/X_test_svm.npy",  X_te_pca)

joblib.dump(pca, f"{OUT_DIR}/pca_model.pkl")
print(f"PCA model saved → {OUT_DIR}/pca_model.pkl")
print("\nStep 6 complete. Role 1 pipeline fully done.")