import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import os, glob, time

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

OUT_DIR = "outputs"
CNN_DIR = "cnn_results"
os.makedirs(CNN_DIR, exist_ok=True)

# ── load MADBase (existing CSVs) ──────────────────────────────────────────────
print("Loading MADBase...")
import pandas as pd
X_mad = pd.read_csv("data/csvTrainImages 60k x 784.csv", header=None).values
y_mad = pd.read_csv("data/csvTrainLabel 60k x 1.csv",   header=None).values.flatten()
X_mad_te = pd.read_csv("data/csvTestImages 10k x 784.csv", header=None).values
y_mad_te = pd.read_csv("data/csvTestLabel 10k x 1.csv",    header=None).values.flatten()

# combine MADBase train+test
X_mad = np.concatenate([X_mad, X_mad_te], axis=0).astype("float32") / 255.0
y_mad = np.concatenate([y_mad, y_mad_te], axis=0)
print(f"MADBase: {X_mad.shape[0]} images")

# ── load Mendeley (folder of PNGs) ───────────────────────────────────────────
print("Loading Mendeley dataset...")
X_men, y_men = [], []
for digit in range(10):
    folder = f"data/mendely/{digit}"
    for fpath in glob.glob(f"{folder}/*.png"):
        img = Image.open(fpath).convert('L')      # grayscale
        img = img.resize((28, 28), Image.LANCZOS) # ensure 28x28
        arr = np.array(img).astype("float32") / 255.0
        X_men.append(arr.flatten())               # flatten to 784
        y_men.append(digit)

X_men = np.array(X_men)
y_men = np.array(y_men)
print(f"Mendeley: {X_men.shape[0]} images")

# ── check if Mendeley is dark-on-white (needs inversion) ─────────────────────
# MADBase is white digit on black — Mendeley might be opposite
sample_mad = X_mad[0]
sample_men = X_men[0]
if sample_men.mean() > 0.5:   # bright background → invert
    print("Inverting Mendeley images (dark digit on white → white on black)...")
    X_men = 1.0 - X_men

# ── merge ─────────────────────────────────────────────────────────────────────
X_all = np.concatenate([X_mad, X_men], axis=0)
y_all = np.concatenate([y_mad, y_men], axis=0)
print(f"Merged total: {X_all.shape[0]} images")

# ── split 70/15/15 ────────────────────────────────────────────────────────────
X_tr, X_temp, y_tr, y_temp = train_test_split(X_all, y_all, test_size=0.30,
                                               random_state=42, stratify=y_all)
X_val, X_te, y_val, y_te   = train_test_split(X_temp, y_temp, test_size=0.50,
                                               random_state=42, stratify=y_temp)
print(f"Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")

# reshape for CNN (N, 1, 28, 28)
def to_loader(X, y, shuffle=False):
    X_t = torch.tensor(X.reshape(-1, 1, 28, 28), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=shuffle)

train_loader = to_loader(X_tr,  y_tr,  shuffle=True)
val_loader   = to_loader(X_val, y_val)
test_loader  = to_loader(X_te,  y_te)

# ── model (same architecture) ─────────────────────────────────────────────────
class ArabicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = ArabicCNN()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ── train ─────────────────────────────────────────────────────────────────────
best_val_acc, patience_count = 0.0, 0
PATIENCE = 5
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

print("\nTraining on merged dataset...\n")
start = time.time()

for epoch in range(1, 51):
    model.train()
    tl, tc = 0.0, 0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        out  = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        tl += loss.item() * len(X_b)
        tc += (out.argmax(1) == y_b).sum().item()

    model.eval()
    vl, vc = 0.0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            out  = model(X_b)
            vl  += criterion(out, y_b).item() * len(X_b)
            vc  += (out.argmax(1) == y_b).sum().item()

    ta = tc / len(train_loader.dataset)
    va = vc / len(val_loader.dataset)
    tl /= len(train_loader.dataset)
    vl /= len(val_loader.dataset)
    history['train_loss'].append(tl)
    history['val_loss'].append(vl)
    history['train_acc'].append(ta)
    history['val_acc'].append(va)

    print(f"Epoch {epoch:02d} | loss {tl:.4f} | acc {ta:.4f} | val_loss {vl:.4f} | val_acc {va:.4f}")

    if va > best_val_acc:
        best_val_acc = va
        patience_count = 0
        torch.save(model.state_dict(), f"{CNN_DIR}/best_cnn.pt")  # overwrite with better model
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

print(f"\nDone in {time.time()-start:.1f}s | Best val acc: {best_val_acc:.4f}")

# ── evaluate ──────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(f"{CNN_DIR}/best_cnn.pt"))
model.eval()
preds, labels = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        preds.extend(model(X_b).argmax(1).numpy())
        labels.extend(y_b.numpy())

preds  = np.array(preds)
labels = np.array(labels)
acc    = (preds == labels).mean()
print(f"Test accuracy: {acc*100:.2f}%")
print(classification_report(labels, preds))

# confusion matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title(f'Merged CNN Confusion Matrix (Test Acc: {acc*100:.2f}%)')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(f"{CNN_DIR}/confusion_matrix_merged.png", dpi=120)
plt.close()
print(f"Saved → {CNN_DIR}/confusion_matrix_merged.png")

# training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_loss'], label='Train'); ax1.plot(history['val_loss'], label='Val')
ax1.set_title('Loss'); ax1.legend()
ax2.plot(history['train_acc'], label='Train'); ax2.plot(history['val_acc'], label='Val')
ax2.set_title('Accuracy'); ax2.legend()
plt.tight_layout()
plt.savefig(f"{CNN_DIR}/training_curves_merged.png", dpi=120)
plt.close()

np.save(f"{OUT_DIR}/y_pred_cnn.npy", preds)
print("\nRetrain complete. best_cnn.pt updated.")