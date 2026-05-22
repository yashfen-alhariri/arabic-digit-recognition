import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from pillow_heif import register_heif_opener
import cv2, glob, os, joblib, time
import matplotlib.pyplot as plt
import seaborn as sns

register_heif_opener()

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
OUT_DIR = "outputs"
CNN_DIR = "cnn_results"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CNN_DIR, exist_ok=True)

# ── 1. MADBase ────────────────────────────────────────────────────────────────
print("Loading MADBase...")
X_mad = pd.read_csv("data/csvTrainImages 60k x 784.csv", header=None).values
y_mad = pd.read_csv("data/csvTrainLabel 60k x 1.csv",   header=None).values.flatten()
X_mad_te = pd.read_csv("data/csvTestImages 10k x 784.csv", header=None).values
y_mad_te = pd.read_csv("data/csvTestLabel 10k x 1.csv",    header=None).values.flatten()
X_mad = np.concatenate([X_mad, X_mad_te]).astype("float32") / 255.0
y_mad = np.concatenate([y_mad, y_mad_te])
print(f"MADBase: {len(X_mad)} images")

# ── 2. Mendeley ───────────────────────────────────────────────────────────────
print("Loading Mendeley...")
X_men, y_men = [], []
for digit in range(10):
    for fpath in glob.glob(f"data/mendely/{digit}/*.png"):
        img = Image.open(fpath).convert('L')
        img = img.resize((28, 28), Image.LANCZOS)
        arr = np.array(img).astype("float32") / 255.0
        X_men.append(arr.flatten())
        y_men.append(digit)
X_men = np.array(X_men)
y_men = np.array(y_men)
if X_men.mean() > 0.5:
    X_men = 1.0 - X_men
print(f"Mendeley: {len(X_men)} images")

# ── 3. Your handwriting ───────────────────────────────────────────────────────
print("Loading your handwriting...")

def preprocess_photo(path):
    img = Image.open(path).convert('L')
    img = np.array(img)
    h, w = img.shape
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    h, w = img.shape
    img = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    img = cv2.GaussianBlur(img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(img, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    filled = binary.copy()
    mask = np.zeros((binary.shape[0]+2, binary.shape[1]+2), np.uint8)
    cv2.floodFill(filled, mask, (0, 0), 255)
    binary = cv2.bitwise_or(binary, cv2.bitwise_not(filled))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels < 2:
        return None
    best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean = np.zeros_like(binary)
    clean[labels == best] = 255
    x  = stats[best, cv2.CC_STAT_LEFT]
    y  = stats[best, cv2.CC_STAT_TOP]
    bw = stats[best, cv2.CC_STAT_WIDTH]
    bh = stats[best, cv2.CC_STAT_HEIGHT]
    cropped = clean[y:y+bh, x:x+bw]
    pad = int(max(bw, bh) * 0.3)
    cropped = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
    return resized.astype("float32") / 255.0

X_mine, y_mine = [], []
for digit in range(10):
    files = (glob.glob(f"my_handwriting/{digit}/*.jpg") +
             glob.glob(f"my_handwriting/{digit}/*.heic") +
             glob.glob(f"my_handwriting/{digit}/*.HEIC") +
             glob.glob(f"my_handwriting/{digit}/*.png"))
    for fpath in files:
        img = preprocess_photo(fpath)
        if img is not None:
            # augment each photo 30x
            for _ in range(30):
                aug = img.copy()
                angle = np.random.uniform(-20, 20)
                M = cv2.getRotationMatrix2D((14,14), angle, 1.0)
                aug = cv2.warpAffine(aug, M, (28,28))
                tx, ty = np.random.randint(-3,3), np.random.randint(-3,3)
                aug = cv2.warpAffine(aug, np.float32([[1,0,tx],[0,1,ty]]), (28,28))
                scale = np.random.uniform(0.85, 1.15)
                aug = cv2.resize(cv2.resize(aug, None, fx=scale, fy=scale), (28,28))
                aug = np.clip(aug + np.random.normal(0, 0.02, aug.shape), 0, 1)
                X_mine.append(aug.flatten())
                y_mine.append(digit)

X_mine = np.array(X_mine)
y_mine = np.array(y_mine)
print(f"Your handwriting (augmented): {len(X_mine)} images")

# ── 4. Merge all ──────────────────────────────────────────────────────────────
X_all = np.concatenate([X_mad, X_men, X_mine])
y_all = np.concatenate([y_mad, y_men, y_mine])
print(f"\nTotal merged: {len(X_all)} images")
for d in range(10):
    print(f"  Digit {d}: {(y_all==d).sum()}")

# ── 5. Split ──────────────────────────────────────────────────────────────────
X_tr, X_temp, y_tr, y_temp = train_test_split(X_all, y_all, test_size=0.30,
                                               random_state=42, stratify=y_all)
X_val, X_te, y_val, y_te   = train_test_split(X_temp, y_temp, test_size=0.50,
                                               random_state=42, stratify=y_temp)
print(f"\nSplit — Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")

# ── 6. Save CNN arrays ────────────────────────────────────────────────────────
np.save(f"{OUT_DIR}/X_train_cnn.npy", X_tr.reshape(-1, 28, 28, 1))
np.save(f"{OUT_DIR}/X_val_cnn.npy",   X_val.reshape(-1, 28, 28, 1))
np.save(f"{OUT_DIR}/X_test_cnn.npy",  X_te.reshape(-1, 28, 28, 1))
np.save(f"{OUT_DIR}/y_train.npy",     y_tr)
np.save(f"{OUT_DIR}/y_val.npy",       y_val)
np.save(f"{OUT_DIR}/y_test.npy",      y_te)

# ── 7. PCA for SVM ────────────────────────────────────────────────────────────
print("\nFitting PCA for SVM...")
pca = PCA(n_components=0.95, random_state=42)
X_tr_pca  = pca.fit_transform(X_tr)
X_val_pca = pca.transform(X_val)
X_te_pca  = pca.transform(X_te)
print(f"PCA components: {pca.n_components_}")
np.save(f"{OUT_DIR}/X_train_svm.npy", X_tr_pca)
np.save(f"{OUT_DIR}/X_val_svm.npy",   X_val_pca)
np.save(f"{OUT_DIR}/X_test_svm.npy",  X_te_pca)
joblib.dump(pca, f"{OUT_DIR}/pca_model.pkl")
print("PCA saved.")

# ── 8. Train CNN ──────────────────────────────────────────────────────────────
class ArabicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128*7*7, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def to_loader(X, y, shuffle=False):
    Xt = torch.tensor(X.reshape(-1,1,28,28), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=shuffle)

train_loader = to_loader(X_tr,  y_tr,  shuffle=True)
val_loader   = to_loader(X_val, y_val)
test_loader  = to_loader(X_te,  y_te)

model     = ArabicCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
best_val, patience = 0.0, 0

print("\nTraining CNN on fully merged dataset...\n")
start = time.time()
for epoch in range(1, 51):
    model.train()
    tl, tc = 0.0, 0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(Xb); loss = criterion(out, yb)
        loss.backward(); optimizer.step()
        tl += loss.item()*len(Xb); tc += (out.argmax(1)==yb).sum().item()
    model.eval()
    vl, vc = 0.0, 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            out = model(Xb)
            vl += criterion(out,yb).item()*len(Xb)
            vc += (out.argmax(1)==yb).sum().item()
    ta = tc/len(train_loader.dataset); va = vc/len(val_loader.dataset)
    tl /= len(train_loader.dataset);   vl /= len(val_loader.dataset)
    print(f"Epoch {epoch:02d} | loss {tl:.4f} | acc {ta:.4f} | val_loss {vl:.4f} | val_acc {va:.4f}")
    if va > best_val:
        best_val = va; patience = 0
        torch.save(model.state_dict(), f"{CNN_DIR}/best_cnn.pt")
    else:
        patience += 1
        if patience >= 5:
            print(f"Early stopping at epoch {epoch}.")
            break

print(f"\nTraining done in {time.time()-start:.1f}s | Best val acc: {best_val:.4f}")

# ── 9. Test ───────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(f"{CNN_DIR}/best_cnn.pt"))
model.eval()
preds, labs = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        preds.extend(model(Xb).argmax(1).numpy())
        labs.extend(yb.numpy())
preds = np.array(preds); labs = np.array(labs)
acc = (preds==labs).mean()
print(f"Test accuracy: {acc*100:.2f}%")
print(classification_report(labs, preds))
np.save(f"{OUT_DIR}/y_pred_cnn.npy", preds)

cm = confusion_matrix(labs, preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title(f'CNN Confusion Matrix — Full Merge (Test Acc: {acc*100:.2f}%)')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(f"{CNN_DIR}/confusion_matrix_final.png", dpi=120)
plt.close()
print(f"\nAll done. best_cnn.pt updated with fully merged training.")
