import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from pillow_heif import register_heif_opener
import cv2, glob, os

register_heif_opener()

CNN_DIR = "cnn_results"

# ── same model architecture ───────────────────────────────────────────────────
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

# ── load existing trained model ───────────────────────────────────────────────
model = ArabicCNN()
model.load_state_dict(torch.load(f"{CNN_DIR}/best_cnn.pt"))
print("Loaded existing model.")

# ── preprocess one photo → 28x28 tensor ──────────────────────────────────────
def preprocess(path):
    img = Image.open(path).convert('L')
    img = np.array(img)
    h, w = img.shape
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    h, w = img.shape
    img = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    img = cv2.GaussianBlur(img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(img, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    filled = binary.copy()
    mask = np.zeros((binary.shape[0]+2, binary.shape[1]+2), np.uint8)
    cv2.floodFill(filled, mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    binary = cv2.bitwise_or(binary, filled_inv)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels < 2:
        return None
    best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean = np.zeros_like(binary)
    clean[labels == best_label] = 255
    x  = stats[best_label, cv2.CC_STAT_LEFT]
    y  = stats[best_label, cv2.CC_STAT_TOP]
    bw = stats[best_label, cv2.CC_STAT_WIDTH]
    bh = stats[best_label, cv2.CC_STAT_HEIGHT]
    cropped = clean[y:y+bh, x:x+bw]
    pad = int(max(bw, bh) * 0.3)
    cropped = cv2.copyMakeBorder(cropped, pad, pad, pad, pad,
                                  cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
    return resized.astype("float32") / 255.0

# ── load my handwriting photos ────────────────────────────────────────────────
print("\nLoading your handwriting photos...")
X_mine, y_mine = [], []
for digit in range(10):
    folder = f"my_handwriting/{digit}"
    files  = glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.HEIC") + glob.glob(f"{folder}/*.png")
    for fpath in files:
        img = preprocess(fpath)
        if img is not None:
            X_mine.append(img.flatten())
            y_mine.append(digit)
        else:
            print(f"  Skipped: {fpath}")

X_mine = np.array(X_mine)
y_mine = np.array(y_mine)
print(f"Loaded {len(X_mine)} personal photos")
for d in range(10):
    print(f"  Digit {d}: {(y_mine==d).sum()} samples")

# ── augment each photo 20x to boost variety ───────────────────────────────────
print("\nAugmenting personal photos...")
X_aug, y_aug = [], []
for img_flat, label in zip(X_mine, y_mine):
    img = img_flat.reshape(28, 28)
    for _ in range(20):
        # random rotation
        angle = np.random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        aug = cv2.warpAffine(img, M, (28, 28))
        # random shift
        tx, ty = np.random.randint(-3, 3), np.random.randint(-3, 3)
        M2 = np.float32([[1,0,tx],[0,1,ty]])
        aug = cv2.warpAffine(aug, M2, (28, 28))
        # random scale
        scale = np.random.uniform(0.85, 1.15)
        aug = cv2.resize(aug, None, fx=scale, fy=scale)
        aug = cv2.resize(aug, (28, 28))
        # add tiny noise
        noise = np.random.normal(0, 0.02, aug.shape).astype("float32")
        aug = np.clip(aug + noise, 0, 1)
        X_aug.append(aug.flatten())
        y_aug.append(label)

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)
print(f"Augmented to {len(X_aug)} samples")

# ── fine-tune: lower LR so we don't forget existing knowledge ─────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 10x lower than training

X_t = torch.tensor(X_aug.reshape(-1, 1, 28, 28), dtype=torch.float32)
y_t = torch.tensor(y_aug, dtype=torch.long)
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

print("\nFine-tuning...")
model.train()
for epoch in range(1, 11):
    total_loss, correct = 0.0, 0
    for X_b, y_b in loader:
        optimizer.zero_grad()
        out  = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_b)
        correct    += (out.argmax(1) == y_b).sum().item()
    acc = correct / len(X_aug)
    print(f"Epoch {epoch:02d} | loss {total_loss/len(X_aug):.4f} | acc {acc:.4f}")

# ── save fine-tuned model ─────────────────────────────────────────────────────
torch.save(model.state_dict(), f"{CNN_DIR}/best_cnn.pt")
print(f"\nFine-tuned model saved → {CNN_DIR}/best_cnn.pt")
print("Now run: python3 predict_photo.py")