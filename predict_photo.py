import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pillow_heif import register_heif_opener
import os, glob

register_heif_opener()  # enables PIL to open HEIC

# -- model ------------------------------------------------------------------
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
model.load_state_dict(torch.load("cnn_results/best_cnn.pt"))
model.eval()

# -- preprocessing: photo -> 28x28 like training data ----------------------
def preprocess(path):
    img = Image.open(path).convert('L')
    img = np.array(img)

    # resize first
    h, w = img.shape
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    h, w = img.shape

    # crop center 60% of the image - digit is always roughly centered
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    img = img[y1:y2, x1:x2]

    # blur + adaptive threshold (handles uneven lighting better than Otsu)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(img, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 10)

    # morphological closing to fill gaps in strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # find all components, pick largest (after center crop, digit dominates)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels < 2:
        return None

    # largest component by area (skip background label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + np.argmax(areas)

    clean = np.zeros_like(binary)
    clean[labels == best_label] = 255

    # crop to bounding box
    x = stats[best_label, cv2.CC_STAT_LEFT]
    y = stats[best_label, cv2.CC_STAT_TOP]
    bw = stats[best_label, cv2.CC_STAT_WIDTH]
    bh = stats[best_label, cv2.CC_STAT_HEIGHT]
    cropped = clean[y:y+bh, x:x+bw]

    # padding + resize
    pad = int(max(bw, bh) * 0.3)
    cropped = cv2.copyMakeBorder(cropped, pad, pad, pad, pad,
                                  cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
    return resized.astype("float32") / 255.0

# -- predict all photos -----------------------------------------------------
photos = sorted(glob.glob("IMG_*.HEIC") + glob.glob("IMG_*.jpg") + glob.glob("IMG_*.png"))
print(f"Found {len(photos)} photos: {photos}\n")

fig, axes = plt.subplots(2, len(photos), figsize=(3 * len(photos), 6))

for i, path in enumerate(photos):
    img = preprocess(path)
    if img is None:
        print(f"{path}: could not find digit")
        continue

    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
        conf   = probs[pred].item()

    print(f"{path} -> Predicted: {pred}  ({conf*100:.1f}% confidence)")

    # top row: preprocessed image
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title(f"Pred: {pred}\n{conf*100:.1f}%", fontsize=10,
                          color='green' if conf > 0.8 else 'orange')
    axes[0, i].axis('off')

    # bottom row: original photo
    orig = np.array(Image.open(path).convert('L'))
    axes[1, i].imshow(orig, cmap='gray')
    axes[1, i].set_title(os.path.basename(path), fontsize=7)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel("Preprocessed", fontsize=9)
axes[1, 0].set_ylabel("Original photo", fontsize=9)
plt.suptitle("Real photo predictions - Arabic digit CNN", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/real_photo_predictions.png", dpi=120)
plt.close()
print("\nSaved -> outputs/real_photo_predictions.png")
