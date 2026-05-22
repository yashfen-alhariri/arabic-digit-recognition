import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── copy the model class (must match cnn_model.py exactly) ───────────────────
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

# ── load model ────────────────────────────────────────────────────────────────
model = ArabicCNN()
model.load_state_dict(torch.load("cnn_results/best_cnn.pt"))
model.eval()
print("Model loaded successfully.")

# ── test on 10 random samples from test set (one per digit) ──────────────────
X_test = np.load("outputs/X_test_cnn.npy")
y_test = np.load("outputs/y_test.npy")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for digit in range(10):
    idx = np.where(y_test == digit)[0][0]
    img = X_test[idx]                                           # (28,28,1)
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)   # (1,1,28,28)

    with torch.no_grad():
        logits = model(tensor)
        pred   = logits.argmax(1).item()
        conf   = torch.softmax(logits, dim=1)[0][pred].item()

    ax = axes[digit // 5, digit % 5]
    ax.imshow(img.reshape(28, 28), cmap='gray')
    color = 'green' if pred == digit else 'red'
    ax.set_title(f"True:{digit}  Pred:{pred}\n{conf*100:.1f}%", color=color, fontsize=9)
    ax.axis('off')

plt.suptitle('CNN test — one sample per digit class', fontsize=12)
plt.tight_layout()
plt.savefig("outputs/cnn_test_result.png", dpi=120)
plt.close()
print("Saved → outputs/cnn_test_result.png")