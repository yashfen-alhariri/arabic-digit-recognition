import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os, time

# ── setup ────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

OUT_DIR = "outputs"
CNN_DIR = "cnn_results"
os.makedirs(CNN_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS     = 50
LR         = 0.001
PATIENCE   = 5
DEVICE     = torch.device("cpu")  # Intel Mac — no GPU
print(f"Device: {DEVICE}")

# ── load data ────────────────────────────────────────────────────────────────
print("\nLoading data...")
X_train = np.load(f"{OUT_DIR}/X_train_cnn.npy")  # (49000, 28, 28, 1)
X_val   = np.load(f"{OUT_DIR}/X_val_cnn.npy")
X_test  = np.load(f"{OUT_DIR}/X_test_cnn.npy")
y_train = np.load(f"{OUT_DIR}/y_train.npy")
y_val   = np.load(f"{OUT_DIR}/y_val.npy")
y_test  = np.load(f"{OUT_DIR}/y_test.npy")

print(f"Train : {X_train.shape} | Val : {X_val.shape} | Test : {X_test.shape}")

# ── convert to PyTorch tensors (N, 1, 28, 28) ────────────────────────────────
def to_tensor(X, y):
    X_t = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)  # (N,1,28,28)
    y_t = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_t, y_t)

train_loader = DataLoader(to_tensor(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(to_tensor(X_val,   y_val),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(to_tensor(X_test,  y_test),  batch_size=BATCH_SIZE)

# ── CNN architecture ─────────────────────────────────────────────────────────
class ArabicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (N, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # (N, 32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (N, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # (N, 64, 7, 7)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# (N, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = ArabicCNN().to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── training ─────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc  = 0.0
patience_count = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

print("\nTraining...\n")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    # train
    model.train()
    train_loss, train_correct = 0.0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * len(X_batch)
        train_correct += (output.argmax(1) == y_batch).sum().item()

    # validate
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            output    = model(X_batch)
            val_loss += criterion(output, y_batch).item() * len(X_batch)
            val_correct += (output.argmax(1) == y_batch).sum().item()

    # metrics
    tl = train_loss / len(train_loader.dataset)
    ta = train_correct / len(train_loader.dataset)
    vl = val_loss / len(val_loader.dataset)
    va = val_correct / len(val_loader.dataset)
    history['train_loss'].append(tl)
    history['val_loss'].append(vl)
    history['train_acc'].append(ta)
    history['val_acc'].append(va)

    print(f"Epoch {epoch:02d}/{EPOCHS} | loss {tl:.4f} | acc {ta:.4f} | val_loss {vl:.4f} | val_acc {va:.4f}")

    # early stopping + save best
    if va > best_val_acc:
        best_val_acc = va
        patience_count = 0
        torch.save(model.state_dict(), f"{CNN_DIR}/best_cnn.pt")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

training_time = time.time() - start_time
print(f"\nTraining complete in {training_time:.1f}s | Best val acc: {best_val_acc:.4f}")

# ── training curves ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_loss'], label='Train loss')
ax1.plot(history['val_loss'],   label='Val loss')
ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
ax2.plot(history['train_acc'], label='Train acc')
ax2.plot(history['val_acc'],   label='Val acc')
ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
plt.tight_layout()
plt.savefig(f"{CNN_DIR}/training_curves.png", dpi=120)
plt.close()
print(f"Saved → {CNN_DIR}/training_curves.png")

# ── evaluate on test set ─────────────────────────────────────────────────────
model.load_state_dict(torch.load(f"{CNN_DIR}/best_cnn.pt"))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
test_acc   = (all_preds == all_labels).mean()
print(f"\nTest accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print("\nClassification report:")
print(classification_report(all_labels, all_preds))

# ── confusion matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title(f'CNN Confusion Matrix (Test Acc: {test_acc*100:.2f}%)')
plt.tight_layout()
plt.savefig(f"{CNN_DIR}/confusion_matrix.png", dpi=120)
plt.close()
print(f"Saved → {CNN_DIR}/confusion_matrix.png")

# ── misclassification examples ───────────────────────────────────────────────
wrong_idx = np.where(all_preds != all_labels)[0][:10]
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, idx in enumerate(wrong_idx):
    ax = axes[i // 5, i % 5]
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"True:{all_labels[idx]} Pred:{all_preds[idx]}", fontsize=9, color='red')
    ax.axis('off')
plt.suptitle('Misclassified examples', fontsize=12)
plt.tight_layout()
plt.savefig(f"{CNN_DIR}/misclassified.png", dpi=120)
plt.close()
print(f"Saved → {CNN_DIR}/misclassified.png")

# ── save predictions for Role 3 ──────────────────────────────────────────────
np.save(f"{OUT_DIR}/y_pred_cnn.npy", all_preds)
print(f"Saved → {OUT_DIR}/y_pred_cnn.npy")

# ── save notes ───────────────────────────────────────────────────────────────
model_size = os.path.getsize(f"{CNN_DIR}/best_cnn.pt") / (1024*1024)
notes = f"""CNN Results
===========
Test accuracy  : {test_acc*100:.2f}%
Parameters     : {sum(p.numel() for p in model.parameters()):,}
Training time  : {training_time:.1f}s
Best val acc   : {best_val_acc*100:.2f}%
Model size     : {model_size:.1f} MB
Architecture   : Conv2D(32) -> Conv2D(64) -> Conv2D(128) -> Dense(256) -> Dense(10)
Extras         : BatchNorm after each conv, Dropout(0.5) before output
"""
with open(f"{CNN_DIR}/cnn_notes.txt", "w") as f:
    f.write(notes)
print(f"Saved → {CNN_DIR}/cnn_notes.txt")
print("\nRole 2 complete.")