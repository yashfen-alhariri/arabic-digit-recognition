import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, time, os

OUT_DIR = "outputs"
SVM_DIR = "svm_results"
os.makedirs(SVM_DIR, exist_ok=True)

# --- 1. Load data ---
print("Loading data...")
X_train = np.load(f"{OUT_DIR}/X_train_svm.npy")
X_val   = np.load(f"{OUT_DIR}/X_val_svm.npy")
X_test  = np.load(f"{OUT_DIR}/X_test_svm.npy")
y_train = np.load(f"{OUT_DIR}/y_train.npy")
y_val   = np.load(f"{OUT_DIR}/y_val.npy")
y_test  = np.load(f"{OUT_DIR}/y_test.npy")
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# --- 2. Train SVM ---
print("Training SVM (RBF kernel, C=10)...")
start = time.time()
svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm.fit(X_train, y_train)
train_time = time.time() - start
print(f"Training done in {train_time:.1f}s")

# --- 3. Validate ---
val_acc = svm.score(X_val, y_val)
print(f"Val accuracy: {val_acc*100:.2f}%")

# --- 4. Test ---
start = time.time()
y_pred = svm.predict(X_test)
inf_time = (time.time() - start) / len(X_test) * 1000
test_acc = (y_pred == y_test).mean()
print(f"Test accuracy: {test_acc*100:.2f}%")
print(f"Inference: {inf_time:.4f} ms/image")
print()
print(classification_report(y_test, y_pred))

# --- 5. Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=range(10), yticklabels=range(10))
plt.title(f"SVM Confusion Matrix - Test Acc: {test_acc*100:.2f}%")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{SVM_DIR}/confusion_matrix.png", dpi=120)
plt.close()
print("Confusion matrix saved.")

# --- 6. Save model and predictions ---
joblib.dump(svm, f"{SVM_DIR}/svm_model.pkl")
np.save(f"{OUT_DIR}/y_pred_svm.npy", y_pred)
print("Model and predictions saved.")

# --- 7. Verify saves ---
svm_check = joblib.load(f"{SVM_DIR}/svm_model.pkl")
pred_check = np.load(f"{OUT_DIR}/y_pred_svm.npy")
assert len(pred_check) == len(y_test), "Prediction count mismatch!"
print("All saves verified OK.")