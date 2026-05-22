import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

OUT_DIR = "outputs"
SVM_DIR = "svm_results"
os.makedirs(SVM_DIR, exist_ok=True)

# Load data
X_train = np.load(f"{OUT_DIR}/X_train_svm.npy")
X_val   = np.load(f"{OUT_DIR}/X_val_svm.npy")
X_test  = np.load(f"{OUT_DIR}/X_test_svm.npy")
y_train = np.load(f"{OUT_DIR}/y_train.npy")
y_val   = np.load(f"{OUT_DIR}/y_val.npy")
y_test  = np.load(f"{OUT_DIR}/y_test.npy")

# Verify shapes
print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)
print("y_train:", y_train.shape)
print("y_val:  ", y_val.shape)
print("y_test: ", y_test.shape)
print("Classes:", np.unique(y_train))
print("Data loaded successfully!")

from sklearn.svm import SVC
import time

print("\nTraining SVM...")
train_start = time.time()
svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm.fit(X_train, y_train)
train_time = time.time() - train_start

val_acc = svm.score(X_val, y_val) * 100
print(f"Training time: {train_time:.1f} seconds")
print(f"Val accuracy:  {val_acc:.2f}%")


from sklearn.metrics import classification_report, confusion_matrix

print("\nEvaluating on test set...")
inf_start = time.time()
y_pred_svm = svm.predict(X_test)
inf_time = (time.time() - inf_start) / len(X_test) * 1000

test_acc = (y_pred_svm == y_test).mean() * 100
print(f"Test accuracy:  {test_acc:.2f}%")
print(f"Inference time: {inf_time:.4f} ms/image")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

import joblib

print("\nSaving model and predictions...")
joblib.dump(svm, f"{SVM_DIR}/svm_model.pkl")
np.save(f"{OUT_DIR}/y_pred_svm.npy", y_pred_svm)
print(f"Model saved to {SVM_DIR}/svm_model.pkl")
print(f"Predictions saved to {OUT_DIR}/y_pred_svm.npy")


import matplotlib.pyplot as plt
import seaborn as sns

print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=range(10), yticklabels=range(10))
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{SVM_DIR}/confusion_matrix.png", dpi=120)
plt.close()
print(f"Confusion matrix saved to {SVM_DIR}/confusion_matrix.png")