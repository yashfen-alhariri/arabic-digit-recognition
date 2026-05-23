import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os

os.makedirs("comparison", exist_ok=True)

y_test     = np.load("outputs/y_test.npy")
y_pred_cnn = np.load("outputs/y_pred_cnn.npy")
y_pred_svm = np.load("outputs/y_pred_svm.npy")

f1_cnn = f1_score(y_test, y_pred_cnn, average=None)
f1_svm = f1_score(y_test, y_pred_svm, average=None)

x = range(10)
plt.figure(figsize=(12, 5))
plt.bar([i - 0.2 for i in x], f1_cnn, width=0.4, label="CNN", color="steelblue")
plt.bar([i + 0.2 for i in x], f1_svm, width=0.4, label="SVM", color="darkorange")
plt.xticks(range(10), [str(i) for i in range(10)])
plt.ylim(0.85, 1.01)
plt.legend()
plt.xlabel("Digit Class")
plt.ylabel("F1 Score")
plt.title("F1 Score per Digit Class — CNN vs SVM")
plt.tight_layout()
plt.savefig("comparison/f1_per_class.png", dpi=120)
plt.close()

print("CNN vs SVM Comparison")
print("-" * 50)
print(f"{'Digit':<10} {'CNN F1':>10} {'SVM F1':>10}")
for i in range(10):
    print(f"{i:<10} {f1_cnn[i]:>10.4f} {f1_svm[i]:>10.4f}")
print(f"\n{'Macro avg':<10} {f1_cnn.mean():>10.4f} {f1_svm.mean():>10.4f}")
print("\nChart saved to comparison/f1_per_class.png")
