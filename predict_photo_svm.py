import matplotlib
matplotlib.use('Agg')

import numpy as np
import joblib
import cv2
from PIL import Image
from pillow_heif import register_heif_opener
import sys

register_heif_opener()

# Load saved model and PCA
svm = joblib.load("svm_results/svm_model.pkl")
pca = joblib.load("outputs/pca_model.pkl")

def preprocess(path):
    img = Image.open(path).convert('L')
    img = np.array(img)

    # Resize if too large
    h, w = img.shape
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    h, w = img.shape

    # Crop center 60%
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    img = img[y1:y2, x1:x2]

    # Blur + adaptive threshold
    img = cv2.GaussianBlur(img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(img, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 10)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels < 2:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + np.argmax(areas)

    clean = np.zeros_like(binary)
    clean[labels == best_label] = 255

    # Crop to bounding box + padding
    x = stats[best_label, cv2.CC_STAT_LEFT]
    y = stats[best_label, cv2.CC_STAT_TOP]
    bw = stats[best_label, cv2.CC_STAT_WIDTH]
    bh = stats[best_label, cv2.CC_STAT_HEIGHT]
    cropped = clean[y:y+bh, x:x+bw]

    pad = int(max(bw, bh) * 0.3)
    cropped = cv2.copyMakeBorder(cropped, pad, pad, pad, pad,
                                  cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
    return resized.astype("float32") / 255.0

# Get image path
image_path = sys.argv[1] if len(sys.argv) > 1 else "IMG_9430.HEIC"

print(f"Testing image: {image_path}")
img = preprocess(image_path)

if img is None:
    print("Could not find digit in image.")
else:
    flattened = img.flatten().reshape(1, -1)
    pca_features = pca.transform(flattened)
    prediction = svm.predict(pca_features)[0]
    print(f"Predicted digit: {prediction}")