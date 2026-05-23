import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
import cv2
from PIL import Image
from pillow_heif import register_heif_opener
import torch
import torch.nn as nn
import io
import base64

register_heif_opener()

app = Flask(__name__)

# ── Load SVM ──────────────────────────────────────────────────────────────────
svm = joblib.load("svm_results/svm_model.pkl")
pca = joblib.load("outputs/pca_model.pkl")

# ── Load CNN ──────────────────────────────────────────────────────────────────
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

cnn_model = ArabicCNN()
cnn_model.load_state_dict(torch.load("cnn_results/best_cnn.pt", map_location="cpu"))
cnn_model.eval()

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = np.array(img)

    h, w = img.shape
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    h, w = img.shape
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    img = img[y1:y2, x1:x2]

    img = cv2.GaussianBlur(img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(img, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels < 2:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + np.argmax(areas)
    clean = np.zeros_like(binary)
    clean[labels == best_label] = 255

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

# ── HTML Page ─────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Arabic Digit Recognition</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
        h1 { text-align: center; color: #333; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; justify-content: center; }
        .tab { padding: 10px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 15px; background: #ddd; }
        .tab.active { background: #4CAF50; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .box { background: white; padding: 30px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        button.predict-btn { background: #4CAF50; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px 5px; }
        button.predict-btn:hover { background: #45a049; }
        button.clear-btn { background: #f44336; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px 5px; }
        .results { display: flex; gap: 20px; margin-top: 30px; }
        .model-box { flex: 1; background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .model-box h2 { color: #555; }
        .digit { font-size: 80px; font-weight: bold; color: #2196F3; margin: 10px 0; }
        .label { font-size: 14px; color: #888; }
        canvas { border: 3px solid #333; border-radius: 8px; cursor: crosshair; background: black; }
        img#preview { max-width: 200px; border-radius: 8px; margin-top: 15px; }
        .loading { display: none; color: #888; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>🔢 Arabic Handwritten Digit Recognition</h1>

    <div class="tabs">
        <button class="tab active" onclick="switchTab('draw')">✏️ Draw Digit</button>
        <button class="tab" onclick="switchTab('upload')">📁 Upload Image</button>
    </div>

    <!-- Draw Tab -->
    <div id="draw-tab" class="tab-content active">
        <div class="box">
            <p>Draw an Arabic digit below (white on black)</p>
            <canvas id="drawCanvas" width="280" height="280"></canvas>
            <br>
            <button class="predict-btn" onclick="predictCanvas()">Predict</button>
            <button class="clear-btn" onclick="clearCanvas()">Clear</button>
            <p class="loading" id="loading-draw">Processing...</p>
        </div>
    </div>

    <!-- Upload Tab -->
    <div id="upload-tab" class="tab-content">
        <div class="box">
            <p>Upload a photo of a handwritten Arabic digit</p>
            <input type="file" id="imageInput" accept="image/*,.heic">
            <br><br>
            <button class="predict-btn" onclick="predictUpload()">Predict</button>
            <p class="loading" id="loading-upload">Processing...</p>
            <img id="preview" style="display:none">
        </div>
    </div>

    <!-- Results -->
    <div class="results" id="results" style="display:none">
        <div class="model-box">
            <h2>🧠 CNN</h2>
            <div class="digit" id="cnn_pred">-</div>
            <div class="label">Deep Learning Model</div>
            <div class="label">Accuracy: 99.18%</div>
        </div>
        <div class="model-box">
            <h2>📊 SVM</h2>
            <div class="digit" id="svm_pred">-</div>
            <div class="label">Classical ML Model</div>
            <div class="label">Accuracy: 94.82%</div>
        </div>
    </div>

    <script>
        // ── Tab switching ──
        function switchTab(tab) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tab + '-tab').classList.add('active');
            event.target.classList.add('active');
            document.getElementById('results').style.display = 'none';
        }

        // ── Drawing canvas ──
        const canvas = document.getElementById('drawCanvas');
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 18;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        let drawing = false;
        canvas.addEventListener('mousedown', e => { drawing = true; ctx.beginPath(); ctx.moveTo(e.offsetX, e.offsetY); });
        canvas.addEventListener('mousemove', e => { if (drawing) { ctx.lineTo(e.offsetX, e.offsetY); ctx.stroke(); } });
        canvas.addEventListener('mouseup', () => drawing = false);
        canvas.addEventListener('mouseleave', () => drawing = false);

        // Touch support
        canvas.addEventListener('touchstart', e => { e.preventDefault(); const t = e.touches[0]; const r = canvas.getBoundingClientRect(); drawing = true; ctx.beginPath(); ctx.moveTo(t.clientX - r.left, t.clientY - r.top); });
        canvas.addEventListener('touchmove', e => { e.preventDefault(); const t = e.touches[0]; const r = canvas.getBoundingClientRect(); if (drawing) { ctx.lineTo(t.clientX - r.left, t.clientY - r.top); ctx.stroke(); } });
        canvas.addEventListener('touchend', () => drawing = false);

        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('results').style.display = 'none';
        }

        function predictCanvas() {
            document.getElementById('loading-draw').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('image', blob, 'drawing.png');
                fetch('/predict', { method: 'POST', body: formData })
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('loading-draw').style.display = 'none';
                        document.getElementById('results').style.display = 'flex';
                        document.getElementById('cnn_pred').textContent = data.cnn;
                        document.getElementById('svm_pred').textContent = data.svm;
                    });
            });
        }

        // ── Upload ──
        function predictUpload() {
            const file = document.getElementById('imageInput').files[0];
            if (!file) { alert('Please select an image first'); return; }
            document.getElementById('loading-upload').style.display = 'block';
            document.getElementById('results').style.display = 'none';

            const reader = new FileReader();
            reader.onload = e => {
                document.getElementById('preview').src = e.target.result;
                document.getElementById('preview').style.display = 'block';
            };
            reader.readAsDataURL(file);

            const formData = new FormData();
            formData.append('image', file);
            fetch('/predict', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    document.getElementById('loading-upload').style.display = 'none';
                    document.getElementById('results').style.display = 'flex';
                    document.getElementById('cnn_pred').textContent = data.cnn;
                    document.getElementById('svm_pred').textContent = data.svm;
                });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image_bytes = file.read()

    img = preprocess(image_bytes)
    if img is None:
        return jsonify({'cnn': '?', 'svm': '?'})

    # CNN prediction
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = cnn_model(tensor)
        cnn_pred = logits.argmax(dim=1).item()

    # SVM prediction
    flattened = img.flatten().reshape(1, -1)
    pca_features = pca.transform(flattened)
    svm_pred = svm.predict(pca_features)[0]

    return jsonify({'cnn': int(cnn_pred), 'svm': int(svm_pred)})

if __name__ == '__main__':
    app.run(debug=True)