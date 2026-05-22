# Arabic Handwritten Digit Recognition

CNN and SVM models for recognizing handwritten Arabic digits (0-9).

## Datasets
- MADBase (Kaggle: mloey1/ahdd1) — 70,000 images
- Mendeley Arabic Numerals — 9,289 images
- Personal handwriting (augmented) — 1,590 images
- Total: 80,879 images

## Models
- CNN (PyTorch): 99.18% test accuracy
- SVM (sklearn): in progress

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn joblib torch torchvision opencv-python pillow-heif jupyter
```

## Run order
```bash
python3 data_pipeline.py       # Role 1 — preprocessing
python3 merge_and_train.py     # Role 2 — CNN training
python3 svm_model.py           # Role 3 — SVM
python3 predict_photo.py       # Test on real photos
```
