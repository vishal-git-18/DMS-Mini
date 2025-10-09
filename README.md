# Track B — DMS Mini

## 🧠 Project Overview

This repository implements an **end-to-end emotion classification pipeline** for driver monitoring systems, built completely from scratch.
It covers everything from dataset preparation and model training to evaluation, visualization, and lightweight deployment.
A simple **seat-belt detection module** (binary CNN) is also implemented as the bonus task.

---

## 🚀 Features & Deliverables

| Deliverable          | Description                                           | Status |
| -------------------- | ----------------------------------------------------- | ------ |
| ✅ Face detection     | Lightweight heuristic + cropping of facial ROI using Mediapipe | ✅ Done |
| ✅ Emotion classifier | CNN trained on a subset of FER2013                   | ✅ Done |
| ✅ Quantized model    | Exported to TFLite / ONNX with INT8 quantization     | ✅ Done |
| ✅ Evaluation metrics | Accuracy, confusion matrix, result visualization     | ✅ Done |
| ✅ Model analysis     | Latency, size, and inference performance noted       | ✅ Done |
| ✅ Bonus task         | Seat-belt ROI heuristic with binary CNN              | ✅ Done |
| ✅ Documentation      | README, `requirements.txt`, and result visualizations| ✅ Done |


---

## 🧩 Repository Structure

```

DMS-Mini/
│
├── data/
│   ├── train/               # Training images (emotion dataset)
│   └── test/                # Test images
│
├── data_cropped/            # Cropped faces using face detector
│   └── seatbelt/            # Bonus task dataset (seatbelt / without_seatbelt)
│
├── models/                  # Saved PyTorch models (.pth)
│
├── results/                 # Emotion model results (confusion matrix, accuracy plot)
│
├── seatbelt_results/        # Seatbelt model results
│
├── src/
│   ├── cnn.py               # Emotion CNN architecture
│   ├── face_detector.py     # Lightweight face detector
│   ├── train.py             # Emotion model training
│   ├── eval.py              # Emotion model evaluation + visualization
│   ├── inference.py         # Emotion inference on single image
│   │
│   └── seatbelt/            # Bonus task (seatbelt detection)
│       ├── seatbelt_cnn.py
│       ├── train_seatbelt.py
│       ├── eval_seatbelt.py
│       └── inference_seatbelt.py
│
├── requirements.txt
└── README.md

````

---

## ⚙️ Environment Setup

```bash
# 1️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate    # (Linux/macOS)
venv\Scripts\activate       # (Windows)

# 2️⃣ Install dependencies
pip install -r requirements.txt
````

---

## 📥 Dataset Links

* **FER2013 (Emotion dataset):** [Kaggle link](https://www.kaggle.com/datasets/msambare/fer2013?resource=download)
* **Seat-belt dataset:** [Kaggle link](https://www.kaggle.com/datasets/yehiahassanain/seat-belt2)

> Download datasets and save under the `data/` folder following this structure:
>
> ```
> data/train/<emotion_class>/
> data/test/<emotion_class>/
> data/seatbelt/train/seatbelt/
> data/seatbelt/train/without_seatbelt/
> data/seatbelt/test/seatbelt/
> data/seatbelt/test/without_seatbelt/
> ```

---

## 🏋️‍♂️ Emotion Classification

### 🧠 Train the model

```bash
python src/train.py
```

* Trains a CNN on cropped facial emotion images
* Saves the model as `models/emotion_model.pth`

---

### 🧪 Evaluate the model

```bash
python src/eval.py
```

* Loads the trained model
* Computes **accuracy** and **confusion matrix**
* Saves results in `results/`:

  * `confusion_matrix.png`
  * `accuracy_chart.png`

---

### 🖼️ Run inference

```bash
python src/inference.py --image "data/test/happy/image_01.jpg"
```

Outputs the **predicted emotion label** for the given image.

---

## 🎯 Bonus Task — Seat-belt Detection

A simple **seat-belt ROI heuristic** with a tiny binary CNN trained on a small dataset.

### Dataset Structure

```
data/
  └── seatbelt/
      ├── train/
      │   ├── seatbelt/
      │   └── without_seatbelt/
      └── test/
          ├── seatbelt/
          └── without_seatbelt/
```

### Train

```bash
python src/seatbelt/train_seatbelt.py
```

### Evaluate

```bash
python src/seatbelt/eval_seatbelt.py
```

* Generates accuracy & confusion matrix in `seatbelt_results/`
* Also generates bar chart of test accuracy

### Inference

```bash
python src/seatbelt/inference_seatbelt.py --image "data/seatbelt/test/seatbelt/image1.jpg"
```

---




## 📂 Results Summary


## 📊 Model Performance Metrics

| Model              | Test Accuracy | Avg. Latency per Frame | Model Size |
| ----------------- | ------------- | -------------------- | ---------- |
| Emotion Classifier | 56%           | 0.0055 s             | 4.80 MB    |
| Seatbelt Detector  | 99%           | 0.0045 s             | 1.20 MB    |



| Component      | Folder              | Output                           |
| -------------- | ------------------- | -------------------------------- |
| Emotion Model  | `results/`          | Confusion Matrix, Accuracy Chart |
| Seatbelt Model | `seatbelt_results/` | Confusion Matrix, Accuracy Chart |
| Models         | `models/`           | `.pth` weights (both models)     |


## 🧾 Notes

* Large datasets are **not committed** — use the Kaggle links above or download your own.
* The repository demonstrates a complete **end-to-end DMS mini-pipeline** with modular design.
* Both models are compact, CPU-friendly, and ready for quantization/deployment.
* All results are reproducible from scripts provided.



