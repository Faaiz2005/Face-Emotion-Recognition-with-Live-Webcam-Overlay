# Face Emotion Recognition with Live Webcam Overlay

**Project Name:** Face Emotion Recognition with Live Webcam Overlay  
**Roll No.:** 23-AI-40  
**Dataset:** FER-2013 — 7 emotions, ~35,000 grayscale 48×48 images  
**Course:** Computer Vision Lab — Lab 14

---

## 📁 Project Structure

```
lab14-fer/
├── lab14_emotion_recognition.ipynb   # Full training pipeline (Google Colab)
├── main.py                 # Inference script — images + webcam (VS Code)
├── fixed_model.keras                 # Best trained model (download from Colab)
└── README.md                         # This file
```

---

## 🧪 Emotion Classes

`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

## ⚙️ Setup Instructions

### 1. Clone / Download the Repo

```bash
git clone https://github.com/Faaiz2005/Face-Emotion-Recognition-with-Live-Webcam-Overlay.git
cd Face-Emotion-Recognition-with-Live-Webcam-Overlay.git
```

### 2. Install Required Libraries

```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn pandas kagglehub
```

> **Note:** Tested with TensorFlow 2.x and OpenCV 4.x. Python 3.9+ recommended.

---

## 🚀 Part 1 — Train the Model (Google Colab)

Open `lab14_emotion_recognition.ipynb` in [Google Colab](https://colab.research.google.com/).

The notebook will:
1. Download FER-2013 via `kagglehub`
2. Preprocess images (grayscale → Gaussian Blur → CLAHE → resize → normalize)
3. Build and train 3 architectures:
   - **CNN from Scratch** (48×48 grayscale)
   - **MobileNetV2 Frozen** (96×96 RGB, feature extractor)
   - **MobileNetV2 Fine-tuned** (96×96 RGB, last 30 layers unfrozen)
   - **VGG-Style Double Conv Block** (48×48 grayscale)
4. Evaluate each model with confusion matrix and per-class F1-score
5. Save the best model as `fixed_model.keras`

After training, download `fixed_model.keras` from Colab and place it in the project root.

---

## 🖥️ Part 2 — Run Inference (VS Code / Local)

Make sure `fixed_model.keras` is in the same folder as `main.py`.

### 🎥 Live Webcam Mode

```bash
python main.py --webcam
```

- Detects faces using Haar Cascade (`haarcascade_frontalface_default.xml`)
- Classifies emotion per face with confidence bar overlay
- Shows stable prediction using a rolling `deque(maxlen=20)`
- Displays real-time FPS on screen
- Press **Q** to quit and save a 30-second demo clip as `demo_clip.avi`

### 🖼️ Single Image Mode

```bash
python main.py --images sad.jpg
```

### OUTPUT

# Happy Test
![Happy](https://raw.githubusercontent.com/Faaiz2005/Face-Emotion-Recognition-with-Live-Webcam-Overlay/main/annotated_Happy.jpg)

# Sad Test
![Sad](https://raw.githubusercontent.com/Faaiz2005/Face-Emotion-Recognition-with-Live-Webcam-Overlay/main/Sad_test.png)

### 🔁 Custom Model Path

```bash
python test_on_images.py --model path/to/your_model.keras --webcam
```

---

## 🏗️ Model Architectures

| Model | Input | Params (approx) | Notes |
|---|---|---|---|
| CNN Scratch | 48×48 grayscale | ~1.2M | 3 double-conv blocks + GAP |
| MobileNetV2 Frozen | 96×96 RGB | ~2.3M trainable | Backbone frozen, only head trained |
| MobileNetV2 Fine-tuned | 96×96 RGB | ~3.4M trainable | Last 30 backbone layers unfrozen |
| VGG-Style | 48×48 grayscale | ~1.8M | VGG-inspired double conv + BN |

---

## 📊 Preprocessing Pipeline

Each face crop goes through:

1. BGR → Grayscale
2. Gaussian Blur (3×3 kernel)
3. CLAHE (clipLimit=2.0, tileGridSize=8×8)
4. Resize to 48×48 (or 96×96 for MobileNet)
5. Normalize to [0, 1]

---

## 📈 Evaluation Outputs

- Per-class F1-score (via `classification_report`)
- 7×7 Confusion matrix (saved as PNG)
- Model comparison table (accuracy, macro F1, parameters, training time)

---



## ⚠️ Known Limitations

- Haar Cascade may miss faces at extreme angles (use DNN-based detector for better results)
- `disgust` class is heavily underrepresented in FER-2013 (~550 samples vs 8000+ for happy)
- Inference speed on CPU is ~5–10 FPS depending on hardware

---

## 📚 References

- [FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- [MobileNetV2 Paper — Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
- OpenCV Haar Cascade — `haarcascade_frontalface_default.xml`
