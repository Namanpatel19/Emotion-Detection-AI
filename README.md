# 😊 Real-Time Human Emotion Detection

A student project that uses a **Convolutional Neural Network (CNN)** trained on the FER-2013 dataset to classify facial expressions from a live webcam feed in real time, powered by **OpenCV** and **TensorFlow/Keras**.

---

## 🎯 Detected Emotions

| Index | Emotion   | Colour on screen |
|-------|-----------|-----------------|
| 0     | 😠 Angry    | Red             |
| 1     | 😨 Fear     | Purple          |
| 2     | 😊 Happy    | Green           |
| 3     | 😢 Sad      | Blue            |
| 4     | 😲 Surprise | Orange          |
| 5     | 😐 Neutral  | Grey            |

---

## 📁 Project Structure

```
AI project/
├── data/
│   └── fer2013/
│       └── fer2013.csv          ← place the dataset here
├── models/
│   └── emotion_model.h5         ← saved after training
├── haarcascades/
│   └── haarcascade_frontalface_default.xml
├── plots/
│   └── training_curves.png      ← generated after training
├── screenshots/                 ← saved when you press 's' during inference
├── src/
│   ├── data_preprocessing.py    ← dataset loading + augmentation
│   ├── model.py                 ← CNN architecture
│   ├── train.py                 ← training loop
│   └── inference.py             ← real-time webcam demo
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone / Open the project

Make sure you are in the `AI project` folder in your terminal.

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on Mac / Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 **GPU users**: Install `tensorflow[and-cuda]` instead of plain `tensorflow` for NVIDIA GPU acceleration.

---

## 📥 Downloading the FER-2013 Dataset

The FER-2013 dataset is available on Kaggle. You have **two options**:

### Option A — Kaggle API (recommended)

1. Log in to [kaggle.com](https://www.kaggle.com) and go to **Account → API → Create New Token**.
2. This downloads a `kaggle.json` file. Place it at:
   - **Windows**: `C:\Users\YourName\.kaggle\kaggle.json`
   - **Mac/Linux**: `~/.kaggle/kaggle.json`
3. Run:
   ```bash
   kaggle datasets download -d msambare/fer2013 -p data/fer2013 --unzip
   ```

### Option B — Manual Download

1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Click **Download** (you need a free Kaggle account).
3. Unzip the file and copy `fer2013.csv` into `data/fer2013/`.

After either option, your file tree should look like:
```
data/
└── fer2013/
    └── fer2013.csv   ✅
```

---

## 🏋️ Training the Model

```bash
cd "AI project"
python src/train.py
```

**What happens:**
- Loads and preprocesses FER-2013 (drops "Disgust" class, normalises, augments)
- Trains the CNN for up to **50 epochs** (EarlyStopping halts early if val_loss plateaus)
- Saves the best model to `models/emotion_model.h5`
- Saves training curves to `plots/training_curves.png`

> ⏱️ **Training time:**
> - CPU (laptop): ~3–6 hours
> - GPU (RTX 3060+): ~15–30 minutes
> - Google Colab (free T4 GPU): ~20–40 minutes ← **recommended for students**

### Expected Results

A well-trained model should achieve roughly:
- **Training accuracy**: ~70–80%
- **Validation accuracy**: ~60–68%

> Note: FER-2013 is a challenging dataset — even state-of-the-art models cap around 72% on it.

---

## 🎥 Running the Live Demo

Make sure you have trained the model first, then:

```bash
# Default webcam (camera index 0)
python src/inference.py

# External USB webcam
python src/inference.py --camera 1

# Run on a video file instead
python src/inference.py --video path/to/my_video.mp4
```

### Controls during the demo

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Save a screenshot (saved to `screenshots/`) |

### What you'll see

- **Coloured bounding box** around each detected face
- **Emotion label + confidence %** above the box
- **Mini bar chart** (right of the box) showing probabilities for all 6 emotions
- **FPS counter** and face count in the top-left corner

---

## 🧠 CNN Architecture

```
Input: (48 × 48 × 1) grayscale face
   │
   ├─ Conv Block 1:  Conv2D(64)  × 2 → BatchNorm → MaxPool → Dropout(0.25)
   ├─ Conv Block 2:  Conv2D(128) × 2 → BatchNorm → MaxPool → Dropout(0.25)
   ├─ Conv Block 3:  Conv2D(256) × 2 → BatchNorm → MaxPool → Dropout(0.25)
   ├─ Conv Block 4:  Conv2D(512) × 2 → BatchNorm → MaxPool → Dropout(0.25)
   │
   ├─ Flatten
   ├─ Dense(256) → BatchNorm → Dropout(0.50)
   ├─ Dense(128) → BatchNorm → Dropout(0.50)
   └─ Dense(6, softmax)  ← emotion probabilities

Loss:      Categorical Cross-Entropy
Optimizer: Adam (lr=1e-3, with ReduceLROnPlateau)
```

---

## 🗂️ Module Reference

| File | Purpose |
|------|---------|
| [`src/data_preprocessing.py`](src/data_preprocessing.py) | Load FER-2013, split train/val, apply augmentation |
| [`src/model.py`](src/model.py) | Define and compile the CNN |
| [`src/train.py`](src/train.py) | Run training, save model, plot curves |
| [`src/inference.py`](src/inference.py) | Live webcam emotion detection |

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: fer2013.csv` | Follow the dataset download steps above |
| `FileNotFoundError: emotion_model.h5` | Run `python src/train.py` first |
| Webcam won't open | Try `--camera 1` or check if another app is using it |
| Very slow on CPU | Use Google Colab or a machine with a GPU |
| Low accuracy | Train for more epochs or reduce EarlyStopping patience |
| ImportError for tensorflow | Run `pip install -r requirements.txt` in your virtual env |

---

## 📚 References

- **FER-2013 Dataset**: https://www.kaggle.com/datasets/msambare/fer2013
- **OpenCV Haar Cascades**: https://github.com/opencv/opencv/tree/master/data/haarcascades
- **TensorFlow / Keras Docs**: https://www.tensorflow.org/api_docs

---

*Built as a student AI project. Feel free to extend — try DNN-based face detection, add more emotions, or experiment with MobileNet transfer learning!*
