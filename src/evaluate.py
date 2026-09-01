import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import tf_preprocess_image, EMOTIONS, NUM_CLASSES

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FER_TEST  = PROJECT_ROOT / "data" / "fer2013" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "emotion_model.keras"
LABEL_PATH = PROJECT_ROOT / "models" / "label_map.json"
PLOTS_DIR  = PROJECT_ROOT / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  EmoSense AI  -  FER2013 Evaluation Pipeline")
print("=" * 65)

# ── Load Label Map ───────────────────────────────────────────────
with open(LABEL_PATH, "r") as f:
    label_map_json = json.load(f)
    # The JSON maps string -> idx. We need to ensure EMOTIONS list matches the indices.
    LABEL_MAP = {k: v for k, v in label_map_json.items()}
    # Re-order EMOTIONS based on LABEL_MAP to be perfectly safe
    ordered_emotions = [""] * NUM_CLASSES
    for name, idx in LABEL_MAP.items():
        ordered_emotions[idx] = name

# ── Load Test Dataset ────────────────────────────────────────────
def get_test_paths():
    paths = []
    labels = []
    for emotion, label_idx in LABEL_MAP.items():
        emotion_dir = FER_TEST / emotion.lower()
        if not emotion_dir.exists():
            continue
        for ext in ["*.jpg", "*.png"]:
            for img_path in emotion_dir.glob(ext):
                paths.append(str(img_path))
                labels.append(label_idx)
    return np.array(paths), np.array(labels, dtype=np.int32)

test_paths, test_labels = get_test_paths()
print(f"Loaded {len(test_paths)} test samples.")

ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
ds = ds.map(tf_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.map(lambda i, l: (i, tf.one_hot(l, NUM_CLASSES)), num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)

# ── Evaluate Model ───────────────────────────────────────────────
print(f"\nLoading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(str(MODEL_PATH))

print("\nRunning Evaluation on Untouched Test Set...")
loss, accuracy = model.evaluate(ds)
print(f"\n[RESULTS] Test Loss: {loss:.4f}")
print(f"[RESULTS] Test Accuracy: {accuracy*100:.2f}%")

# ── Generate Confusion Matrix & Report ───────────────────────────
print("\nGenerating Predictions for Classification Report...")
y_true = test_labels
y_pred_probs = model.predict(ds)
y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(y_true, y_pred, target_names=ordered_emotions)
print("\nClassification Report:")
print(report)

with open(PLOTS_DIR / "classification_report.txt", "w") as f:
    f.write(report)
    f.write(f"\nTest Accuracy: {accuracy*100:.2f}%\n")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ordered_emotions, yticklabels=ordered_emotions)
plt.title("Confusion Matrix (FER-2013 Test Set)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig(PLOTS_DIR / "confusion_matrix.png")
plt.close()

print(f"\nPlots and reports saved to {PLOTS_DIR}")
