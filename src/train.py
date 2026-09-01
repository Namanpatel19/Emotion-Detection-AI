import os
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from data_preprocessing import tf_preprocess_image, augment_image, EMOTIONS, NUM_CLASSES
from model import build_model, unfreeze_backbone

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FER_TRAIN = PROJECT_ROOT / "data" / "fer2013" / "train"
FER_TEST  = PROJECT_ROOT / "data" / "fer2013" / "test"
MODEL_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
LOGS_DIR  = PROJECT_ROOT / "training_logs"

MODEL_PATH = MODEL_DIR / "emotion_model.keras"
LABEL_PATH = MODEL_DIR / "label_map.json"

# Create directories
for d in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS_STAGE_1 = 15
EPOCHS_STAGE_2 = 25
VAL_SPLIT = 0.15
SEED = 42

LABEL_MAP = {e: i for i, e in enumerate(EMOTIONS)}

# ── Setup GPU & Mixed Precision ──────────────────────────────────
print("=" * 65)
print("  EmoSense AI  -  FER2013 Training Pipeline")
print("=" * 65)

print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {[g.name for g in gpus]}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
    # Mixed precision for faster training on modern GPUs
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
else:
    print("No GPU detected. Training will run on CPU.")

# ── Load Dataset Paths ───────────────────────────────────────────
def get_image_paths(dataset_dir):
    paths = []
    labels = []
    for emotion, label_idx in LABEL_MAP.items():
        emotion_dir = dataset_dir / emotion.lower()
        if not emotion_dir.exists():
            continue
        for ext in ["*.jpg", "*.png"]:
            for img_path in emotion_dir.glob(ext):
                paths.append(str(img_path))
                labels.append(label_idx)
    return np.array(paths), np.array(labels, dtype=np.int32)

print("\nScanning FER2013 dataset...")
train_all_paths, train_all_labels = get_image_paths(FER_TRAIN)
test_paths, test_labels = get_image_paths(FER_TEST)

# Stratified Train/Val Split from the train directory
idx = np.arange(len(train_all_paths))
train_idx, val_idx = train_test_split(idx, test_size=VAL_SPLIT, stratify=train_all_labels, random_state=SEED)

train_paths = train_all_paths[train_idx]
train_labels = train_all_labels[train_idx]
val_paths = train_all_paths[val_idx]
val_labels = train_all_labels[val_idx]

print(f"Input shape: (224, 224, 3)")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Train samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")
print(f"Test samples (untouched): {len(test_paths)}")

# ── Handle Class Imbalance ───────────────────────────────────────
cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train_labels)
class_weights = {i: float(v) for i, v in enumerate(cw)}

print("\nClass Distribution and Weights:")
for i, emotion in enumerate(EMOTIONS):
    count = np.sum(train_labels == i)
    print(f"  {emotion:10s} | Samples: {count:5d} | Weight: {class_weights[i]:.2f}")

with open(LABEL_PATH, "w") as f:
    json.dump(LABEL_MAP, f, indent=2)

# ── Build tf.data Pipeline ───────────────────────────────────────
def create_dataset(paths, labels, is_training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(len(paths), seed=SEED)
    
    ds = ds.map(tf_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
    ds = ds.map(lambda i, l: (i, tf.one_hot(l, NUM_CLASSES)), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(train_paths, train_labels, is_training=True)
val_ds = create_dataset(val_paths, val_labels, is_training=False)

# ── Build Model ──────────────────────────────────────────────────
model, backbone = build_model(NUM_CLASSES)
model.summary()

# ── Stage 1: Train Head ──────────────────────────────────────────
print("\n--- STAGE 1: Training Classification Head (Frozen Backbone) ---")
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_1 = [
    tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor="val_accuracy", mode="max", verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
]

start_time = time.time()
hist1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE_1,
    class_weight=class_weights,
    callbacks=callbacks_1
)

# ── Stage 2: Fine-Tuning ─────────────────────────────────────────
print("\n--- STAGE 2: Fine-Tuning Backbone ---")
# Unfreeze the top layers of EfficientNetB0 (e.g., leaving the first 100 layers frozen)
unfreeze_backbone(backbone, unfreeze_from_layer=150) # EfficientNetB0 has ~238 layers

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_2 = [
    tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor="val_accuracy", mode="max", verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

hist2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE_2,
    class_weight=class_weights,
    callbacks=callbacks_2
)
total_time = time.time() - start_time

# ── Plot Training History ────────────────────────────────────────
acc = hist1.history['accuracy'] + hist2.history['accuracy']
val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
loss = hist1.history['loss'] + hist2.history['loss']
val_loss = hist1.history['val_loss'] + hist2.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train')
plt.plot(val_acc, label='Validation')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Train')
plt.plot(val_loss, label='Validation')
plt.title('Loss')
plt.legend()
plt.savefig(PLOTS_DIR / 'training_history.png')
plt.close()

# ── Save Experiment Log ──────────────────────────────────────────
best_val_acc = max(val_acc)
log_data = {
    "architecture": "EfficientNetB0",
    "optimizer_stage1": "AdamW(1e-3)",
    "optimizer_stage2": "AdamW(1e-5)",
    "batch_size": BATCH_SIZE,
    "class_weights": class_weights,
    "best_val_accuracy": round(float(best_val_acc), 4),
    "total_training_time_seconds": round(total_time, 2)
}
log_file = LOGS_DIR / f"exp_{int(time.time())}.json"
with open(log_file, "w") as f:
    json.dump(log_data, f, indent=4)

print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"Model saved to: {MODEL_PATH}")
print(f"Run `python src/evaluate.py` to evaluate on the test set.")
