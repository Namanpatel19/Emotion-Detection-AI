"""
train.py
--------
Entry point for training the emotion-recognition CNN.

What this script does:
  1. Loads the FER-2013 dataset via data_preprocessing.py
  2. Builds the CNN from model.py
  3. Sets up smart training callbacks:
       • ModelCheckpoint — saves the best model weights automatically
       • EarlyStopping   — stops training when val_loss stops improving
       • ReduceLROnPlateau — lowers the learning rate when training stalls
  4. Trains the model and saves it to models/emotion_model.h5
  5. Plots training/validation accuracy and loss curves, saves to plots/

Usage
─────
  cd "AI project"
  python src/train.py

Optional flags (edit CONSTANTS section below):
  EPOCHS      — maximum training epochs (default 50, EarlyStopping may stop earlier)
  BATCH_SIZE  — mini-batch size (default 64)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Allow importing sibling modules regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import get_data_generators, EMOTION_LABELS, EMOTION_DISPLAY
from model import build_emotion_cnn

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

# ── Constants — tweak these if needed ────────────────────────────────────────
EPOCHS      = 50        # Max epochs; EarlyStopping will likely halt earlier
BATCH_SIZE  = 64        # Reduce to 32 if you run out of GPU/CPU memory
RANDOM_SEED = 42

# Output paths (relative to project root)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
MODEL_DIR    = os.path.join(PROJECT_ROOT, 'models')
PLOT_DIR     = os.path.join(PROJECT_ROOT, 'plots')
MODEL_PATH   = os.path.join(MODEL_DIR, 'emotion_model.h5')


def setup_callbacks(model_path: str):
    """
    Return a list of Keras callbacks that help training automatically.

    ModelCheckpoint  → saves model only when val_accuracy improves
    EarlyStopping    → stops after 10 epochs of no val_loss improvement
    ReduceLROnPlateau → halves the learning rate after 5 plateau epochs
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,         # only save when accuracy improves
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,                 # stop if val_loss doesn't improve for 10 epochs
            restore_best_weights=True,   # roll back to the best checkpoint
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,                  # multiply LR by 0.5 when plateau detected
            patience=5,                  # wait 5 epochs before reducing
            min_lr=1e-6,
            verbose=1
        ),
    ]
    return callbacks


def plot_history(history, save_dir: str):
    """
    Generate and save training/validation accuracy & loss curves.

    Parameters
    ----------
    history  : tf.keras.callbacks.History object returned by model.fit()
    save_dir : directory to save the PNG file
    """
    os.makedirs(save_dir, exist_ok=True)

    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs   = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History — Emotion CNN', fontsize=14, fontweight='bold')

    # ── Accuracy subplot ──
    axes[0].plot(epochs, acc,     label='Train Accuracy', color='steelblue',   linewidth=2)
    axes[0].plot(epochs, val_acc, label='Val Accuracy',   color='darkorange',  linewidth=2, linestyle='--')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1])

    # ── Loss subplot ──
    axes[1].plot(epochs, loss,     label='Train Loss', color='steelblue',  linewidth=2)
    axes[1].plot(epochs, val_loss, label='Val Loss',   color='darkorange', linewidth=2, linestyle='--')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n[INFO] Training curves saved to: {save_path}")
    plt.show()


def train():
    """Full training pipeline — call this to start training."""

    print("=" * 60)
    print("   Emotion Detection CNN — Training Script")
    print("=" * 60)

    # ── GPU check ──
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[INFO] GPU detected: {[g.name for g in gpus]}")
    else:
        print("[INFO] No GPU detected — training on CPU (this may be slow).")
        print("[TIP]  Use Google Colab for free GPU training.\n")

    # ── Load data ──
    print("\n[STEP 1/4] Loading and preprocessing FER-2013 dataset...")
    train_gen, val_gen, class_weights, num_classes, label_map = get_data_generators(
        batch_size=BATCH_SIZE
    )

    # Save the label map next to the model so inference can use it
    import json
    label_map_path = os.path.join(MODEL_DIR, 'label_map.json')
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"[INFO] Label map saved to: {label_map_path}")

    # ── Build model ──
    print("\n[STEP 2/4] Building CNN architecture...")
    model = build_emotion_cnn(num_classes=num_classes)
    model.summary()

    # ── Callbacks ──
    print("\n[STEP 3/4] Setting up training callbacks...")
    callbacks = setup_callbacks(MODEL_PATH)

    # ── Train ──
    print(f"\n[STEP 4/4] Training for up to {EPOCHS} epochs "
          f"(EarlyStopping may stop earlier)...")
    print(f"           Model will be saved to: {MODEL_PATH}\n")

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        validation_steps=len(val_gen),
        callbacks=callbacks,
        class_weight=class_weights,   # handle class imbalance
        verbose=1
    )

    # ── Final metrics ──
    best_val_acc  = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Best val_accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.1f}%)")
    print(f"  Best val_loss     : {best_val_loss:.4f}")
    print(f"  Model saved to    : {MODEL_PATH}")
    print("=" * 60)

    # ── Plot curves ──
    plot_history(history, PLOT_DIR)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train()
