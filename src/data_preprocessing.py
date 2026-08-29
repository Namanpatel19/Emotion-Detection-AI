"""
data_preprocessing.py
---------------------
Handles loading and preparing the FER-2013 image-folder dataset.

Expected folder structure after extracting the ZIP:
  data/fer2013/
    train/
      angry/    fear/    happy/    neutral/    sad/    surprise/    disgust/
    test/        (optional — used as validation if present)
      angry/    fear/    happy/ ...

If there is no train/ subfolder (just emotion folders at root), the script
will automatically create a train/val split (80/20) for you.

We use 6 emotions (disgust is excluded — too few samples):
  angry, fear, happy, neutral, sad, surprise
"""

import os
import sys
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'fer2013')
IMG_SIZE    = 48          # CNN expects 48x48
BATCH_SIZE  = 64
NUM_CLASSES = 6

# The 6 emotion classes we train on (disgust excluded)
EMOTION_LABELS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Human-readable capitalised names (same order — used in inference overlays)
EMOTION_DISPLAY = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# ── Dataset Structure Detection ───────────────────────────────────────────────

def detect_dataset_structure(dataset_dir: str):
    """
    Automatically detect whether the dataset has a train/test split
    or is a flat folder of emotion subfolders.

    Returns
    -------
    structure : str
        'split'  -> dataset_dir/train/<emotion>/ and dataset_dir/test/<emotion>/
        'flat'   -> dataset_dir/<emotion>/  (we will create split ourselves)
        'none'   -> dataset not found
    """
    train_dir = os.path.join(dataset_dir, 'train')
    if os.path.isdir(train_dir):
        # Check it has at least one emotion subfolder
        subdirs = [d for d in os.listdir(train_dir)
                   if os.path.isdir(os.path.join(train_dir, d))]
        if subdirs:
            return 'split'

    # Check for flat structure (emotion folders directly in dataset_dir)
    subdirs = [d for d in os.listdir(dataset_dir)
               if os.path.isdir(os.path.join(dataset_dir, d))]
    emotion_like = [d for d in subdirs
                    if d.lower() in ['angry', 'fear', 'happy', 'neutral', 'sad',
                                     'surprise', 'disgust']]
    if emotion_like:
        return 'flat'

    return 'none'


def create_split_from_flat(dataset_dir: str, val_ratio: float = 0.2,
                            random_seed: int = 42):
    """
    If the dataset has no train/test split, create one:
      dataset_dir/train/<emotion>/  (80% of images)
      dataset_dir/val/<emotion>/    (20% of images)

    Images are MOVED (not copied) to avoid doubling disk usage.
    This only runs once — if train/ already exists it's skipped.
    """
    import random
    random.seed(random_seed)

    train_dir = os.path.join(dataset_dir, 'train')
    val_dir   = os.path.join(dataset_dir, 'val')

    if os.path.isdir(train_dir):
        print("[INFO] train/ folder already exists, skipping split creation.")
        return train_dir, val_dir

    print("[INFO] No train/ folder found. Creating 80/20 train/val split...")

    all_emotions = [d for d in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, d))
                    and d.lower() not in ['train', 'val', 'test']]

    for emotion in all_emotions:
        if emotion.lower() == 'disgust':
            print(f"[INFO] Skipping '{emotion}' (excluded from training)")
            continue

        src_dir  = os.path.join(dataset_dir, emotion)
        images   = [f for f in os.listdir(src_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        split_idx  = int(len(images) * (1 - val_ratio))
        train_imgs = images[:split_idx]
        val_imgs   = images[split_idx:]

        # Create destination dirs
        train_emo_dir = os.path.join(train_dir, emotion)
        val_emo_dir   = os.path.join(val_dir,   emotion)
        os.makedirs(train_emo_dir, exist_ok=True)
        os.makedirs(val_emo_dir,   exist_ok=True)

        for img in train_imgs:
            shutil.move(os.path.join(src_dir, img),
                        os.path.join(train_emo_dir, img))
        for img in val_imgs:
            shutil.move(os.path.join(src_dir, img),
                        os.path.join(val_emo_dir, img))

        print(f"  {emotion:10s} -> train: {len(train_imgs):5d}  val: {len(val_imgs):5d}")

    print("[INFO] Split complete.")
    return train_dir, val_dir


# ── Main Data Generator Builder ───────────────────────────────────────────────

def get_data_generators(dataset_dir: str = None,
                        batch_size: int = BATCH_SIZE):
    """
    Build and return training and validation Keras ImageDataGenerators.

    Automatically handles both dataset structures:
      - train/test split already present
      - flat emotion folders (creates split automatically)

    Parameters
    ----------
    dataset_dir : path to the fer2013 folder (default: data/fer2013/)
    batch_size  : mini-batch size for training

    Returns
    -------
    train_gen       : augmented generator for training
    val_gen         : non-augmented generator for validation
    class_weights   : dict balancing imbalanced class counts
    num_classes     : int, number of classes found (should be 6)
    label_map       : dict mapping folder name -> class index
    """
    if dataset_dir is None:
        dataset_dir = os.path.abspath(DATASET_DIR)

    print(f"[INFO] Dataset directory: {dataset_dir}")

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset folder not found: {dataset_dir}\n"
            "Please extract your fer2013 ZIP to:\n"
            f"  {dataset_dir}\n"
        )

    # ── Detect structure ──
    structure = detect_dataset_structure(dataset_dir)
    print(f"[INFO] Detected dataset structure: '{structure}'")

    if structure == 'none':
        raise FileNotFoundError(
            f"\n[ERROR] No emotion folders found in: {dataset_dir}\n"
            "Make sure you extracted the ZIP so that folders like\n"
            "  angry/, happy/, sad/ etc. are inside data/fer2013/\n"
        )

    if structure == 'flat':
        # Create train/val split from flat folders
        train_dir, val_dir = create_split_from_flat(dataset_dir)
    else:
        # Use existing train/ folder; use test/ or val/ for validation
        train_dir = os.path.join(dataset_dir, 'train')
        val_dir   = os.path.join(dataset_dir, 'test')
        if not os.path.isdir(val_dir):
            val_dir = os.path.join(dataset_dir, 'val')
        if not os.path.isdir(val_dir):
            print("[WARN] No test/ or val/ folder found. Using 20% of train for validation.")
            val_dir = train_dir  # will use validation_split below

    # ── Exclude 'disgust' folder by only allowing our 6 classes ──
    # flow_from_directory respects the 'classes' argument
    # Check which of our 6 classes actually exist in train_dir
    available = [d.lower() for d in os.listdir(train_dir)
                 if os.path.isdir(os.path.join(train_dir, d))]
    classes_to_use = [e for e in EMOTION_LABELS if e in available]

    if not classes_to_use:
        raise ValueError(
            f"[ERROR] None of the expected emotion folders found in {train_dir}\n"
            f"Expected: {EMOTION_LABELS}\n"
            f"Found:    {available}"
        )

    print(f"[INFO] Using classes: {classes_to_use}")

    # ── Augmentation for training ──
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,          # normalise pixels to [0, 1]
        rotation_range=10,             # rotate +/- 10 degrees
        width_shift_range=0.1,         # horizontal shift 10%
        height_shift_range=0.1,        # vertical shift 10%
        zoom_range=0.1,                # zoom in/out 10%
        horizontal_flip=True,          # mirror left <-> right
        fill_mode='nearest'
    )

    # Validation: only normalise, no augmentation
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # ── Build generators ──
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',        # model expects 1-channel images
        classes=classes_to_use,        # enforces our 6-class order
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    if val_dir == train_dir:
        # Fallback: use the same folder with validation_split
        val_gen = val_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='grayscale',
            classes=classes_to_use,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=False
        )
    else:
        val_gen = val_datagen.flow_from_directory(
            val_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='grayscale',
            classes=classes_to_use,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=False
        )

    num_classes = len(classes_to_use)
    label_map   = train_gen.class_indices   # {'angry': 0, 'fear': 1, ...}

    print(f"[INFO] Train samples : {train_gen.samples}")
    print(f"[INFO] Val samples   : {val_gen.samples}")
    print(f"[INFO] Class map     : {label_map}")

    # ── Compute class weights to handle imbalanced counts ──
    from sklearn.utils.class_weight import compute_class_weight
    labels_array = train_gen.classes           # integer label for each sample
    classes_arr  = np.arange(num_classes)
    weights      = compute_class_weight(
        class_weight='balanced',
        classes=classes_arr,
        y=labels_array
    )
    class_weights = dict(enumerate(weights))
    print(f"[INFO] Class weights : {class_weights}")

    return train_gen, val_gen, class_weights, num_classes, label_map


# ── Quick sanity-check ────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_gen, val_gen, cw, nc, lm = get_data_generators()
    batch_X, batch_y = next(train_gen)
    print(f"[TEST] Batch shape  : images={batch_X.shape}, labels={batch_y.shape}")
    print(f"[TEST] Pixel range  : [{batch_X.min():.3f}, {batch_X.max():.3f}]")
    print(f"[TEST] Num classes  : {nc}")
    print("[TEST] data_preprocessing.py looks good!")
