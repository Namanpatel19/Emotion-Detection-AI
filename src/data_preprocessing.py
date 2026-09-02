import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
NUM_CLASSES = 7
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def preprocess_face(img_bgr):
    """
    Standardized preprocessing for inference — matched exactly to training pipeline.
    Takes raw BGR face crop, converts to RGB, resizes, and casts to float32.
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    # Step 1: Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Step 2: Resize to 224x224 (EfficientNet input size) matching tf.image.resize lanczos3
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)

    # Step 3: EfficientNet expects inputs in [0, 255]
    return img_resized.astype(np.float32)

def tf_preprocess_image(file_path, label):
    """
    TensorFlow dataset mapping function for training/validation.
    """
    img = tf.io.read_file(file_path)
    # decode_image handles both jpg and png
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    
    # Ensure shape is defined after decode_image
    img.set_shape([None, None, 3])
    
    # Resize
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method='lanczos3')
    
    # EfficientNet expects inputs in [0, 255], tf.image.resize returns float32
    return img, label

# Instantiate globally to prevent tf.function from recreating variables
AUG_MODEL = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(factor=0.05), # +/- 18 degrees
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
])

def augment_image(img, label):
    """
    Applies augmentation and color jitter.
    """
    img = AUG_MODEL(img, training=True)
    
    # Slight color jitter
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    
    # Clip to [0, 255] just in case brightness pushes it over
    img = tf.clip_by_value(img, 0.0, 255.0)
    
    return img, label
