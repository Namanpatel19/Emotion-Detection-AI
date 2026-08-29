"""
inference.py
────────────
Real-time facial emotion detection using a trained CNN + OpenCV webcam feed.

What this script does:
  1. Loads the saved Keras model (models/emotion_model.h5)
  2. Loads OpenCV's Haar Cascade face detector
  3. Opens your webcam
  4. For every video frame:
       a. Convert to grayscale (face detection + model input both need it)
       b. Detect all faces with detectMultiScale
       c. For each face:
            • Crop the face region
            • Resize to 48×48 (model input size)
            • Normalise pixels to [0, 1]
            • Run the CNN → get emotion probabilities
            • Pick the highest-probability emotion
       d. Draw a coloured bounding box around the face
       e. Overlay the emotion label + confidence percentage
       f. Show a small confidence bar chart for all 6 emotions
  5. Press 'q' to quit

Usage
─────
  cd "AI project"
  python src/inference.py

  # Use a different webcam index (0 is usually built-in, 1 is external):
  python src/inference.py --camera 1

  # Run on a video file instead of webcam:
  python src/inference.py --video path/to/video.mp4
"""

import os
import sys
import argparse
import cv2
import numpy as np

# Allow importing sibling modules
sys.path.insert(0, os.path.dirname(__file__))

# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT    = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATH      = os.path.join(PROJECT_ROOT, 'models', 'emotion_model.h5')
LABEL_MAP_PATH  = os.path.join(PROJECT_ROOT, 'models', 'label_map.json')
CASCADE_PATH    = os.path.join(PROJECT_ROOT, 'haarcascades',
                               'haarcascade_frontalface_default.xml')

IMG_SIZE        = 48       # CNN expects 48×48 grayscale input
DETECTION_SCALE = 1.3      # Haar Cascade scaleFactor
MIN_NEIGHBOURS  = 5        # Haar Cascade minNeighbors (higher = fewer false positives)
MIN_FACE_SIZE   = (30, 30) # Smallest face to detect (pixels)

# Default emotion labels — overridden at runtime by label_map.json
# Order must match the class indices from training
EMOTION_LABELS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Colour for each emotion (BGR format for OpenCV)
EMOTION_COLORS = {
    'Angry':    (0,   0,   220),   # red
    'Fear':     (130, 0,   130),   # purple
    'Happy':    (0,   200, 100),   # green
    'Sad':      (220, 100, 0  ),   # blue-ish
    'Surprise': (0,   165, 255),   # orange
    'Neutral':  (180, 180, 180),   # grey
}


# ── Model & Cascade Loaders ───────────────────────────────────────────────────

def load_model(model_path: str):
    """Load the trained Keras emotion model and its label map from disk."""
    import json
    import tensorflow as tf
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"\n[ERROR] Trained model not found at: {model_path}\n"
            "Please run the training script first:\n"
            "  .venv\\Scripts\\python.exe src\\train.py\n"
        )
    print(f"[INFO] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("[INFO] Model loaded successfully.")

    # Load label map to ensure inference order matches training order
    global EMOTION_LABELS
    if os.path.isfile(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)   # {'angry': 0, 'fear': 1, ...}
        # Invert: index -> display name
        EMOTION_LABELS = [None] * len(label_map)
        for folder_name, idx in label_map.items():
            EMOTION_LABELS[idx] = folder_name.capitalize()
        print(f"[INFO] Emotion labels: {EMOTION_LABELS}")
    else:
        print("[WARN] label_map.json not found — using default label order.")

    return model


def load_face_cascade(cascade_path: str):
    """
    Load OpenCV Haar Cascade for frontal face detection.

    If the local XML file is missing, fall back to OpenCV's bundled version.
    """
    if os.path.isfile(cascade_path):
        cascade = cv2.CascadeClassifier(cascade_path)
    else:
        # Fall back to OpenCV's own bundled Haar Cascade
        print(f"[WARN] Cascade not found at {cascade_path}. Using OpenCV built-in.")
        builtin = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(builtin)

    if cascade.empty():
        raise RuntimeError(
            "[ERROR] Failed to load Haar Cascade face detector.\n"
            "Download haarcascade_frontalface_default.xml from:\n"
            "https://github.com/opencv/opencv/tree/master/data/haarcascades\n"
            f"and place it in: {os.path.dirname(cascade_path)}"
        )
    print("[INFO] Face cascade loaded.")
    return cascade


# ── Frame Processing ──────────────────────────────────────────────────────────

def detect_faces(gray_frame, cascade):
    """
    Detect faces in a grayscale frame using Haar Cascade.

    Returns
    -------
    faces : list of (x, y, w, h) tuples — bounding boxes of detected faces
    """
    faces = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=DETECTION_SCALE,
        minNeighbors=MIN_NEIGHBOURS,
        minSize=MIN_FACE_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces if len(faces) > 0 else []


def preprocess_face(gray_frame, x, y, w, h):
    """
    Extract and preprocess a face region for CNN inference.

    Steps:
      1. Crop the face from the grayscale frame
      2. Resize to IMG_SIZE × IMG_SIZE
      3. Normalise pixel values to [0, 1]
      4. Reshape to (1, IMG_SIZE, IMG_SIZE, 1) — batch of 1

    Returns
    -------
    face_input : np.ndarray, shape (1, 48, 48, 1)
    """
    face_roi = gray_frame[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face_norm = face_resized.astype(np.float32) / 255.0
    face_input = face_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)  # add batch & channel dims
    return face_input


def predict_emotion(model, face_input):
    """
    Run the CNN on a preprocessed face and return the predicted emotion.

    Returns
    -------
    emotion     : str   — predicted emotion label
    confidence  : float — probability of the predicted class (0.0–1.0)
    all_probs   : list  — probabilities for all 6 emotions
    """
    predictions = model.predict(face_input, verbose=0)[0]  # shape: (6,)
    emotion_idx = np.argmax(predictions)
    emotion     = EMOTION_LABELS[emotion_idx]
    confidence  = predictions[emotion_idx]
    return emotion, confidence, predictions


# ── Overlay Drawing ───────────────────────────────────────────────────────────

def draw_face_overlay(frame, x, y, w, h, emotion, confidence, all_probs):
    """
    Draw bounding box, emotion label, confidence score, and mini bar chart
    on the video frame for a detected face.

    Parameters
    ----------
    frame      : BGR frame to draw on (modified in-place)
    x, y, w, h : face bounding box
    emotion    : predicted emotion string
    confidence : float, 0.0–1.0
    all_probs  : array of probabilities for all 6 emotions
    """
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))

    # ── Bounding box ──
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # ── Emotion label + confidence above the box ──
    label_text = f"{emotion}: {confidence*100:.1f}%"
    label_y = y - 10 if y - 10 > 10 else y + h + 25

    # Dark background for text readability
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(frame,
                  (x, label_y - text_h - 6),
                  (x + text_w + 6, label_y + 4),
                  (30, 30, 30), -1)                         # filled dark rectangle
    cv2.putText(frame, label_text,
                (x + 3, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                color, 2, cv2.LINE_AA)

    # ── Mini confidence bar chart (right side of face box) ──
    bar_x_start = x + w + 10
    bar_max_w   = 100
    bar_h       = 12
    bar_gap     = 4

    for i, (emo, prob) in enumerate(zip(EMOTION_LABELS, all_probs)):
        bar_y = y + i * (bar_h + bar_gap)
        # Filled bar proportional to probability
        bar_filled = int(bar_max_w * prob)
        bar_color  = EMOTION_COLORS.get(emo, (200, 200, 200))

        cv2.rectangle(frame,
                      (bar_x_start, bar_y),
                      (bar_x_start + bar_max_w, bar_y + bar_h),
                      (60, 60, 60), -1)                     # background
        cv2.rectangle(frame,
                      (bar_x_start, bar_y),
                      (bar_x_start + bar_filled, bar_y + bar_h),
                      bar_color, -1)                        # filled portion
        cv2.putText(frame, f"{emo[:3]} {prob*100:.0f}%",
                    (bar_x_start + bar_max_w + 5, bar_y + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    bar_color, 1, cv2.LINE_AA)


def draw_hud(frame, fps: float, face_count: int):
    """Draw a heads-up display with FPS and detection count in the corner."""
    h, w = frame.shape[:2]

    # FPS counter (top-left)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    # Face count (top-left, second line)
    cv2.putText(frame, f"Faces: {face_count}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    # Instructions (bottom-left)
    cv2.putText(frame, "Press 'q' to quit | 's' to screenshot",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1, cv2.LINE_AA)


# ── Main Inference Loop ───────────────────────────────────────────────────────

def run_inference(camera_index: int = 0, video_path: str = None):
    """
    Open the webcam (or a video file) and run real-time emotion detection.

    Parameters
    ----------
    camera_index : int  — webcam device index (0 = default camera)
    video_path   : str  — if provided, process this video file instead of webcam
    """

    # ── Load model and face detector ──
    model   = load_model(MODEL_PATH)
    cascade = load_face_cascade(CASCADE_PATH)

    # ── Open video source ──
    source = video_path if video_path else camera_index
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"[ERROR] Could not open video source: {source}\n"
            "  • Check that your webcam is connected and not in use by another app.\n"
            "  • Try a different camera index: python src/inference.py --camera 1"
        )

    print(f"\n[INFO] Video source opened: {'Webcam ' + str(source) if not video_path else video_path}")
    print("[INFO] Starting real-time inference...")
    print("[INFO] Controls: 'q' = quit | 's' = save screenshot\n")

    import time
    screenshot_dir = os.path.join(PROJECT_ROOT, 'screenshots')
    os.makedirs(screenshot_dir, exist_ok=True)
    screenshot_count = 0

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video or camera disconnected.")
            break

        # ── FPS calculation ──
        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 1e-9)
        prev_time = curr_time

        # ── Convert frame to grayscale for face detection ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Histogram equalisation improves detection under poor lighting ──
        gray_eq = cv2.equalizeHist(gray)

        # ── Detect faces ──
        faces = detect_faces(gray_eq, cascade)

        # ── Process each detected face ──
        for (x, y, w, h) in faces:
            try:
                # Preprocess the face region for the CNN
                face_input = preprocess_face(gray, x, y, w, h)

                # Predict emotion
                emotion, confidence, all_probs = predict_emotion(model, face_input)

                # Draw results on the frame
                draw_face_overlay(frame, x, y, w, h, emotion, confidence, all_probs)

            except Exception as e:
                # Don't crash on a bad face crop; just skip it
                print(f"[WARN] Could not process face at ({x},{y}): {e}")

        # ── Draw HUD ──
        draw_hud(frame, fps, len(faces))

        # ── Display frame ──
        cv2.imshow('Real-Time Emotion Detection — Press Q to Quit', frame)

        # ── Handle key presses ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' pressed — quitting.")
            break
        elif key == ord('s'):
            screenshot_count += 1
            filename = os.path.join(screenshot_dir, f'screenshot_{screenshot_count:04d}.jpg')
            cv2.imwrite(filename, frame)
            print(f"[INFO] Screenshot saved: {filename}")

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Inference session ended.")


# ── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Real-Time Emotion Detection using a trained CNN + OpenCV'
    )
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Webcam device index (0 = default built-in camera)'
    )
    parser.add_argument(
        '--video', type=str, default=None,
        help='Path to a video file (overrides --camera if provided)'
    )
    args = parser.parse_args()

    run_inference(camera_index=args.camera, video_path=args.video)
