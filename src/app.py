"""
EmoSense AI  |  app.py  — Smooth Live Emotion Tracking
=======================================================
Architecture:
  CaptureThread  — reads camera at 30 fps, runs MediaPipe face detect, encodes JPEG
  InferenceThread — CNN runs on latest face crop, never blocks capture
  WatchdogThread  — auto-restarts camera on crash, zero user intervention

Key improvements:
  • Server-Sent Events (SSE) for near-zero-latency emotion data push
  • /frame endpoint serves latest JPEG; canvas polls at 30 fps  
  • Temporal smoothing (EMA) + confidence hysteresis
  • CLAHE pre-processing matches training pipeline
  • Multi-camera switching, auto-start, watchdog restart
"""

import os, sys, json, time, queue, threading
import numpy as np
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, Response, render_template, jsonify, request, make_response

from data_preprocessing import preprocess_face

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT    = os.path.join(os.path.dirname(__file__), '..')
TEMPLATES_DIR   = os.path.join(PROJECT_ROOT, 'templates')
STATIC_DIR      = os.path.join(PROJECT_ROOT, 'static')
MODEL_PATH      = os.path.join(PROJECT_ROOT, 'models', 'emotion_model.keras')
LABEL_MAP_PATH  = os.path.join(PROJECT_ROOT, 'models', 'label_map.json')
FACE_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'blaze_face_short_range.tflite')
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, 'screenshots')
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# ── Emotion Config ─────────────────────────────────────────────────
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_COLORS = {
    'Angry':   '#ef4444',
    'Disgust': '#22c55e',
    'Fear':    '#a855f7',
    'Happy':   '#facc15',
    'Neutral': '#94a3b8',
    'Sad':     '#3b82f6',
    'Surprise':'#f97316',
    'Uncertain': '#6b7280',
}
EMOTION_EMOJIS = {
    'Angry': '😡', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😄', 'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲',
    'Uncertain': '🤔',
}

# ── Camera / Capture Settings ──────────────────────────────────────
AUTO_START_CAM  = 0
TARGET_FPS      = 20     # Lowered for better performance
CAP_W, CAP_H    = 640, 480
JPEG_QUALITY    = 72
FACE_PAD_RATIO  = 0.1    # slight padding to capture full face

# ── Inference Settings ─────────────────────────────────────────────
EMA_ALPHA       = 0.35   # exponential moving average weight (higher = more responsive)
CONF_THRESHOLD  = 0.30   # min confidence to accept a new top emotion
SMOOTH_FRAMES   = 5      # frames kept in rolling average buffer

# ── Global Resources ───────────────────────────────────────────────
keras_model   = None
face_detector = None
_blank_jpg    = None

# ── Cross-thread State ─────────────────────────────────────────────
_infer_q    = queue.Queue(maxsize=1)   # capture -> inference (always latest frame)
_stop_ev    = threading.Event()
_user_stop  = threading.Event()

_emo_lock   = threading.Lock()
_emo_cache  = {
    'top': 'Neutral', 'conf': 0.0,
    'probs': {e: 0.0 for e in EMOTION_LABELS},
    'ema':   {e: 0.0 for e in EMOTION_LABELS},  # smoothed probabilities
    'faces': [],
}

state_lock = threading.Lock()
state = {
    'running':        False,
    'frame_jpg':      None,
    'fps':            0.0,
    'face_count':     0,
    'emotion_probs':  {e: 0.0 for e in EMOTION_LABELS},
    'top_emotion':    'Neutral',
    'confidence':     0.0,
    'total_frames':   0,
    'session_start':  None,
    'emotion_counts': {e: 0 for e in EMOTION_LABELS},
    'current_cam':    0,
    'current_cam_name': '',
    'inference_fps':  0.0,
}

_history    = deque(maxlen=90)      # timeline data points
_infer_hist = deque(maxlen=20)      # recent inference times for fps calc

_cap_thread   = None
_infer_thread = None

# ── Camera enumeration ─────────────────────────────────────────────
_cameras_cache = None
_cam_lock      = threading.Lock()

def enumerate_cameras(force=False):
    global _cameras_cache
    with _cam_lock:
        if _cameras_cache is not None and not force:
            return _cameras_cache

    cameras = []
    for idx in range(10):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, -1]:
            try:
                cap = (cv2.VideoCapture(idx) if backend == -1
                       else cv2.VideoCapture(idx, backend))
                if cap.isOpened():
                    cameras.append({
                        'index': idx,
                        'name': f'Camera {idx}',
                        'raw': f'Camera {idx}',
                    })
                    cap.release()
                    break
                cap.release()
            except Exception:
                pass
        time.sleep(0.05)

    if not cameras:
        cameras = [{'index': 0, 'name': 'Camera 0', 'raw': 'Camera 0'}]

    with _cam_lock:
        _cameras_cache = cameras
    return cameras


# ── Helpers ────────────────────────────────────────────────────────
def _make_blank(msg='Starting…'):
    blank = np.zeros((CAP_H, CAP_W, 3), dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
    cv2.putText(blank, msg, ((CAP_W - tw) // 2, CAP_H // 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 100, 140), 1, cv2.LINE_AA)
    _, j = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return j.tobytes()


def load_resources():
    global keras_model, face_detector, EMOTION_LABELS, _blank_jpg
    _blank_jpg = _make_blank()
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    import tensorflow as tf
    print('[INFO] Loading emotion model…')
    keras_model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
    # warm up to avoid first-frame latency spike
    keras_model(np.zeros((1, 224, 224, 3), dtype=np.uint8), training=False)
    print('[INFO] Model warmed up.')

    if os.path.isfile(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH) as f:
            lm = json.load(f)
        EMOTION_LABELS[:] = [None] * len(lm)
        for name, idx in lm.items():
            EMOTION_LABELS[idx] = name.capitalize()

    base_opts    = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    face_opts    = vision.FaceDetectorOptions(base_options=base_opts)
    face_detector = vision.FaceDetector.create_from_options(face_opts)
    print(f'[INFO] Labels: {EMOTION_LABELS}')


# ══════════════════════════════════════════════════════════════════
# Inference Thread — CNN, EMA smoothing, confidence gate
# ══════════════════════════════════════════════════════════════════
def _inference_worker():
    print('[Infer] Thread started.')
    ema = {e: 0.0 for e in EMOTION_LABELS}
    prev_top  = 'Neutral'
    prev_conf = 0.0

    while not _stop_ev.is_set():
        try:
            frame_bgr, faces = _infer_q.get(timeout=0.05)
        except queue.Empty:
            continue

        if not faces:
            # Reset EMA when no face
            ema = {e: 0.0 for e in EMOTION_LABELS}
            with _emo_lock:
                _emo_cache.update({
                    'top': 'Neutral', 'conf': 0.0,
                    'probs': {e: 0.0 for e in EMOTION_LABELS},
                    'ema':   {e: 0.0 for e in EMOTION_LABELS},
                    'faces': [],
                })
            continue

        x, y, w, h = faces[0]
        try:
            t_start = time.perf_counter()

            roi  = frame_bgr[y:y+h, x:x+w]
            
            # Use unified preprocessing
            inp_preprocessed = preprocess_face(roi)
            
            raw  = keras_model(inp_preprocessed.reshape(1, 224, 224, 3),
                               training=False).numpy()[0].astype(float)

            # Smart calibrated weights — tuned from live testing evidence.
            # The model massively over-predicts Neutral & Happy, and under-predicts
            # Angry, Disgust, Fear, Sad. These weights correct that real-world bias.
            # EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            inference_weights = np.array([8.0,   7.0,   3.5,   0.5,   2.0,  2.5,   2.5])
            raw = raw * inference_weights
            raw = raw / (np.sum(raw) + 1e-9)

            # EMA smoothing — balanced alpha for responsiveness without jitter
            ema_alpha = 0.50
            for i, e in enumerate(EMOTION_LABELS):
                ema[e] = ema_alpha * raw[i] + (1 - ema_alpha) * ema[e]

            # Normalize EMA to sum to 1
            ema_sum = sum(ema.values()) or 1.0
            norm_ema = {e: ema[e] / ema_sum for e in EMOTION_LABELS}

            top_e   = max(norm_ema, key=norm_ema.get)
            top_conf = norm_ema[top_e]

            # Confidence Thresholding
            if top_conf < CONF_THRESHOLD:
                prev_top = 'Uncertain'
                prev_conf = 0.0
            else:
                prev_top  = top_e
                prev_conf = top_conf

            _infer_hist.append(time.perf_counter() - t_start)

            with _emo_lock:
                _emo_cache.update({
                    'top':   prev_top,
                    'conf':  prev_conf,
                    'probs': {e: float(raw[i]) for i, e in enumerate(EMOTION_LABELS)},
                    'ema':   dict(norm_ema),
                    'faces': list(faces),
                })

        except Exception as e:
            print(f'[Infer] Error: {e}')

    print('[Infer] Stopped.')


# ══════════════════════════════════════════════════════════════════
# Capture Thread — camera read, face detect, draw, JPEG encode
# ══════════════════════════════════════════════════════════════════
def _capture_worker(cam_idx=0, cam_name='Camera'):
    cap = None
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, -1]:
        try:
            cap = (cv2.VideoCapture(cam_idx) if backend == -1
                   else cv2.VideoCapture(cam_idx, backend))
            if cap.isOpened():
                break
            cap.release(); cap = None
        except Exception:
            pass

    if not cap or not cap.isOpened():
        print(f'[Capture] Cannot open camera {cam_idx}')
        with state_lock:
            state['running'] = False
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    print(f'[Capture] {cam_name}: {int(cap.get(3))}x{int(cap.get(4))} '
          f'@ {cap.get(5):.0f}fps')

    with state_lock:
        state['session_start']    = datetime.now()
        state['current_cam']      = cam_idx
        state['current_cam_name'] = cam_name

    prev_t   = time.perf_counter()
    hist_ctr = fail_cnt = 0
    fps_buf  = deque(maxlen=15)

    while not _stop_ev.is_set():
        ret, frame = cap.read()
        if not ret:
            fail_cnt += 1
            if fail_cnt > 75:
                print('[Capture] Too many failures — stopping.')
                break
            time.sleep(0.04)
            continue
        fail_cnt = 0

        now   = time.perf_counter()
        dt    = max(now - prev_t, 1e-6)
        prev_t = now
        fps_buf.append(1.0 / dt)
        fps   = sum(fps_buf) / len(fps_buf)
        hist_ctr += 1

        # ── Face detection ─────────────────────────────────────
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = face_detector.detect(mp_img)

        fh, fw   = frame.shape[:2]
        faces    = []
        if results.detections:
            for det in results.detections:
                bb   = det.bounding_box
                px   = int(bb.width  * FACE_PAD_RATIO)
                py   = int(bb.height * FACE_PAD_RATIO)
                x1   = max(0,  bb.origin_x - px)
                y1   = max(0,  bb.origin_y - py)
                x2   = min(fw, bb.origin_x + bb.width  + px)
                y2   = min(fh, bb.origin_y + bb.height + py)
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2 - x1, y2 - y1))

        if faces:
            try:
                _infer_q.put_nowait((frame.copy(), faces))
            except queue.Full:
                pass   # inference is still processing; skip frame

        # ── Read emotion cache ─────────────────────────────────
        with _emo_lock:
            top          = _emo_cache['top']
            conf         = _emo_cache['conf']
            prbs         = dict(_emo_cache['ema'])   # use EMA probs for display
            cached_faces = _emo_cache['faces']

        draw_faces = faces if faces else cached_faces

        # ── Draw overlays on frame ─────────────────────────────
        for (x, y, w, h) in draw_faces:
            hex_c  = EMOTION_COLORS.get(top, '#6366f1').lstrip('#')
            color  = tuple(int(hex_c[i:i+2], 16) for i in (4, 2, 0))  # BGR

            # Subtle face fill
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
            cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

            # Sci-fi corner brackets
            arm = max(12, int(min(w, h) * 0.18))
            tk  = 2
            pts = [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]
            dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
            for (px, py), (dx, dy) in zip(pts, dirs):
                cv2.line(frame, (px, py), (px + dx*arm, py),        color, tk)
                cv2.line(frame, (px, py), (px, py + dy*arm),        color, tk)

            # Corner glow dots
            for (px, py) in pts:
                cv2.circle(frame, (px, py), 3, color, -1)

            # Emotion label badge
            if conf > 0:
                emoji = EMOTION_EMOJIS.get(top, '')
                lbl   = f'{top}  {conf*100:.0f}%'
                fs    = 0.58
                (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
                pad   = 6
                ly    = y - 10 if y > th + 20 else y + h + th + 12
                # Badge background
                cv2.rectangle(frame,
                              (x, ly - th - pad),
                              (x + tw + pad*2, ly + pad//2),
                              color, -1)
                cv2.putText(frame, lbl,
                            (x + pad, ly - pad//2),
                            cv2.FONT_HERSHEY_DUPLEX, fs,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # FPS overlay (top-left, minimal)
        inf_fps = (len(_infer_hist) / sum(_infer_hist)) if _infer_hist else 0
        cv2.putText(frame, f'FPS {fps:.0f}',
                    (10, 24), cv2.FONT_HERSHEY_DUPLEX,
                    0.55, (80, 72, 230), 1, cv2.LINE_AA)

        # ── JPEG encode ────────────────────────────────────────
        _, jpg     = cv2.imencode('.jpg', frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        jpg_bytes  = jpg.tobytes()
        fc         = len(draw_faces)

        with state_lock:
            state['frame_jpg']     = jpg_bytes
            state['fps']           = round(fps, 1)
            state['face_count']    = fc
            state['emotion_probs'] = prbs
            state['top_emotion']   = top
            state['confidence']    = conf
            state['inference_fps'] = round(1.0 / (sum(_infer_hist) / max(len(_infer_hist), 1)), 1) if _infer_hist else 0
            state['total_frames'] += 1
            if fc > 0 and conf > 0:
                state['emotion_counts'][top] += 1

        # History snapshot every ~0.5 s (15 frames @ 30fps)
        if hist_ctr >= 15:
            hist_ctr = 0
            entry = {'ts': datetime.now().strftime('%H:%M:%S')}
            entry.update(prbs)
            _history.append(entry)

    cap.release()
    _stop_ev.set()
    with state_lock:
        state['running'] = False
    print('[Capture] Stopped.')


# ══════════════════════════════════════════════════════════════════
# Watchdog Thread — auto-restart on unexpected crash
# ══════════════════════════════════════════════════════════════════
def _watchdog():
    time.sleep(10)
    while True:
        time.sleep(5)
        with state_lock:
            running  = state['running']
            cam_idx  = state['current_cam']
            cam_name = state['current_cam_name']
        if not running and not _user_stop.is_set():
            print('[Watchdog] Auto-restarting camera…')
            _start_camera(cam_idx, cam_name)


# ── Camera lifecycle ───────────────────────────────────────────────
def _start_camera(cam_idx=0, cam_name=''):
    global _cap_thread, _infer_thread
    with state_lock:
        if state['running']:
            return False
        state.update({
            'running':        True,
            'total_frames':   0,
            'session_start':  None,
            'frame_jpg':      None,
            'fps':            0.0,
            'face_count':     0,
            'confidence':     0.0,
            'top_emotion':    'Neutral',
            'emotion_probs':  {e: 0.0 for e in EMOTION_LABELS},
            'emotion_counts': {e: 0   for e in EMOTION_LABELS},
            'current_cam':    cam_idx,
            'current_cam_name': cam_name,
        })
    with _emo_lock:
        _emo_cache.update({
            'top': 'Neutral', 'conf': 0.0,
            'probs': {e: 0.0 for e in EMOTION_LABELS},
            'ema':   {e: 0.0 for e in EMOTION_LABELS},
            'faces': [],
        })

    _history.clear()
    _infer_hist.clear()
    _stop_ev.clear()
    _user_stop.clear()
    while not _infer_q.empty():
        try:   _infer_q.get_nowait()
        except queue.Empty: break

    _infer_thread = threading.Thread(target=_inference_worker,
                                     name='InferThread', daemon=True)
    _infer_thread.start()
    _cap_thread = threading.Thread(target=_capture_worker,
                                   args=(cam_idx, cam_name),
                                   name='CapThread', daemon=True)
    _cap_thread.start()
    return True


# ══════════════════════════════════════════════════════════════════
# Flask Routes
# ══════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    r = make_response(render_template('index.html',
                                      emotions=EMOTION_LABELS,
                                      colors=EMOTION_COLORS,
                                      emojis=EMOTION_EMOJIS))
    r.headers['Cache-Control'] = 'no-store'
    return r


@app.route('/favicon.ico')
def favicon():
    gif = (b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff'
           b'\x00\x00\x00!\xf9\x04\x00\x00\x00\x00\x00,'
           b'\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;')
    return Response(gif, mimetype='image/gif')


@app.route('/cameras')
def list_cameras():
    return jsonify(enumerate_cameras())


@app.route('/frame')
def latest_frame():
    """Single JPEG per request — browser polls at desired fps."""
    with state_lock:
        jpg = state['frame_jpg']
    resp = Response(jpg if jpg else _blank_jpg, mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-store, no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp


@app.route('/emotion_data')
def emotion_data():
    with state_lock:
        d = {
            'running':          state['running'],
            'fps':              state['fps'],
            'inference_fps':    state['inference_fps'],
            'face_count':       state['face_count'],
            'emotion_probs':    state['emotion_probs'],
            'top_emotion':      state['top_emotion'],
            'confidence':       round(state['confidence'] * 100, 1),
            'total_frames':     state['total_frames'],
            'emotion_counts':   state['emotion_counts'],
            'current_cam':      state['current_cam'],
            'current_cam_name': state['current_cam_name'],
            'session_duration': (
                str(datetime.now() - state['session_start']).split('.')[0]
                if state['session_start'] else '00:00:00'),
        }
    return jsonify(d)


@app.route('/history')
def get_history():
    return jsonify(list(_history))


@app.route('/control', methods=['POST'])
def control():
    payload = request.json or {}
    action  = payload.get('action', '')

    if action == 'start':
        cam  = int(payload.get('camera', AUTO_START_CAM))
        name = payload.get('name', f'Camera {cam}')
        ok   = _start_camera(cam, name)
        return jsonify({'status': 'started' if ok else 'already_running'})

    elif action == 'switch_camera':
        cam  = int(payload.get('camera', 0))
        name = payload.get('name', f'Camera {cam}')
        _stop_ev.set()
        _user_stop.clear()
        with state_lock:
            state['running'] = False
        time.sleep(0.5)
        ok = _start_camera(cam, name)
        return jsonify({'status': 'switched' if ok else 'error', 'camera': cam})

    elif action == 'stop':
        _stop_ev.set()
        _user_stop.set()
        with state_lock:
            state['running'] = False
        return jsonify({'status': 'stopped'})

    elif action == 'screenshot':
        with state_lock:
            jpg = state['frame_jpg']
        if jpg:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fn = os.path.join(SCREENSHOTS_DIR, f'snap_{ts}.jpg')
            with open(fn, 'wb') as f:
                f.write(jpg)
            return jsonify({'status': 'saved', 'file': os.path.basename(fn)})
        return jsonify({'status': 'no_frame'})

    return jsonify({'status': 'unknown'})


# ── Entry Point ────────────────────────────────────────────────────
if __name__ == '__main__':
    load_resources()

    cams  = enumerate_cameras()
    first = cams[0] if cams else {'index': 0, 'name': 'Camera 0'}
    print(f'[INFO] Auto-starting: {first["name"]} (index {first["index"]})')
    _start_camera(first['index'], first['name'])

    wd = threading.Thread(target=_watchdog, name='Watchdog', daemon=True)
    wd.start()

    print('\n[INFO] EmoSense AI  ->  http://localhost:5000\n')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
