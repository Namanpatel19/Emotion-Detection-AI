"""
EmoSense AI  |  app.py  |  Multi-camera + 30 FPS
=================================================
Threads:
  CaptureThread  — 30 fps camera read, face detect, annotate, encode JPEG
  InferenceThread — CNN model, never blocks capture
  WatchdogThread  — auto-restarts camera on crash

New features:
  /cameras  — enumerate all webcams including Camo virtual camera
  /control  — action=switch_camera for live camera switching
  /frame    — single JPEG per request (zero browser buffering)
  Auto-start + watchdog auto-restart
"""

import os, sys, json, time, queue, threading, subprocess
import numpy as np
from collections import deque
from datetime import datetime

import cv2
from flask import Flask, Response, render_template, jsonify, request, make_response

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = os.path.join(os.path.dirname(__file__), '..')
TEMPLATES_DIR   = os.path.join(PROJECT_ROOT, 'templates')
STATIC_DIR      = os.path.join(PROJECT_ROOT, 'static')
MODEL_PATH      = os.path.join(PROJECT_ROOT, 'models', 'emotion_model.h5')
LABEL_MAP_PATH  = os.path.join(PROJECT_ROOT, 'models', 'label_map.json')
CASCADE_PATH    = os.path.join(PROJECT_ROOT, 'haarcascades',
                               'haarcascade_frontalface_default.xml')
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, 'screenshots')
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# ── Emotion Config ─────────────────────────────────────────────────────────────
EMOTION_LABELS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_COLORS = {
    'Angry':    '#ef4444', 'Fear':    '#a855f7',
    'Happy':    '#22c55e', 'Neutral': '#64748b',
    'Sad':      '#3b82f6', 'Surprise':'#f97316',
}

# ── Settings ───────────────────────────────────────────────────────────────────
AUTO_START_CAM = 0
TARGET_FPS     = 30
CAP_W, CAP_H   = 640, 480
JPEG_QUALITY   = 68

# ── Camera enumeration ─────────────────────────────────────────────────────────
_cameras_cache = None
_cameras_lock  = threading.Lock()

def _get_device_names():
    """Use PowerShell Get-PnpDevice to get friendly names for camera devices."""
    try:
        cmd = [
            'powershell', '-NoProfile', '-Command',
            'Get-PnpDevice -Class Camera -Status OK | '
            'Sort-Object InstanceId | '
            'Select-Object FriendlyName | ConvertTo-Json'
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
        if r.returncode != 0 or not r.stdout.strip():
            return {}
        data = json.loads(r.stdout)
        if isinstance(data, dict):
            data = [data]
        # Map order index → name (Windows assigns camera indices by InstanceId order)
        return {i: d.get('FriendlyName', f'Camera {i}') for i, d in enumerate(data)}
    except Exception as e:
        print(f'[Cameras] Name lookup failed: {e}')
        return {}

def enumerate_cameras(force=False):
    """
    Probe camera indices 0-5, return working ones with friendly names.
    Camo virtual camera is detected via PowerShell Get-PnpDevice.
    Result is cached after first scan.
    """
    global _cameras_cache
    with _cameras_lock:
        if _cameras_cache is not None and not force:
            return _cameras_cache

    print('[Cameras] Scanning cameras ...')
    device_names = _get_device_names()
    print(f'[Cameras] Device names: {device_names}')

    cameras = []
    for idx in range(6):
        opened = False
        for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, -1]:
            try:
                cap = (cv2.VideoCapture(idx) if backend == -1
                       else cv2.VideoCapture(idx, backend))
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        pos   = len(cameras)
                        name  = device_names.get(pos, f'Camera {idx}')
                        if 'camo' in name.lower():
                            label = 'Camo (iPhone)'
                        elif 'hp' in name.lower() or 'wide' in name.lower():
                            label = f'Built-in ({name})'
                        else:
                            label = name
                        cameras.append({'index': idx, 'name': label, 'raw': name})
                        print(f'[Cameras]  [{idx}] {label} (backend={backend})')
                        opened = True
                    cap.release()
                    if opened:
                        break
            except Exception as e:
                pass
        time.sleep(0.06)

    if not cameras:
        cameras = [{'index': 0, 'name': 'Camera 0', 'raw': 'Camera 0'}]

    with _cameras_lock:
        _cameras_cache = cameras
    return cameras


# ── Cross-thread primitives ────────────────────────────────────────────────────
_infer_queue  = queue.Queue(maxsize=1)
_stop_event   = threading.Event()
_user_stopped = threading.Event()

_emotion_lock  = threading.Lock()
_emotion_cache = {
    'top': 'Neutral', 'conf': 0.0,
    'probs': {e: 0.0 for e in EMOTION_LABELS}, 'faces': [],
}

state_lock = threading.Lock()
state = {
    'running': False, 'frame_jpg': None, 'fps': 0.0, 'face_count': 0,
    'emotion_probs': {e: 0.0 for e in EMOTION_LABELS},
    'top_emotion': 'Neutral', 'confidence': 0.0,
    'total_frames': 0, 'session_start': None,
    'emotion_counts': {e: 0 for e in EMOTION_LABELS},
    'current_cam': 0, 'current_cam_name': '',
}
history = deque(maxlen=60)

keras_model = face_cascade = None
_blank_jpg  = None
_cap_thread = _infer_thread = None


def _make_blank(msg='Starting camera\u2026'):
    blank = np.zeros((CAP_H, CAP_W, 3), dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
    cv2.putText(blank, msg, ((CAP_W - tw) // 2, CAP_H // 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (120, 120, 160), 1, cv2.LINE_AA)
    _, j = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return j.tobytes()


def load_resources():
    global keras_model, face_cascade, EMOTION_LABELS, _blank_jpg
    _blank_jpg = _make_blank()
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    import tensorflow as tf
    print('[INFO] Loading model ...')
    keras_model = tf.keras.models.load_model(MODEL_PATH)
    keras_model(np.zeros((1, 48, 48, 1), dtype=np.float32), training=False)
    print('[INFO] Model pre-warmed.')

    if os.path.isfile(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH) as f:
            lm = json.load(f)
        EMOTION_LABELS[:] = [None] * len(lm)
        for name, idx in lm.items():
            EMOTION_LABELS[idx] = name.capitalize()

    cf = CASCADE_PATH if os.path.isfile(CASCADE_PATH) \
        else cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cf)
    print(f'[INFO] Labels: {EMOTION_LABELS}')

    # Pre-scan cameras in background so first request is instant
    threading.Thread(target=enumerate_cameras, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# Inference Thread
# ══════════════════════════════════════════════════════════════════════════════
def inference_worker():
    print('[Infer] Thread started.')
    while not _stop_event.is_set():
        try:
            gray, faces = _infer_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        if not faces:
            with _emotion_lock:
                _emotion_cache.update({
                    'top': 'Neutral', 'conf': 0.0,
                    'probs': {e: 0.0 for e in EMOTION_LABELS}, 'faces': [],
                })
            continue
        x, y, w, h = faces[0]
        try:
            roi   = gray[y:y+h, x:x+w]
            inp   = cv2.resize(roi, (48, 48),
                               interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            preds = keras_model(inp.reshape(1, 48, 48, 1),
                                training=False).numpy()[0]
            idx   = int(np.argmax(preds))
            with _emotion_lock:
                _emotion_cache.update({
                    'top':   EMOTION_LABELS[idx],
                    'conf':  float(preds[idx]),
                    'probs': {EMOTION_LABELS[i]: float(preds[i])
                              for i in range(len(EMOTION_LABELS))},
                    'faces': list(faces),
                })
        except Exception as e:
            print(f'[Infer] Error: {e}')
    print('[Infer] Stopped.')


# ══════════════════════════════════════════════════════════════════════════════
# Capture Thread
# ══════════════════════════════════════════════════════════════════════════════
def capture_worker(camera_index=0, cam_name='Camera'):
    cap = None
    for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, -1]:
        try:
            cap = (cv2.VideoCapture(camera_index) if backend == -1
                   else cv2.VideoCapture(camera_index, backend))
            if cap.isOpened():
                break
            cap.release()
        except Exception:
            pass

    if cap is None or not cap.isOpened():
        print(f'[Capture] Cannot open camera {camera_index}')
        with state_lock:
            state['running'] = False
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    print(f'[Capture] {cam_name} ({camera_index}): '
          f'{int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.0f}fps')

    with state_lock:
        state['session_start']   = datetime.now()
        state['current_cam']     = camera_index
        state['current_cam_name']= cam_name

    prev_t = time.perf_counter()
    hist_ctr = consec_fail = 0

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            consec_fail += 1
            if consec_fail > 10:
                print('[Capture] Read failures — stopping.')
                break
            time.sleep(0.05)
            continue
        consec_fail = 0

        now   = time.perf_counter()
        fps   = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        hist_ctr += 1

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        raw     = face_cascade.detectMultiScale(
            gray_eq, 1.2, 4, cv2.CASCADE_SCALE_IMAGE, (40, 40))
        faces = list(raw) if len(raw) > 0 else []

        if faces:
            try:
                _infer_queue.put_nowait((gray.copy(), faces))
            except queue.Full:
                pass

        with _emotion_lock:
            top          = _emotion_cache['top']
            conf         = _emotion_cache['conf']
            prbs         = dict(_emotion_cache['probs'])
            cached_faces = _emotion_cache['faces']

        draw_faces = faces if faces else cached_faces

        for (x, y, w, h) in draw_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (99, 102, 241), 2)
            if conf > 0:
                lbl = f'{top}  {conf*100:.0f}%'
                (tw, th), _ = cv2.getTextSize(
                    lbl, cv2.FONT_HERSHEY_DUPLEX, 0.62, 1)
                ly = y - 8 if y - 8 > 8 else y + h + 20
                cv2.rectangle(frame,
                              (x, ly - th - 5), (x + tw + 8, ly + 4),
                              (240, 242, 255), -1)
                cv2.putText(frame, lbl, (x + 4, ly - 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.62,
                            (79, 70, 229), 1, cv2.LINE_AA)

        # HUD (subtle on light-style feed)
        cv2.putText(frame, f'FPS {fps:.0f}', (10, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, (79, 70, 229), 1, cv2.LINE_AA)

        _, jpg     = cv2.imencode('.jpg', frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        jpg_bytes  = jpg.tobytes()
        fc = len(draw_faces)

        with state_lock:
            state['frame_jpg']     = jpg_bytes
            state['fps']           = round(fps, 1)
            state['face_count']    = fc
            state['emotion_probs'] = prbs
            state['top_emotion']   = top
            state['confidence']    = conf
            state['total_frames'] += 1
            if fc > 0 and conf > 0:
                state['emotion_counts'][top] += 1

        if hist_ctr >= 15:
            hist_ctr = 0
            entry = {'ts': datetime.now().strftime('%H:%M:%S')}
            entry.update(prbs)
            history.append(entry)

    cap.release()
    _stop_event.set()
    with state_lock:
        state['running'] = False
    print('[Capture] Stopped.')


# ══════════════════════════════════════════════════════════════════════════════
# Watchdog Thread
# ══════════════════════════════════════════════════════════════════════════════
def watchdog_worker():
    time.sleep(8)
    while True:
        time.sleep(4)
        with state_lock:
            running  = state['running']
            cam_idx  = state['current_cam']
            cam_name = state['current_cam_name']
        if not running and not _user_stopped.is_set():
            print('[Watchdog] Auto-restarting camera ...')
            start_camera(cam_idx, cam_name)


# ── Camera start helper ────────────────────────────────────────────────────────
def start_camera(cam_idx=0, cam_name=''):
    global _cap_thread, _infer_thread
    with state_lock:
        if state['running']:
            return False
        state.update({
            'running': True, 'total_frames': 0, 'session_start': None,
            'frame_jpg': None, 'fps': 0.0, 'face_count': 0,
            'confidence': 0.0, 'top_emotion': 'Neutral',
            'emotion_probs':  {e: 0.0 for e in EMOTION_LABELS},
            'emotion_counts': {e: 0   for e in EMOTION_LABELS},
            'current_cam': cam_idx, 'current_cam_name': cam_name,
        })
    with _emotion_lock:
        _emotion_cache.update({
            'top': 'Neutral', 'conf': 0.0,
            'probs': {e: 0.0 for e in EMOTION_LABELS}, 'faces': [],
        })
    history.clear()
    _stop_event.clear()
    _user_stopped.clear()
    while not _infer_queue.empty():
        try:   _infer_queue.get_nowait()
        except queue.Empty: break

    _infer_thread = threading.Thread(
        target=inference_worker, name='InferThread', daemon=True)
    _infer_thread.start()
    _cap_thread = threading.Thread(
        target=capture_worker, args=(cam_idx, cam_name),
        name='CapThread', daemon=True)
    _cap_thread.start()
    return True


# ── Flask Routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    r = make_response(render_template('index.html',
                                      emotions=EMOTION_LABELS,
                                      colors=EMOTION_COLORS))
    r.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return r


@app.route('/favicon.ico')
def favicon():
    gif = (b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff'
           b'\x00\x00\x00!\xf9\x04\x00\x00\x00\x00\x00,'
           b'\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;')
    return Response(gif, mimetype='image/gif')


@app.route('/cameras')
def list_cameras():
    """Return list of all available cameras including Camo."""
    return jsonify(enumerate_cameras())


@app.route('/frame')
def latest_frame():
    with state_lock:
        jpg = state['frame_jpg']
    data = jpg if jpg else _blank_jpg
    resp = Response(data, mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-store, no-cache'
    return resp


def gen_frames():
    while True:
        with state_lock:
            jpg = state['frame_jpg']
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + (jpg or _blank_jpg) + b'\r\n')
        time.sleep(0.033)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion_data')
def emotion_data():
    with state_lock:
        d = {
            'running':        state['running'],
            'fps':            state['fps'],
            'face_count':     state['face_count'],
            'emotion_probs':  state['emotion_probs'],
            'top_emotion':    state['top_emotion'],
            'confidence':     round(state['confidence'] * 100, 1),
            'total_frames':   state['total_frames'],
            'emotion_counts': state['emotion_counts'],
            'current_cam':    state['current_cam'],
            'current_cam_name': state['current_cam_name'],
            'session_duration': (
                str(datetime.now() - state['session_start']).split('.')[0]
                if state['session_start'] else '00:00:00'),
        }
    return jsonify(d)


@app.route('/history')
def get_history():
    return jsonify(list(history))


@app.route('/control', methods=['POST'])
def control():
    payload = request.json or {}
    action  = payload.get('action', '')

    if action == 'start':
        cam  = int(payload.get('camera', AUTO_START_CAM))
        name = payload.get('name', f'Camera {cam}')
        ok   = start_camera(cam, name)
        return jsonify({'status': 'started' if ok else 'already_running'})

    elif action == 'switch_camera':
        # Stop current, start new camera
        cam  = int(payload.get('camera', 0))
        name = payload.get('name', f'Camera {cam}')
        _stop_event.set()
        _user_stopped.clear()   # don't treat as user-stopped
        with state_lock:
            state['running'] = False
        time.sleep(0.4)          # wait for threads to wind down
        ok = start_camera(cam, name)
        return jsonify({'status': 'switched' if ok else 'error', 'camera': cam})

    elif action == 'stop':
        _stop_event.set()
        _user_stopped.set()
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


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_resources()
    # Start default camera
    cams = enumerate_cameras()
    first = cams[0] if cams else {'index': 0, 'name': 'Camera 0'}
    print(f'[INFO] Auto-starting: {first["name"]} (index {first["index"]})')
    start_camera(first['index'], first['name'])

    wd = threading.Thread(target=watchdog_worker, name='Watchdog', daemon=True)
    wd.start()

    print('\n[INFO] EmoSense AI  ->  http://localhost:5000\n')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
