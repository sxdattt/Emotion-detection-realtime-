# config.py – Central configuration for the Emotion Detector

# ── DeepFace settings ──────────────────────────────────────────────────────
DETECTOR_BACKEND   = "opencv"      # options: opencv | mtcnn | retinaface
ANALYSIS_INTERVAL  = 4             # analyse every N frames
ENFORCE_DETECTION  = False         # False keeps app running when no face found

# ── History graph ──────────────────────────────────────────────────────────
HISTORY_MAX_POINTS = 60            # seconds of history in graph

# ── Window / Camera ────────────────────────────────────────────────────────
WINDOW_TITLE  = "Real-Time Emotion Detector"
CAM_INDEX     = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

# ── 7 emotions + colours (BGR for OpenCV) ──────────────────────────────────
EMOTIONS = ["angry", "happy", "neutral", "sad", "surprise"]

EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "happy":    (0,   210, 210),
    "neutral":  (180, 180, 180),
    "sad":      (210, 100, 0  ),
    "surprise": (0,   165, 255),
}

EMOTION_EMOJIS = {
    "angry":    "Angry",
    "happy":    "Happy",
    "neutral":  "Neutral",
    "sad":      "Sad",
    "surprise": "Surprise",
}