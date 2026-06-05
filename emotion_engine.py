# emotion_engine.py – Wraps DeepFace in a background thread

import threading
import time
from collections import deque

from deepface import DeepFace
from config import (
    ANALYSIS_INTERVAL, DETECTOR_BACKEND,
    ENFORCE_DETECTION, EMOTIONS, HISTORY_MAX_POINTS,
)

# Boost non-neutral emotions, suppress neutral
SENSITIVITY = {
    "angry":    1.6,
    "happy":    1.3,
    "neutral":  0.6,
    "sad":      1.6,
    "surprise": 1.4,
}


class EmotionEngine:
    """Runs DeepFace.analyze() in a daemon thread so video loop never blocks."""

    def __init__(self):
        self.latest_result = None
        self.lock = threading.Lock()
        self.history = deque(maxlen=HISTORY_MAX_POINTS)
        self._frame_counter = 0
        self._busy = False

    def process_frame(self, frame):
        """Call once per frame. Triggers async analysis every N frames."""
        self._frame_counter += 1
        if self._frame_counter % ANALYSIS_INTERVAL == 0 and not self._busy:
            thread = threading.Thread(
                target=self._analyse, args=(frame.copy(),), daemon=True
            )
            thread.start()

    def get_latest(self):
        with self.lock:
            return self.latest_result

    def get_history(self):
        with self.lock:
            return list(self.history)

    def session_dominant_emotion(self):
        """Return the most frequent emotion across the entire session."""
        history = self.get_history()
        if not history:
            return "none"
        counts = {e: 0 for e in EMOTIONS}
        for _, emotion, _ in history:
            counts[emotion] = counts.get(emotion, 0) + 1
        return max(counts, key=counts.get)

    def _analyse(self, frame):
        self._busy = True
        try:
            results = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=ENFORCE_DETECTION,
                silent=True,
            )
            data = results[0] if isinstance(results, list) else results
            raw_scores = data.get("emotion", {})

            # Apply sensitivity only for our 5 chosen emotions
            boosted = {
                e: raw_scores.get(e, 0.0) * SENSITIVITY.get(e, 1.0)
                for e in EMOTIONS
            }

            # Pick dominant from boosted scores
            dominant = max(boosted, key=boosted.get)

            entry = {
                "dominant": dominant,
                "scores":   raw_scores,    # raw scores for display bars
                "region":   data.get("region", {}),
            }
            with self.lock:
                self.latest_result = entry
                self.history.append((time.time(), dominant, raw_scores))

        except Exception:
            pass
        finally:
            self._busy = False