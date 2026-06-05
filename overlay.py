# overlay.py – All OpenCV drawing helpers

import cv2
from config import EMOTION_COLORS, EMOTION_EMOJIS, EMOTIONS


def draw_face_box(frame, result: dict):
    """Bounding box + emotion label over detected face."""
    region   = result.get("region", {})
    dominant = result.get("dominant", "")
    color    = EMOTION_COLORS.get(dominant, (255, 255, 255))
    label    = f"{EMOTION_EMOJIS.get(dominant, '')} {dominant.upper()}"

    if region and region.get("w", 0) > 0:
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)
        cv2.putText(frame, label, (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def draw_confidence_bars(frame, result: dict):
    """Mini bar chart on the right side for all 7 emotion scores."""
    scores    = result.get("scores", {})
    if not scores:
        return

    bar_x     = frame.shape[1] - 180
    bar_y0    = 20
    bar_h     = 18
    bar_gap   = 6
    max_width = 160

    cv2.rectangle(frame,
                  (bar_x - 10, bar_y0 - 5),
                  (frame.shape[1] - 5,
                   bar_y0 + len(EMOTIONS) * (bar_h + bar_gap) + 5),
                  (30, 30, 30), -1)

    for i, emotion in enumerate(EMOTIONS):
        score  = scores.get(emotion, 0.0)
        filled = int((score / 100.0) * max_width)
        y      = bar_y0 + i * (bar_h + bar_gap)
        color  = EMOTION_COLORS.get(emotion, (200, 200, 200))

        cv2.rectangle(frame, (bar_x, y), (bar_x + max_width, y + bar_h), (70, 70, 70), -1)
        if filled > 0:
            cv2.rectangle(frame, (bar_x, y), (bar_x + filled, y + bar_h), color, -1)
        cv2.putText(frame, f"{emotion[:3].upper()} {score:4.1f}%",
                    (bar_x - 5, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)


def draw_session_info(frame, dominant_session: str):
    """Bottom-left: session dominant emotion."""
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 35), (320, h), (30, 30, 30), -1)
    cv2.putText(frame,
                f"Session mood: {dominant_session.upper()}",
                (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_no_face_notice(frame):
    cv2.putText(frame, "No face detected", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 100, 255), 2)