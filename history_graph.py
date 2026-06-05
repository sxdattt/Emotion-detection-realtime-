# history_graph.py – Renders live emotion history graph as OpenCV image

import time
import cv2
import matplotlib
matplotlib.use("Agg")                  # no extra window
import matplotlib.pyplot as plt
import numpy as np

from config import EMOTION_COLORS, EMOTIONS

_FIG_W, _FIG_H = 4.2, 1.6
_DPI            = 80


def build_graph_image(history: list):
    """Convert history → small BGR numpy array. Returns None if < 2 points."""
    if len(history) < 2:
        return None

    now        = time.time()
    timestamps = [t - now for t, _, _ in history]
    scores_by  = {e: [sc.get(e, 0.0) for _, _, sc in history] for e in EMOTIONS}

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), dpi=_DPI)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    for emotion in EMOTIONS:
        bgr = EMOTION_COLORS[emotion]
        rgb = (bgr[2]/255, bgr[1]/255, bgr[0]/255)
        ax.plot(timestamps, scores_by[emotion], label=emotion, color=rgb, linewidth=1.4)

    ax.set_xlim(timestamps[0], 0)
    ax.set_ylim(0, 105)
    ax.set_xlabel("seconds ago", color="#aaaaaa", fontsize=7)
    ax.set_ylabel("score %",     color="#aaaaaa", fontsize=7)
    ax.tick_params(colors="#aaaaaa", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.legend(fontsize=5.5, loc="upper left",
              facecolor="#1e1e1e", labelcolor="white",
              ncol=4, framealpha=0.6)

    fig.tight_layout(pad=0.4)
    fig.canvas.draw()

    # Fixed: use buffer_rgba() instead of deprecated tostring_rgb()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def overlay_graph(frame, graph_img: np.ndarray):
    """Paste graph_img into the bottom-right corner of frame."""
    gh, gw = graph_img.shape[:2]
    fh, fw = frame.shape[:2]
    x1, y1 = fw - gw - 5, fh - gh - 40
    if x1 >= 0 and y1 >= 0:
        frame[y1:y1+gh, x1:x1+gw] = graph_img