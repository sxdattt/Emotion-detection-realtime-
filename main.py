# main.py – Entry point: video loop + UI assembly

import cv2
from config import CAM_INDEX, FRAME_HEIGHT, FRAME_WIDTH, WINDOW_TITLE
from emotion_engine import EmotionEngine
from history_graph import build_graph_image, overlay_graph
from overlay import (
    draw_confidence_bars, draw_face_box,
    draw_no_face_notice, draw_session_info,
)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check CAM_INDEX in config.py")
        return

    engine      = EmotionEngine()
    graph_cache = None

    print(f"[INFO] Starting {WINDOW_TITLE}  –  press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 1. Trigger async analysis
        engine.process_frame(frame)

        # 2. Draw overlays
        result = engine.get_latest()
        if result:
            draw_face_box(frame, result)
            draw_confidence_bars(frame, result)
        else:
            draw_no_face_notice(frame)

        # 3. Session mood
        draw_session_info(frame, engine.session_dominant_emotion())

        # 4. History graph
        new_graph = build_graph_image(engine.get_history())
        if new_graph is not None:
            graph_cache = new_graph
        if graph_cache is not None:
            overlay_graph(frame, graph_cache)

        # 5. Display
        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()