from __future__ import annotations

from .geometry import Detection
from .hand_tracking import HandLandmarks


def draw_detections(frame_bgr, detections: list[Detection]):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python to draw detections") from exc

    frame = frame_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(v) for v in detection.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            frame,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_hand_points(frame_bgr, hands: list[HandLandmarks]):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python to draw hand landmarks") from exc

    height, width = frame_bgr.shape[:2]
    frame = frame_bgr.copy()
    for hand in hands:
        for x, y, _ in hand.points:
            cv2.circle(frame, (int(x * width), int(y * height)), 3, (80, 220, 80), -1)
    return frame
