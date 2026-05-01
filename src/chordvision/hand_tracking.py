from __future__ import annotations

from dataclasses import dataclass

from .geometry import Point2D, Point3D

FINGER_TIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}


@dataclass(frozen=True)
class HandLandmarks:
    points: tuple[Point3D, ...]
    handedness: str = "unknown"

    def fingertip(self, finger: str) -> Point2D:
        idx = FINGER_TIP_IDS[finger]
        x, y, _ = self.points[idx]
        return (x, y)

    def fingertip_points(self) -> dict[str, Point2D]:
        return {finger: self.fingertip(finger) for finger in FINGER_TIP_IDS}


class HandTracker:
    """Small wrapper around MediaPipe Hands.

    It returns normalized landmark coordinates, so later code can scale them to
    the frame size or map them to a fretboard coordinate system.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("Install mediapipe to use HandTracker: pip install mediapipe") from exc

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr) -> list[HandLandmarks]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install opencv-python to use HandTracker.detect") from exc

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)
        if not result.multi_hand_landmarks:
            return []

        hands: list[HandLandmarks] = []
        handedness_labels = []
        if result.multi_handedness:
            handedness_labels = [item.classification[0].label for item in result.multi_handedness]

        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            points = tuple((lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark)
            handedness = handedness_labels[idx] if idx < len(handedness_labels) else "unknown"
            hands.append(HandLandmarks(points=points, handedness=handedness))
        return hands
