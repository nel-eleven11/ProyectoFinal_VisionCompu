from __future__ import annotations

from pathlib import Path

from .geometry import Detection


class InstrumentDetector:
    """Wrapper around Ultralytics YOLO for instrument-region detection."""

    def __init__(self, weights_path: str | Path, confidence: float = 0.35) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics to use YOLO: pip install ultralytics") from exc

        self.weights_path = Path(weights_path)
        self.confidence = confidence
        self.model = YOLO(str(self.weights_path))

    def detect(self, frame_bgr) -> list[Detection]:
        results = self.model(frame_bgr, conf=self.confidence, verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = str(names.get(cls_id, cls_id))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(Detection(label=label, confidence=conf, xyxy=(x1, y1, x2, y2)))
        return detections
