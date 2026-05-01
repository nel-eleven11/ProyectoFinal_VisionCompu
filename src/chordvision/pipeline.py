from __future__ import annotations

from dataclasses import dataclass

from .chord_matching import ChordMatch, match_chord
from .config import PipelineConfig
from .geometry import Detection
from .hand_tracking import HandLandmarks
from .instrument import ChordShape


@dataclass(frozen=True)
class FrameResult:
    frame_index: int
    time_s: float
    detections: list[Detection]
    hands: list[HandLandmarks]
    observed_frets: tuple[int, ...] | None
    chord: ChordMatch | None


class ChordRecognitionPipeline:
    """High-level orchestration for one video frame.

    The geometry stage is intentionally left as a replaceable piece because it
    depends on the camera angle, the YOLO labels, and the calibration strategy.
    """

    def __init__(
        self,
        config: PipelineConfig,
        chord_shapes: list[ChordShape],
        hand_tracker=None,
        instrument_detector=None,
    ) -> None:
        self.config = config
        self.chord_shapes = chord_shapes
        self.hand_tracker = hand_tracker
        self.instrument_detector = instrument_detector

    def process_frame(self, frame_bgr, frame_index: int, fps: float) -> FrameResult:
        detections = self.instrument_detector.detect(frame_bgr) if self.instrument_detector else []
        hands = self.hand_tracker.detect(frame_bgr) if self.hand_tracker else []

        observed_frets = self.estimate_frets_from_frame(detections, hands)
        chord = match_chord(observed_frets, self.chord_shapes) if observed_frets else None

        return FrameResult(
            frame_index=frame_index,
            time_s=frame_index / fps if fps else 0.0,
            detections=detections,
            hands=hands,
            observed_frets=observed_frets,
            chord=chord,
        )

    def estimate_frets_from_frame(
        self,
        detections: list[Detection],
        hands: list[HandLandmarks],
    ) -> tuple[int, ...] | None:
        """Placeholder for the finger-to-string/fret mapping.

        Next implementation step:
        1. Use YOLO detections to locate fretboard/string regions.
        2. Convert MediaPipe fingertip landmarks to pixel coordinates.
        3. Project fingertips to the fretboard axis.
        4. Return one fret per string, using -1 for muted/unused strings.
        """
        return None
