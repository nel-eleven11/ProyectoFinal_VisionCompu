from __future__ import annotations

from dataclasses import dataclass
from math import hypot

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> Point2D:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance(a: Point2D, b: Point2D) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def nearest_index(value: float, candidates: list[float] | tuple[float, ...]) -> int:
    if not candidates:
        raise ValueError("At least one candidate is required")
    return min(range(len(candidates)), key=lambda idx: abs(value - candidates[idx]))


def nearest_string_index(point: Point2D, string_y_positions: list[float] | tuple[float, ...]) -> int:
    """Map a fingertip to the nearest string using horizontal string centers."""
    return nearest_index(point[1], string_y_positions)


def fret_from_x(point_x: float, fret_x_positions: list[float] | tuple[float, ...]) -> int:
    """Map a point x-coordinate to a fret interval.

    fret_x_positions should contain fret boundaries ordered from nut to bridge.
    If there are N boundaries, the returned fret is in [0, N].
    """
    if len(fret_x_positions) < 2:
        raise ValueError("At least two fret positions are required")

    ordered = sorted(fret_x_positions)
    if point_x <= ordered[0]:
        return 0
    for idx in range(1, len(ordered)):
        if point_x <= ordered[idx]:
            return idx
    return len(ordered)


def project_point_to_axis(point: Point2D, origin: Point2D, end: Point2D) -> float:
    """Return normalized projection of point over the fretboard axis."""
    ox, oy = origin
    ex, ey = end
    px, py = point
    axis_x = ex - ox
    axis_y = ey - oy
    denom = axis_x * axis_x + axis_y * axis_y
    if denom == 0:
        raise ValueError("Axis origin and end cannot be the same point")
    return ((px - ox) * axis_x + (py - oy) * axis_y) / denom


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
