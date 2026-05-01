from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .instrument import ChordShape, frets_to_notes


@dataclass(frozen=True)
class ChordMatch:
    label: str
    score: float
    expected_frets: tuple[int, ...]
    observed_frets: tuple[int, ...]
    notes: tuple[str, ...]


def load_chord_shapes(path: str | Path, instrument: str | None = None) -> list[ChordShape]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    shapes: list[ChordShape] = []
    for item in raw["shapes"]:
        if instrument and item["instrument"] != instrument:
            continue
        notes = tuple(note for note in item.get("notes", []))
        shapes.append(
            ChordShape(
                label=item["label"],
                instrument=item["instrument"],
                frets=tuple(int(fret) for fret in item["frets"]),
                notes=notes,
            )
        )
    return shapes


def _fret_error(observed: tuple[int, ...], expected: tuple[int, ...]) -> float:
    if len(observed) != len(expected):
        return float("inf")

    total = 0.0
    compared = 0
    for obs, exp in zip(observed, expected):
        if obs < 0 and exp < 0:
            compared += 1
            continue
        if obs < 0 or exp < 0:
            total += 3.0
        else:
            total += abs(obs - exp)
        compared += 1

    return total / max(compared, 1)


def match_chord(
    observed_frets: list[int] | tuple[int, ...],
    shapes: list[ChordShape],
    max_average_fret_error: float = 0.75,
) -> ChordMatch | None:
    """Match observed fret positions against predefined chord/note shapes."""
    observed = tuple(int(fret) for fret in observed_frets)
    if not shapes:
        return None

    best_shape = min(shapes, key=lambda shape: _fret_error(observed, shape.frets))
    error = _fret_error(observed, best_shape.frets)
    if error > max_average_fret_error:
        return None

    score = max(0.0, 1.0 - error / max_average_fret_error)
    notes = best_shape.notes or tuple(note for note in frets_to_notes(best_shape.instrument, best_shape.frets) if note)
    return ChordMatch(
        label=best_shape.label,
        score=score,
        expected_frets=best_shape.frets,
        observed_frets=observed,
        notes=tuple(notes),
    )
