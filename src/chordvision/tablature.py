from __future__ import annotations

from dataclasses import dataclass

from .instrument import get_tuning


@dataclass(frozen=True)
class TabEvent:
    time_s: float
    label: str
    frets: tuple[int, ...]
    duration_s: float = 0.0


def merge_repeated_events(events: list[TabEvent], min_gap_s: float = 0.15) -> list[TabEvent]:
    """Collapse consecutive equal events to keep the generated tab readable."""
    if not events:
        return []

    merged: list[TabEvent] = [events[0]]
    for event in events[1:]:
        previous = merged[-1]
        same_shape = event.label == previous.label and event.frets == previous.frets
        close_enough = event.time_s - previous.time_s <= min_gap_s
        if same_shape and close_enough:
            merged[-1] = TabEvent(
                time_s=previous.time_s,
                label=previous.label,
                frets=previous.frets,
                duration_s=max(previous.duration_s, event.time_s - previous.time_s),
            )
        else:
            merged.append(event)
    return merged


def _format_fret(fret: int) -> str:
    if fret < 0:
        return "x"
    return str(fret)


def render_ascii_tab(events: list[TabEvent], instrument: str) -> str:
    """Render a simple ASCII tablature from detected fret events."""
    tuning = get_tuning(instrument)
    lines = {string_idx: [] for string_idx in range(len(tuning))}

    for event in events:
        for string_idx, fret in enumerate(event.frets):
            token = _format_fret(fret)
            lines[string_idx].append(token.rjust(2, "-"))

    rendered = []
    for string_idx in reversed(range(len(tuning))):
        label = tuning[string_idx].ljust(3)
        rendered.append(f"{label}|{'-'.join(lines[string_idx])}|")
    return "\n".join(rendered)
