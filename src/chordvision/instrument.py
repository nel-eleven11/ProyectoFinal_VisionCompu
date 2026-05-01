from __future__ import annotations

from dataclasses import dataclass

NOTE_NAMES_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NOTE_TO_PC = {name: idx for idx, name in enumerate(NOTE_NAMES_SHARP)}
NOTE_TO_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})

# Strings are stored from lowest pitch to highest pitch.
STANDARD_TUNINGS = {
    "guitar": ("E2", "A2", "D3", "G3", "B3", "E4"),
    "bass": ("E1", "A1", "D2", "G2"),
}


@dataclass(frozen=True)
class ChordShape:
    label: str
    instrument: str
    frets: tuple[int, ...]
    notes: tuple[str, ...] = ()


def get_tuning(instrument: str) -> tuple[str, ...]:
    try:
        return STANDARD_TUNINGS[instrument]
    except KeyError as exc:
        valid = ", ".join(sorted(STANDARD_TUNINGS))
        raise ValueError(f"Unsupported instrument '{instrument}'. Valid options: {valid}") from exc


def split_note(note: str) -> tuple[str, int]:
    if len(note) < 2:
        raise ValueError(f"Invalid note name: {note}")

    if len(note) >= 3 and note[1] in ("#", "b"):
        name = note[:2]
        octave = int(note[2:])
    else:
        name = note[0]
        octave = int(note[1:])

    if name not in NOTE_TO_PC:
        raise ValueError(f"Invalid note name: {note}")
    return name, octave


def note_to_midi(note: str) -> int:
    name, octave = split_note(note)
    return 12 * (octave + 1) + NOTE_TO_PC[name]


def midi_to_note(midi: int) -> str:
    octave = midi // 12 - 1
    name = NOTE_NAMES_SHARP[midi % 12]
    return f"{name}{octave}"


def transpose_note(note: str, semitones: int) -> str:
    return midi_to_note(note_to_midi(note) + semitones)


def frets_to_notes(instrument: str, frets: list[int] | tuple[int, ...]) -> list[str | None]:
    """Convert string frets to note names. -1 means muted or unused."""
    tuning = get_tuning(instrument)
    if len(frets) != len(tuning):
        raise ValueError(f"{instrument} expects {len(tuning)} strings, got {len(frets)} frets")

    notes: list[str | None] = []
    for open_note, fret in zip(tuning, frets):
        if fret < 0:
            notes.append(None)
        else:
            notes.append(transpose_note(open_note, fret))
    return notes
