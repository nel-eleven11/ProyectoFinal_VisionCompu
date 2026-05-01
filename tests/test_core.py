from chordvision.chord_matching import load_chord_shapes, match_chord
from chordvision.instrument import frets_to_notes
from chordvision.tablature import TabEvent, render_ascii_tab


def test_guitar_note_conversion():
    assert frets_to_notes("guitar", [0, 2, 2, 0, 0, 0]) == [
        "E2",
        "B2",
        "E3",
        "G3",
        "B3",
        "E4",
    ]


def test_match_example_chord():
    shapes = load_chord_shapes("data/chord_shapes.example.json", instrument="guitar")
    match = match_chord([0, 2, 2, 0, 0, 0], shapes)
    assert match is not None
    assert match.label == "Em"


def test_render_ascii_tab():
    tab = render_ascii_tab([TabEvent(time_s=0.0, label="Em", frets=(0, 2, 2, 0, 0, 0))], "guitar")
    assert "E4" in tab
    assert "|-0|" in tab
