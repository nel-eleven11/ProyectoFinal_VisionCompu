from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    yolo_weights: Path | None = None
    yolo_confidence: float = 0.35
    max_hands: int = 1


@dataclass(frozen=True)
class InstrumentConfig:
    instrument_type: str = "guitar"
    tuning: str = "standard"
    max_fret: int = 24


@dataclass(frozen=True)
class PipelineConfig:
    video_path: Path | None = None
    output_dir: Path = Path("outputs")
    chord_dataset: Path = Path("data/chord_shapes.example.json")
    models: ModelConfig = ModelConfig()
    instrument: InstrumentConfig = InstrumentConfig()


def _as_path(value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if value == "" or value.lower() in {"null", "none"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Tiny YAML subset parser for this project's simple config file."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_scalar(value)

    return root


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return _load_simple_yaml(text)
    return yaml.safe_load(text) or {}


def load_config(path: str | Path) -> PipelineConfig:
    """Load the YAML config used by notebooks and scripts."""
    raw = _read_yaml(Path(path))
    model_raw: dict[str, Any] = raw.get("models", {})
    instrument_raw: dict[str, Any] = raw.get("instrument", {})

    return PipelineConfig(
        video_path=_as_path(raw.get("video_path")),
        output_dir=Path(raw.get("output_dir", "outputs")),
        chord_dataset=Path(raw.get("chord_dataset", "data/chord_shapes.example.json")),
        models=ModelConfig(
            yolo_weights=_as_path(model_raw.get("yolo_weights")),
            yolo_confidence=float(model_raw.get("yolo_confidence", 0.35)),
            max_hands=int(model_raw.get("max_hands", 1)),
        ),
        instrument=InstrumentConfig(
            instrument_type=str(instrument_raw.get("instrument_type", "guitar")),
            tuning=str(instrument_raw.get("tuning", "standard")),
            max_fret=int(instrument_raw.get("max_fret", 24)),
        ),
    )
