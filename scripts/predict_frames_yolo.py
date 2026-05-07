from __future__ import annotations

import argparse
import csv
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained Ultralytics YOLO classifier on extracted video frames.")
    parser.add_argument("--model", required=True, help="Path to trained classifier, for example runs/classify/train/weights/best.pt.")
    parser.add_argument("--source", default="data/video_frames", help="Folder of extracted frames.")
    parser.add_argument("--out", default="outputs/frame_predictions.csv", help="Output CSV.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics before prediction: pip install ultralytics") from exc

    model = YOLO(args.model)
    source_path = Path(args.source)
    if source_path.is_dir():
        sources = sorted(str(path) for path in source_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        sources = [str(source_path)]

    if not sources:
        raise RuntimeError(f"No image frames found under {source_path}")

    rows: list[dict[str, str | float]] = []
    for source, result in zip(sources, model.predict(source=sources, stream=True, verbose=False), strict=False):
        if result.probs is None:
            continue
        top1 = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        label = result.names[top1]
        rows.append(
            {
                "frame_path": str(result.path),
                "source_frame_path": source,
                "predicted_label": label,
                "confidence": round(confidence, 6),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_path", "source_frame_path", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()
