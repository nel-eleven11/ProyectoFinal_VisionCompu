from __future__ import annotations

import argparse
import csv
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def get_video_metadata(path: Path) -> dict[str, str | float | int]:
    metadata: dict[str, str | float | int] = {
        "file": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "duration_s": "",
        "fps": "",
        "frame_count": "",
        "width": "",
        "height": "",
        "source": "internet",
        "ground_truth": "unknown",
        "use_case": "robustness_test",
        "notes": "",
    }

    try:
        import cv2
    except ImportError:
        return metadata

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return metadata

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    metadata["fps"] = round(float(fps), 3) if fps else ""
    metadata["frame_count"] = frame_count
    metadata["duration_s"] = round(frame_count / fps, 3) if fps else ""
    metadata["width"] = width
    metadata["height"] = height
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manifest for local guitar/bass test videos.")
    parser.add_argument("--video-dir", default="data", help="Directory containing downloaded videos.")
    parser.add_argument("--out", default="data/video_manifest.csv", help="Manifest CSV path.")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    videos = sorted(path for path in video_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    rows = [get_video_metadata(path) for path in videos]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "size_bytes", "duration_s", "fps", "frame_count", "width", "height", "source", "ground_truth", "use_case", "notes"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} video records to {out_path}")


if __name__ == "__main__":
    main()
