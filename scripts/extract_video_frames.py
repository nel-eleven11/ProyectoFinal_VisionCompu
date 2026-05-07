from __future__ import annotations

import argparse
import csv
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def extract_frames(video_path: Path, out_dir: Path, every_seconds: float) -> list[dict[str, str | int | float]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python before extracting frames: pip install opencv-python") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_step = max(1, int(round(fps * every_seconds)))
    stem_dir = out_dir / video_path.stem
    stem_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    frame_index = 0
    saved_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_step == 0:
            time_s = frame_index / fps
            image_path = stem_dir / f"frame_{saved_index:05d}.jpg"
            cv2.imwrite(str(image_path), frame)
            rows.append(
                {
                    "video": video_path.as_posix(),
                    "frame_path": image_path.as_posix(),
                    "frame_index": frame_index,
                    "time_s": round(float(time_s), 3),
                }
            )
            saved_index += 1

        frame_index += 1

    cap.release()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sampled frames from local videos.")
    parser.add_argument("--video-dir", default="data", help="Folder containing videos.")
    parser.add_argument("--out", default="data/video_frames", help="Output frame folder.")
    parser.add_argument("--manifest", default="data/video_frames_manifest.csv", help="Frame manifest CSV.")
    parser.add_argument("--every-seconds", type=float, default=0.5, help="Sampling interval in seconds.")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    videos = sorted(path for path in video_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    all_rows: list[dict[str, str | int | float]] = []
    for video_path in videos:
        rows = extract_frames(video_path, Path(args.out), args.every_seconds)
        all_rows.extend(rows)
        print(f"{video_path}: {len(rows)} frames")

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["video", "frame_path", "frame_index", "time_s"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} frame records to {manifest_path}")


if __name__ == "__main__":
    main()
