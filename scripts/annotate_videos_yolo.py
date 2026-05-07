from __future__ import annotations

import argparse
import csv
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def draw_label(frame, label: str, confidence: float, time_s: float):
    import cv2

    annotated = frame.copy()
    text = f"{label}  {confidence:.2f}  t={time_s:.2f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = 24, 40

    cv2.rectangle(
        annotated,
        (x - 10, y - text_h - 12),
        (x + text_w + 10, y + baseline + 10),
        (0, 0, 0),
        -1,
    )
    cv2.putText(annotated, text, (x, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    return annotated


def predict_frame(model, frame):
    result = model.predict(source=frame, verbose=False)[0]
    top1 = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    label = str(result.names[top1])
    return label, confidence


def annotate_video(
    model,
    video_path: Path,
    output_dir: Path,
    csv_dir: Path,
    predict_every: int,
    show: bool = False,
    display_scale: float = 1.0,
) -> dict[str, str | int]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"{video_path.stem}_yolo_annotated.mp4"
    out_csv = csv_dir / f"{video_path.stem}_predictions.csv"

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {out_video}")

    rows: list[dict[str, str | int | float]] = []
    frame_index = 0
    last_label = "unknown"
    last_confidence = 0.0
    window_name = f"YOLO chord demo - {video_path.name}"
    delay_ms = max(1, int(1000 / fps))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        time_s = frame_index / fps
        if frame_index % predict_every == 0:
            last_label, last_confidence = predict_frame(model, frame)

        annotated = draw_label(frame, last_label, last_confidence, time_s)
        writer.write(annotated)

        if show:
            display_frame = annotated
            if display_scale != 1.0:
                display_frame = cv2.resize(annotated, None, fx=display_scale, fy=display_scale)
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break

        rows.append(
            {
                "video": video_path.as_posix(),
                "frame_index": frame_index,
                "time_s": round(time_s, 3),
                "predicted_label": last_label,
                "confidence": round(last_confidence, 6),
            }
        )
        frame_index += 1

    cap.release()
    writer.release()
    if show:
        cv2.destroyWindow(window_name)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=["video", "frame_index", "time_s", "predicted_label", "confidence"])
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    return {
        "input": video_path.as_posix(),
        "output_video": out_video.as_posix(),
        "output_csv": out_csv.as_posix(),
        "frames": frame_index,
        "source_frames": frame_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate videos with YOLO chord-classification predictions.")
    parser.add_argument("--model", default="runs/classify/chord_cls_yolo/weights/best.pt", help="Trained YOLO classification model.")
    parser.add_argument("--video-dir", default="data", help="Folder containing videos.")
    parser.add_argument("--video-path", default=None, help="Optional single video path for a focused demo.")
    parser.add_argument("--out-dir", default="outputs/annotated_videos", help="Output folder for annotated videos.")
    parser.add_argument("--csv-dir", default="outputs/video_predictions", help="Output folder for per-frame CSV files.")
    parser.add_argument("--predict-every", type=int, default=5, help="Run YOLO every N frames and hold the last label between predictions.")
    parser.add_argument("--show", action="store_true", help="Show an OpenCV preview while the video is being annotated. Press q or Esc to stop.")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Scale used only for the live preview window.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics before annotation: pip install ultralytics") from exc

    model = YOLO(args.model)
    if args.video_path:
        videos = [Path(args.video_path)]
    else:
        video_dir = Path(args.video_dir)
        videos = sorted(path for path in video_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise RuntimeError(f"No videos found under {args.video_dir}")

    summary = []
    for video_path in videos:
        result = annotate_video(
            model,
            video_path,
            Path(args.out_dir),
            Path(args.csv_dir),
            max(1, args.predict_every),
            show=args.show,
            display_scale=args.display_scale,
        )
        summary.append(result)
        print(f"{video_path} -> {result['output_video']} ({result['frames']} frames)")

    summary_path = Path(args.csv_dir) / "annotation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input", "output_video", "output_csv", "frames", "source_frames"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
