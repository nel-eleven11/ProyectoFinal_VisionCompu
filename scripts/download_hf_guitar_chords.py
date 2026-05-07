from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

DATASET = "dduka/guitar-chords"
CONFIG = "default"
BASE_URL = "https://datasets-server.huggingface.co"
LABELS = ("A", "Am", "B", "Bm", "C", "Cm", "D", "Dm", "E", "Em", "F", "Fm", "G", "Gm")
SPLIT_DIRS = {"train": "train", "validation": "val", "test": "test"}


def fetch_json(endpoint: str, params: dict[str, str | int], retries: int = 5) -> dict:
    url = f"{BASE_URL}/{endpoint}?{urlencode(params)}"
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError):
            if attempt == retries:
                raise
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"Could not fetch {url}")


def download_file(url: str, out_path: Path, retries: int = 3) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=90) as response:
                tmp_path.write_bytes(response.read())
            tmp_path.replace(out_path)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.5 * attempt)


def split_sizes() -> dict[str, int]:
    payload = fetch_json("size", {"dataset": DATASET})
    return {
        item["split"]: int(item["num_rows"])
        for item in payload["size"]["splits"]
        if item["config"] == CONFIG
    }


def download_split(split: str, out_dir: Path, max_rows: int | None = None, page_size: int = 100) -> list[dict[str, str | int]]:
    sizes = split_sizes()
    total = sizes[split]
    if max_rows is not None:
        total = min(total, max_rows)

    records: list[dict[str, str | int]] = []
    for offset in range(0, total, page_size):
        length = min(page_size, total - offset)
        payload = fetch_json(
            "rows",
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": split,
                "offset": offset,
                "length": length,
            },
        )

        for item in payload["rows"]:
            row_idx = int(item["row_idx"])
            row = item["row"]
            label_id = int(row["label"])
            label = LABELS[label_id]
            src = row["image"]["src"]
            split_dir = SPLIT_DIRS[split]
            image_path = out_dir / split_dir / label / f"{row_idx:06d}.jpg"
            download_file(src, image_path)
            records.append(
                {
                    "dataset": DATASET,
                    "split": split,
                    "row_idx": row_idx,
                    "label_id": label_id,
                    "label": label,
                    "relative_path": image_path.relative_to(out_dir).as_posix(),
                    "width": int(row["image"]["width"]),
                    "height": int(row["image"]["height"]),
                }
            )

        print(f"{split}: {min(offset + length, total)}/{total} images", flush=True)

    return records


def write_manifest(records: list[dict[str, str | int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "split", "row_idx", "label_id", "label", "relative_path", "width", "height"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def ensure_class_dirs(out_dir: Path) -> None:
    for split_dir in SPLIT_DIRS.values():
        for label in LABELS:
            (out_dir / split_dir / label).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dduka/guitar-chords into an Ultralytics classification folder.")
    parser.add_argument("--out", default="data/hf_guitar_chords", help="Output folder.")
    parser.add_argument("--max-per-split", type=int, default=None, help="Optional cap for quick experiments.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_class_dirs(out_dir)
    all_records: list[dict[str, str | int]] = []
    for split in ("train", "validation", "test"):
        all_records.extend(download_split(split, out_dir, max_rows=args.max_per_split))

    write_manifest(all_records, out_dir / "manifest.csv")
    (out_dir / "labels.txt").write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    print(f"Done. Wrote {len(all_records)} images to {out_dir}")


if __name__ == "__main__":
    main()
