from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def parse_7_floats(txt_path: Path) -> Tuple[float, float, float, float, float, float, float]:
    raw = txt_path.read_text(encoding="utf-8").strip().replace(",", ".")
    parts = [p for p in raw.split() if p]
    if len(parts) < 7:
        raise ValueError(f"Expected 7 floats in {txt_path}, got {len(parts)}: {raw[:200]}")
    vals = list(map(float, parts[:7]))
    return vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]


def build_split(dataset_root: Path, split: str, image_ext: str, overwrite: bool) -> Path:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    out_csv = labels_dir / f"_{split}_annotations.csv"

    if out_csv.exists() and not overwrite:
        print(f"[skip] {out_csv} exists (use --overwrite to rebuild)")
        return out_csv

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels dir: {labels_dir}")

    images = sorted(images_dir.glob(f"*{image_ext}"))
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir} with ext='{image_ext}'")

    rows: List[dict] = []
    missing_txt = 0
    for img_path in images:
        stem = img_path.stem  # img_01234
        txt_path = labels_dir / f"{stem}.txt"
        if not txt_path.exists():
            missing_txt += 1
            continue

        a, b, c, qx, qy, qz, qw = parse_7_floats(txt_path)

        # Схема по умолчанию:
        # первые 3 = lat, lon, alt
        # последние 4 = quaternion (x,y,z,w)
        rows.append(
            {
                "filename": img_path.name,
                "lat": a,
                "lon": b,
                "alt": c,
                "x": qx,
                "y": qy,
                "z": qz,
                "w": qw,
            }
        )

    df = pd.DataFrame(rows, columns=["filename", "lat", "lon", "alt", "x", "y", "z", "w"])
    df.to_csv(out_csv, index=False)

    print(f"[ok] split={split} images={len(images)} rows_written={len(df)} missing_txt={missing_txt} -> {out_csv}")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="dataset_v1")
    ap.add_argument("--image_ext", type=str, default=".png", help="Image extension, e.g. .png or .jpg")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    for split in args.splits:
        build_split(dataset_root, split, args.image_ext, args.overwrite)


if __name__ == "__main__":
    main()
