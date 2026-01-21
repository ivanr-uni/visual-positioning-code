from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common import ensure_dir, load_labels_csv, save_json


REQUIRED_COLS = ["filename", "lat", "lon", "alt", "x", "y", "z", "w"]


def summarize_split(split: str, images_dir: Path, labels_csv: Path) -> Dict:
    df = load_labels_csv(labels_csv)

    summary: Dict = {
        "split": split,
        "labels_csv": str(labels_csv),
        "images_dir": str(images_dir),
        "n_rows": int(len(df)),
        "n_unique_filenames": int(df["filename"].nunique()),
        "missing_columns": [],
        "nan_counts": {},
        "numeric_stats": {},
        "missing_images": 0,
        "missing_images_examples": [],
        "duplicate_filenames": int(df["filename"].duplicated().sum()),
    }

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    summary["missing_columns"] = missing_cols

    # NaN counts for existing columns
    for c in [c for c in REQUIRED_COLS if c in df.columns]:
        summary["nan_counts"][c] = int(df[c].isna().sum())

    # Basic stats for numeric cols that exist
    for c in [c for c in REQUIRED_COLS if c in df.columns and c != "filename"]:
        s = pd.to_numeric(df[c], errors="coerce")
        summary["numeric_stats"][c] = {
            "min": float(np.nanmin(s)),
            "max": float(np.nanmax(s)),
            "mean": float(np.nanmean(s)),
            "std": float(np.nanstd(s)),
        }

    # Missing images check (only first 50 examples for speed in huge sets)
    missing = []
    if images_dir.exists():
        for fname in df["filename"].astype(str).tolist():
            if not (images_dir / fname).exists():
                missing.append(fname)
                if len(missing) >= 50:
                    break
    summary["missing_images"] = int(len(missing))
    summary["missing_images_examples"] = missing

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit dataset_v1 structure and CSV consistency.")
    ap.add_argument("--dataset_root", type=str, default="dataset_v1", help="Path to dataset root (dataset_v1).")
    ap.add_argument("--outdir", type=str, default="results/dataset_audit", help="Where to save audit outputs.")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    outdir = ensure_dir(Path(args.outdir))

    splits = ["train", "val", "test"]
    all_summaries: List[Dict] = []
    for split in splits:
        images_dir = dataset_root / "images" / split
        labels_csv = dataset_root / "labels" / split / f"_{split}_annotations.csv"
        if not labels_csv.exists():
            print(f"[skip] Missing CSV for split='{split}': {labels_csv}")
            continue

        print(f"[audit] split={split}")
        summary = summarize_split(split, images_dir, labels_csv)
        all_summaries.append(summary)

    save_json({"dataset_root": str(dataset_root), "splits": all_summaries}, outdir / "dataset_audit_summary.json")

    # Create a convenient table (CSV + XLSX)
    if all_summaries:
        rows = []
        for s in all_summaries:
            row = {
                "split": s["split"],
                "n_rows": s["n_rows"],
                "n_unique_filenames": s["n_unique_filenames"],
                "duplicate_filenames": s["duplicate_filenames"],
                "missing_images": s["missing_images"],
            }
            # add nan counts
            for k, v in s.get("nan_counts", {}).items():
                row[f"nan_{k}"] = v
            rows.append(row)

        df_table = pd.DataFrame(rows)
        df_table.to_csv(outdir / "dataset_audit_table.csv", index=False)
        try:
            df_table.to_excel(outdir / "dataset_audit_table.xlsx", index=False)
        except Exception as e:
            print(f"[warn] Could not write xlsx: {e}")

    print(f"✅ Done. Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
