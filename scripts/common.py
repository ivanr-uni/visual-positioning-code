from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    paths.sort(key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")
    return paths


def load_labels_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {path}")
    df = pd.read_csv(path)
    if "filename" not in df.columns:
        raise ValueError("Labels CSV must contain column 'filename'")
    df["filename"] = df["filename"].astype(str)
    return df


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]) -> Dict[str, dict]:
    """Returns MAE/MSE/RMSE per-dimension and totals."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err ** 2

    mae = abs_err.mean(axis=0)
    mse = sq_err.mean(axis=0)
    rmse = np.sqrt(mse)

    out = {
        "per_param": {
            "MAE": {n: float(v) for n, v in zip(names, mae)},
            "MSE": {n: float(v) for n, v in zip(names, mse)},
            "RMSE": {n: float(v) for n, v in zip(names, rmse)},
        },
        "total": {
            "MAE": float(mae.mean()),
            "MSE": float(mse.mean()),
            "RMSE": float(rmse.mean()),
        }
    }
    return out


def compute_pos_error(lat_true: np.ndarray, lon_true: np.ndarray, lat_pred: np.ndarray, lon_pred: np.ndarray) -> Dict[str, float]:
    """Euclidean 2D error. Units are the same as lat/lon units in the dataset (often meters in simulator coords)."""
    dlat = lat_pred - lat_true
    dlon = lon_pred - lon_true
    e = np.sqrt(dlat * dlat + dlon * dlon)

    def pct(p: float) -> float:
        return float(np.percentile(e, p))

    return {
        "mean": float(np.mean(e)),
        "median": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "p90": pct(90.0),
        "p95": pct(95.0),
        "p99": pct(99.0),
        "min": float(np.min(e)),
        "max": float(np.max(e)),
    }


def make_pos_error_plots(pos_error: np.ndarray, outdir: Path, prefix: str = "pos_error") -> None:
    """
    Saves:
      - histogram: <prefix>_hist.png
      - CDF:       <prefix>_cdf.png
    """
    import matplotlib.pyplot as plt

    ensure_dir(outdir)

    # Histogram
    plt.figure()
    plt.hist(pos_error, bins=50)
    plt.xlabel("Position error (same units as lat/lon)")
    plt.ylabel("Count")
    plt.title("Position error histogram")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_hist.png", dpi=150)
    plt.close()

    # CDF
    xs = np.sort(pos_error)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Position error (same units as lat/lon)")
    plt.ylabel("CDF")
    plt.title("Position error CDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_cdf.png", dpi=150)
    plt.close()
