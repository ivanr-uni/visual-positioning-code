from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from common import (
    compute_pos_error,
    compute_regression_metrics,
    ensure_dir,
    list_images,
    load_labels_csv,
    make_pos_error_plots,
    save_json,
)


PARAM_NAMES = ["lat", "lon", "alt", "x", "y", "z", "w"]


class ResNet18Regressor(nn.Module):
    def __init__(self, num_params: int = 7):
        super().__init__()
        # Do NOT download weights by default (offline-friendly). We will load our own state_dict.
        try:
            self.base_model = models.resnet18(weights=None)
        except TypeError:
            self.base_model = models.resnet18(pretrained=False)

        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        self.regressor = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_params),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        return self.regressor(features)


class ImageListDataset(Dataset):
    def __init__(self, image_paths: List[Path], img_size: int = 224):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x, p.name


def load_model(weights: Path, device: torch.device) -> nn.Module:
    model = ResNet18Regressor(num_params=len(PARAM_NAMES))
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_folder(model: nn.Module, images_dir: Path, device: torch.device, batch_size: int, num_workers: int, img_size: int) -> pd.DataFrame:
    image_paths = list_images(images_dir)
    ds = ImageListDataset(image_paths, img_size=img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    rows = []
    for xb, fnames in dl:
        xb = xb.to(device)
        out = model(xb).detach().cpu().numpy()
        for fname, pred in zip(fnames, out):
            rows.append([fname] + pred.tolist())

    pred_df = pd.DataFrame(rows, columns=["filename"] + PARAM_NAMES)
    return pred_df


def evaluate(pred_df: pd.DataFrame, labels_csv: Path) -> tuple[dict, pd.DataFrame]:
    true_df = load_labels_csv(labels_csv)

    # Ensure required columns exist in GT
    missing = [c for c in ["filename"] + PARAM_NAMES if c not in true_df.columns]
    if missing:
        raise ValueError(f"GT CSV is missing columns: {missing}")

    merged = pd.merge(
        pred_df,
        true_df[["filename"] + PARAM_NAMES],
        on="filename",
        how="inner",
        suffixes=("_pred", "_true"),
    )
    if len(merged) == 0:
        raise ValueError("No matching filenames between predictions and ground truth.")

    y_true = merged[[f"{c}_true" for c in PARAM_NAMES]].to_numpy(dtype=np.float32)
    y_pred = merged[[f"{c}_pred" for c in PARAM_NAMES]].to_numpy(dtype=np.float32)

    metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred, names=PARAM_NAMES)

    # Position error (2D)
    lat_true = merged["lat_true"].to_numpy(dtype=np.float32)
    lon_true = merged["lon_true"].to_numpy(dtype=np.float32)
    lat_pred = merged["lat_pred"].to_numpy(dtype=np.float32)
    lon_pred = merged["lon_pred"].to_numpy(dtype=np.float32)
    pos_stats = compute_pos_error(lat_true, lon_true, lat_pred, lon_pred)
    metrics["pos_error_2d"] = pos_stats

    # Report
    report = pd.DataFrame({"filename": merged["filename"]})
    for c in PARAM_NAMES:
        report[f"{c}_true"] = merged[f"{c}_true"]
        report[f"{c}_pred"] = merged[f"{c}_pred"]
        report[f"{c}_err"] = merged[f"{c}_pred"] - merged[f"{c}_true"]

    # Add pos error per-frame
    pos_error = np.sqrt((lat_pred - lat_true) ** 2 + (lon_pred - lon_true) ** 2)
    report["pos_error_2d"] = pos_error

    metrics["n_matched"] = int(len(merged))

    return metrics, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ResNet18 7D regressor on dataset_v1 split.")
    ap.add_argument("--dataset_root", type=str, default="dataset_v1", help="Path to dataset root.")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to run.")
    ap.add_argument("--weights", type=str, required=True, help="Path to .pth weights.")
    ap.add_argument("--outdir", type=str, default="results/resnet18_7d", help="Where to save outputs.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="", help="cpu/cuda or empty for auto.")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    images_dir = dataset_root / "images" / args.split
    labels_csv = dataset_root / "labels" / args.split / f"_{args.split}_annotations.csv"

    outdir = ensure_dir(Path(args.outdir))
    weights = Path(args.weights)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_csv}")
    print(f"Weights: {weights}")

    model = load_model(weights, device)

    pred_df = predict_folder(
        model=model,
        images_dir=images_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    pred_path = outdir / f"predictions_{args.split}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"✅ Saved predictions: {pred_path}")

    metrics, report_df = evaluate(pred_df, labels_csv)
    report_path = outdir / f"report_{args.split}.csv"
    report_df.to_csv(report_path, index=False)
    print(f"✅ Saved report: {report_path}")

    # Plots
    make_pos_error_plots(report_df["pos_error_2d"].to_numpy(dtype=np.float32), outdir, prefix=f"resnet18_7d_{args.split}")

    # Metrics JSON
    metrics_out = {
        "model_name": "ResNet18Regressor",
        "task": "7D regression (lat,lon,alt,qx,qy,qz,qw)",
        "split": args.split,
        "weights": str(weights),
        "dataset_root": str(dataset_root),
        "device": str(device),
        **metrics,
    }
    save_json(metrics_out, outdir / f"metrics_{args.split}.json")
    print(f"✅ Saved metrics: {outdir / f'metrics_{args.split}.json'}")

    print("\nSummary:")
    print(f"  matched: {metrics_out['n_matched']}")
    print(f"  total MAE: {metrics_out['total']['MAE']:.6f}")
    print(f"  pos_error_mean: {metrics_out['pos_error_2d']['mean']:.3f} (same units as lat/lon)")


if __name__ == "__main__":
    main()
