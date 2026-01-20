from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from common import (
    compute_pos_error,
    compute_regression_metrics,
    ensure_dir,
    load_labels_csv,
    make_pos_error_plots,
    save_json,
)

PARAM_NAMES = ["lat", "lon", "alt"]


def create_model(output_size: int = 3) -> nn.Module:
    # Offline-friendly init; our state_dict will override weights anyway.
    try:
        model = models.resnext50_32x4d(weights=None)
    except TypeError:
        model = models.resnext50_32x4d(pretrained=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, output_size)
    return model


class GeoPoseDataset(Dataset):
    def __init__(self, images_dir: Path, labels_csv: Path, img_size: int = 224, scaler: Optional[StandardScaler] = None):
        self.images_dir = images_dir
        self.df = load_labels_csv(labels_csv)

        required_cols = ["filename"] + PARAM_NAMES
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Labels CSV is missing required columns: {missing}")

        # drop NaNs in target cols
        self.df = self.df.dropna(subset=PARAM_NAMES).reset_index(drop=True)

        self.scaler = scaler
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        fname = str(row["filename"])
        img_path = self.images_dir / fname
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)

        target = np.array([row[c] for c in PARAM_NAMES], dtype=np.float32)

        if self.scaler is not None:
            # scaler expects 2D
            target_scaled = self.scaler.transform(pd.DataFrame([target], columns=PARAM_NAMES))[0].astype(np.float32)
        else:
            target_scaled = target

        return x, torch.tensor(target_scaled, dtype=torch.float32), fname


def fit_scaler(train_csv: Path, val_csv: Path) -> StandardScaler:
    train_df = load_labels_csv(train_csv)[PARAM_NAMES]
    val_df = load_labels_csv(val_csv)[PARAM_NAMES]
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(all_df)
    return scaler


def load_state_dict_safely(weights_path: Path, device: torch.device) -> dict:
    obj = torch.load(weights_path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: Optional[StandardScaler],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    model.eval()

    outs = []
    tgts = []
    fnames = []

    for xb, yb, names in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)

        outs.append(out.detach().cpu().numpy())
        tgts.append(yb.detach().cpu().numpy())
        fnames.extend(list(names))

    y_pred_scaled = np.vstack(outs).astype(np.float32)
    y_true_scaled = np.vstack(tgts).astype(np.float32)

    if scaler is not None:
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_true = scaler.inverse_transform(y_true_scaled)
    else:
        y_pred = y_pred_scaled
        y_true = y_true_scaled

    metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred, names=PARAM_NAMES)

    # pos error from lat/lon
    lat_true = y_true[:, 0]
    lon_true = y_true[:, 1]
    lat_pred = y_pred[:, 0]
    lon_pred = y_pred[:, 1]

    pos_stats = compute_pos_error(lat_true, lon_true, lat_pred, lon_pred)
    metrics["pos_error_2d"] = pos_stats
    metrics["n_samples"] = int(len(fnames))

    # predictions df
    pred_df = pd.DataFrame({
        "filename": fnames,
        "lat": y_pred[:, 0],
        "lon": y_pred[:, 1],
        "alt": y_pred[:, 2],
    })

    pos_error = np.sqrt((lat_pred - lat_true) ** 2 + (lon_pred - lon_true) ** 2)
    report_df = pd.DataFrame({
        "filename": fnames,
        "lat_true": y_true[:, 0],
        "lon_true": y_true[:, 1],
        "alt_true": y_true[:, 2],
        "lat_pred": y_pred[:, 0],
        "lon_pred": y_pred[:, 1],
        "alt_pred": y_pred[:, 2],
        "lat_err": y_pred[:, 0] - y_true[:, 0],
        "lon_err": y_pred[:, 1] - y_true[:, 1],
        "alt_err": y_pred[:, 2] - y_true[:, 2],
        "pos_error_2d": pos_error,
    })

    return metrics, pred_df, report_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ResNeXt50 3D regressor (lat/lon/alt) on dataset_v1 split.")
    ap.add_argument("--dataset_root", type=str, default="dataset_v1")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="results/resnext50_3d")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--no_scaler", action="store_true", help="Disable StandardScaler (only if model was trained without it).")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    images_dir = dataset_root / "images" / args.split
    labels_csv = dataset_root / "labels" / args.split / f"_{args.split}_annotations.csv"

    # scaler from train+val by default, matching main.py logic
    scaler = None
    if not args.no_scaler:
        train_csv = dataset_root / "labels" / "train" / "_train_annotations.csv"
        val_csv = dataset_root / "labels" / "val" / "_val_annotations.csv"
        if not train_csv.exists() or not val_csv.exists():
            raise FileNotFoundError(
                "To use scaler you must have train+val CSV files. "
                f"Missing: {train_csv if not train_csv.exists() else ''} {val_csv if not val_csv.exists() else ''}"
            )
        print("Fitting StandardScaler on train + val labels (lat/lon/alt)...")
        scaler = fit_scaler(train_csv, val_csv)
        print("Scaler fitted.")

    outdir = ensure_dir(Path(args.outdir))
    weights = Path(args.weights)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    model = create_model(output_size=3)
    state = load_state_dict_safely(weights, device)
    model.load_state_dict(state)
    model.to(device)

    ds = GeoPoseDataset(images_dir=images_dir, labels_csv=labels_csv, img_size=args.img_size, scaler=scaler)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    metrics, pred_df, report_df = evaluate(model, dl, device, scaler)

    pred_path = outdir / f"predictions_{args.split}.csv"
    report_path = outdir / f"report_{args.split}.csv"
    pred_df.to_csv(pred_path, index=False)
    report_df.to_csv(report_path, index=False)

    make_pos_error_plots(report_df["pos_error_2d"].to_numpy(dtype=np.float32), outdir, prefix=f"resnext50_3d_{args.split}")

    metrics_out = {
        "model_name": "ResNeXt50_32x4d",
        "task": "3D regression (lat,lon,alt)",
        "split": args.split,
        "weights": str(weights),
        "dataset_root": str(dataset_root),
        "device": str(device),
        "scaler_used": bool(scaler is not None),
        **metrics,
    }
    save_json(metrics_out, outdir / f"metrics_{args.split}.json")

    print(f"✅ Saved predictions: {pred_path}")
    print(f"✅ Saved report: {report_path}")
    print(f"✅ Saved metrics: {outdir / f'metrics_{args.split}.json'}")

    print("\nSummary:")
    print(f"  n_samples: {metrics_out['n_samples']}")
    print(f"  total MAE: {metrics_out['total']['MAE']:.6f}")
    print(f"  pos_error_mean: {metrics_out['pos_error_2d']['mean']:.3f} (same units as lat/lon)")


if __name__ == "__main__":
    main()
