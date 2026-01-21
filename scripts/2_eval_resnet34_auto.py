from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_COLS_BY_D = {
    2: ["lat", "lon"],
    3: ["lat", "lon", "alt"],
    7: ["lat", "lon", "alt", "x", "y", "z", "w"],
}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError(f"Unsupported checkpoint format in {path}")


def normalize_state_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Handles common prefixes / naming differences:
    # - DataParallel: "module."
    # - Some scripts: "backbone." instead of "base_model."
    new_state: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("backbone."):
            k = "base_model." + k[len("backbone."):]
        new_state[k] = v
    return new_state


def infer_output_dim_from_state(state: Dict[str, torch.Tensor]) -> int:
    if "regressor.3.weight" in state:
        return int(state["regressor.3.weight"].shape[0])

    candidates: List[Tuple[int, str]] = []
    for k, v in state.items():
        if "regressor" in k and k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            parts = k.split(".")
            if len(parts) >= 3 and parts[-1] == "weight":
                try:
                    idx = int(parts[-2])
                    candidates.append((idx, k))
                except Exception:
                    pass
    if candidates:
        candidates.sort(key=lambda x: x[0])
        last_key = candidates[-1][1]
        return int(state[last_key].shape[0])

    raise KeyError("Could not infer output dim from checkpoint.")


class LabeledImages(Dataset):
    def __init__(self, images_dir: Path, df: pd.DataFrame, target_cols: List[str], tfm):
        self.images_dir = images_dir
        self.df = df.reset_index(drop=True)
        self.target_cols = target_cols
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fname = str(row["filename"])
        img_path = self.images_dir / fname
        img = Image.open(img_path).convert("RGB")
        x = self.tfm(img)
        y = torch.tensor([float(row[c]) for c in self.target_cols], dtype=torch.float32)
        return x, y, fname


class ResNet34Regressor(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        try:
            self.base_model = models.resnet34(weights=None)
        except TypeError:
            self.base_model = models.resnet34(pretrained=False)

        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        self.regressor = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.base_model(x)
        return self.regressor(feats)


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def pos_error_stats(lat_true, lon_true, lat_pred, lon_pred) -> Dict[str, float]:
    err = np.sqrt((lat_pred - lat_true) ** 2 + (lon_pred - lon_true) ** 2)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "rmse": rmse,
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--num_params", type=int, default=0, help="Override out_dim. If 0, infer from checkpoint.")
    ap.add_argument("--target_cols", type=str, default="", help="Comma-separated override. Example: lat,lon")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    images_dir = dataset_root / "images" / args.split
    labels_csv = dataset_root / "labels" / args.split / f"_{args.split}_annotations.csv"
    weights = Path(args.weights)
    outdir = ensure_dir(Path(args.outdir))

    print(f"Device: {args.device}")
    print(f"Images:  {images_dir}")
    print(f"Labels:  {labels_csv}")
    print(f"Weights: {weights}")

    df = pd.read_csv(labels_csv)

    state = normalize_state_keys(load_checkpoint_state(weights))
    inferred_dim = infer_output_dim_from_state(state)
    out_dim = args.num_params if args.num_params > 0 else inferred_dim

    if args.target_cols.strip():
        target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()]
    else:
        if out_dim not in DEFAULT_COLS_BY_D:
            raise ValueError(f"Unsupported out_dim={out_dim}. Provide --target_cols explicitly.")
        target_cols = DEFAULT_COLS_BY_D[out_dim]

    for c in ["filename"] + target_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {labels_csv}. Columns={list(df.columns)}")

    print(f"[info] checkpoint out_dim={inferred_dim} -> using out_dim={out_dim}")
    print(f"[info] target_cols={target_cols}")

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    ds = LabeledImages(images_dir, df, target_cols, tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = ResNet34Regressor(out_dim)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    preds_all = []
    fnames_all = []
    truths_all = []

    with torch.no_grad():
        for xb, yb, fnames in dl:
            xb = xb.to(device)
            yhat = model(xb).cpu().numpy()
            preds_all.append(yhat)
            truths_all.append(yb.numpy())
            fnames_all.extend(list(fnames))

    preds = np.concatenate(preds_all, axis=0)
    truths = np.concatenate(truths_all, axis=0)

    # predictions
    pred_df = pd.DataFrame(preds, columns=[f"pred_{c}" for c in target_cols])
    pred_df.insert(0, "filename", fnames_all)
    pred_path = outdir / f"predictions_{args.split}.csv"
    pred_df.to_csv(pred_path, index=False)

    # report
    gt_df = df[["filename"] + target_cols].copy()
    rep = gt_df.merge(pred_df, on="filename", how="inner")
    if len(rep) == 0:
        raise ValueError("No matching filenames between predictions and ground truth.")

    # metrics
    metrics: Dict[str, float] = {}
    for c in target_cols:
        true = rep[c].to_numpy(dtype=float)
        pred = rep[f"pred_{c}"].to_numpy(dtype=float)
        abs_err = np.abs(pred - true)
        sq_err = (pred - true) ** 2

        rep[f"abs_err_{c}"] = abs_err
        rep[f"sq_err_{c}"] = sq_err

        metrics[f"MAE_{c}"] = float(np.mean(abs_err))
        metrics[f"MSE_{c}"] = float(np.mean(sq_err))
        metrics[f"RMSE_{c}"] = float(np.sqrt(np.mean(sq_err)))

    metrics["MAE_mean"] = float(np.mean([metrics[f"MAE_{c}"] for c in target_cols]))
    metrics["RMSE_mean"] = float(np.mean([metrics[f"RMSE_{c}"] for c in target_cols]))
    metrics["n_images"] = int(len(rep))

    # pos error for comparison (always computed if lat/lon present)
    if "lat" in target_cols and "lon" in target_cols:
        lat_true = rep["lat"].to_numpy(dtype=float)
        lon_true = rep["lon"].to_numpy(dtype=float)
        lat_pred = rep["pred_lat"].to_numpy(dtype=float)
        lon_pred = rep["pred_lon"].to_numpy(dtype=float)
        metrics["pos_error_2d_mean"] = float(np.mean(np.sqrt((lat_pred-lat_true)**2 + (lon_pred-lon_true)**2)))
        metrics["pos_error_2d_median"] = float(np.median(np.sqrt((lat_pred-lat_true)**2 + (lon_pred-lon_true)**2)))
        metrics["pos_error_2d_rmse"] = float(np.sqrt(np.mean(((lat_pred-lat_true)**2 + (lon_pred-lon_true)**2))))
        metrics["pos_error_2d_p90"] = float(np.percentile(np.sqrt((lat_pred-lat_true)**2 + (lon_pred-lon_true)**2), 90))
        metrics["pos_error_2d_p95"] = float(np.percentile(np.sqrt((lat_pred-lat_true)**2 + (lon_pred-lon_true)**2), 95))

    rep_path = outdir / f"report_{args.split}.csv"
    rep.to_csv(rep_path, index=False)

    metrics_path = outdir / f"metrics_{args.split}.json"
    save_json(metrics, metrics_path)

    # plots
    for c in target_cols:
        plt.figure()
        plt.scatter(rep[c].to_numpy(dtype=float), rep[f"pred_{c}"].to_numpy(dtype=float), s=3)
        plt.xlabel(f"true {c}")
        plt.ylabel(f"pred {c}")
        plt.title(f"ResNet34 ({out_dim}D) {args.split}: true vs pred ({c})")
        plt.tight_layout()
        plt.savefig(outdir / f"resnet34_{out_dim}d_{args.split}_scatter_{c}.png", dpi=200)
        plt.close()

        plt.figure()
        plt.hist(rep[f"abs_err_{c}"].to_numpy(dtype=float), bins=60)
        plt.xlabel(f"abs error {c}")
        plt.ylabel("count")
        plt.title(f"ResNet34 ({out_dim}D) {args.split}: abs error histogram ({c})")
        plt.tight_layout()
        plt.savefig(outdir / f"resnet34_{out_dim}d_{args.split}_hist_{c}.png", dpi=200)
        plt.close()

    print("✅ Done.")
    print(f"Predictions: {pred_path}")
    print(f"Report:      {rep_path}")
    print(f"Metrics:     {metrics_path}")


if __name__ == "__main__":
    main()
